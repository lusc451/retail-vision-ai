import cv2
import time
import csv
import os
import json
import numpy as np
import torch
from datetime import datetime
from threading import Thread, Lock
from flask import Flask, Response, render_template_string, jsonify, redirect, url_for, send_file
from flask_cors import CORS
from ultralytics import YOLO

# ===================== Configurações =====================
FRAME_WIDTH = 640          # resolução de processamento (ajuste p/ desempenho)
CONF_THRESH = 0.5          # confiança mínima das detecções
DRAW_CONF = True           # mostra confiança no rótulo
LOG_EVERY_SEC = 0.5        # registra no CSV no máx. 2x por segundo
MODEL_NAME = "yolov8n.pt"  # modelo leve; cobre pessoas e objetos COCO
JSON_EVERY_SEC = 60.0      # salva resumo JSON a cada 60s

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# ===================== Estado global =====================
lock = Lock()
running = True
latest_frame = None            # último frame com caixas/rótulos
class_counts = {}              # contagem acumulada por classe (desde o start)
last_log_time = 0.0            # controle de frequência de log

# CSV de relatório (salvo na pasta backend/reports)
BASE_DIR = os.path.dirname(__file__)
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
os.makedirs(REPORTS_DIR, exist_ok=True)
REPORT_PATH = os.path.join(REPORTS_DIR, f"detections-{datetime.now().strftime('%Y%m%d-%H%M%S')}.csv")

# JSON (resumos por minuto)
JSON_DIR = os.path.join(REPORTS_DIR, "summaries")
os.makedirs(JSON_DIR, exist_ok=True)
latest_summary_path = None
last_json_dump_ts = time.time()
last_json_counts_snapshot = {}         # snapshot anterior para calcular delta (minute_counts)
last_json_period_start = datetime.utcnow()

# ===================== Utilitários =====================
def init_csv(path):
    new_file = not os.path.exists(path)
    f = open(path, "a", newline="", encoding="utf-8")
    writer = csv.writer(f)
    if new_file:
        writer.writerow(["timestamp_iso", "class", "confidence", "x1", "y1", "x2", "y2", "frame_w", "frame_h"])
        f.flush()
    return f, writer

csv_file, csv_writer = init_csv(REPORT_PATH)

def safe_put_label(frame, text, x1, y1, font_scale=0.5, thickness=1):
    (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    tx = max(0, min(x1, frame.shape[1] - tw - 2))
    ty = max(th + 4, y1 - 4)
    cv2.rectangle(frame, (tx - 2, ty - th - 2), (tx + tw + 2, ty + 2), (0, 0, 0), -1)
    cv2.putText(frame, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

def placeholder_frame(size, text):
    W, H = size
    img = np.zeros((H, W, 3), dtype=np.uint8)
    cv2.putText(img, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2, cv2.LINE_AA)
    return img

def sort_counts(d):
    return dict(sorted(d.items(), key=lambda x: (-x[1], x[0])))

def dump_minute_summary():
    """Gera um JSON com cumulative_counts e minute_counts (delta do último minuto)."""
    global latest_summary_path, last_json_counts_snapshot, last_json_dump_ts, last_json_period_start

    now_dt = datetime.utcnow()
    with lock:
        current_counts = class_counts.copy()

    # deltas do período
    keys = set(current_counts.keys()) | set(last_json_counts_snapshot.keys())
    minute_counts = {k: current_counts.get(k, 0) - last_json_counts_snapshot.get(k, 0) for k in keys}
    # remove zeros
    minute_counts = {k: v for k, v in minute_counts.items() if v != 0}

    payload = {
        "generated_at": now_dt.isoformat() + "Z",
        "period_start": last_json_period_start.isoformat() + "Z",
        "period_end": now_dt.isoformat() + "Z",
        "total_cumulative": int(sum(current_counts.values())),
        "total_minute": int(sum(minute_counts.values())),
        "cumulative_counts": sort_counts(current_counts),
        "minute_counts": sort_counts(minute_counts),
    }

    fname = f"summary-{now_dt.strftime('%Y%m%d-%H%M%S')}.json"
    path = os.path.join(JSON_DIR, fname)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)

    latest_summary_path = path
    last_json_counts_snapshot = current_counts
    last_json_dump_ts = time.time()
    last_json_period_start = now_dt

# ===================== Modelo YOLO =====================
model = YOLO(MODEL_NAME)
model.to(DEVICE)

# ===================== Webcam (abertura robusta) =====================
def open_camera():
    backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, 0]  # 0 = default
    for cam_index in range(0, 4):
        for backend in backends:
            cap = cv2.VideoCapture(cam_index, backend)
            if not cap or not cap.isOpened():
                if cap: cap.release()
                continue

            for (w, h) in [(1280, 720), (1024, 576), (640, 480)]:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
                ok, test = cap.read()
                if ok and test is not None:
                    fh, fw = test.shape[:2]
                    scale = FRAME_WIDTH / float(fw)
                    target_size = (FRAME_WIDTH, int(fh * scale))
                    print(f"[camera] index={cam_index} backend={backend} capt={fw}x{fh} -> proc={target_size}")
                    return cap, target_size

            ok, test = cap.read()
            if ok and test is not None:
                fh, fw = test.shape[:2]
                scale = FRAME_WIDTH / float(fw)
                target_size = (FRAME_WIDTH, int(fh * scale))
                print(f"[camera] index={cam_index} backend={backend} capt={fw}x{fh} -> proc={target_size}")
                return cap, target_size

            cap.release()
    return None, None

cap, target_size = open_camera()
if cap is None:
    print("[camera] Não consegui abrir a webcam. Feche apps que a estejam usando e tente novamente.")

# ===================== Loop de processamento =====================
def processing_loop():
    global latest_frame, class_counts, last_log_time, running

    if cap is None:
        while running:
            with lock:
                latest_frame = placeholder_frame((FRAME_WIDTH, 360), "Sem webcam: verifique conexao/permicoes")
            # ainda assim gera JSON por minuto (vazio)
            if (time.time() - last_json_dump_ts) >= JSON_EVERY_SEC:
                dump_minute_summary()
            time.sleep(0.5)
        return

    ok, frame = cap.read()
    if not ok or frame is None:
        print("[camera] Falha na leitura inicial.")
        running = False
        return

    frame = cv2.resize(frame, target_size)

    while running:
        ok, frame = cap.read()
        if not ok or frame is None:
            # mesmo sem frame novo, mantemos o schedule do JSON
            if (time.time() - last_json_dump_ts) >= JSON_EVERY_SEC:
                dump_minute_summary()
            time.sleep(0.02)
            continue

        frame = cv2.resize(frame, target_size)
        H, W = frame.shape[:2]

        results = model.predict(
            source=frame,
            imgsz=FRAME_WIDTH,
            conf=CONF_THRESH,
            device=DEVICE,
            verbose=False
        )

        detections = []
        if len(results):
            r = results[0]
            names = r.names if hasattr(r, "names") else model.names
            if r.boxes is not None and len(r.boxes) > 0:
                for b in r.boxes:
                    cls_id = int(b.cls[0].item()) if b.cls is not None else -1
                    conf = float(b.conf[0].item()) if b.conf is not None else 0.0
                    x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 230, 0), 2)
                    label = names.get(cls_id, str(cls_id)) if isinstance(names, dict) else (names[cls_id] if 0 <= cls_id < len(names) else str(cls_id))
                    text = f"{label} {conf:.2f}" if DRAW_CONF else label
                    font_scale = max(0.45, FRAME_WIDTH / 1600.0)
                    safe_put_label(frame, text, x1, y1, font_scale=font_scale, thickness=1)

                    detections.append((label, conf, x1, y1, x2, y2))

        now = time.time()
        if detections and (now - last_log_time) >= LOG_EVERY_SEC:
            timestamp = datetime.utcnow().isoformat()
            with lock:
                for (label, conf, x1, y1, x2, y2) in detections:
                    class_counts[label] = class_counts.get(label, 0) + 1
                    csv_writer.writerow([timestamp, label, f"{conf:.4f}", x1, y1, x2, y2, W, H])
                csv_file.flush()
                last_log_time = now

        # dump JSON por minuto (mesmo se não teve detecção)
        if (now - last_json_dump_ts) >= JSON_EVERY_SEC:
            dump_minute_summary()

        with lock:
            latest_frame = frame

        time.sleep(0.01)

# ===================== MJPEG Streaming =====================
def mjpeg_generator():
    while True:
        with lock:
            frame = latest_frame
        if frame is None:
            frame = placeholder_frame((FRAME_WIDTH, 360), "Inicializando stream...")
        ok, jpg = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if ok:
            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" +
                   jpg.tobytes() + b"\r\n")
        time.sleep(0.03)

# ===================== Flask App =====================
app = Flask(__name__)
CORS(app)

PANEL_HTML = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Retail Vision AI - Painel</title>
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <style>
    body { margin:0; font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; background:#111; color:#eee; }
    header { padding:12px 16px; background:#1a1a1a; position:sticky; top:0; }
    main { padding:12px; }
    .card { background:#181818; border-radius:12px; padding:12px; }
    img { width:100%; height:auto; border-radius:8px; display:block; }
    .row { display:grid; gap:12px; }
    @media (min-width: 900px){ .row { grid-template-columns: 1fr 320px; } }
    pre { background:#0f0f0f; padding:12px; border-radius:8px; overflow:auto; }
    a { color:#8ab4ff; }
    code { background:#0f0f0f; padding:2px 6px; border-radius:6px; }
  </style>
</head>
<body>
  <header>
    <strong>Retail Vision AI</strong> — detecção ao vivo + relatórios
  </header>
  <main class="row">
    <section class="card">
      <h3>Stream</h3>
      <img src="/video_feed" alt="video stream" />
    </section>
    <section class="card">
      <h3>Relatórios &amp; APIs</h3>
      <p><a href="/report">Baixar relatório CSV</a></p>
      <p><a href="/summary" target="_blank">Ver resumo (JSON ao vivo)</a></p>
      <p><a href="/report_json_latest">Baixar <em>último JSON por minuto</em></a></p>
      <p style="opacity:.85">CSV atual: <code>{{report_name}}</code><br/>Último JSON: <code>{{latest_json_name}}</code></p>
      <details>
        <summary>Como usar</summary>
        <pre>/video_feed          → stream MJPEG
/report              → CSV das detecções (timestamp, classe, confiança, bbox)
/summary             → contagem acumulada por classe (ao vivo)
/report_json_latest  → baixa o último JSON gerado no ciclo de 60s</pre>
      </details>
    </section>
  </main>
</body>
</html>
"""

@app.route("/")
def index():
    return redirect(url_for("panel"))

@app.route("/health")
def health():
    return jsonify(status="ok")

@app.route("/panel")
def panel():
    latest_name = os.path.basename(latest_summary_path) if latest_summary_path else "—"
    return render_template_string(PANEL_HTML,
                                  report_name=os.path.basename(REPORT_PATH),
                                  latest_json_name=latest_name)

@app.route("/video_feed")
def video_feed():
    return Response(mjpeg_generator(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/summary")
def summary():
    with lock:
        return jsonify(sort_counts(class_counts))

@app.route("/report")
def report():
    return send_file(REPORT_PATH, mimetype="text/csv", as_attachment=True, download_name=os.path.basename(REPORT_PATH))

@app.route("/report_json_latest")
def report_json_latest():
    if latest_summary_path and os.path.exists(latest_summary_path):
        return send_file(latest_summary_path, mimetype="application/json", as_attachment=True,
                         download_name=os.path.basename(latest_summary_path))
    return jsonify(error="Nenhum resumo JSON gerado ainda. Aguarde até o próximo minuto."), 404

def main():
    print(f"[device] usando: {DEVICE}")
    print(f"[report] CSV em: {REPORT_PATH}")
    print(f"[report] JSON dir: {JSON_DIR}")

    t = Thread(target=processing_loop, daemon=True)
    t.start()
    try:
        app.run(host="0.0.0.0", port=5000, threaded=True)
    finally:
        global running
        running = False
        time.sleep(0.2)
        if cap: cap.release()
        cv2.destroyAllWindows()
        try:
            csv_file.close()
        except Exception:
            pass

if __name__ == "__main__":
    main()
