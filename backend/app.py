import os
import cv2
import time
import csv
import json
import numpy as np
from datetime import datetime
from threading import Thread, Lock
from flask import Flask, Response, render_template_string, jsonify, redirect, url_for, send_file
from flask_cors import CORS

# Torch (para Ultralytics)
try:
    import torch
    DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
except Exception:
    DEVICE = "cpu"

from ultralytics import YOLO

# ===== Atributos (opcionais) =====
ENABLE_ATTRIBUTES = True        # <<< deixe False p/ máximo FPS; ligue quando precisar
PREFER_INSIGHTFACE = False        # se True e insightface estiver instalado, usa ele; senão tenta DeepFace

HAS_INSIGHT = False
HAS_DEEPFACE = False
INSIGHT_APP = None

if ENABLE_ATTRIBUTES and PREFER_INSIGHTFACE:
    try:
        from insightface.app import FaceAnalysis
        import onnxruntime  # noqa: F401
        HAS_INSIGHT = True
    except Exception:
        HAS_INSIGHT = False

if ENABLE_ATTRIBUTES and not HAS_INSIGHT:
    try:
        from deepface import DeepFace
        HAS_DEEPFACE = True
    except Exception:
        HAS_DEEPFACE = False

# ===== Calibradores (opcionais) =====
HAS_JOBLIB = False
gender_cal = None
agegrp_cal = None
if ENABLE_ATTRIBUTES:
    try:
        import joblib
        HAS_JOBLIB = True
    except Exception:
        HAS_JOBLIB = False

# ===================== Configs de desempenho =====================
FRAME_WIDTH = 1280           # 512 acelera muito no CPU; 640 se sua máquina aguentar
CONF_THRESH = 0.5
DRAW_CONF = True
MAX_DETECTIONS = 30         # ignora “chuva” de detecções
MODEL_NAME = "yolov8n.pt"
JPEG_QUALITY = 72

# Controle de inferência (YOLO) — limite de FPS
TARGET_INFER_FPS = 10       # 8–12 é um bom compromisso em CPU
INFER_INTERVAL = 1.0 / TARGET_INFER_FPS

# Relatórios
LOG_EVERY_SEC = 0.5         # CSV no máx. 2x/s
JSON_EVERY_SEC = 60.0

# Deduplicação de pessoas (janela)
DEDUP_WINDOW_SEC = 600

# Atributos: limites agressivos p/ manter FPS quando ativados
ATTR_SAMPLE_SEC = 8.0       # só tenta atributos a cada N segundos por pessoa
ATTR_MAX_PER_INFER = 1      # no máx. 1 pessoa por inferência terá atributos calculados
MIN_FACE_PIX = 112           # recorte de cabeça mínimo p/ estimar bem
EMA_ALPHA = 0.2             # suavização de idade

# ===================== Estado global =====================
lock = Lock()
running = True
latest_frame = None

class_counts = {}
people_seen_times = {}       # track_id -> last_seen
people_first_seen = {}       # track_id -> first_seen
track_attrs = {}             # track_id -> meta (gender, age, age_group, next_attr_time, age_ema, gender_votes)

last_log_time = 0.0
last_json_dump_ts = time.time()
last_json_counts_snapshot = {}
last_json_period_start = datetime.utcnow()
last_infer_ts = 0.0          # controle da taxa de inferência

# Saída
BASE_DIR = os.path.dirname(__file__)
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
os.makedirs(REPORTS_DIR, exist_ok=True)
REPORT_PATH = os.path.join(REPORTS_DIR, f"detections-{datetime.now().strftime('%Y%m%d-%H%M%S')}.csv")

JSON_DIR = os.path.join(REPORTS_DIR, "summaries")
os.makedirs(JSON_DIR, exist_ok=True)
latest_summary_path = None

# Calibradores (opcionais)
MODELS_DIR = os.path.join(BASE_DIR, "models")
if ENABLE_ATTRIBUTES and HAS_JOBLIB:
    gc_p = os.path.join(MODELS_DIR, "gender_calibrator.pkl")
    ag_p = os.path.join(MODELS_DIR, "agegrp_calibrator.pkl")
    if os.path.exists(gc_p):
        try:
            gender_cal = joblib.load(gc_p)
            print("[calib] gender_calibrator carregado.")
        except Exception as e:
            print(f"[calib] falha ao carregar gender_calibrator.pkl: {e}")
    if os.path.exists(ag_p):
        try:
            agegrp_cal = joblib.load(ag_p)
            print("[calib] agegrp_calibrator carregado.")
        except Exception as e:
            print(f"[calib] falha ao carregar agegrp_calibrator.pkl: {e}")

# ===================== Utilitários =====================
def init_csv(path):
    new_file = not os.path.exists(path)
    f = open(path, "a", newline="", encoding="utf-8")
    w = csv.writer(f)
    if new_file:
        w.writerow(["timestamp_iso", "class", "confidence", "track_id",
                    "gender", "age", "age_group",
                    "x1","y1","x2","y2","frame_w","frame_h"])
        f.flush()
    return f, w

csv_file, csv_writer = init_csv(REPORT_PATH)

def placeholder_frame(size, text):
    W, H = size
    img = np.zeros((H, W, 3), dtype=np.uint8)
    cv2.putText(img, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2, cv2.LINE_AA)
    return img

def safe_put_label(frame, text, x1, y1, font_scale=0.5, thickness=1):
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    tx = max(0, min(x1, frame.shape[1] - tw - 2))
    ty = max(th + 4, y1 - 4)
    cv2.rectangle(frame, (tx - 2, ty - th - 2), (tx + tw + 2, ty + 2), (0,0,0), -1)
    cv2.putText(frame, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255,255,255), thickness, cv2.LINE_AA)

def sort_counts(d): return dict(sorted(d.items(), key=lambda x: (-x[1], x[0])))

def age_to_group(age):
    if age is None: return "unknown"
    a = int(round(age))
    if a <= 12: return "0-12"
    if a <= 17: return "13-17"
    if a <= 24: return "18-24"
    if a <= 34: return "25-34"
    if a <= 44: return "35-44"
    if a <= 54: return "45-54"
    if a <= 64: return "55-64"
    return "65+"

def dump_minute_summary():
    global latest_summary_path, last_json_counts_snapshot, last_json_dump_ts, last_json_period_start
    now_dt = datetime.utcnow()
    with lock:
        cum = class_counts.copy()
        gender_counts = {"male":0,"female":0,"unknown":0}
        age_group_counts = {}
        for tid, attrs in track_attrs.items():
            g = (attrs.get("gender") or "unknown").lower()
            if g not in gender_counts: g = "unknown"
            gender_counts[g] += 1
            ag = attrs.get("age_group") or age_to_group(attrs.get("age"))
            age_group_counts[ag] = age_group_counts.get(ag, 0) + 1

    keys = set(cum.keys()) | set(last_json_counts_snapshot.keys())
    minute_counts = {k: cum.get(k,0) - last_json_counts_snapshot.get(k,0) for k in keys}
    minute_counts = {k:v for k,v in minute_counts.items() if v!=0}

    payload = {
        "generated_at": now_dt.isoformat()+"Z",
        "period_start": last_json_period_start.isoformat()+"Z",
        "period_end": now_dt.isoformat()+"Z",
        "total_cumulative": int(sum(cum.values())),
        "total_minute": int(sum(minute_counts.values())),
        "cumulative_counts": sort_counts(cum),
        "minute_counts": sort_counts(minute_counts),
        "people_attributes_snapshot": {
            "gender_counts": gender_counts,
            "age_group_counts": sort_counts(age_group_counts)
        }
    }
    fname = f"summary-{now_dt.strftime('%Y%m%d-%H%M%S')}.json"
    path = os.path.join(JSON_DIR, fname)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)
    latest_summary_path = path
    last_json_counts_snapshot = cum
    last_json_dump_ts = time.time()
    last_json_period_start = now_dt

# ====== Atributos (implementações) ======
def get_head_crop(frame, x1,y1,x2,y2):
    w = max(1, x2-x1); h = max(1, y2-y1)
    hy2 = y1 + int(h*0.40)
    x1c = max(0, x1); y1c = max(0, y1)
    x2c = min(frame.shape[1], x2); y2c = min(frame.shape[0], hy2)
    return frame[y1c:y2c, x1c:x2c]

def ensure_insightface():
    global INSIGHT_APP
    if INSIGHT_APP is None:
        try:
            from insightface.app import FaceAnalysis
            app = FaceAnalysis(name="buffalo_l", providers=['CPUExecutionProvider'])
            app.prepare(ctx_id=0, det_size=(320, 320))
            INSIGHT_APP = app
            print("[attrs] InsightFace inicializado (CPU).")
        except Exception as e:
            print(f"[attrs] Falha InsightFace: {e}")

def attrs_via_insightface(bgr_crop):
    # retorna (gender, age)
    ensure_insightface()
    if INSIGHT_APP is None:
        return 'unknown', None
    try:
        faces = INSIGHT_APP.get(bgr_crop)
        if not faces:
            return 'unknown', None
        # pega a maior face do crop
        f = max(faces, key=lambda x: (x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]))
        # APIs variam, tentamos 'gender' e 'sex'
        g = getattr(f, "gender", None)
        if g is None: g = getattr(f, "sex", None)
        # convenção: alguns modelos usam 0=woman,1=man
        gender = 'unknown'
        if g is not None:
            gender = 'male' if int(round(float(g))) == 1 else 'female'
        a = getattr(f, "age", None)
        age = int(a) if isinstance(a, (int,float)) else None
        return gender, age
    except Exception:
        return 'unknown', None

def attrs_via_deepface(bgr_crop):
    try:
        from deepface import DeepFace
        rgb = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2RGB)
        res = DeepFace.analyze(img_path=rgb, actions=['gender','age'],
                               detector_backend='retinaface', enforce_detection=False, prog_bar=False)
        data = res[0] if isinstance(res, list) else res
        g = data.get('dominant_gender') or data.get('gender')
        gender = 'unknown'
        if isinstance(g, str):
            gender = 'male' if g.lower().startswith('m') else ('female' if g.lower().startswith('f') else 'unknown')
        a = data.get('age', None)
        age = int(a) if isinstance(a, (int,float)) else None
        return gender, age
    except Exception:
        return 'unknown', None

def estimate_attrs(bgr_crop):
    if not ENABLE_ATTRIBUTES:
        return 'unknown', None, 'unknown'
    if bgr_crop.shape[1] < MIN_FACE_PIX:
        return 'unknown', None, 'unknown'

    # Ordem de preferência
    if HAS_INSIGHT:
        g, a = attrs_via_insightface(bgr_crop)
    elif HAS_DEEPFACE:
        g, a = attrs_via_deepface(bgr_crop)
    else:
        return 'unknown', None, 'unknown'

    return g, a, age_to_group(a)

# ===================== Modelo YOLO =====================
model = YOLO(MODEL_NAME)
model.to(DEVICE)

# ===================== Webcam =====================
def open_camera():
    backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, 0]
    for cam_index in range(0,4):
        for backend in backends:
            cap = cv2.VideoCapture(cam_index, backend)
            if not cap or not cap.isOpened():
                if cap: cap.release()
                continue
            for (w,h) in [(1280,720),(1024,576),(640,480)]:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
                ok, test = cap.read()
                if ok and test is not None:
                    fh, fw = test.shape[:2]
                    scale = FRAME_WIDTH / float(fw)
                    target_size = (FRAME_WIDTH, int(fh*scale))
                    print(f"[camera] index={cam_index} backend={backend} capt={fw}x{fh} -> proc={target_size}")
                    return cap, target_size
            ok, test = cap.read()
            if ok and test is not None:
                fh, fw = test.shape[:2]
                scale = FRAME_WIDTH / float(fw)
                target_size = (FRAME_WIDTH, int(fh*scale))
                print(f"[camera] index={cam_index} backend={backend} capt={fw}x{fh} -> proc={target_size}")
                return cap, target_size
            cap.release()
    return None, None

cap, target_size = open_camera()
if cap is None:
    print("[camera] Não consegui abrir a webcam. Feche apps que a estejam usando e tente novamente.")

# ===================== Loop principal =====================
def processing_loop():
    global latest_frame, last_log_time, last_json_dump_ts, last_infer_ts, running

    if cap is None:
        while running:
            with lock:
                latest_frame = placeholder_frame((FRAME_WIDTH, 360), "Sem webcam: verifique conexao/permicoes")
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
            if (time.time() - last_json_dump_ts) >= JSON_EVERY_SEC:
                dump_minute_summary()
            time.sleep(0.01)
            continue

        frame = cv2.resize(frame, target_size)
        now = time.time()

        # throttle de inferência YOLO
        do_infer = (now - last_infer_ts) >= INFER_INTERVAL

        if do_infer:
            # ===== YOLO + ByteTrack =====
            results = model.track(
                source=frame,
                imgsz=FRAME_WIDTH,
                conf=CONF_THRESH,
                max_det=MAX_DETECTIONS,
                classes=[0],                 # apenas "person"
                tracker="bytetrack.yaml",
                persist=True,
                device=DEVICE,
                verbose=False
            )
            last_infer_ts = now

            detections = []
            if len(results):
                r = results[0]
                boxes = r.boxes
                ids = getattr(boxes, "id", None)
                # limita o nº de pessoas para atributos por inferência
                attrs_left = ATTR_MAX_PER_INFER

                for i in range(len(boxes)):
                    x1, y1, x2, y2 = map(int, boxes.xyxy[i].tolist())
                    conf = float(boxes.conf[i].item()) if boxes.conf is not None else 0.0
                    track_id = int(ids[i].item()) if (ids is not None and ids[i] is not None) else -1

                    # Desenha bbox
                    cv2.rectangle(frame, (x1,y1), (x2,y2), (0,230,0), 2)

                    # Dedup
                    people_seen_times[track_id] = now
                    if track_id not in people_first_seen:
                        people_first_seen[track_id] = now
                        class_counts["person_unique"] = class_counts.get("person_unique", 0) + 1

                    # Atributos (opcional) — limitador agressivo
                    gtxt, atxt, agroup = "unknown", None, "unknown"
                    meta = track_attrs.get(track_id) or {}
                    next_attr_time = meta.get("next_attr_time", 0.0)

                    if ENABLE_ATTRIBUTES and attrs_left > 0 and now >= next_attr_time:
                        crop = get_head_crop(frame, x1, y1, x2, y2)
                        if crop.size > 0 and crop.shape[1] >= MIN_FACE_PIX:
                            g, a, ag = estimate_attrs(crop)
                            # EMA idade
                            prev_ema = meta.get("age_ema")
                            if a is not None:
                                meta["age_ema"] = a if prev_ema is None else (EMA_ALPHA*a + (1-EMA_ALPHA)*prev_ema)
                                meta["age"] = int(round(meta["age_ema"]))
                            # votos de gênero
                            gv = meta.get("gender_votes", {"male":0,"female":0,"unknown":0})
                            gv[g] = gv.get(g, 0) + 1
                            meta["gender_votes"] = gv
                            meta["gender"] = max(gv, key=gv.get)
                            # faixa etária
                            meta["age_group"] = ag if isinstance(ag, str) else age_to_group(meta.get("age"))
                            attrs_left -= 1

                        meta["next_attr_time"] = now + ATTR_SAMPLE_SEC
                        track_attrs[track_id] = meta

                    # valores atuais (podem vir de execuções passadas)
                    gtxt = (track_attrs.get(track_id, {}).get("gender") or "unknown")
                    a_val = track_attrs.get(track_id, {}).get("age")
                    atxt = a_val if a_val is not None else None
                    agroup = track_attrs.get(track_id, {}).get("age_group") or age_to_group(a_val)

                    # Rótulo
                    conf_txt = f"{conf:.2f}" if DRAW_CONF else ""
                    label = f"ID {track_id if track_id>=0 else '-'}"
                    if conf_txt: label += f" | {conf_txt}"
                    label += f" | {gtxt[:1].upper() if isinstance(gtxt,str) else '?'} | {agroup}"
                    font_scale = max(0.45, FRAME_WIDTH / 1600.0)
                    safe_put_label(frame, label, x1, y1, font_scale=font_scale, thickness=1)

                    detections.append(("person", conf, track_id, x1, y1, x2, y2, gtxt, atxt, agroup))

            # Limpa IDs muito antigos (fora da janela de dedup)
            stale = [tid for tid, last in people_seen_times.items() if now - last > DEDUP_WINDOW_SEC]
            for tid in stale:
                people_seen_times.pop(tid, None)
                people_first_seen.pop(tid, None)
                # track_attrs mantém snapshot; remova se quiser
                # track_attrs.pop(tid, None)

            # Logging CSV limitado
            if detections and (now - last_log_time) >= LOG_EVERY_SEC:
                ts = datetime.utcnow().isoformat()
                with lock:
                    for (cls, conf, tid, x1,y1,x2,y2, gtxt, atxt, agroup) in detections:
                        class_counts[cls] = class_counts.get(cls, 0) + 1
                        class_counts[f"person_gender_{(gtxt or 'unknown').lower()}"] = \
                            class_counts.get(f"person_gender_{(gtxt or 'unknown').lower()}", 0) + 1
                        class_counts[f"person_age_{agroup}"] = \
                            class_counts.get(f"person_age_{agroup}", 0) + 1
                        csv_writer.writerow([ts, cls, f"{conf:.4f}", tid,
                                             (gtxt or "unknown").lower(),
                                             (atxt if atxt is not None else ""),
                                             agroup, x1, y1, x2, y2, frame.shape[1], frame.shape[0]])
                    csv_file.flush()
                    last_log_time = now

            # Dump JSON por minuto
            if (now - last_json_dump_ts) >= JSON_EVERY_SEC:
                dump_minute_summary()

            # atualiza frame anotado
            with lock:
                latest_frame = frame

        else:
            # Sem nova inferência: apenas publica o último frame anotado
            with lock:
                if latest_frame is None:
                    latest_frame = frame  # primeiro preenchimento
            # ainda assim podemos disparar JSON por minuto
            if (now - last_json_dump_ts) >= JSON_EVERY_SEC:
                dump_minute_summary()
            time.sleep(0.002)

# ===================== MJPEG Streaming =====================
def mjpeg_generator():
    while True:
        with lock:
            frame = latest_frame
        if frame is None:
            frame = placeholder_frame((FRAME_WIDTH, 360), "Inicializando stream...")
        ok, jpg = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
        if ok:
            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" +
                   jpg.tobytes() + b"\r\n")
        time.sleep(0.02)

# ===================== Flask =====================
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
    <strong>Retail Vision AI</strong> — foco em FPS (atributos opcionais)
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
        <summary>Perfis</summary>
        <pre>FRAME_WIDTH={{fw}} | TARGET_INFER_FPS={{fps}} | MAX_DETECTIONS={{maxdet}}
ENABLE_ATTRIBUTES={{attrs}} | PREFER_INSIGHTFACE={{insight}}</pre>
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
    return render_template_string(
        PANEL_HTML,
        report_name=os.path.basename(REPORT_PATH),
        latest_json_name=latest_name,
        fw=FRAME_WIDTH, fps=TARGET_INFER_FPS, maxdet=MAX_DETECTIONS,
        attrs=ENABLE_ATTRIBUTES, insight=PREFER_INSIGHTFACE
    )

@app.route("/video_feed")
def video_feed():
    return Response(mjpeg_generator(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/summary")
def summary():
    with lock:
        return jsonify(sort_counts(class_counts))

@app.route("/report")
def report():
    return send_file(REPORT_PATH, mimetype="text/csv", as_attachment=True,
                     download_name=os.path.basename(REPORT_PATH))

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
    if ENABLE_ATTRIBUTES:
        if HAS_INSIGHT:
            print("[attrs] InsightFace preferido (CPU).")
        elif HAS_DEEPFACE:
            print("[attrs] DeepFace em fallback (CPU).")
        else:
            print("[attrs] Nenhum backend de atributos disponível; retornará 'unknown'.")
    t = Thread(target=processing_loop, daemon=True)
    t.start()
    try:
        app.run(host="0.0.0.0", port=5000, threaded=True)
    finally:
        global running
        running = False
        time.sleep(0.2)
        try:
            if cap: cap.release()
        except Exception:
            pass
        cv2.destroyAllWindows()
        try:
            csv_file.close()
        except Exception:
            pass

if __name__ == "__main__":
    main()
