import cv2
import time
import numpy as np
import torch
from threading import Thread, Lock
from flask import Flask, Response, render_template_string, jsonify, redirect, url_for
from flask_cors import CORS
from ultralytics import YOLO

# ========= Configs =========
FRAME_WIDTH = 640
CONF_THRESH = 0.5
DECAY = 0.96
KERNEL_RADIUS = 24
ALPHA_OVERLAY = 0.45
MODEL_NAME = "yolov8n.pt"

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

lock = Lock()
latest_overlay = None
latest_heat_only = None
running = True

model = YOLO(MODEL_NAME)
model.to(DEVICE)

# ========= Abrir webcam (robusto no Windows) =========
def open_camera():
    """
    Tenta abrir a webcam testando índices 0..3 e backends típicos do Windows.
    Retorna (cap, target_size) onde cap é o VideoCapture já configurado.
    """
    backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, 0]  # 0 = default
    for cam_index in range(0, 4):
        for backend in backends:
            cap = cv2.VideoCapture(cam_index, backend)
            if not cap or not cap.isOpened():
                if cap: cap.release()
                continue
            # tenta ler 1 frame
            ok, frame = cap.read()
            if not ok or frame is None:
                cap.release()
                continue
            # ajusta resolução de trabalho
            h, w = frame.shape[:2]
            scale = FRAME_WIDTH / float(w)
            target_size = (FRAME_WIDTH, int(h * scale))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, target_size[0])
            # altura em muitos drivers é só informativa; redimensionaremos manualmente
            print(f"[camera] OK em index={cam_index} backend={backend} -> {target_size}")
            return cap, target_size
    return None, None

cap, target_size = open_camera()
# for (w, h) in [(1920, 1080), (1280, 720), (1024, 576), (640, 480)]:
#     cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
#     cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
#     ok, test = cap.read()
#     if ok and test is not None and test.shape[1] >= w-8 and test.shape[0] >= h-8:
#         frame_h, frame_w = test.shape[:2]
#         scale = FRAME_WIDTH / float(frame_w)  # FRAME_WIDTH é só a largura de processamento
#         target_size = (FRAME_WIDTH, int(frame_h * scale))
#         print(f"[camera] usando {frame_w}x{frame_h} -> processando em {target_size}")
#         break

if cap is None:
    print("[camera] Não consegui abrir a webcam (indices 0..3). Feche apps que usam a câmera e tente novamente.")

# ========= Heatmap =========
heatmap_accum = None

def _ensure_heatmap(shape):
    global heatmap_accum
    if heatmap_accum is None or heatmap_accum.shape[:2] != shape[:2]:
        heatmap_accum = np.zeros((shape[0], shape[1]), dtype=np.float32)
    return heatmap_accum

def add_detection_to_heatmap(accum, center_xy):
    cv2.circle(accum, center_xy, KERNEL_RADIUS, 1.0, -1)
    

def make_heat_overlay(frame, accum):
    hm = accum.copy()
    mx = hm.max()
    if mx > 0:
        hm = (hm / mx) * 255.0
    hm = hm.astype(np.uint8)
    hm_color = cv2.applyColorMap(hm, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(frame, 1.0 - ALPHA_OVERLAY, hm_color, ALPHA_OVERLAY, 0)
    return overlay, hm_color

def placeholder_frame(size, text):
    W, H = size
    img = np.zeros((H, W, 3), dtype=np.uint8)
    cv2.putText(img, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2, cv2.LINE_AA)
    return img

# ========= Loop de processamento =========
def processing_loop():
    global latest_overlay, latest_heat_only, running

    if cap is None:
        # Mantém o servidor no ar, com placeholder
        while running:
            ph = placeholder_frame((FRAME_WIDTH, 360), "Sem webcam: verifique conexao/permicoes")
            with lock:
                latest_overlay = ph
                latest_heat_only = ph
            time.sleep(0.5)
        return

    # Primeira leitura e inicializações
    ok, frame = cap.read()
    if not ok or frame is None:
        print("[camera] Falha na leitura inicial.")
        running = False
        return

    frame = cv2.resize(frame, target_size)
    H, W = frame.shape[:2]
    _ensure_heatmap((H, W, 3))

    while running:
        ok, frame = cap.read()
        if not ok or frame is None:
            time.sleep(0.02)
            continue

        frame = cv2.resize(frame, target_size)

        results = model.predict(
            source=frame,
            imgsz=FRAME_WIDTH,
            conf=CONF_THRESH,
            classes=[0],
            device=DEVICE,
            verbose=False
        )

        heatmap_accum[:] = heatmap_accum * DECAY

        if len(results):
            r = results[0]
            if r.boxes is not None and len(r.boxes) > 0:
                for b in r.boxes:
                    cls_id = int(b.cls[0].item()) if b.cls is not None else -1
                    if cls_id != 0:
                        continue
                    x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 230, 0), 2)
                    cx = (x1 + x2) // 2
                    cy = (y1 + y2) // 2
                    add_detection_to_heatmap(heatmap_accum, (cx, cy))

        overlay, heat_color = make_heat_overlay(frame, heatmap_accum)

        with lock:
            latest_overlay = overlay
            latest_heat_only = heat_color

        time.sleep(0.01)

# ========= MJPEG Streaming =========
def mjpeg_generator(which="overlay"):
    while True:
        with lock:
            frame = latest_overlay if which == "overlay" else latest_heat_only
        if frame is None:
            frame = placeholder_frame((FRAME_WIDTH, 360), "Inicializando stream...")
        ok, jpg = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if ok:
            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" +
                   jpg.tobytes() + b"\r\n")
        time.sleep(0.03)

# ========= Flask App =========
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
    main { display:grid; gap:12px; padding:12px; }
    .card { background:#181818; border-radius:12px; padding:12px; }
    img { width:100%; height:auto; border-radius:8px; display:block; }
    @media (min-width: 900px){ main { grid-template-columns: 1fr 1fr; } }
  </style>
</head>
<body>
  <header>
    <div><strong>Retail Vision AI</strong> — visão computacional + heatmap (tempo real)</div>
  </header>
  <main>
    <section class="card">
      <h3>Detecções + Heatmap</h3>
      <img src="/video_feed" alt="video stream" />
    </section>
    <section class="card">
      <h3>Heatmap (apenas)</h3>
      <img src="/heatmap_feed" alt="heatmap stream" />
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
    return render_template_string(PANEL_HTML)

@app.route("/video_feed")
def video_feed():
    return Response(mjpeg_generator("overlay"),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/heatmap_feed")
def heatmap_feed():
    return Response(mjpeg_generator("heat"),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

def main():
    print("[flask] rotas registradas:")
    for r in app.url_map.iter_rules():
        print(f"  {r.methods} {r}")
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

if __name__ == "__main__":
    main()
