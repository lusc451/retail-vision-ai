import cv2
import time
import numpy as np
from threading import Thread, Lock
from flask import Flask, Response, render_template_string
from flask_cors import CORS
from ultralytics import YOLO

# ========= Configs =========
CAM_INDEX = 0          # webcam padrão
FRAME_WIDTH = 640      # largura de processamento (mantém desempenho)
CONF_THRESH = 0.5      # confiança mínima
DECAY = 0.96           # 0.96~0.99: decaimento do heatmap por frame
KERNEL_RADIUS = 24     # “tinta” por detecção (raio do círculo no heatmap)
ALPHA_OVERLAY = 0.45   # mistura heatmap x frame
MODEL_NAME = "yolov8n.pt"  # leve e rápido

# ========= Estado global seguro =========
lock = Lock()
latest_overlay = None    # frame com caixas + heatmap
latest_heat_only = None  # imagem do heatmap colorido
running = True

# ========= Inicializa modelo =========
model = YOLO(MODEL_NAME)  # baixa pesos na 1ª vez

# ========= Captura da câmera =========
cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_DSHOW)  # CAP_DSHOW (Windows) evita travas
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
# altura será ajustada pelo OpenCV; pegamos após a primeira leitura

# ========= Heatmap =========
heatmap_accum = None  # inicializa após 1º frame

def _ensure_heatmap(shape):
    global heatmap_accum
    if heatmap_accum is None or heatmap_accum.shape[:2] != shape[:2]:
        # matriz float32 para acumular intensidades
        return np.zeros((shape[0], shape[1]), dtype=np.float32)
    return heatmap_accum

def add_detection_to_heatmap(accum, center_xy):
    # pinta um círculo no acumulador (mais barato que convolução por kernel)
    cv2.circle(accum, center_xy, KERNEL_RADIUS, 1.0, -1)

def make_heat_overlay(frame, accum):
    # normaliza e colore
    hm = accum.copy()
    maxv = hm.max()
    if maxv > 0:
        hm = (hm / maxv) * 255.0
    hm = hm.astype(np.uint8)
    hm_color = cv2.applyColorMap(hm, cv2.COLORMAP_JET)
    # mistura com a imagem original
    overlay = cv2.addWeighted(frame, 1.0 - ALPHA_OVERLAY, hm_color, ALPHA_OVERLAY, 0)
    return overlay, hm_color

# ========= Loop de processamento =========
def processing_loop():
    global latest_overlay, latest_heat_only, heatmap_accum, running

    # descobrimos a altura real após a 1ª leitura
    ret, frame = cap.read()
    if not ret:
        print("Não consegui acessar a webcam. Feche outros apps que a estejam usando.")
        running = False
        return

    # redimensiona para a largura-alvo
    h, w = frame.shape[:2]
    scale = FRAME_WIDTH / float(w)
    target_size = (FRAME_WIDTH, int(h * scale))

    heatmap_accum = _ensure_heatmap((target_size[1], target_size[0], 3))

    while running:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.01)
            continue

        frame = cv2.resize(frame, target_size)
        H, W = frame.shape[:2]

        # YOLO: só "person" (classe 0 no COCO)
        # Dica: aumentar imgsz=640 para melhor precisão, custo ↑
        results = model.predict(
            source=frame,
            imgsz=FRAME_WIDTH,
            conf=CONF_THRESH,
            classes=[0],            # só pessoas
            verbose=False,
            device=0 if hasattr(model, "device") else None  # usa GPU se disponível
        )

        # aplica decaimento “tempo-real”
        heatmap_accum *= DECAY

        # desenha caixas e alimenta heatmap
        if len(results):
            r = results[0]
            if r.boxes is not None and len(r.boxes) > 0:
                for b in r.boxes:
                    cls_id = int(b.cls[0].item()) if b.cls is not None else -1
                    if cls_id != 0:
                        continue
                    x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
                    # bounding box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 230, 0), 2)
                    # centro para o heatmap
                    cx = (x1 + x2) // 2
                    cy = (y1 + y2) // 2
                    add_detection_to_heatmap(heatmap_accum, (cx, cy))

        # gera overlay e heat-only
        overlay, heat_color = make_heat_overlay(frame, heatmap_accum)

        with lock:
            latest_overlay = overlay
            latest_heat_only = heat_color

        # dá uma folguinha à CPU
        time.sleep(0.01)

# ========= MJPEG Streaming =========
def mjpeg_generator(which="overlay"):
    global latest_overlay, latest_heat_only, running
    while running:
        with lock:
            frame = latest_overlay if which == "overlay" else latest_heat_only
            if frame is None:
                # evita enviar antes do 1º frame
                pass
        if frame is not None:
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
    body { margin: 0; font-family: sans-serif; background: #111; color: #eee; }
    header { padding: 12px 16px; background: #1a1a1a; position: sticky; top: 0; }
    main { display: grid; gap: 12px; padding: 12px; }
    .card { background: #181818; border-radius: 12px; padding: 12px; }
    img { width: 100%; height: auto; border-radius: 8px; display: block; }
    @media (min-width: 900px){
      main { grid-template-columns: 1fr 1fr; }
    }
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
    t = Thread(target=processing_loop, daemon=True)
    t.start()
    try:
        # host='0.0.0.0' permite acesso do celular via IP da sua máquina
        app.run(host="0.0.0.0", port=5000, threaded=True)
    finally:
        global running
        running = False
        time.sleep(0.2)
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
