import os
import cv2
import time
from ultralytics import YOLO

PROJECT_DIR = os.path.dirname(__file__)
VIDEO_DIR = os.path.join(PROJECT_DIR, 'video')
OUT_DIR = os.path.join(PROJECT_DIR, 'out')

os.makedirs(OUT_DIR, exist_ok=True)

USE_WEBCAM = False

if USE_WEBCAM:
    cap = cv2.VideoCapture(0)
else:
    VIDEO_PATH = os.path.join(VIDEO_DIR, 'animals.mp4')
    cap = cv2.VideoCapture(VIDEO_PATH)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out_video_path = os.path.join(OUT_DIR, 'result.mp4')
out_writer = None

model = YOLO('yolov8n.pt')

CONF_TRESHOLD = 0.6
RESIZE_WIDTH = 960

prev_time = time.time()
fps = 0.0

while True:
    ret, frame = cap.read()

    if not ret:
        break

    if RESIZE_WIDTH is not None:
        h, w = frame.shape[:2]
        scale = RESIZE_WIDTH / (w*3)
        new_w = int(scale * w)
        new_h = int(scale * h)
        frame = cv2.resize(frame, (new_w, new_h))

    if out_writer is None:
        h, w = frame.shape[:2]
        out_writer = cv2.VideoWriter(out_video_path, fourcc, 30.0, (w, h))

    result = model(frame, conf=CONF_TRESHOLD, verbose=False)
    cat_count = 0
    dog_count = 0

    CAT_CLASS_ID = 15
    DOG_CLASS_ID = 16

    for r in result:
        boxes = r.boxes
        if boxes is None:
            continue

        for box in boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            if cls == CAT_CLASS_ID or cls == DOG_CLASS_ID:
                if cls == CAT_CLASS_ID:
                    cat_count += 1
                    label = "Cat"
                else:
                    dog_count += 1
                    label = "Dog"

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                txt = f'{label} {conf:.2f}'
                cv2.putText(frame, txt, (x1, max(20, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    now = time.time()
    dt = now - prev_time
    prev_time = now
    if dt > 0:
        fps = 1.0 / dt

    total_animals = cat_count + dog_count
    cv2.putText(frame, f'Cats: {cat_count}', (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
    cv2.putText(frame, f'Dogs: {dog_count}', (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
    cv2.putText(frame, f'Total: {total_animals}', (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
    cv2.putText(frame, f'FPS: {fps:.1f}', (20, 160), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

    if out_writer is not None:
        out_writer.write(frame)

    cv2.imshow("YOLO", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
if out_writer is not None:
    out_writer.release()
cv2.destroyAllWindows()