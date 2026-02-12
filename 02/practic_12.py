import cv2
import numpy as np
import time
import csv
import os
from ultralytics import YOLO
import yt_dlp

YOUTUBE_URL = 'https://www.youtube.com/watch?v=Lxqcg1qt0XU'
MODEL_PATH = 'yolov8n.pt'
DISTANCE_METERS = 10.0


L1 = [950, 340, 1700, 390]
L2 = [800, 420, 1600, 520]


def get_live_url(url):
    ydl_opts = {'format': 'best', 'quiet': True, 'no_warnings': True, 'nocheckcertificate': True}
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            return ydl.extract_info(url, download=False)['url']
    except:
        return None


def is_below(point, line):
    x, y = point
    x1, y1, x2, y2 = line
    return (y2 - y1) * x - (x2 - x1) * y + x2 * y1 - y2 * x1 > 0


source_url = get_live_url(YOUTUBE_URL)
if not source_url: exit()

model = YOLO(MODEL_PATH)


tracker_data = {}
final_speeds_list = []
total_counted = 0

os.makedirs('output', exist_ok=True)
CSV_PATH = 'output/traffic_data.csv'
with open(CSV_PATH, mode='w', newline='') as f:
    csv.writer(f).writerow(['ID', 'Class', 'Speed_kmh'])

cap = cv2.VideoCapture(source_url)

while True:
    ret, frame = cap.read()
    if not ret: break


    frame = cv2.resize(frame, (frame.shape[1] // 2, frame.shape[0] // 2))
    scale = 0.5
    l1_s = [int(v * scale) for v in L1]
    l2_s = [int(v * scale) for v in L2]

    cv2.line(frame, (l1_s[0], l1_s[1]), (l1_s[2], l1_s[3]), (0, 0, 255), 1)
    cv2.line(frame, (l2_s[0], l2_s[1]), (l2_s[2], l2_s[3]), (0, 0, 255), 1)

    results = model.track(frame, conf=0.3, tracker="bytetrack.yaml", persist=True, verbose=False)

    if results[0].boxes is not None and results[0].boxes.id is not None:
        boxes = results[0].boxes
        for box, tid_raw, cls_id in zip(boxes.xyxy.cpu().numpy(), boxes.id.cpu().numpy(), boxes.cls.cpu().numpy()):

            tid = int(tid_raw)
            class_name = model.names[int(cls_id)]
            if class_name not in ['car', 'truck', 'bus', 'motorcycle']:
                continue

            x1, y1, x2, y2 = box.astype(int)

            center = ((x1 + x2) // 2, (y1 + y2) // 2)
            curr_time = time.time()

            if tid not in tracker_data:
                tracker_data[tid] = {'t1': None, 'speed': None, 'counted': False}

            data = tracker_data[tid]
            b1, b2 = is_below(center, l1_s), is_below(center, l2_s)

            if data['t1'] is None and (b1 != b2):
                data['t1'] = curr_time
                data['side'] = b1

            elif data['t1'] is not None and not data['counted']:
                if b1 == b2:
                    dt = curr_time - data['t1']
                    if dt > 0.1:
                        v = round((DISTANCE_METERS / dt) * 3.6, 2)


                        if 1.0 < v < 160.0:
                            data['speed'] = v
                            data['counted'] = True


                            total_counted += 1
                            final_speeds_list.append(v)

                            with open(CSV_PATH, mode='a', newline='') as f:
                                csv.writer(f).writerow([tid, class_name, v])


            label = f"{class_name}"
            if data['speed']: label += f", {data['speed']} km/h"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


    if len(final_speeds_list) > 0:
        avg_speed = sum(final_speeds_list) / len(final_speeds_list)
    else:
        avg_speed = 0

    cv2.putText(frame, f"Total Registered: {total_counted}", (20, 45), 1, 2.0, (255, 255, 255), 2)
    cv2.putText(frame, f"Avg Speed: {avg_speed:.2f} km/h", (20, 90), 1, 2.0, (255, 255, 255), 2)

    cv2.imshow('Traffic Monitor', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()
