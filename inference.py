import cv2
from ultralytics import YOLO
import csv
import sys
import os

if len(sys.argv) < 2:
    print("Usage: python inference.py <video_filename>")
    sys.exit(1)

video_path_arg = sys.argv[1]
video_name = os.path.basename(video_path_arg)
name_no_ext = os.path.splitext(video_name)[0]

MODEL_PATH = "runs/detect/train-3/weights/best.pt"
VIDEO_PATH = video_path_arg
OUTPUT_VIDEO = f"TestRecs/annotated ouputs/{name_no_ext}_annotated.avi"
OUTPUT_CSV = f"TestRecs/pos results/{name_no_ext}_positions.csv"

model = YOLO(MODEL_PATH)

cap = cv2.VideoCapture(VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*"MJPG")
out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (width, height))

results_log = []
frame_num = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, verbose=False)[0]

    best_box = None
    best_conf = 0
    for box in results.boxes:
        conf = float(box.conf[0])
        if conf > best_conf:
            best_conf = conf
            best_box = box

    if best_box is not None:
        x1, y1, x2, y2 = best_box.xyxy[0].tolist()
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2

        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 165, 255), 2)
        cv2.circle(frame, (int(cx), int(cy)), 4, (0, 0, 255), -1)

        results_log.append([frame_num, cx, cy, best_conf])
    else:
        results_log.append([frame_num, None, None, None])

    out.write(frame)
    frame_num += 1

cap.release()
out.release()

with open(OUTPUT_CSV, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["frame", "x", "y", "confidence"])
    writer.writerows(results_log)

print(f"Done. Processed {frame_num} frames.")
print(f"Annotated video saved to {OUTPUT_VIDEO}")
print(f"Ball positions saved to {OUTPUT_CSV}")