from ultralytics import YOLO

model = YOLO("runs/detect/train-3/weights/best.pt")
results = model("data/valid/images", save=True, conf=0.25)

# Find and print filenames of images with 3+ detections, so we can inspect them
for r in results:
    if len(r.boxes) >= 3:
        print(r.path)