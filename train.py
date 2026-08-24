from ultralytics import YOLO

model = YOLO("yolov8n.pt")

model.train(
    data="data/Tennis Ball detection.v1i.yolov8/data.yaml",
    epochs=2,
    imgsz=640,
    batch=8,
    device="cpu",
    fraction=0.05
)