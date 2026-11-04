from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO("runs/detect/train-3/weights/last.pt")
    model.train(resume=True)