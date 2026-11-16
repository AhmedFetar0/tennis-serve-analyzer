"""
train.py

Trains the ball detector. These are the settings that produced the model the
pipeline uses (runs/detect/train-6, mAP50 0.733).

Why these values:
    imgsz=1280   The ball is about 25 px wide in a 1920 px frame. YOLO resizes
                 every image to imgsz before looking at it, so the default of
                 640 shrinks the ball to about 8 px, which is too small to
                 detect reliably. 1280 keeps it at about 17 px.
    batch=4      Larger images need more GPU memory per image. 4 is what fits
                 in 8 GB at this resolution. Lower the batch before lowering
                 the resolution: resolution is what fixes the small-ball
                 problem.
    workers=4    Each data-loading worker builds a 2560x2560 mosaic canvas in
                 system RAM. 8 workers ran out of RAM (not GPU memory) during
                 the first epoch.
    patience=20  Stop early if validation has not improved for 20 epochs.

Training starts from the stock yolov8n.pt on the MERGED dataset (public
images plus our court footage), not from an earlier run's weights. Fine-tuning
old weights on the new frames alone risks the model forgetting what it had
learned from the larger public set.

Run from the repo root:
    python training/train.py
"""

from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO("yolov8n.pt")

    model.train(
        data="data/data.yaml",
        epochs=100,
        imgsz=1280,
        batch=4,
        workers=4,
        device=0,
        patience=20,
    )
