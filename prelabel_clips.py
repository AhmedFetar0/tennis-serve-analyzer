"""
prelabel_clips.py

Phase 2 helper: turn raw court video clips into a YOLO-format dataset,
pre-labeled by your existing best.pt model, ready for review/correction
in Roboflow (or any YOLO-format annotation tool).

Workflow this script implements:
  1. Sample each clip at a low, fixed FPS (default 2) -> avoids near-duplicate
     frames that add labeling work without adding information.
  2. Run your current YOLOv8n model on each sampled frame at a low confidence
     threshold -> catches weak-but-real detections for you to keep, at the
     cost of some false positives you'll reject during review.
  3. Write YOLO-format labels (class x_center y_center width height, normalized)
     next to each image, plus an annotated preview copy so you can eyeball
     quality before importing anything into Roboflow.
  4. Flag every frame with zero detections into needs_review.txt -- these are
     the frames most likely to contain a missed ball you'll need to box by hand.

Usage:
    python prelabel_clips.py --videos_dir ./raw_clips --weights ./best.pt \
        --output_dir ./phase2_dataset --fps 2 --conf 0.15

Output layout:
    phase2_dataset/
        images/        <video_name>_f000123.jpg   (real image, upload this)
        labels/        <video_name>_f000123.txt    (YOLO label, upload this)
        preview/       <video_name>_f000123.jpg    (boxes drawn in, for QA only)
        needs_review.txt
"""

import argparse
import sys
from pathlib import Path

import cv2

try:
    from ultralytics import YOLO
except ImportError:
    print("Missing dependency. Install with: pip install ultralytics opencv-python")
    sys.exit(1)


def extract_frames(video_path: Path, target_fps: float):
    """Yield (frame_index, frame) sampled at approximately target_fps."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  [skip] could not open {video_path.name}")
        return

    source_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(int(round(source_fps / target_fps)), 1)

    frame_idx = 0
    saved_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % step == 0:
            yield saved_idx, frame
            saved_idx += 1
        frame_idx += 1
    cap.release()


def run_pipeline(videos_dir: Path, weights: Path, output_dir: Path,
                  fps: float, conf: float, class_id: int, class_name: str):
    images_dir = output_dir / "images"
    labels_dir = output_dir / "labels"
    preview_dir = output_dir / "preview"
    for d in (images_dir, labels_dir, preview_dir):
        d.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(weights))

    video_paths = sorted(
        p for p in videos_dir.iterdir()
        if p.suffix.lower() in {".mp4", ".mov", ".m4v", ".avi", ".mkv"}
    )
    if not video_paths:
        print(f"No video files found in {videos_dir}")
        return

    needs_review = []
    total_frames = 0
    total_detections = 0

    for video_path in video_paths:
        print(f"Processing {video_path.name} ...")
        stem = video_path.stem
        video_frame_count = 0
        video_zero_detect = 0

        for saved_idx, frame in extract_frames(video_path, fps):
            frame_name = f"{stem}_f{saved_idx:06d}"
            img_path = images_dir / f"{frame_name}.jpg"
            cv2.imwrite(str(img_path), frame)

            h, w = frame.shape[:2]
            results = model.predict(frame, conf=conf, verbose=False)[0]

            lines = []
            preview = frame.copy()
            for box in results.boxes:
                cx, cy, bw, bh = box.xywh[0].tolist()
                lines.append(
                    f"{class_id} {cx / w:.6f} {cy / h:.6f} {bw / w:.6f} {bh / h:.6f}"
                )
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                cv2.rectangle(preview, (int(x1), int(y1)), (int(x2), int(y2)),
                              (0, 255, 0), 2)
                conf_val = float(box.conf[0])
                cv2.putText(preview, f"{class_name} {conf_val:.2f}",
                            (int(x1), max(int(y1) - 6, 0)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            (labels_dir / f"{frame_name}.txt").write_text("\n".join(lines))
            cv2.imwrite(str(preview_dir / f"{frame_name}.jpg"), preview)

            if not lines:
                needs_review.append(frame_name)
                video_zero_detect += 1

            video_frame_count += 1
            total_detections += len(lines)

        total_frames += video_frame_count
        print(f"  {video_frame_count} frames sampled, "
              f"{video_zero_detect} with zero detections")

    review_path = output_dir / "needs_review.txt"
    existing = review_path.read_text().splitlines() if review_path.exists() else []
    existing_set = set(existing)
    combined = existing + [n for n in needs_review if n not in existing_set]
    review_path.write_text("\n".join(combined))

    print("\n--- Summary ---")
    print(f"Videos processed:    {len(video_paths)}")
    print(f"Frames sampled:      {total_frames}")
    print(f"Total detections:    {total_detections}")
    print(f"Zero-detection frames flagged in needs_review.txt: {len(needs_review)}")
    print(f"\nUpload images/ + labels/ to Roboflow to review and fix boxes.")
    print(f"Check preview/ locally first if you want a quick visual sanity check.")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--videos_dir", required=True, type=Path,
                         help="Folder containing your raw clips")
    parser.add_argument("--weights", required=True, type=Path,
                         help="Path to your trained best.pt")
    parser.add_argument("--output_dir", required=True, type=Path,
                         help="Where to write images/labels/preview")
    parser.add_argument("--fps", type=float, default=2.0,
                         help="Sampling rate in frames per second (default 2)")
    parser.add_argument("--conf", type=float, default=0.15,
                         help="Confidence threshold for pre-labeling (default 0.15, "
                              "kept low deliberately so you catch weak detections to "
                              "review rather than missing them entirely)")
    parser.add_argument("--class_id", type=int, default=0,
                         help="Class id to write into labels (default 0)")
    parser.add_argument("--class_name", default="tennis-ball",
                         help="Class name, used only for preview image text")
    args = parser.parse_args()

    run_pipeline(args.videos_dir, args.weights, args.output_dir,
                 args.fps, args.conf, args.class_id, args.class_name)


if __name__ == "__main__":
    main()
