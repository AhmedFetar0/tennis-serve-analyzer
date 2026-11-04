"""
evaluate_per_clip.py

Compares model predictions against your hand-corrected ground-truth labels,
frame by frame, and breaks results down per clip (serve1..serve5) -- turns
"serve5 looked worse" into an actual precision/recall number per clip.

Usage:
  1. Generate predictions with confidence scores and YOLO-format txt output:
     yolo detect predict model="runs\\detect\\train-6\\weights\\best.pt" \
         source="phase2_dataset\\images" imgsz=1280 conf=0.25 \
         save_txt=True save_conf=True project=runs\\detect name=eval_phase2

  2. python evaluation/evaluate_per_clip.py --gt_dir phase2_dataset\\labels \
         --pred_dir runs\\detect\\eval_phase2\\labels --iou_thresh 0.5
"""

import argparse
from collections import defaultdict
from pathlib import Path


def load_boxes(path: Path, has_conf: bool = False):
    boxes = []
    if not path.exists():
        return boxes
    for line in path.read_text().splitlines():
        parts = line.split()
        if not parts:
            continue
        min_len = 6 if has_conf else 5
        if len(parts) < min_len:
            continue
        cx, cy, w, h = (float(p) for p in parts[1:5])
        boxes.append((cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2))
    return boxes


def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def match(gt_boxes, pred_boxes, iou_thresh):
    """Greedy best-IoU matching. Returns (true_positives, false_positives,
    false_negatives) for one frame."""
    matched_gt = set()
    tp = 0
    for pb in pred_boxes:
        best_iou, best_idx = 0.0, -1
        for i, gb in enumerate(gt_boxes):
            if i in matched_gt:
                continue
            v = iou(pb, gb)
            if v > best_iou:
                best_iou, best_idx = v, i
        if best_iou >= iou_thresh:
            matched_gt.add(best_idx)
            tp += 1
    fp = len(pred_boxes) - tp
    fn = len(gt_boxes) - len(matched_gt)
    return tp, fp, fn


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gt_dir", required=True, type=Path)
    parser.add_argument("--pred_dir", required=True, type=Path)
    parser.add_argument("--iou_thresh", type=float, default=0.5)
    parser.add_argument("--list_errors", action="store_true",
                         help="Also print the exact frame names that were "
                              "missed (FN) or false-positived (FP), per clip")
    args = parser.parse_args()

    stats = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0, "frames": 0})
    missed = defaultdict(list)
    false_pos = defaultdict(list)

    gt_files = sorted(args.gt_dir.glob("*.txt"))
    if not gt_files:
        print(f"No ground-truth label files found in {args.gt_dir}")
        return

    for gt_path in gt_files:
        name = gt_path.stem
        clip = name.split("_f")[0]
        gt_boxes = load_boxes(gt_path)
        pred_boxes = load_boxes(args.pred_dir / f"{name}.txt", has_conf=True)

        tp, fp, fn = match(gt_boxes, pred_boxes, args.iou_thresh)
        s = stats[clip]
        s["tp"] += tp
        s["fp"] += fp
        s["fn"] += fn
        s["frames"] += 1
        if fn > 0:
            missed[clip].append(name)
        if fp > 0:
            false_pos[clip].append(name)

    header = f"{'Clip':<10}{'Frames':>8}{'GT':>7}{'TP':>6}{'FP':>6}{'FN':>6}{'Precision':>11}{'Recall':>9}"
    print(header)
    print("-" * len(header))

    totals = {"tp": 0, "fp": 0, "fn": 0, "gt": 0, "frames": 0}
    for clip in sorted(stats):
        s = stats[clip]
        gt_count = s["tp"] + s["fn"]
        precision = s["tp"] / (s["tp"] + s["fp"]) if (s["tp"] + s["fp"]) else float("nan")
        recall = s["tp"] / gt_count if gt_count else float("nan")
        print(f"{clip:<10}{s['frames']:>8}{gt_count:>7}{s['tp']:>6}{s['fp']:>6}{s['fn']:>6}"
              f"{precision:>11.3f}{recall:>9.3f}")
        totals["tp"] += s["tp"]
        totals["fp"] += s["fp"]
        totals["fn"] += s["fn"]
        totals["gt"] += gt_count
        totals["frames"] += s["frames"]

    print("-" * len(header))
    overall_p = totals["tp"] / (totals["tp"] + totals["fp"]) if (totals["tp"] + totals["fp"]) else float("nan")
    overall_r = totals["tp"] / totals["gt"] if totals["gt"] else float("nan")
    print(f"{'TOTAL':<10}{totals['frames']:>8}{totals['gt']:>7}{totals['tp']:>6}{totals['fp']:>6}{totals['fn']:>6}"
          f"{overall_p:>11.3f}{overall_r:>9.3f}")

    if args.list_errors:
        print("\n--- Frames with a missed detection (FN) ---")
        for clip in sorted(missed):
            print(f"{clip}: {', '.join(missed[clip])}")
        print("\n--- Frames with a false positive (FP) ---")
        for clip in sorted(false_pos):
            print(f"{clip}: {', '.join(false_pos[clip])}")


if __name__ == "__main__":
    main()
