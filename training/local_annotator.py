"""
local_annotator.py

Fully offline bounding-box annotator/reviewer for the dataset produced by
prelabel_clips.py. Nothing leaves your machine -- no upload, no account.

Loads each image, shows any existing pre-labeled box from labels/*.txt, and
lets you fix it with the mouse. Frames listed in needs_review.txt (if
present) are shown first and flagged in red, since those are the ones your
model missed entirely.

A magnifier in the top-right corner follows your cursor, zoomed in on the
original resolution, with a crosshair marking the exact point a click would
land -- meant to make placing tight boxes around a small, fast-moving ball
easier than eyeballing it at full-frame scale.

Controls:
    Left-click + drag     draw a new box (shown live in the magnifier too)
    Right-click on a box   delete that box
    c                      clear all boxes on this frame
    m                      toggle the magnifier on/off
    n / Space / Enter      save this frame and go to the next image
    b / Backspace          save this frame and go to the previous image
    q / Esc                save and quit

Usage:
    python training/local_annotator.py --dataset_dir ./phase2_dataset --class_id 0
"""

import argparse
from pathlib import Path

import cv2
import numpy as np

MAX_DISPLAY_W = 1280
MAX_DISPLAY_H = 800


class Box:
    __slots__ = ("x1", "y1", "x2", "y2")

    def __init__(self, x1, y1, x2, y2):
        self.x1, self.y1, self.x2, self.y2 = x1, y1, x2, y2

    def normalized(self, class_id, img_w, img_h):
        cx = (self.x1 + self.x2) / 2 / img_w
        cy = (self.y1 + self.y2) / 2 / img_h
        w = abs(self.x2 - self.x1) / img_w
        h = abs(self.y2 - self.y1) / img_h
        return f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"

    def contains(self, x, y):
        return (min(self.x1, self.x2) <= x <= max(self.x1, self.x2) and
                min(self.y1, self.y2) <= y <= max(self.y1, self.y2))


def load_boxes(label_path: Path, img_w, img_h):
    boxes = []
    if label_path.exists():
        for line in label_path.read_text().splitlines():
            parts = line.split()
            if len(parts) != 5:
                continue
            _, cx, cy, w, h = (float(p) for p in parts)
            x1 = (cx - w / 2) * img_w
            y1 = (cy - h / 2) * img_h
            x2 = (cx + w / 2) * img_w
            y2 = (cy + h / 2) * img_h
            boxes.append(Box(x1, y1, x2, y2))
    return boxes


def save_boxes(label_path: Path, boxes, class_id, img_w, img_h):
    lines = [b.normalized(class_id, img_w, img_h) for b in boxes]
    label_path.write_text("\n".join(lines))


def get_magnifier(img, cx, cy, crop_size, zoom, preview_box=None):
    """Return a zoomed-in crop of img centered near (cx, cy) (original-image
    coordinates), with a crosshair at the exact cursor position and, while a
    box is being drawn, a preview of it rendered in the same zoomed space."""
    h, w = img.shape[:2]
    x0 = int(max(0, min(cx - crop_size / 2, w - crop_size)))
    y0 = int(max(0, min(cy - crop_size / 2, h - crop_size)))
    x1 = min(w, x0 + crop_size)
    y1 = min(h, y0 + crop_size)
    crop = img[y0:y1, x0:x1]
    if crop.size == 0:
        return None
    mag = cv2.resize(crop, (crop.shape[1] * zoom, crop.shape[0] * zoom),
                      interpolation=cv2.INTER_LINEAR)
    cross_x = int((cx - x0) * zoom)
    cross_y = int((cy - y0) * zoom)
    cv2.drawMarker(mag, (cross_x, cross_y), (0, 255, 255),
                    markerType=cv2.MARKER_CROSS, markerSize=16, thickness=1)
    if preview_box is not None:
        bx1, by1, bx2, by2 = preview_box
        p1 = (int((min(bx1, bx2) - x0) * zoom), int((min(by1, by2) - y0) * zoom))
        p2 = (int((max(bx1, bx2) - x0) * zoom), int((max(by1, by2) - y0) * zoom))
        cv2.rectangle(mag, p1, p2, (0, 255, 255), 1)
    return mag


def ordered_image_list(images_dir: Path, review_list_path: Path):
    all_images = sorted(p.stem for p in images_dir.glob("*.jpg"))
    all_set = set(all_images)
    if review_list_path.exists():
        priority = [n for n in review_list_path.read_text().splitlines()
                    if n and n in all_set]
        priority_set = set(priority)
        rest = [n for n in all_images if n not in priority_set]
        return priority + rest
    return all_images


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset_dir", required=True, type=Path)
    parser.add_argument("--class_id", type=int, default=0)
    parser.add_argument("--zoom_crop", type=int, default=60,
                         help="Region size in original-image px shown in the "
                              "magnifier (default 60 -- roughly 2-3x a ball's "
                              "width, enough context to place a box against)")
    parser.add_argument("--zoom_factor", type=int, default=6,
                         help="How much to enlarge the magnifier crop (default 6x)")
    parser.add_argument("--start_at", type=int, default=1,
                         help="1-indexed position to jump straight to, matching "
                              "the 'N/Total' counter shown in-app (default 1)")
    args = parser.parse_args()

    images_dir = args.dataset_dir / "images"
    labels_dir = args.dataset_dir / "labels"
    labels_dir.mkdir(exist_ok=True)
    review_list_path = args.dataset_dir / "needs_review.txt"
    review_set = (set(review_list_path.read_text().splitlines())
                  if review_list_path.exists() else set())

    names = ordered_image_list(images_dir, review_list_path)
    if not names:
        print(f"No images found in {images_dir}")
        return

    idx = max(0, min(args.start_at - 1, len(names) - 1))
    state = {"boxes": [], "drawing": False, "start": (0.0, 0.0), "scale": 1.0,
             "mouse_orig": (0.0, 0.0), "show_magnifier": True}

    def mouse_cb(event, x, y, flags, userdata):
        if x > state.get("disp_w", 10 ** 9):
            return  # click/hover landed in the side panel, not the image
        ox, oy = x / state["scale"], y / state["scale"]
        state["mouse_orig"] = (ox, oy)
        if event == cv2.EVENT_LBUTTONDOWN:
            state["drawing"] = True
            state["start"] = (ox, oy)
        elif event == cv2.EVENT_LBUTTONUP and state["drawing"]:
            state["drawing"] = False
            x0, y0 = state["start"]
            if abs(ox - x0) > 3 and abs(oy - y0) > 3:
                state["boxes"].append(Box(x0, y0, ox, oy))
        elif event == cv2.EVENT_RBUTTONDOWN:
            for b in list(state["boxes"]):
                if b.contains(ox, oy):
                    state["boxes"].remove(b)
                    break

    cv2.namedWindow("annotator")
    cv2.setMouseCallback("annotator", mouse_cb)

    print("Controls: drag=new box | right-click=delete | c=clear | "
          "m=toggle magnifier | n/space=next | b/backspace=prev | "
          "q/esc=save+quit\n")

    while 0 <= idx < len(names):
        name = names[idx]
        img_path = images_dir / f"{name}.jpg"
        label_path = labels_dir / f"{name}.txt"
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  [skip] could not read {img_path}")
            idx += 1
            continue

        h, w = img.shape[:2]
        scale = min(1.0, MAX_DISPLAY_W / w, MAX_DISPLAY_H / h)
        state["scale"] = scale
        state["boxes"] = load_boxes(label_path, w, h)
        flagged = name in review_set

        panel_w = args.zoom_crop * args.zoom_factor + 40

        while True:
            disp_w, disp_h = int(w * scale), int(h * scale)
            state["disp_w"] = disp_w
            disp = cv2.resize(img, (disp_w, disp_h)) if scale != 1.0 else img.copy()
            for b in state["boxes"]:
                p1 = (int(b.x1 * scale), int(b.y1 * scale))
                p2 = (int(b.x2 * scale), int(b.y2 * scale))
                cv2.rectangle(disp, p1, p2, (0, 255, 0), 2)

            preview_box = None
            if state["drawing"]:
                sx, sy = state["start"]
                mx, my = state["mouse_orig"]
                preview_box = (sx, sy, mx, my)
                p1 = (int(min(sx, mx) * scale), int(min(sy, my) * scale))
                p2 = (int(max(sx, mx) * scale), int(max(sy, my) * scale))
                cv2.rectangle(disp, p1, p2, (0, 255, 255), 1)

            # Side panel keeps the magnifier and text off the actual image --
            # the full frame stays visible no matter what.
            canvas_h = max(disp_h, panel_w)
            canvas = np.zeros((canvas_h, disp_w + panel_w, 3), dtype=np.uint8)
            canvas[0:disp_h, 0:disp_w] = disp

            lines = [f"{idx + 1}/{len(names)}", name,
                     f"boxes: {len(state['boxes'])}"]
            if flagged:
                lines.append("NEEDS REVIEW")
            ty = 22
            for i, line in enumerate(lines):
                color = (0, 0, 255) if (flagged and i == len(lines) - 1) else (255, 255, 255)
                cv2.putText(canvas, line, (disp_w + 10, ty),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
                ty += 24

            if state["show_magnifier"]:
                mx, my = state["mouse_orig"]
                mag = get_magnifier(img, mx, my, args.zoom_crop,
                                     args.zoom_factor, preview_box)
                if mag is not None:
                    mh, mw = mag.shape[:2]
                    x_off, y_off = disp_w + 10, ty + 10
                    if y_off + mh <= canvas_h and x_off + mw <= canvas.shape[1]:
                        canvas[y_off:y_off + mh, x_off:x_off + mw] = mag
                        cv2.rectangle(canvas, (x_off - 1, y_off - 1),
                                      (x_off + mw, y_off + mh), (255, 255, 255), 1)

            cv2.imshow("annotator", canvas)

            key = cv2.waitKey(20) & 0xFF
            if key in (ord("n"), 13, 32):          # next
                save_boxes(label_path, state["boxes"], args.class_id, w, h)
                review_set.discard(name)
                idx += 1
                break
            elif key in (ord("b"), 8):             # previous
                save_boxes(label_path, state["boxes"], args.class_id, w, h)
                review_set.discard(name)
                idx = max(idx - 1, 0)
                break
            elif key == ord("c"):                  # clear
                state["boxes"] = []
            elif key == ord("m"):                  # toggle magnifier
                state["show_magnifier"] = not state["show_magnifier"]
            elif key in (ord("q"), 27):            # quit
                save_boxes(label_path, state["boxes"], args.class_id, w, h)
                review_set.discard(name)
                review_list_path.write_text("\n".join(sorted(review_set)))
                cv2.destroyAllWindows()
                print("Saved and exited.")
                return

    review_list_path.write_text("\n".join(sorted(review_set)))
    cv2.destroyAllWindows()
    print("Done -- reached the end of the image list.")


if __name__ == "__main__":
    main()
