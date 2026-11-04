"""
calibrate_court.py

Builds an image -> court-plane homography from named court landmarks clicked
on ONE frame of the clip being measured.

Why a new tool rather than calibrate.py:
    calibrate.py clicks 4 points on court.jpg, a photo of a different court.
    A homography is only valid for the exact view it was solved on. Using more
    than 4 points also gives residuals, so a bad click shows up as a number
    instead of silently bending the mapping.

Court coordinates (metres), same convention as calibrate.py:
    origin = near-left SINGLES baseline corner, as seen from the camera
    x      = across the court, y = down the court, away from the near baseline

Output is a JSON file of clicked points. The homography itself is recomputed
on load, so the file cannot hold a matrix that disagrees with its points.

Usage:
    # Click landmarks (s = skip one that isn't visible, u = undo, q = finish)
    python pipeline/calibrate_court.py --video "TestRecs\\oncourt test 1\\240\\serve3.MOV" `
        --frame 1014 --out tracking\\serve3\\court_f1014.json

    # Check an existing calibration: print residuals, draw the court on the frame
    python pipeline/calibrate_court.py --video "TestRecs\\oncourt test 1\\240\\serve3.MOV" `
        --check tracking\\serve3\\court_f1014.json --preview court_check.jpg
"""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

SINGLES_W = 8.23
DOUBLES_MARGIN = 1.37
COURT_L = 23.77
SERVICE_D = 5.485            # baseline to service line
CENTRE_X = SINGLES_W / 2
FAR_SERVICE_Y = COURT_L - SERVICE_D
NET_Y = COURT_L / 2

COURT_LANDMARKS = {
    "near_left_doubles": (-DOUBLES_MARGIN, 0.0),
    "near_left_singles": (0.0, 0.0),
    "near_center_mark": (CENTRE_X, 0.0),
    "near_right_singles": (SINGLES_W, 0.0),
    "near_right_doubles": (SINGLES_W + DOUBLES_MARGIN, 0.0),
    "near_service_left": (0.0, SERVICE_D),
    "near_T": (CENTRE_X, SERVICE_D),
    "near_service_right": (SINGLES_W, SERVICE_D),
    "far_service_left": (0.0, FAR_SERVICE_Y),
    "far_T": (CENTRE_X, FAR_SERVICE_Y),
    "far_service_right": (SINGLES_W, FAR_SERVICE_Y),
    "far_left_doubles": (-DOUBLES_MARGIN, COURT_L),
    "far_left_singles": (0.0, COURT_L),
    "far_right_singles": (SINGLES_W, COURT_L),
    "far_right_doubles": (SINGLES_W + DOUBLES_MARGIN, COURT_L),
}

# Court markings as world-coordinate segments, shared with the plots.
COURT_LINES = [
    ((-DOUBLES_MARGIN, 0), (SINGLES_W + DOUBLES_MARGIN, 0)),
    ((-DOUBLES_MARGIN, COURT_L), (SINGLES_W + DOUBLES_MARGIN, COURT_L)),
    ((-DOUBLES_MARGIN, 0), (-DOUBLES_MARGIN, COURT_L)),
    ((SINGLES_W + DOUBLES_MARGIN, 0), (SINGLES_W + DOUBLES_MARGIN, COURT_L)),
    ((0, 0), (0, COURT_L)),
    ((SINGLES_W, 0), (SINGLES_W, COURT_L)),
    ((0, SERVICE_D), (SINGLES_W, SERVICE_D)),
    ((0, FAR_SERVICE_Y), (SINGLES_W, FAR_SERVICE_Y)),
    ((CENTRE_X, SERVICE_D), (CENTRE_X, FAR_SERVICE_Y)),
]

MIN_POINTS = 6   # 4 solve it exactly; extra points are what make errors visible


@dataclass
class CourtCalibration:
    path: Path
    frame: int
    image_size: tuple
    points: dict             # landmark name -> (x_px, y_px)
    H: np.ndarray            # image px -> court metres
    rms_px: float            # reprojection residual, image pixels
    loo_rms_m: float         # leave-one-out landmark prediction error, metres
    ground_sign: float       # sign of homogeneous w for points on the court

    def to_court(self, px):
        """Project reference-frame pixels to the court plane.

        Returns (xy, on_ground). on_ground is False for pixels at or above the
        horizon: their ray never meets the court in front of the camera, so
        the xy returned for them is meaningless.
        """
        px = np.asarray(px, dtype=float).reshape(-1, 2)
        homog = np.c_[px, np.ones(len(px))] @ self.H.T
        w = homog[:, 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            xy = homog[:, :2] / w[:, None]
        return xy, w * self.ground_sign > 0

    def horizon(self):
        """Image line a*x + b*y + c = 0 where the court plane meets infinity."""
        return self.H[2]


def _solve(image_pts, world_pts):
    H, _ = cv2.findHomography(np.float64(image_pts), np.float64(world_pts), 0)
    if H is None:
        raise SystemExit("Homography solve failed -- are the points collinear?")
    return H


def _apply(H, pts):
    return cv2.perspectiveTransform(np.float64(pts).reshape(-1, 1, 2), H).reshape(-1, 2)


def load_calibration(path: Path) -> CourtCalibration:
    if not path.is_file():
        raise SystemExit(
            f"Court calibration not found: {path}\n"
            "Create one for THIS clip with calibrate_court.py. homography.npy "
            "from calibrate.py will not work: it was solved on court.jpg.")
    data = json.loads(path.read_text())

    unknown = set(data["points"]) - set(COURT_LANDMARKS)
    if unknown:
        raise SystemExit(f"{path}: unknown landmark(s): {sorted(unknown)}")
    if len(data["points"]) < MIN_POINTS:
        raise SystemExit(f"{path}: {len(data['points'])} landmarks, need at "
                         f"least {MIN_POINTS} so that errors can be detected.")

    names = list(data["points"])
    image_pts = np.float64([data["points"][n] for n in names])
    world_pts = np.float64([COURT_LANDMARKS[n] for n in names])

    H = _solve(image_pts, world_pts)
    reproj = _apply(np.linalg.inv(H), world_pts)
    rms_px = float(np.sqrt(((reproj - image_pts) ** 2).sum(axis=1).mean()))

    # Each landmark predicted by a homography that never saw it: an honest
    # measure of accuracy, where the fit residual is flattered by the fit.
    loo = []
    for i in range(len(names)):
        keep = np.arange(len(names)) != i
        pred = _apply(_solve(image_pts[keep], world_pts[keep]), image_pts[i])[0]
        loo.append(np.linalg.norm(pred - world_pts[i]))
    loo_rms_m = float(np.sqrt(np.mean(np.square(loo))))

    w = np.c_[image_pts, np.ones(len(names))] @ H[2]
    return CourtCalibration(path, int(data["frame"]), tuple(data["image_size"]),
                            {n: tuple(data["points"][n]) for n in names}, H,
                            rms_px, loo_rms_m, float(np.sign(np.median(w))))


def read_frame(video: Path, frame_idx: int):
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise SystemExit(f"Could not open {video}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise SystemExit(f"Could not read frame {frame_idx} of {video}")
    return frame


def draw_court(image, calib: CourtCalibration, colour=(0, 255, 255)):
    """Draw the court model back onto the image: the quickest visual check."""
    H_inv = np.linalg.inv(calib.H)
    for a, b in COURT_LINES:
        line = _apply(H_inv, np.linspace(a, b, 30))
        cv2.polylines(image, [line.astype(np.int32)], False, colour, 1, cv2.LINE_AA)
    for name, px in calib.points.items():
        cv2.drawMarker(image, (int(px[0]), int(px[1])), (0, 0, 255),
                       cv2.MARKER_CROSS, 12, 1)
    return image


def click_landmarks(frame):
    """Prompt for each landmark in turn. Returns {name: (x, y)}."""
    window = "Court calibration"
    h, w = frame.shape[:2]
    state = {"mouse": (w // 2, h // 2), "click": None}

    def on_mouse(event, x, y, flags, param):
        state["mouse"] = (x, y)
        if event == cv2.EVENT_LBUTTONDOWN:
            state["click"] = (x, y)

    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window, on_mouse)

    names = list(COURT_LANDMARKS)
    points = {}
    i = 0
    while i < len(names):
        canvas = frame.copy()
        for name, (x, y) in points.items():
            cv2.drawMarker(canvas, (x, y), (0, 0, 255), cv2.MARKER_CROSS, 12, 1)

        # Far-court landmarks are a few pixels apart: magnify under the cursor.
        mx, my = state["mouse"]
        r, zoom = 30, 8
        x0, y0 = min(max(mx - r, 0), w - 2 * r), min(max(my - r, 0), h - 2 * r)
        inset = cv2.resize(frame[y0:y0 + 2 * r, x0:x0 + 2 * r], None,
                           fx=zoom, fy=zoom, interpolation=cv2.INTER_CUBIC)
        centre = ((mx - x0) * zoom, (my - y0) * zoom)
        cv2.drawMarker(inset, centre, (0, 255, 255), cv2.MARKER_CROSS, 24, 1)
        canvas[:inset.shape[0], :inset.shape[1]] = inset

        cv2.putText(canvas, f"{names[i]}   (s skip, u undo, q finish)",
                    (500, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
        cv2.imshow(window, canvas)
        key = cv2.waitKey(15) & 0xFF

        if state["click"] is not None:
            points[names[i]] = state["click"]
            state["click"] = None
            i += 1
        elif key == ord("s"):
            i += 1
        elif key == ord("u") and i > 0:
            i -= 1
            points.pop(names[i], None)
        elif key in (ord("q"), 27):
            break

    cv2.destroyWindow(window)
    return points


def report(calib: CourtCalibration):
    print(f"Calibration: {calib.path}  (frame {calib.frame}, "
          f"{len(calib.points)} landmarks)")
    print(f"  reprojection RMS:   {calib.rms_px:.2f} px")
    print(f"  leave-one-out RMS:  {calib.loo_rms_m:.3f} m")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--video", required=True, type=Path)
    p.add_argument("--frame", type=int,
                   help="Frame to calibrate on. Pick one where the far-court "
                        "lines are unobstructed.")
    p.add_argument("--out", type=Path, help="JSON file to write clicked points to")
    p.add_argument("--check", type=Path,
                   help="Validate an existing calibration instead of clicking")
    p.add_argument("--preview", type=Path,
                   help="Save the frame with the court model drawn on it")
    args = p.parse_args()

    if args.check:
        calib = load_calibration(args.check)
    else:
        if args.frame is None or args.out is None:
            p.error("--frame and --out are required unless using --check")
        frame = read_frame(args.video, args.frame)
        points = click_landmarks(frame)
        if len(points) < MIN_POINTS:
            raise SystemExit(f"Only {len(points)} landmarks clicked; need at "
                             f"least {MIN_POINTS}. Nothing saved.")
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({
            "video": args.video.name,
            "frame": args.frame,
            "image_size": [frame.shape[1], frame.shape[0]],
            "points": {k: list(v) for k, v in points.items()},
        }, indent=2))
        calib = load_calibration(args.out)

    report(calib)
    if args.preview:
        image = draw_court(read_frame(args.video, calib.frame), calib)
        cv2.imwrite(str(args.preview), image)
        print(f"  preview: {args.preview}")


if __name__ == "__main__":
    main()
