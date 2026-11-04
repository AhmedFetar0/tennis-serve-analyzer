"""
track_ball.py

Runs the detector on EVERY frame of a clip, then feeds the detection sequence
through a Kalman filter to produce a smoothed, gap-filled trajectory with
velocity estimates.

Why this needs full-video inference (not phase2_dataset):
    phase2_dataset frames were SAMPLED (2 fps / 10 fps), so consecutive files
    are 30 or 24 source frames apart. A Kalman filter models frame-to-frame
    motion -- it needs consecutive frames at the native rate to work at all.

State vector (constant acceleration):
    [x, y, vx, vy, ax, ay]
Measurement:
    [x, y]  -- the detected box centre

What the filter buys you:
  * Gap filling      -- missed detections get a predicted position, with
                        widened uncertainty, instead of a hole
  * Outlier gating   -- a detection too far from the prediction (Mahalanobis
                        distance) is rejected rather than yanking the track.
                        This is what kills the "stationary ball behind the
                        baseline" false positives
  * Velocity         -- vx, vy come out as first-class state, fitted across
                        the whole observation history rather than differenced
                        between two noisy frames

IMPORTANT -- velocity here is in PIXELS PER FRAME, not real-world speed.
Converting to m/s requires the 3D geometry work (camera intrinsics + court
pose). This script is the trajectory front-end for that, not a speedometer.

Usage:
    python pipeline/track_ball.py --video "TestRecs\\oncourt test 1\\240\\serve3.MOV" \
        --weights "runs\\detect\\train-6\\weights\\best.pt" \
        --out_dir .\\tracking\\serve3 --imgsz 1280 --conf 0.15
"""

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np

try:
    from ultralytics import YOLO
except ImportError:
    raise SystemExit("Install deps first: pip install ultralytics opencv-python")


MIN_STREAK_TO_COAST = 5


class BallKalman:
    """Constant-acceleration Kalman filter over image coordinates.

    Working in units of frames (dt = 1) keeps the numbers simple; convert to
    seconds afterwards by dividing velocities by the frame interval.
    """

    def __init__(self, meas_noise_px=3.0, process_noise=0.5, gate_chi2=9.21):
        dt = 1.0
        # State transition: position += v*dt + 0.5*a*dt^2, velocity += a*dt
        self.F = np.array([
            [1, 0, dt, 0, 0.5 * dt * dt, 0],
            [0, 1, 0, dt, 0, 0.5 * dt * dt],
            [0, 0, 1, 0, dt, 0],
            [0, 0, 0, 1, 0, dt],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1],
        ], dtype=float)

        # We only ever observe position
        self.H = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
        ], dtype=float)

        # Measurement noise: how much do we trust a detected box centre?
        # ~3 px is realistic for a ~25 px ball with a tight box.
        self.R = np.eye(2) * (meas_noise_px ** 2)

        # Process noise: how much the ball may deviate from constant
        # acceleration (spin, drag changes, detection bias). Acceleration
        # terms get the most slack since that's what actually varies.
        q = process_noise
        self.Q = np.diag([q * 0.25, q * 0.25, q, q, q * 4, q * 4])

        # Chi-square gate on 2 DOF: 5.99 = 95%, 9.21 = 99%.
        # Looser gate = more detections accepted, including bad ones.
        self.gate_chi2 = gate_chi2

        self.x = None          # state estimate
        self.P = None          # state covariance
        self.initialized = False
        self.misses = 0        # consecutive frames with no accepted measurement
        self.streak = 0        # unbroken run of accepted measurements

    def initialize(self, z):
        self.x = np.array([z[0], z[1], 0.0, 0.0, 0.0, 0.0], dtype=float)
        # Large initial uncertainty on velocity/acceleration: we know where it
        # is, we have no idea yet how fast it's going.
        self.P = np.diag([9.0, 9.0, 400.0, 400.0, 100.0, 100.0])
        self.initialized = True
        self.misses = 0
        self.streak = 0

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x.copy()

    def gate(self, z):
        """Mahalanobis distance^2 of a measurement against the prediction.

        Small value = consistent with where we expected the ball. Large value
        = this detection is somewhere the ball could not plausibly have gone.
        """
        y = np.asarray(z, dtype=float) - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        return float(y @ np.linalg.inv(S) @ y)

    def update(self, z):
        z = np.asarray(z, dtype=float)
        y = z - self.H @ self.x                       # innovation
        S = self.H @ self.P @ self.H.T + self.R       # innovation covariance
        K = self.P @ self.H.T @ np.linalg.inv(S)      # Kalman gain
        self.x = self.x + K @ y
        I = np.eye(len(self.x))
        self.P = (I - K @ self.H) @ self.P
        self.streak = self.streak + 1 if self.misses == 0 else 1
        self.misses = 0

    def drop_acceleration(self):
        """Coast at constant velocity.

        A noisy acceleration estimate extrapolates quadratically, so with no
        measurements it dominates within a few frames. Real acceleration is
        negligible over a short coast (gravity is ~0.04 px/frame^2 at 240 fps).
        Clearing its covariance too keeps the gate from widening on it.
        """
        self.x[4:] = 0.0
        self.P[4:, :] = 0.0
        self.P[:, 4:] = 0.0

    def position(self):
        return self.x[0], self.x[1]

    def velocity(self):
        return self.x[2], self.x[3]

    def speed_px_per_frame(self):
        vx, vy = self.velocity()
        return float(np.hypot(vx, vy))


def detect_sequence(video_path: Path, weights: Path, imgsz: int, conf: float):
    """Run the detector on every frame. Returns (fps, size, detections) where
    detections[i] is a list of (cx, cy, confidence) for frame i."""
    model = YOLO(str(weights))
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise SystemExit(f"Could not open {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    detections = []
    frames_done = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        res = model.predict(frame, imgsz=imgsz, conf=conf, verbose=False)[0]
        per_frame = []
        for box in res.boxes:
            cx, cy, _, _ = box.xywh[0].tolist()
            per_frame.append((cx, cy, float(box.conf[0])))
        detections.append(per_frame)
        frames_done += 1
        if frames_done % 200 == 0:
            print(f"  detected through frame {frames_done}")
    cap.release()
    return fps, (w, h), detections


def run_filter(detections, max_misses=15, max_coast=24, **kf_kwargs):
    """Feed the detection sequence through the filter.

    Returns a list of per-frame dicts describing what happened, which is what
    makes this debuggable: you can see exactly which frames were coasted
    through and which detections were thrown out.

    Coasting is only trustworthy briefly. With no measurements, a constant-
    acceleration model extrapolates quadratically and its uncertainty balloons,
    so the position flies off and the gate starts accepting anything. After
    max_coast misses the track goes 'stale': no position is reported, and the
    next detection starts a fresh track. Restarting is safe because a new track
    has zero velocity and may not coast, so a wrong restart cannot draw a path.
    """
    kf = BallKalman(**kf_kwargs)
    rows = []

    for i, per_frame in enumerate(detections):
        row = {
            "frame": i,
            "raw_x": "", "raw_y": "", "raw_conf": "",
            "status": "", "mahalanobis": "",
            "filt_x": "", "filt_y": "", "vx": "", "vy": "", "speed_px": "",
        }

        # Velocity is only trusted after an unbroken run of detections. A new
        # track has not converged, and one detection after a gap may be a hop
        # to a different object, which the filter would read as a huge velocity.
        coast_limit = max_coast if kf.streak >= MIN_STREAK_TO_COAST else 0
        stale = kf.initialized and kf.misses > 0 and kf.misses >= coast_limit

        if not kf.initialized or (stale and per_frame):
            if per_frame:
                # Start the track on the highest-confidence detection
                cx, cy, c = max(per_frame, key=lambda d: d[2])
                kf.initialize((cx, cy))
                row.update(raw_x=f"{cx:.2f}", raw_y=f"{cy:.2f}",
                           raw_conf=f"{c:.3f}", status="init",
                           filt_x=f"{cx:.2f}", filt_y=f"{cy:.2f}",
                           vx="0.00", vy="0.00", speed_px="0.00")
            else:
                row["status"] = "waiting"
            rows.append(row)
            continue

        if not stale:
            kf.predict()

        # Of the detections this frame, take the one most consistent with the
        # prediction -- not simply the most confident one. A confident
        # detection of the wrong object still loses to a plausible one.
        best = None
        if per_frame:
            scored = [(kf.gate((d[0], d[1])), d) for d in per_frame]
            scored.sort(key=lambda s: s[0])
            best = scored[0]

        if best is not None and best[0] <= kf.gate_chi2:
            d2, (cx, cy, c) = best
            kf.update((cx, cy))
            row.update(raw_x=f"{cx:.2f}", raw_y=f"{cy:.2f}", raw_conf=f"{c:.3f}",
                       status="updated", mahalanobis=f"{d2:.2f}")
        else:
            kf.misses += 1
            kf.drop_acceleration()
            if best is not None:
                d2, (cx, cy, c) = best
                row.update(raw_x=f"{cx:.2f}", raw_y=f"{cy:.2f}",
                           raw_conf=f"{c:.3f}", mahalanobis=f"{d2:.2f}",
                           status="rejected")   # detection existed, didn't fit
            else:
                row["status"] = "coasted"        # no detection at all

            if kf.misses >= max_misses:
                # Track is lost (ball left frame, or a discontinuity the model
                # can't follow). Drop it and wait to re-acquire.
                kf.initialized = False
                row["status"] = "lost"
                rows.append(row)
                continue

            if stale:
                if row["status"] == "coasted":
                    row["status"] = "stale"
                rows.append(row)
                continue

        fx, fy = kf.position()
        vx, vy = kf.velocity()
        row.update(filt_x=f"{fx:.2f}", filt_y=f"{fy:.2f}",
                   vx=f"{vx:.2f}", vy=f"{vy:.2f}",
                   speed_px=f"{kf.speed_px_per_frame():.2f}")
        rows.append(row)

    return rows


def write_overlay(video_path: Path, rows, out_path: Path, size, out_fps=30.0):
    """Annotated video: raw detections orange (red if rejected), filtered
    track green, with a fading trail so you can see the trajectory shape.

    out_fps is the PLAYBACK rate, not the source rate. Two reasons it is not
    just the source fps: the mp4 container rejects a 240 fps timebase
    outright (the denominator exceeds what MPEG-4 allows, and VideoWriter
    fails silently), and 240 fps playback of a 15 s clip is unwatchable
    anyway. Writing every frame at 30 fps gives 8x slow motion, which is what
    you want for inspecting contact and bounce.
    """
    cap = cv2.VideoCapture(str(video_path))

    writer = None
    for fourcc in ("mp4v", "avc1", "MJPG"):
        path = out_path if fourcc != "MJPG" else out_path.with_suffix(".avi")
        w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*fourcc),
                            out_fps, size)
        if w.isOpened():
            writer, out_path = w, path
            break
        w.release()
    if writer is None:
        cap.release()
        print("  could not open any video writer -- skipping overlay")
        return None
    trail = []
    for row in rows:
        ok, frame = cap.read()
        if not ok:
            break

        if row["raw_x"]:
            rx, ry = float(row["raw_x"]), float(row["raw_y"])
            colour = (0, 0, 255) if row["status"] == "rejected" else (0, 140, 255)
            cv2.circle(frame, (int(rx), int(ry)), 14, colour, 2)

        if row["filt_x"]:
            fx, fy = float(row["filt_x"]), float(row["filt_y"])
            trail.append((int(fx), int(fy)))
            trail = trail[-60:]
            cv2.circle(frame, (int(fx), int(fy)), 8, (0, 255, 0), 2)
            for j in range(1, len(trail)):
                cv2.line(frame, trail[j - 1], trail[j], (0, 255, 0), 1)
        else:
            trail = []

        label = f"{row['frame']}  {row['status']}"
        if row["speed_px"]:
            label += f"  {row['speed_px']} px/f"
        cv2.putText(frame, label, (12, 34), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (255, 255, 255), 2)
        writer.write(frame)

    cap.release()
    writer.release()
    return out_path


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--video", required=True, type=Path)
    p.add_argument("--weights", required=True, type=Path)
    p.add_argument("--out_dir", required=True, type=Path)
    p.add_argument("--imgsz", type=int, default=1280)
    p.add_argument("--conf", type=float, default=0.15,
                   help="Detector threshold. Kept low on purpose: the filter's "
                        "gate can throw out bad detections, but it can't "
                        "recover ones the detector never reported.")
    p.add_argument("--meas_noise_px", type=float, default=3.0)
    p.add_argument("--process_noise", type=float, default=0.5)
    p.add_argument("--gate_chi2", type=float, default=9.21)
    p.add_argument("--max_misses", type=int, default=120,
                   help="Consecutive frames with no accepted detection before "
                        "the track is dropped (default 120). This is in FRAMES, "
                        "so at 240 fps it is half a second -- a value tuned for "
                        "30 fps would drop the track after a few milliseconds.")
    p.add_argument("--max_coast", type=int, default=24,
                   help="Consecutive misses to keep predicting through before "
                        "the track goes stale and reports no position (default "
                        "24 frames, 100 ms at 240 fps). Must be below --max_misses.")
    p.add_argument("--out_fps", type=float, default=30.0,
                   help="Playback rate of the overlay video (default 30). Every "
                        "source frame is written; this only sets how fast it "
                        "plays back, so 30 gives 8x slow motion from 240 fps "
                        "footage.")
    p.add_argument("--no_video", action="store_true")
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Detecting on every frame of {args.video.name} ...")
    fps, size, detections = detect_sequence(args.video, args.weights,
                                             args.imgsz, args.conf)
    print(f"  {len(detections)} frames at {fps:.1f} fps")

    rows = run_filter(detections,
                      max_misses=args.max_misses,
                      max_coast=args.max_coast,
                      meas_noise_px=args.meas_noise_px,
                      process_noise=args.process_noise,
                      gate_chi2=args.gate_chi2)

    csv_path = args.out_dir / "track.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    counts = {}
    for r in rows:
        counts[r["status"]] = counts.get(r["status"], 0) + 1

    raw_frames = sum(1 for r in rows if r["raw_x"])
    tracked = sum(1 for r in rows if r["filt_x"])

    print("\n--- Tracking summary ---")
    print(f"Frames:                 {len(rows)}")
    print(f"Frames with a raw detection: {raw_frames} "
          f"({raw_frames / len(rows):.1%})")
    print(f"Frames with a track position: {tracked} "
          f"({tracked / len(rows):.1%})   <- gap filling gain")
    for status in ("init", "updated", "coasted", "stale", "rejected", "lost",
                   "waiting"):
        if status in counts:
            print(f"  {status:<10}{counts[status]}")
    print(f"\nPer-frame detail: {csv_path}")

    if not args.no_video:
        out_video = args.out_dir / "tracked.mp4"
        print(f"Writing overlay video ({args.out_fps:.0f} fps playback) ...")
        written = write_overlay(args.video, rows, out_video, size, args.out_fps)
        if written:
            print(f"Overlay: {written}")


if __name__ == "__main__":
    main()
