"""
estimate_speed.py

Rough serve speed from a ball track and a court calibration, plus a top-down
path plot. No camera intrinsics are used.

Method (contact-to-bounce):
    speed = straight-line 3D distance(contact, bounce) / elapsed time

    bounce   is on the court, so the homography locates it directly
    contact  is airborne, so it is built from the server's ground position
             plus an assumed height

This gives AVERAGE FLIGHT SPEED. A radar gun reports peak speed off the racket,
which is higher because drag slows the ball throughout the flight. A separate,
clearly labelled drag-corrected figure is printed alongside it.

Inputs you supply by eye (step through the video to find them):
    --contact_frame        last frame before the ball deflects off the racket
    --bounce_frame         frame where the ball reaches its lowest image point
    --contact_ground_px    a point on the court under the contact point. The
                           server is usually airborne at contact, so use the
                           landing foot and name its frame with
                           --contact_ground_frame
    --bounce_px            BOTTOM of the ball at the bounce, where it touches
                           the court, not its centre. The centre is 3.3 cm up,
                           which at a low camera angle projects ~0.5 m long.
                           Needed whenever the detector missed the ball:
                           coasted rows are predictions, and are never
                           accepted as a measurement

Handheld footage: a homography only holds for the frame it was solved on, and
at the far court one pixel can be tens of centimetres. --camera handheld
therefore matches background features between each frame and the calibrated
frame, and maps pixels back to it first. This assumes the camera mostly
rotates; it does not recover large walking movement.

Usage (PowerShell):
    python pipeline/estimate_speed.py --track tracking\\serve3\\track.csv `
        --calibration tracking\\serve3\\court_f1014.json `
        --camera handheld --video "TestRecs\\oncourt test 1\\240\\serve3.MOV" `
        --contact_frame 824 --bounce_frame 1012 --bounce_px 1043 603 `
        --contact_ground_px 943 887 --contact_ground_frame 880 `
        --out_dir tracking\\serve3\\serve_a
"""

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from calibrate_court import (COURT_L, COURT_LINES, DOUBLES_MARGIN, NET_Y, SINGLES_W,
                             CourtCalibration, draw_court, load_calibration,
                             read_frame)

MS_TO_KMH = 3.6
MS_TO_MPH = 2.23694

BALL_RADIUS_M = 0.0335
BALL_MASS_KG = 0.0577
AIR_DENSITY = 1.2

MIN_MOTION_INLIERS = 100
MIN_POINTS_FOR_COLOURBAR = 5
MEASURED_STATUSES = {"updated", "init"}


# --- Speed: knows nothing about how its endpoints were obtained ---------------

@dataclass
class Endpoint:
    """A located 3D ball position at a known frame.

    Any source can produce one. A court-line plane crossing would be another
    producer feeding the same estimator, with no change to the speed maths.
    """
    frame: int
    xyz: np.ndarray          # metres, court coordinates
    position_sigma_m: float  # 1-sigma position error along the flight line
    frame_sigma: float       # 1-sigma timing error, frames
    source: str


@dataclass
class SpeedEstimate:
    distance_m: float
    elapsed_s: float
    speed_ms: float
    sigma_ms: float


def average_flight_speed(start: Endpoint, end: Endpoint, fps: float) -> SpeedEstimate:
    distance = float(np.linalg.norm(end.xyz - start.xyz))
    frames = end.frame - start.frame
    elapsed = frames / fps
    speed = distance / elapsed
    rel_distance = math.hypot(start.position_sigma_m, end.position_sigma_m) / distance
    rel_time = math.hypot(start.frame_sigma, end.frame_sigma) / frames
    return SpeedEstimate(distance, elapsed, speed,
                         speed * math.hypot(rel_distance, rel_time))


def drag_corrected_launch_speed(distance_m, elapsed_s, cd):
    """Launch speed implied by an average speed under quadratic drag.

    dv/dt = -k v^2 gives v(s) = v0 exp(-k s), and integrating time over
    distance gives v0 = (exp(k D) - 1) / (k T). Ignores spin and gravity.
    """
    area = math.pi * BALL_RADIUS_M ** 2
    k = AIR_DENSITY * cd * area / (2 * BALL_MASS_KG)
    return math.expm1(k * distance_m) / (k * elapsed_s)


# --- Inputs ------------------------------------------------------------------

def read_track(path: Path):
    if not path.is_file():
        raise SystemExit(f"Track file not found: {path}")
    with path.open(newline="") as f:
        return {int(r["frame"]): r for r in csv.DictReader(f)}


def require_row(track, frame, label):
    if frame not in track:
        raise SystemExit(f"{label} frame {frame} is not in the track "
                         f"(frames {min(track)}-{max(track)}).")
    return track[frame]


def bounce_pixel(track, frame, override):
    """A measured ball position at the bounce, or a clear failure."""
    row = require_row(track, frame, "Bounce")
    if override is not None:
        return tuple(override), "manual pixel"
    if row["status"] in MEASURED_STATUSES:
        return ((float(row["raw_x"]), float(row["raw_y"])),
                "detector box centre (biased long: prefer --bounce_px)")
    raise SystemExit(
        f"Bounce frame {frame} has status '{row['status']}': no detection backs "
        "it, so the track position there is a prediction, not a measurement.\n"
        "Find the ball in that frame by eye and pass --bounce_px X Y.")


def estimate_camera_motion(video: Path, ref_frame: int, frames):
    """Homographies mapping each frame's pixels into the reference frame.

    Every frame is matched directly to the reference rather than chained
    frame-to-frame, so errors do not accumulate. RANSAC discards the moving
    server and ball.
    """
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise SystemExit(f"Could not open {video}")

    orb = cv2.ORB_create(3000)
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    cap.set(cv2.CAP_PROP_POS_FRAMES, ref_frame)
    ok, ref = cap.read()
    if not ok:
        raise SystemExit(f"Could not read reference frame {ref_frame}")
    ref_kp, ref_desc = orb.detectAndCompute(cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY), None)

    wanted = sorted(set(frames))
    motion = {}
    cap.set(cv2.CAP_PROP_POS_FRAMES, wanted[0])
    position = wanted[0]
    for target in wanted:
        while position < target:
            cap.grab()
            position += 1
        ok, image = cap.read()
        position += 1
        if not ok:
            raise SystemExit(f"Could not read frame {target}")
        if target == ref_frame:
            motion[target] = (np.eye(3), len(ref_kp))
            continue

        kp, desc = orb.detectAndCompute(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), None)
        matches = matcher.match(desc, ref_desc)
        src = np.float32([kp[m.queryIdx].pt for m in matches])
        dst = np.float32([ref_kp[m.trainIdx].pt for m in matches])
        T, mask = cv2.findHomography(src, dst, cv2.RANSAC, 2.0)
        inliers = 0 if mask is None else int(mask.sum())
        if inliers < MIN_MOTION_INLIERS:
            raise SystemExit(
                f"Camera motion failed at frame {target}: {inliers} inliers "
                f"(need {MIN_MOTION_INLIERS}). The view has changed too much "
                f"from calibration frame {ref_frame}; calibrate closer to it.")
        motion[target] = (T, inliers)
    cap.release()
    return motion


def to_reference(px, frame, motion):
    """Map a pixel in `frame` into the calibration frame's pixel coordinates."""
    if motion is None:
        return np.asarray(px, dtype=float)
    T = motion[frame][0]
    return cv2.perspectiveTransform(np.float64(px).reshape(1, 1, 2), T)[0, 0]


def ground_point(calib: CourtCalibration, px_ref, label):
    xy, on_ground = calib.to_court(px_ref)
    if not on_ground[0]:
        raise SystemExit(f"{label} pixel ({px_ref[0]:.1f}, {px_ref[1]:.1f}) is above "
                         "the horizon, so it cannot be a point on the court.")
    return xy[0]


def metres_per_pixel(calib: CourtCalibration, px_ref, direction):
    """Court distance along `direction` covered by one pixel, combining x and y.

    Far from the camera this is dominated by image y, and it is the main
    source of error in the bounce position.
    """
    base = calib.to_court(px_ref)[0][0]
    along_x = (calib.to_court(np.add(px_ref, (1, 0)))[0][0] - base) @ direction
    along_y = (calib.to_court(np.add(px_ref, (0, 1)))[0][0] - base) @ direction
    return math.hypot(along_x, along_y)


# --- Outputs -----------------------------------------------------------------

def project_flight(track, flight_frames, calib, motion):
    """Detector-backed flight positions, flagged by whether they can be projected."""
    if not flight_frames:
        return np.array([]), np.empty((0, 2)), np.empty((0, 2)), np.array([], bool)
    ref_px = np.array([
        to_reference((float(track[f]["raw_x"]), float(track[f]["raw_y"])), f, motion)
        for f in flight_frames])
    xy, on_ground = calib.to_court(ref_px)
    return np.array(flight_frames), ref_px, xy, on_ground


def write_top_down_plot(path, title, contact_xy, bounce_xy, frames, xy, on_ground,
                        speed_label):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x_lim = (-5.0, SINGLES_W + 5.0)
    y_lim = (-3.0, COURT_L + 5.0)
    inside = (on_ground & (xy[:, 0] >= x_lim[0]) & (xy[:, 0] <= x_lim[1])
              & (xy[:, 1] >= y_lim[0]) & (xy[:, 1] <= y_lim[1]))
    above_horizon = int((~on_ground).sum())
    off_plot = int((on_ground & ~inside).sum())

    ink, muted, line = "#1f2328", "#59636e", "#8c959f"
    fig, ax = plt.subplots(figsize=(6.4, 9.6), dpi=150)
    ax.add_patch(plt.Rectangle((-DOUBLES_MARGIN, 0), SINGLES_W + 2 * DOUBLES_MARGIN,
                               COURT_L, color="#eef1f4", zorder=0))
    for a, b in COURT_LINES:
        ax.plot(*zip(a, b), color=line, lw=1.2, zorder=1)
    ax.plot([-DOUBLES_MARGIN - 0.9, SINGLES_W + DOUBLES_MARGIN + 0.9], [NET_Y, NET_Y],
            color=muted, lw=2, zorder=1)
    ax.text(SINGLES_W + DOUBLES_MARGIN + 1.0, NET_Y, "net", color=muted, fontsize=9,
            va="center")

    ax.plot([contact_xy[0], bounce_xy[0]], [contact_xy[1], bounce_xy[1]],
            color=ink, lw=1.5, ls=(0, (5, 3)), zorder=2,
            label="Straight line, contact to bounce")
    if inside.any():
        points = ax.scatter(xy[inside, 0], xy[inside, 1], c=frames[inside], cmap="Blues",
                            vmin=frames.min() - 0.4 * np.ptp(frames), vmax=frames.max(),
                            s=36, edgecolors=ink, linewidths=0.4, zorder=3,
                            label="Ball track projected onto the court")
        if inside.sum() >= MIN_POINTS_FOR_COLOURBAR:
            fig.colorbar(points, ax=ax, shrink=0.45, pad=0.02, label="Frame")

    ax.scatter(*contact_xy, marker="s", s=70, color=ink, zorder=4,
               label="Ground under contact (from landing foot)")
    ax.scatter(*bounce_xy, marker="X", s=110, color="#c2410c", edgecolors="white",
               linewidths=0.8, zorder=4, label="Bounce (on the court plane)")

    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)
    ax.set_aspect("equal")
    ax.set_xlabel("Across court (m)", color=muted)
    ax.set_ylabel("Down court (m)", color=muted)
    ax.tick_params(colors=muted)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.06), frameon=False,
              fontsize=9, ncol=1)
    ax.set_title(f"{title}\nGround-plane projection of ball track "
                 "(not true 3D position)", color=ink, fontsize=11)
    fig.text(0.5, 0.012,
             f"{speed_label}\n"
             f"Of {len(frames)} tracked flight frames: {above_horizon} above the "
             "horizon (cannot be projected),\n"
             f"{off_plot} projected off this plot, {int(inside.sum())} shown. "
             "Airborne points land beyond the ball's true ground position.",
             ha="center", va="bottom", fontsize=8.5, color=muted)
    fig.subplots_adjust(top=0.93, bottom=0.22)
    fig.savefig(path)
    plt.close(fig)


def label(image, text, origin, colour):
    """Text with a dark outline, so it stays legible over any background."""
    for ink, thickness in (((0, 0, 0), 4), (colour, 2)):
        cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.6, ink,
                    thickness, cv2.LINE_AA)


def write_image_overlay(path, video, calib, flight_px, contact_ref_ground, bounce_ref):
    """Stabilised ball path on the calibration frame: what the camera really saw."""
    image = draw_court(read_frame(video, calib.frame), calib, colour=(200, 200, 200))
    a, b, c = calib.horizon()
    w = image.shape[1]
    cv2.line(image, (0, int(-c / b)), (w, int(-(a * w + c) / b)), (255, 200, 0), 1,
             cv2.LINE_AA)
    label(image, "horizon: a ball above this cannot be projected to the court",
          (20, int(-c / b) - 12), (255, 200, 0))
    for x, y in flight_px:
        cv2.circle(image, (int(x), int(y)), 3, (0, 255, 0), -1, cv2.LINE_AA)
    contact_px = tuple(int(v) for v in contact_ref_ground)
    bounce_px = tuple(int(v) for v in bounce_ref)
    cv2.drawMarker(image, contact_px, (255, 255, 255), cv2.MARKER_SQUARE, 14, 2)
    cv2.drawMarker(image, bounce_px, (0, 80, 255), cv2.MARKER_TILTED_CROSS, 18, 2)
    label(image, "ground under contact", (contact_px[0] + 14, contact_px[1] + 6),
          (255, 255, 255))
    label(image, "bounce", (bounce_px[0] - 90, bounce_px[1] - 12), (0, 160, 255))
    cv2.imwrite(str(path), image)


# --- Main --------------------------------------------------------------------

def build_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--track", required=True, type=Path,
                   help="track.csv produced by track_ball.py")
    p.add_argument("--calibration", required=True, type=Path,
                   help="Court calibration JSON from calibrate_court.py")
    p.add_argument("--camera", required=True, choices=("handheld", "tripod"),
                   help="handheld compensates camera motion and requires --video. "
                        "tripod uses the calibration on every frame as it stands.")
    p.add_argument("--video", type=Path,
                   help="Source clip. Needed for handheld; also supplies fps, and "
                        "enables the image overlay.")
    p.add_argument("--fps", type=float,
                   help="Frame rate. Defaults to the rate read from --video.")
    p.add_argument("--events", type=Path,
                   help="events.json from detect_events.py. Supplies the contact "
                        "frame, bounce frame and bounce pixel; use with --serve")
    p.add_argument("--serve", type=int, default=0,
                   help="Which serve in --events to measure, from 0 (default 0)")
    p.add_argument("--contact_frame", type=int,
                   help="Last frame before the ball deflects off the racket")
    p.add_argument("--bounce_frame", type=int,
                   help="Frame of the ball's lowest image point at the bounce")
    p.add_argument("--contact_ground_px", required=True, type=float, nargs=2,
                   metavar=("X", "Y"),
                   help="Pixel of a court point under the contact point")
    p.add_argument("--contact_ground_frame", type=int,
                   help="Frame --contact_ground_px was read on, if not the contact "
                        "frame (the server is usually airborne at contact)")
    p.add_argument("--bounce_px", type=float, nargs=2, metavar=("X", "Y"),
                   help="Bottom of the ball at the bounce (not its centre). "
                        "Required if the detector missed it")
    p.add_argument("--contact_height", type=float, default=2.6,
                   help="Assumed contact height in metres (default 2.6)")
    p.add_argument("--contact_ground_sigma_m", type=float, default=0.5,
                   help="1-sigma error of the contact ground position in metres "
                        "(default 0.5)")
    p.add_argument("--bounce_px_sigma", type=float, default=2.0,
                   help="1-sigma error of the bounce pixel (default 2.0)")
    p.add_argument("--frame_sigma", type=float, default=1.5,
                   help="1-sigma error of each event frame (default 1.5)")
    p.add_argument("--drag_cd", type=float, default=0.55,
                   help="Drag coefficient for the launch estimate (default 0.55)")
    p.add_argument("--exclude_frames", type=int, nargs="*", default=[], metavar="N",
                   help="Track frames confirmed by eye to be false detections. "
                        "Removed from the plots and listed in the report; the "
                        "speed does not use them")
    p.add_argument("--label", default="Serve", help="Title for the outputs")
    p.add_argument("--out_dir", required=True, type=Path)
    return p


def apply_events(args):
    """Fill event frames from detect_events.py, or stop if a serve has no bounce."""
    if args.events is None:
        if args.contact_frame is None or args.bounce_frame is None:
            raise SystemExit("Pass --events, or both --contact_frame and --bounce_frame.")
        args.event_source = "picked by eye"
        return

    if not args.events.is_file():
        raise SystemExit(f"Events file not found: {args.events}")
    serves = json.loads(args.events.read_text())["serves"]
    if not 0 <= args.serve < len(serves):
        raise SystemExit(f"--serve {args.serve}: {args.events} has {len(serves)} serve(s).")
    serve = serves[args.serve]
    if serve["bounce"] is None:
        raise SystemExit(f"No speed for serve {args.serve}: no bounce was detected "
                         f"({serve['no_bounce_reason']}).")
    args.contact_frame = serve["contact_frame"]
    args.bounce_frame = serve["bounce"]["frame"]
    args.bounce_px = serve["bounce"]["px"]
    args.event_source = f"auto-detected ({serve['bounce']['source']})"


def main():
    args = build_parser().parse_args()
    apply_events(args)

    if args.camera == "handheld" and args.video is None:
        raise SystemExit("--camera handheld needs --video: without it, one homography "
                         "would be applied to frames it does not fit.")
    if args.bounce_frame <= args.contact_frame:
        raise SystemExit("--bounce_frame must be after --contact_frame.")

    fps = args.fps
    if fps is None:
        if args.video is None:
            raise SystemExit("Pass --fps or --video: the frame rate sets the speed "
                             "directly and will not be guessed.")
        cap = cv2.VideoCapture(str(args.video))
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        if not fps:
            raise SystemExit(f"Could not read a frame rate from {args.video}")

    calib = load_calibration(args.calibration)
    track = read_track(args.track)

    ground_frame = args.contact_ground_frame or args.contact_frame
    contact_row = require_row(track, args.contact_frame, "Contact")
    raw_bounce_px, bounce_source = bounce_pixel(track, args.bounce_frame, args.bounce_px)

    flight_frames = [f for f in range(args.contact_frame, args.bounce_frame + 1)
                     if f in track and track[f]["status"] in MEASURED_STATUSES
                     and f not in args.exclude_frames]

    motion = None
    if args.camera == "handheld":
        print("Estimating camera motion ...")
        motion = estimate_camera_motion(
            args.video, calib.frame,
            flight_frames + [args.bounce_frame, ground_frame])

    bounce_ref = to_reference(raw_bounce_px, args.bounce_frame, motion)
    ground_ref = to_reference(args.contact_ground_px, ground_frame, motion)
    bounce_xy = ground_point(calib, bounce_ref, "Bounce")
    contact_xy = ground_point(calib, ground_ref, "Contact ground")

    margin = 3.0
    if not (-DOUBLES_MARGIN - margin <= bounce_xy[0] <= SINGLES_W + DOUBLES_MARGIN + margin
            and NET_Y <= bounce_xy[1] <= COURT_L + margin):
        raise SystemExit(f"Bounce projects to {np.round(bounce_xy, 2)}, which is not on "
                         "the far side of the court. Check --bounce_px and the "
                         "calibration.")

    direction = (bounce_xy - contact_xy) / np.linalg.norm(bounce_xy - contact_xy)
    px_scale = metres_per_pixel(calib, bounce_ref, direction)
    bounce_sigma = math.hypot(px_scale * args.bounce_px_sigma, calib.loo_rms_m)

    contact = Endpoint(args.contact_frame, np.r_[contact_xy, args.contact_height],
                       args.contact_ground_sigma_m, args.frame_sigma,
                       f"landing foot (frame {ground_frame}) + assumed height")
    bounce = Endpoint(args.bounce_frame, np.r_[bounce_xy, 0.0], bounce_sigma,
                      args.frame_sigma, f"homography, {bounce_source}")

    est = average_flight_speed(contact, bounce, fps)
    launch = drag_corrected_launch_speed(est.distance_m, est.elapsed_s, args.drag_cd)
    launch_rel = est.sigma_ms / est.speed_ms

    def speeds(v):
        return f"{v:.1f} m/s = {v * MS_TO_KMH:.0f} km/h = {v * MS_TO_MPH:.0f} mph"

    motion_note = "none (tripod: calibration applied to every frame)"
    if motion:
        inliers = [i for _, i in motion.values()]
        motion_note = (f"ORB homography to frame {calib.frame} on {len(motion)} frames, "
                       f"min {min(inliers)} inliers")

    report = f"""
=== {args.label}: rough speed estimate ===

AVERAGE FLIGHT SPEED (contact to bounce): {speeds(est.speed_ms)}
  1-sigma uncertainty: +/- {est.sigma_ms:.1f} m/s (+/- {est.sigma_ms * MS_TO_KMH:.0f} km/h)
  This is NOT a radar-comparable serve speed.

ESTIMATED LAUNCH SPEED (drag-corrected, model-based): {speeds(launch)}
  1-sigma uncertainty: +/- {launch * launch_rel * MS_TO_KMH:.0f} km/h, plus model error
  Assumes quadratic drag, Cd = {args.drag_cd}, no spin, no gravity.

Measurement
  distance:     {est.distance_m:.2f} m (straight line, 3D)
  elapsed:      {est.elapsed_s:.3f} s ({args.bounce_frame - args.contact_frame} frames at {fps:.2f} fps)
  contact:      frame {args.contact_frame}, court ({contact_xy[0]:.2f}, {contact_xy[1]:.2f}) m, height {args.contact_height:.2f} m
                from {contact.source}; track status '{contact_row['status']}'
  bounce:       frame {args.bounce_frame}, court ({bounce_xy[0]:.2f}, {bounce_xy[1]:.2f}) m
                from {bounce.source}; track status '{track[args.bounce_frame]['status']}'

Assumptions
  camera:       {args.camera}
  motion comp:  {motion_note}
  homography:   {calib.path} (frame {calib.frame}, {len(calib.points)} landmarks,
                {calib.rms_px:.2f} px RMS, {calib.loo_rms_m:.3f} m leave-one-out RMS)
  event frames: {args.event_source}, +/- {args.frame_sigma} frames each
  excluded:     {args.exclude_frames or 'none'} (false detections, plots only)
  contact pos:  +/- {args.contact_ground_sigma_m} m along the flight line
  bounce pos:   +/- {bounce_sigma:.2f} m ({px_scale:.3f} m per pixel, {args.bounce_px_sigma} px)

Caveats
  * Rough estimate: planar court model, no camera pose, no lens correction.
  * Not validated against radar. The uncertainty above covers the listed inputs,
    not errors in the method itself.
  * Contact height is assumed, not measured.
""".rstrip() + "\n"
    if args.camera == "handheld":
        report += ("  * Handheld footage: motion compensation assumes the camera mostly "
                   "rotates.\n")
    print(report)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "speed_report.txt").write_text(report, encoding="utf-8")

    frames, flight_px, flight_xy, on_ground = project_flight(
        track, flight_frames, calib, motion)
    label = (f"Average flight speed {est.speed_ms * MS_TO_KMH:.0f} "
             f"± {est.sigma_ms * MS_TO_KMH:.0f} km/h (rough estimate, not radar speed)")
    plot_path = args.out_dir / "path_top_down.png"
    write_top_down_plot(plot_path, args.label, contact_xy, bounce_xy, frames, flight_xy,
                        on_ground, label)
    print(f"Top-down plot: {plot_path}")

    if args.video:
        overlay_path = args.out_dir / "path_overlay.jpg"
        write_image_overlay(overlay_path, args.video, calib, flight_px, ground_ref,
                            bounce_ref)
        print(f"Image overlay: {overlay_path}")


if __name__ == "__main__":
    main()
