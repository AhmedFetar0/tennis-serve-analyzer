"""
serve_mechanics.py

First pass at serve mechanics: how high the server jumps, how deep they load
their legs, how high they reach, how high the toss goes, and the timing
between those moments. All in metres and seconds, from the same single clip.

How pixels become metres here:
    A pose model (YOLO pose) marks 17 body joints in each frame, in pixels.
    One camera cannot say how far away a joint is. But the trajectory fit
    already found WHERE the ball was struck, in 3D. The server is standing
    there at contact. And while the server stands still before the toss,
    their feet are on the court, which pins down where they are standing
    (the ankle's ray is cut at ankle height). Between those two moments the
    server moves forward, and their distance is taken to change steadily.
    Every joint is then assumed to lie in a vertical "body plane" at that
    distance, facing the camera:

        joint pixel -> camera ray -> where the ray crosses the body plane

    That gives each joint a height above the court.

How wrong the plane can be, and what it does:
    The first version used one fixed plane through the contact point. It put
    the feet 0.25 m BELOW the court, because the server starts about a metre
    behind where the ball is struck. A joint that is really nearer than the
    plane is read as too low, and the error grows with how far the joint is
    below the camera: (camera height - joint height) * (distance error /
    distance). Shoulders, near camera height, are barely affected. Ankles
    are affected most. So ankle-based numbers (the jump) are the least
    certain here, and are reported with that warning.

What is deliberately NOT reported:
    Joint ANGLES such as knee bend in degrees. From behind, a knee bends
    toward or away from the camera, which a single view flattens. Leg
    loading is reported as how far the hips drop instead, which is a vertical
    distance and survives any viewing direction.

Built-in check:
    The hitting wrist at contact should be roughly a racket-length below the
    ball (about 0.45 to 0.65 m from wrist to the middle of the strings). The
    report prints that gap. If it is far off, the body plane or the pose is
    wrong, and the other numbers should not be trusted.

Needs a pose model, downloaded automatically on first use:
    weights/yolov8s-pose.pt

Usage (PowerShell), after run_pipeline.py has produced the fit:
    python pipeline/serve_mechanics.py `
        --video "TestRecs\\oncourt test 1\\240\\serve3.MOV" --camera handheld `
        --track tracking\\serve3_auto\\track.csv `
        --calibration tracking\\serve3_auto\\court_serve0.json `
        --trajectory tracking\\serve3_auto\\serve0\\trajectory.json `
        --out_dir tracking\\serve3_auto\\serve0
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from calibrate_court import load_calibration
from camera_pose import solve_camera_pose
from estimate_speed import MEASURED_STATUSES, estimate_camera_motion, read_track

# COCO keypoint order, as produced by YOLO pose models.
JOINTS = ("nose", "left_eye", "right_eye", "left_ear", "right_ear",
          "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
          "left_wrist", "right_wrist", "left_hip", "right_hip",
          "left_knee", "right_knee", "left_ankle", "right_ankle")
J = {name: i for i, name in enumerate(JOINTS)}
SKELETON = [("left_shoulder", "right_shoulder"), ("left_hip", "right_hip"),
            ("left_shoulder", "left_hip"), ("right_shoulder", "right_hip"),
            ("left_shoulder", "left_elbow"), ("left_elbow", "left_wrist"),
            ("right_shoulder", "right_elbow"), ("right_elbow", "right_wrist"),
            ("left_hip", "left_knee"), ("left_knee", "left_ankle"),
            ("right_hip", "right_knee"), ("right_knee", "right_ankle")]

BEFORE_CONTACT_S = 1.6       # how much of the motion to analyse
AFTER_CONTACT_S = 0.35
ANALYSIS_RATE_HZ = 120       # joints are sampled at about this rate, whatever the
                             # clip's frame rate: plenty for a body, and 2x faster
                             # than using every frame of 240 fps footage
MIN_JOINT_CONFIDENCE = 0.5
MIN_PERSON_HEIGHT_PX = 220   # below this the joints are too coarse to measure
STANCE_WINDOW_S = 0.35       # the earliest part of the window: standing still
ANKLE_JOINT_HEIGHT_M = 0.09  # ankle joint above the sole of a shoe on the ground
MAX_STANCE_SPREAD_M = 0.25   # more foot movement than this is not "standing still"
SMOOTH_SAMPLES = 5           # median filter length for joint heights
PLAUSIBLE_WRIST_TO_BALL_M = (0.30, 0.85)


class BodyPlane:
    """A vertical plane that faces the camera and passes through a point on
    the court. The point may differ for every pixel lifted, which is how the
    plane follows the server as they move forward."""

    def __init__(self, pose, ground_xy):
        self.pose = pose
        self.ground_xy = np.atleast_2d(np.asarray(ground_xy, dtype=float))

    def distance(self):
        """Horizontal distance from the camera to the plane, per point."""
        return np.linalg.norm(self.ground_xy - self.pose.centre[:2], axis=1)

    def lift(self, px):
        """Pixels (N, 2) -> 3D points (N, 3) on the plane, court coordinates.

        Each pixel's ray is centre + s * direction. On the plane,
        (X - point) . normal = 0, which gives s directly. The normal is the
        horizontal direction from the camera to the plane's point.
        """
        px = np.asarray(px, dtype=float).reshape(-1, 2)
        rays = self.pose.ray(px)
        towards = np.zeros((len(self.ground_xy), 3))
        towards[:, :2] = self.ground_xy - self.pose.centre[:2]
        normal = towards / np.linalg.norm(towards, axis=1, keepdims=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            s = (towards * normal).sum(axis=1) / (rays * normal).sum(axis=1)
        s = np.where(np.isfinite(s) & (s > 0), s, np.nan)
        return self.pose.centre + s[:, None] * rays


def server_ground_track(pose, keypoints, times, contact_xy):
    """Where the server is standing on the court at each analysed time.

    Stance: both ankle rays are cut at ankle height, which is valid while
    the feet are on the ground. The median over the stance window is used.
    Contact: the fitted contact point.
    In between: a straight line in time from one to the other.

    Returns (ground_xy (F, 2), stance_xy, stance_spread_m).
    """
    stance = times <= times[0] + STANCE_WINDOW_S
    feet = []
    for name in ("left_ankle", "right_ankle"):
        on_ground = pose.ray_at_height(keypoints[stance, J[name]], ANKLE_JOINT_HEIGHT_M)
        feet.append(on_ground[:, :2])
    feet = np.mean(feet, axis=0)                        # midpoint of the two ankles
    feet = feet[np.all(np.isfinite(feet), axis=1)]
    if len(feet) < 5:
        raise SystemExit("No mechanics: the server's feet were not seen while "
                         "standing before the serve.")
    stance_xy = np.median(feet, axis=0)
    spread = float(np.percentile(np.linalg.norm(feet - stance_xy, axis=1), 90))

    stance_end = times[0] + STANCE_WINDOW_S
    progress = np.clip((times - stance_end) / (0.0 - stance_end), 0.0, 1.0)
    ground = stance_xy + progress[:, None] * (np.asarray(contact_xy) - stance_xy)
    return ground, stance_xy, spread


def median_filter(values, length=SMOOTH_SAMPLES):
    """Running median that ignores NaN. Removes single-frame joint glitches
    without rounding off a real peak the way an average would."""
    values = np.asarray(values, dtype=float)
    half = length // 2
    out = np.full_like(values, np.nan)
    for i in range(len(values)):
        window = values[max(0, i - half): i + half + 1]
        if np.any(np.isfinite(window)):
            out[i] = np.nanmedian(window)
    return out


def pick_server(boxes, keypoints, near_px):
    """Index of the detected person the ball was struck by: the one whose
    box is horizontally closest to the contact pixel. None if nobody."""
    if len(boxes) == 0:
        return None
    centre_x = (boxes[:, 0] + boxes[:, 2]) / 2
    return int(np.argmin(np.abs(centre_x - near_px[0])))


def collect_keypoints(video, weights, frames, motion, contact_px_ref, imgsz=1280):
    """Run the pose model. Returns (keypoints, heights_px):
    keypoints (F, 17, 2) in calibration-frame pixels, NaN where a joint was
    not seen confidently; heights_px (F,) the server's box height."""
    try:
        from ultralytics import YOLO
    except ImportError:
        raise SystemExit("Install deps first: pip install -r requirements.txt")
    model = YOLO(str(weights))

    cap = cv2.VideoCapture(str(video))
    keypoints = np.full((len(frames), 17, 2), np.nan)
    box_heights = np.full(len(frames), np.nan)
    for i, frame in enumerate(frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
        ok, image = cap.read()
        if not ok:
            continue
        result = model.predict(image, imgsz=imgsz, conf=0.25, verbose=False)[0]
        boxes = result.boxes.xyxy.cpu().numpy()
        if not len(boxes):
            continue
        points = result.keypoints.data.cpu().numpy()          # (people, 17, 3)
        # The contact pixel is in calibration-frame coordinates; map it into
        # this frame's own pixels before comparing with this frame's boxes.
        near = contact_px_ref
        if motion is not None:
            near = cv2.perspectiveTransform(
                np.float64(contact_px_ref).reshape(1, 1, 2),
                np.linalg.inv(motion[frame][0]))[0, 0]
        who = pick_server(boxes, points, near)
        xy, confidence = points[who, :, :2].astype(float), points[who, :, 2]
        if motion is not None:
            xy = cv2.perspectiveTransform(xy.reshape(-1, 1, 2),
                                          motion[frame][0]).reshape(-1, 2)
        xy[confidence < MIN_JOINT_CONFIDENCE] = np.nan
        keypoints[i] = xy
        box_heights[i] = boxes[who, 3] - boxes[who, 1]
    cap.release()
    return keypoints, box_heights


def measure(times, heights, ball_times, ball_heights, contact_height_m,
            ankle_heights_far=None):
    """Mechanics from joint heights over time.

    times        (F,) seconds relative to contact (negative = before)
    heights      dict of joint name -> (F,) height above the court, metres.
                 Ankles should be placed at the STANCE distance: the feet do
                 not move until take-off.
    ankle_heights_far  optional (2, F) ankle heights placed at the moving
                 body-plane distance instead. The feet at contact are
                 somewhere between the two, so the two give a range for the
                 jump.
    ball_times, ball_heights   the toss, lifted onto the body plane
    Returns a dict of results. Anything that could not be measured is None.
    """
    def series(*names):
        return median_filter(np.nanmean([heights[n] for n in names], axis=0))

    with np.errstate(all="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            hips = series("left_hip", "right_hip")
            ankles = median_filter(np.nanmin([heights["left_ankle"],
                                              heights["right_ankle"]], axis=0))
            shoulders = series("left_shoulder", "right_shoulder")
            wrists = np.array([median_filter(heights["left_wrist"]),
                               median_filter(heights["right_wrist"])])

    stance = times <= times[0] + STANCE_WINDOW_S
    before = times <= 0
    out = {}

    def value_at(values, t):
        i = int(np.argmin(np.abs(times - t)))
        return float(values[i]) if np.isfinite(values[i]) else None

    # Leg loading: how far the hips sink below their standing height before
    # the upward drive, and how long before contact that lowest point is.
    stance_hip = float(np.nanmedian(hips[stance])) if np.any(np.isfinite(hips[stance])) else None
    out["stance_hip_height_m"] = stance_hip
    if stance_hip is not None and np.any(np.isfinite(hips[before])):
        i = int(np.nanargmin(np.where(before, hips, np.nan)))
        out["hip_drop_m"] = stance_hip - float(hips[i])
        out["deepest_load_before_contact_s"] = float(-times[i])
    else:
        out["hip_drop_m"] = out["deepest_load_before_contact_s"] = None

    # Jump: the lower ankle's rise above where it was while standing.
    stance_ankle = float(np.nanmedian(ankles[stance])) if np.any(np.isfinite(ankles[stance])) else None
    out["stance_ankle_height_m"] = stance_ankle
    # Feet off the ground at contact. Placed at the stance distance this
    # reads slightly HIGH (the feet have drifted forward in the air, and a
    # further-away foot looks higher). Placed at the contact distance it
    # reads LOW (the feet trail the ball). The truth is between the two.
    out["jump_height_at_contact_m"] = out["jump_height_low_m"] = None
    if stance_ankle is not None:
        at_contact = value_at(ankles, 0.0)
        if at_contact is not None:
            out["jump_height_at_contact_m"] = at_contact - stance_ankle
        if ankle_heights_far is not None:
            far = median_filter(np.nanmin(ankle_heights_far, axis=0))
            far_at_contact = value_at(far, 0.0)
            if far_at_contact is not None:
                out["jump_height_low_m"] = far_at_contact - stance_ankle
    # The hips rise with the jump too, and sit much nearer camera height, so
    # this second figure is far less sensitive to the body-plane distance.
    # It also includes the legs straightening, so it reads a little higher.
    hips_at_contact = value_at(hips, 0.0)
    out["hip_rise_at_contact_m"] = (hips_at_contact - stance_hip
                                    if None not in (hips_at_contact, stance_hip) else None)

    # Reach: the higher wrist at contact is the hitting hand.
    at_contact = [value_at(w, 0.0) for w in wrists]
    seen = [h for h in at_contact if h is not None]
    out["hitting_side"] = None
    out["wrist_height_at_contact_m"] = out["wrist_to_ball_m"] = None
    if seen:
        out["hitting_side"] = ("left", "right")[int(np.argmax(
            [h if h is not None else -np.inf for h in at_contact]))]
        out["wrist_height_at_contact_m"] = max(seen)
        out["wrist_to_ball_m"] = contact_height_m - max(seen)
    out["shoulder_height_at_contact_m"] = value_at(shoulders, 0.0)
    out["contact_height_m"] = contact_height_m

    # Toss: how far above the contact point the ball went, and for how long
    # it fell. A ball that falls a long way is moving fast when it is struck,
    # which makes timing harder.
    out["toss_peak_height_m"] = out["toss_drop_before_contact_m"] = None
    out["toss_peak_before_contact_s"] = None
    toss = (ball_times < 0) & np.isfinite(ball_heights)
    if toss.sum() >= 10:
        i = int(np.argmax(np.where(toss, ball_heights, -np.inf)))
        out["toss_peak_height_m"] = float(ball_heights[i])
        out["toss_drop_before_contact_m"] = float(ball_heights[i]) - contact_height_m
        out["toss_peak_before_contact_s"] = float(-ball_times[i])

    curves = {"hips": hips, "ankles": ankles, "shoulders": shoulders,
              "hitting_wrist": wrists[int(out["hitting_side"] == "right")]
              if out["hitting_side"] else wrists[0]}
    return out, curves


def write_plot(path, label, times, curves, ball_times, ball_heights, results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ink, muted, grid = "#1f2328", "#59636e", "#d0d7de"
    colours = {"hitting_wrist": "#1d4ed8", "shoulders": "#0f766e", "hips": "#9a3412",
               "ankles": "#6d28d9"}
    names = {"hitting_wrist": "Hitting wrist", "shoulders": "Shoulders", "hips": "Hips",
             "ankles": "Lower ankle"}
    fig, ax = plt.subplots(figsize=(9.5, 5.6), dpi=150)
    toss = (ball_times < 0) & np.isfinite(ball_heights)
    ax.scatter(ball_times[toss], ball_heights[toss], s=6, color="#a3a800", zorder=2,
               label="Ball (toss)")
    for key in ("hitting_wrist", "shoulders", "hips", "ankles"):
        ax.plot(times, curves[key], color=colours[key], lw=2, zorder=3, label=names[key])
    ax.axvline(0, color=ink, lw=1, ls=(0, (4, 3)))
    ax.text(0.01, ax.get_ylim()[1], " contact", color=ink, fontsize=9, va="top")
    ax.scatter([0], [results["contact_height_m"]], marker="X", s=90, color="#c2410c",
               edgecolors="white", linewidths=0.8, zorder=4)
    ax.set_xlabel("Time relative to contact (s)", color=muted)
    ax.set_ylabel("Height above the court (m)", color=muted)
    ax.set_xlim(times[0] - 0.03, times[-1] + 0.03)
    ax.set_ylim(bottom=min(0.0, np.nanmin(curves["ankles"]) - 0.05))
    ax.axhline(0, color=muted, lw=1)
    ax.grid(axis="y", color=grid, lw=0.6)
    ax.tick_params(colors=muted)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.legend(loc="upper left", frameon=False, fontsize=9, ncol=2)
    ax.set_title(f"{label}: joint heights through the serve", color=ink, fontsize=11)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def write_pose_image(path, video, frame, keypoints_ref, motion):
    """The skeleton at contact, drawn on its own frame."""
    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
    ok, image = cap.read()
    cap.release()
    if not ok:
        return
    points = keypoints_ref.copy()
    if motion is not None:
        valid = np.all(np.isfinite(points), axis=1)
        points[valid] = cv2.perspectiveTransform(
            points[valid].reshape(-1, 1, 2), np.linalg.inv(motion[frame][0])).reshape(-1, 2)
    for a, b in SKELETON:
        pa, pb = points[J[a]], points[J[b]]
        if np.all(np.isfinite(pa)) and np.all(np.isfinite(pb)):
            cv2.line(image, tuple(int(v) for v in pa), tuple(int(v) for v in pb),
                     (255, 160, 40), 3, cv2.LINE_AA)
    for p in points[5:]:
        if np.all(np.isfinite(p)):
            cv2.circle(image, (int(p[0]), int(p[1])), 5, (255, 255, 255), -1, cv2.LINE_AA)
    cv2.imwrite(str(path), image)


def fmt(value, unit="m", places=2, signed=False):
    if value is None:
        return "not measured"
    return f"{value:{'+' if signed else ''}.{places}f} {unit}"


def build_report(label, r, coverage, person_px, stance_distance_m, contact_distance_m,
                 stance_spread_m):
    gap = r["wrist_to_ball_m"]
    if gap is None:
        check = "could not be made (hitting wrist not seen at contact)"
    elif PLAUSIBLE_WRIST_TO_BALL_M[0] <= gap <= PLAUSIBLE_WRIST_TO_BALL_M[1]:
        check = "PASSED: consistent with a racket in the hand"
    else:
        check = ("FAILED: not a racket-length. Treat every number in this report "
                 "with suspicion")
    stance_note = ("." if stance_spread_m <= MAX_STANCE_SPREAD_M else
                   ": the server was NOT standing still, so the leg figures are unreliable.")
    return f"""=== {label}: serve mechanics (first pass) ===

Legs
  hips drop while loading:      {fmt(r['hip_drop_m'])}
  deepest point of the load:    {fmt(r['deepest_load_before_contact_s'], 's')} before contact
  hips rise, stance to contact: {fmt(r['hip_rise_at_contact_m'])} (jump plus legs straightening)
  feet off ground at contact:   between {fmt(r['jump_height_low_m'])} and {fmt(r['jump_height_at_contact_m'])} (see below)

Reach
  ball struck at:               {fmt(r['contact_height_m'])} (from the trajectory fit)
  hitting wrist at contact:     {fmt(r['wrist_height_at_contact_m'])} ({r['hitting_side'] or 'unknown'} hand)
  shoulders at contact:         {fmt(r['shoulder_height_at_contact_m'])}

Toss
  peak height:                  {fmt(r['toss_peak_height_m'])}
  ball falls before contact:    {fmt(r['toss_drop_before_contact_m'])}
  peak to contact:              {fmt(r['toss_peak_before_contact_s'], 's')}

Check: wrist to ball at contact = {fmt(gap)}
  {check}

How these were measured
  Joints from a pose model, placed on a vertical plane facing the camera.
  The plane starts where the server stands before the toss ({stance_distance_m:.1f} m from the
  camera, from their feet on the court) and moves steadily to the fitted
  contact point ({contact_distance_m:.1f} m) by the moment of contact.
  The player is {person_px:.0f} px tall in the frame. Joints were found in {coverage:.0%}
  of the frames analysed.
  Feet during the stance window moved {stance_spread_m:.2f} m{stance_note}
  The feet-off-ground figure is a range because the feet are far below the
  camera, which makes their height sensitive to how far away they are. The
  upper value assumes the feet are still at the stance distance, the lower
  that they are level with the ball. They are somewhere between. The hips
  are much less affected.
  Not validated against any other measurement.
"""


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--video", required=True, type=Path)
    p.add_argument("--camera", required=True, choices=("handheld", "tripod"))
    p.add_argument("--track", required=True, type=Path)
    p.add_argument("--calibration", required=True, type=Path)
    p.add_argument("--trajectory", required=True, type=Path,
                   help="trajectory.json written by fit_trajectory.py for this serve")
    p.add_argument("--pose_weights", type=Path, default=Path("weights/yolov8s-pose.pt"))
    p.add_argument("--label", default=None)
    p.add_argument("--out_dir", required=True, type=Path)
    args = p.parse_args(argv)

    fit = json.loads(args.trajectory.read_text())
    label = args.label or fit.get("label", "Serve")
    fps, contact_frame = fit["fps"], fit["contact_frame"]
    summary = fit["summary"]
    calib = load_calibration(args.calibration)
    pose = solve_camera_pose(calib)
    contact_xy = (summary["contact_x_m"], summary["contact_y_m"])
    contact_xyz = [summary["contact_x_m"], summary["contact_y_m"],
                   summary["contact_height_m"]]
    contact_px_ref = pose.project([contact_xyz])[0]

    first = max(0, contact_frame - int(BEFORE_CONTACT_S * fps))
    last = contact_frame + int(AFTER_CONTACT_S * fps)
    frames = list(range(first, last + 1, max(1, round(fps / ANALYSIS_RATE_HZ))))
    track = read_track(args.track)
    frames = [f for f in frames if f in track]
    toss_frames = [f for f in range(first, contact_frame)
                   if f in track and track[f]["status"] in MEASURED_STATUSES]

    motion = None
    if args.camera == "handheld":
        print("Estimating camera motion ...")
        motion = estimate_camera_motion(args.video, calib.frame,
                                        frames + toss_frames + [contact_frame])

    print(f"Finding the server's joints in {len(frames)} frames ...")
    keypoints, box_heights = collect_keypoints(args.video, args.pose_weights, frames,
                                               motion, contact_px_ref)
    seen = np.isfinite(keypoints[:, [J["left_hip"], J["right_hip"]], 0]).any(axis=1)
    if seen.mean() < 0.5:
        raise SystemExit(f"No mechanics: the server's hips were found in only "
                         f"{seen.mean():.0%} of frames.")
    person_px = float(np.nanmedian(box_heights))
    if person_px < MIN_PERSON_HEIGHT_PX:
        raise SystemExit(
            f"No mechanics: the server is only {person_px:.0f} px tall in the frame "
            f"(need {MIN_PERSON_HEIGHT_PX}). Joints cannot be placed accurately "
            "enough from this distance; film from the server's end.")

    times = (np.array(frames) - contact_frame) / fps
    ground, stance_xy, stance_spread = server_ground_track(pose, keypoints, times,
                                                           contact_xy)
    body = BodyPlane(pose, ground)
    heights = {}
    for name in JOINTS:
        heights[name] = body.lift(keypoints[:, J[name]])[:, 2]
    # Feet stay where they are until take-off, so ankles are placed at the
    # stance distance. The moving-plane version is kept as the other end of
    # the range for the jump.
    ankle_heights_far = np.array([heights["left_ankle"], heights["right_ankle"]])
    stance_plane = BodyPlane(pose, stance_xy)
    for name in ("left_ankle", "right_ankle"):
        heights[name] = stance_plane.lift(keypoints[:, J[name]])[:, 2]
    # The ball is tossed toward where it will be struck, so it is placed on
    # the plane through the contact point for the whole toss.
    plane = BodyPlane(pose, contact_xy)

    ball_px = np.array([[float(track[f]["raw_x"]), float(track[f]["raw_y"])]
                        for f in toss_frames]).reshape(-1, 2)
    if motion is not None and len(ball_px):
        ball_px = np.array([cv2.perspectiveTransform(
            np.float64(px).reshape(1, 1, 2), motion[f][0])[0, 0]
            for px, f in zip(ball_px, toss_frames)])
    ball_times = (np.array(toss_frames) - contact_frame) / fps
    ball_heights = plane.lift(ball_px)[:, 2] if len(ball_px) else np.array([])
    # The tracker sometimes sits on a ball lying on the court before the toss.
    # Only detections above head height count as the toss.
    ball_heights = np.where(ball_heights > 1.0, ball_heights, np.nan)

    results, curves = measure(times, heights, ball_times, ball_heights,
                              summary["contact_height_m"], ankle_heights_far)
    stance_distance = float(np.linalg.norm(stance_xy - pose.centre[:2]))
    contact_distance = float(plane.distance()[0])
    results["stance_position_m"] = [float(v) for v in stance_xy]
    report = build_report(label, results, float(seen.mean()), person_px, stance_distance,
                          contact_distance, stance_spread)
    print("\n" + report)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "mechanics_report.txt").write_text(report, encoding="utf-8")
    (args.out_dir / "mechanics.json").write_text(json.dumps(
        {"label": label, "results": results, "player_height_px": person_px,
         "stance_distance_m": stance_distance, "contact_distance_m": contact_distance,
         "times_s": [round(float(t), 4) for t in times],
         "curves_m": {k: [None if not np.isfinite(v) else round(float(v), 3) for v in c]
                      for k, c in curves.items()}}, indent=1), encoding="utf-8")
    plot_path = args.out_dir / "mechanics_heights.png"
    write_plot(plot_path, label, times, curves, ball_times, ball_heights, results)
    print(f"Plot: {plot_path}")
    index = int(np.argmin(np.abs(np.array(frames) - contact_frame)))
    image_path = args.out_dir / "pose_contact.jpg"
    write_pose_image(image_path, args.video, frames[index], keypoints[index], motion)
    print(f"Pose at contact: {image_path}")
    return results


if __name__ == "__main__":
    main()
