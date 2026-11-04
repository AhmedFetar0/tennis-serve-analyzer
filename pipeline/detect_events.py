"""
detect_events.py

Finds serve contact and bounce frames automatically from a ball track.

CONTACT (uses track.csv only):
    A sudden velocity change is not enough: ball bounces before the serve and
    catches produce them too. A contact must also follow a toss, meaning a clean
    free-fall arc, and must not reverse upward as a ground bounce does. The frame
    is where the ball first leaves the parabola fitted to the toss.

BOUNCE (uses the video):
    Near the bounce the ball is a few pixels wide and the detector usually loses
    it. It is still the small bright thing moving against a still background, so
    frames are stabilised and differenced against neighbours to get candidates.
    The bounce is where a descent line, anchored on the real track, meets a
    separately found ascent line.

    If either line lacks evidence, the serve reports NO bounce. It does not fall
    back to a guess, and estimate_speed.py gives no speed for that serve.

Output is events.json, which estimate_speed.py reads with --events.

Usage (PowerShell):
    python pipeline/detect_events.py --track tracking\\serve3_fixed\\track.csv `
        --video "TestRecs\\oncourt test 1\\240\\serve3.MOV" `
        --out tracking\\serve3_fixed\\events.json
"""

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np

from estimate_speed import estimate_camera_motion

MEASURED_STATUSES = {"updated", "init"}

# Times are in seconds so the rules hold at other frame rates.
JUMP_WINDOW_S = 0.04        # velocity measured over this long on each side
MIN_FALL_S = 0.20           # toss peak to contact: real tosses fall at least this
MAX_FALL_S = 0.80           # long, and are not looked for further back than this
MIN_FLIGHT_S = 0.35         # contact-to-bounce limits for a real serve
MAX_FLIGHT_S = 1.30

MIN_JUMP_PX = 0.6           # px/frame velocity change worth considering, and
MIN_JUMP_RATIO = 0.5        # relative to fall speed: a far ball moves few pixels
MIN_TOSS_PX = 40            # height of the toss arc on each side of its peak
MAX_TOSS_RMS_PX = 6.0       # parabola fit limit; handheld shake adds a few px
TOSS_COVERAGE = 0.6         # fraction of toss frames needing a detection

BLOB_THRESHOLD = 12         # grey levels brighter than both neighbours
DIFF_GAP_FRAMES = 6
MIN_BLOB_AREA, MAX_BLOB_AREA, MAX_BLOB_SIDE = 4, 60, 12

LINE_TOLERANCE_PX = 2.5
MIN_LINE_POINTS = 8
MIN_ASCENT_SPAN_S = 0.08
MAX_DESCENT_GAP_S = 0.10
TRAILING_WINDOW_FRAMES = 8
TRAILING_MIN_POINTS = 3
MAX_STEP_RATIO = 6          # px/frame step, against the flight's median step
ASCENT_SEARCH_PX = 60       # how far across the image an ascent may be
MEET_BEFORE_FRAMES = 6      # the bounce lies this close to the last descent point
MEET_AFTER_FRAMES = 25
MEET_TOLERANCE_PX = 8       # ascent and descent must meet this closely in x


def load_detections(track_path: Path):
    with track_path.open(newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["status"] in MEASURED_STATUSES]
    return np.array([[int(r["frame"]), float(r["raw_x"]), float(r["raw_y"])]
                     for r in rows])


def _slope(frames, values):
    return np.polyfit(frames, values, 1)[0]


def velocity_jumps(det, half_window):
    """(frame, jump, vy_before, vy_after) wherever both sides have enough data."""
    frames = det[:, 0]
    need = max(3, int(0.7 * half_window))
    jumps = []
    for f in range(int(frames[0]), int(frames[-1]) + 1):
        before = (frames >= f - half_window) & (frames < f)
        after = (frames > f) & (frames <= f + half_window)
        if before.sum() < need or after.sum() < need:
            continue
        vb = [_slope(frames[before], det[before, i]) for i in (1, 2)]
        va = [_slope(frames[after], det[after, i]) for i in (1, 2)]
        jumps.append((f, float(np.hypot(va[0] - vb[0], va[1] - vb[1])), vb[1], va[1]))
    return jumps


def find_apex(det, frame, fps):
    """Frame of the toss peak before `frame`, or None if there was no real toss.

    Image y grows downward, so the peak is the smallest y. It must come well
    before the jump, with the ball clearly lower on both sides of it.
    """
    frames = det[:, 0]
    window = (frames >= frame - MAX_FALL_S * fps) & (frames < frame)
    if window.sum() < TOSS_COVERAGE * MIN_FALL_S * fps:
        return None
    apex_i = np.where(window)[0][np.argmin(det[window, 2])]
    apex, apex_y = det[apex_i, 0], det[apex_i, 2]
    if frame - apex < MIN_FALL_S * fps:
        return None
    rise = det[(frames >= apex - 2 * MIN_FALL_S * fps) & (frames < apex)]
    fall = det[(frames > apex) & (frames < frame)]
    if not len(rise) or not len(fall):
        return None
    if rise[:, 2].max() - apex_y < MIN_TOSS_PX or fall[:, 2].max() - apex_y < MIN_TOSS_PX:
        return None
    return int(apex)


def is_contact(det, frame, vy_before, vy_after, half_window, fps):
    """True if a velocity jump at `frame` follows a toss and is not a ground bounce."""
    frames = det[:, 0]
    apex = find_apex(det, frame, fps)
    if apex is None or vy_before <= 0 or vy_after < -0.25 * abs(vy_before):
        return False                 # no toss, not falling, or a ground bounce

    fit_end = frame - half_window // 2
    toss = (frames >= apex) & (frames < fit_end)
    if toss.sum() < TOSS_COVERAGE * (fit_end - apex):
        return False
    t = frames[toss] - fit_end
    fit_y = np.polyfit(t, det[toss, 2], 2)
    rms = float(np.sqrt(np.mean((np.polyval(fit_y, t) - det[toss, 2]) ** 2)))
    return rms <= MAX_TOSS_RMS_PX and fit_y[0] > 0


def find_contacts(det, fps):
    half_window = max(3, round(JUMP_WINDOW_S * fps))
    jumps = sorted(velocity_jumps(det, half_window), key=lambda j: -j[1])

    contacts, seen = [], []
    for frame, jump, vy_before, vy_after in jumps:
        if jump < MIN_JUMP_PX:
            break
        if jump < MIN_JUMP_RATIO * abs(vy_before):
            continue
        if any(abs(frame - s) <= 2 * half_window for s in seen):
            continue
        seen.append(frame)
        # Jumps are visited largest first, so the first in a neighbourhood is
        # its peak, which is the contact frame.
        if (is_contact(det, frame, vy_before, vy_after, half_window, fps)
                and all(abs(frame - c) > MIN_FALL_S * fps for c in contacts)):
            contacts.append(frame)
    return sorted(contacts)


def blob_candidates(video, ref_frame, first, last, roi, seed_frames):
    """Small bright moving blobs, in the stabilised pixels of ref_frame.

    Returns rows of (frame, x, y, height) and the camera motion, which also
    covers seed_frames. A blob must be brighter than the same place both before
    and after, which removes the ghost plain differencing leaves at the ball's
    previous position.
    """
    frames = list(range(first - DIFF_GAP_FRAMES, last + DIFF_GAP_FRAMES + 1))
    motion = estimate_camera_motion(video, ref_frame, frames + seed_frames)
    x0, y0, x1, y1 = roi

    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frames[0])
    stable = {}
    for f in frames:
        ok, image = cap.read()
        if not ok:
            break
        grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        warped = cv2.warpPerspective(grey, motion[f][0], grey.shape[::-1])
        stable[f] = warped[y0:y1, x0:x1].astype(np.int16)
    cap.release()

    found = []
    for f in range(first, last + 1):
        if f - DIFF_GAP_FRAMES not in stable or f + DIFF_GAP_FRAMES not in stable:
            continue
        diff = np.minimum(stable[f] - stable[f - DIFF_GAP_FRAMES],
                          stable[f] - stable[f + DIFF_GAP_FRAMES])
        count, _, stats, centres = cv2.connectedComponentsWithStats(
            (diff > BLOB_THRESHOLD).astype(np.uint8))
        for i in range(1, count):
            _, _, w, h, area = stats[i]
            if MIN_BLOB_AREA <= area <= MAX_BLOB_AREA and max(w, h) <= MAX_BLOB_SIDE:
                found.append((f, centres[i][0] + x0, centres[i][1] + y0, h))
    return np.array(found, dtype=float).reshape(-1, 4), motion


def fit_descent(seed, candidates, max_gap):
    """Grow the seed track forward through candidates that stay on its line.

    max_gap is about how long the ball can hide behind the net tape. Any longer
    and a stray blob on the extended line is more likely than the ball.
    """
    points = seed[:, :3].copy()
    for row in candidates[np.argsort(candidates[:, 0])]:
        recent = points[-30:]
        fx = np.polyfit(recent[:, 0], recent[:, 1], 1)
        fy = np.polyfit(recent[:, 0], recent[:, 2], 1)
        gap = row[0] - points[-1, 0]
        # The ball's sideways drift is small and poorly known from the seed, so
        # x gets a looser limit than y, especially across gaps behind the net.
        y_tolerance = LINE_TOLERANCE_PX + 0.05 * gap
        x_tolerance = LINE_TOLERANCE_PX + 0.6 * gap
        if 0 < gap <= max_gap and (abs(np.polyval(fx, row[0]) - row[1]) < x_tolerance
                                   and abs(np.polyval(fy, row[0]) - row[2]) < y_tolerance):
            points = np.vstack([points, row[:3]])

    # A real descent ends on a dense run of points. Lone points past it are
    # stray blobs that happened to sit on the extended line.
    while len(points) > len(seed):
        close = (points[:-1, 0] >= points[-1, 0] - TRAILING_WINDOW_FRAMES).sum()
        if close >= TRAILING_MIN_POINTS:
            break
        points = points[:-1]
    return points[len(seed):], points


def fit_ascent(candidates, descent_end, fit_x, fit_y, min_span):
    """Most-supported rising line that meets the end of the descent.

    Meeting it matters: other things move upward in the image, and without this
    the search happily returns one of them. Exhaustive pair search.
    """
    end_frame = descent_end[0]
    pts = candidates[(candidates[:, 0] > end_frame - MEET_BEFORE_FRAMES)
                     & (np.abs(candidates[:, 1] - descent_end[1]) < ASCENT_SEARCH_PX)]
    best = None
    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            span = pts[j, 0] - pts[i, 0]
            if span < 8:
                continue
            vy = (pts[j, 2] - pts[i, 2]) / span
            vx = (pts[j, 1] - pts[i, 1]) / span
            if not -4.0 < vy < -0.3 or abs(vx) > 3.0:
                continue
            # Where this line crosses the descent, in time and across the image.
            meet = (pts[i, 2] - vy * pts[i, 0] - fit_y[1]) / (fit_y[0] - vy)
            if not end_frame - MEET_BEFORE_FRAMES <= meet <= end_frame + MEET_AFTER_FRAMES:
                continue
            x_gap = pts[i, 1] + vx * (meet - pts[i, 0]) - np.polyval(fit_x, meet)
            if abs(x_gap) > MEET_TOLERANCE_PX:
                continue
            dt = pts[:, 0] - pts[i, 0]
            near = (np.abs(pts[i, 2] + vy * dt - pts[:, 2]) < LINE_TOLERANCE_PX) & \
                   (np.abs(pts[i, 1] + vx * dt - pts[:, 1]) < LINE_TOLERANCE_PX)
            if best is None or near.sum() > best.sum():
                best = near
    if best is None:
        return None
    inliers = pts[best]
    if (len(np.unique(inliers[:, 0])) < MIN_LINE_POINTS
            or np.ptp(inliers[:, 0]) < min_span):
        return None
    return inliers


def bounce_from_track(flight, fps):
    """Bounce seen directly by the detector, or None.

    Needs the ball falling into its lowest image point and rising out of it.
    Returns the box centre: the track has no box size to find the ball's bottom.
    """
    low = int(np.argmax(flight[:, 2]))
    window = 0.25 * fps
    before = flight[(flight[:, 0] < flight[low, 0]) & (flight[:, 0] >= flight[low, 0] - window)]
    after = flight[(flight[:, 0] > flight[low, 0]) & (flight[:, 0] <= flight[low, 0] + window)]
    if len(before) < MIN_LINE_POINTS or len(after) < MIN_LINE_POINTS:
        return None
    if _slope(before[:, 0], before[:, 2]) <= 0 or _slope(after[:, 0], after[:, 2]) >= 0:
        return None
    return {"frame": int(flight[low, 0]),
            "px": [round(float(flight[low, 1]), 1), round(float(flight[low, 2]), 1)],
            "descent_points": len(before), "ascent_points": len(after),
            "source": "track, box centre"}


def find_bounce(det, contact, video, fps):
    """Returns (bounce dict, None) or (None, reason it was not found)."""
    flight = det[(det[:, 0] > contact) & (det[:, 0] <= contact + MAX_FLIGHT_S * fps)]
    if len(flight) < MIN_LINE_POINTS:
        return None, "too few detections after contact"

    # The real flight ends at a long gap, or at a jump far faster than the ball
    # was moving: that is the track hopping to something else.
    dt = np.diff(flight[:, 0])
    step = np.hypot(np.diff(flight[:, 1]), np.diff(flight[:, 2])) / dt
    breaks = np.where((dt > 0.05 * fps) | (step > MAX_STEP_RATIO * np.median(step) + 2))[0]
    if len(breaks):
        flight = flight[:breaks[0] + 1]
    direct = bounce_from_track(flight, fps)
    if direct:
        return direct, None
    ref = int(flight[-1, 0])
    last_x, last_y = flight[-1, 1:3]

    first = ref + 1
    last = int(contact + MAX_FLIGHT_S * fps)
    roi = (max(int(last_x - 130), 0), max(int(last_y - 50), 0),
           int(last_x + 130), int(last_y + 190))
    seed = flight[-min(30, len(flight)):]
    candidates, motion = blob_candidates(video, ref, first, last, roi,
                                         [int(f) for f in seed[:, 0]])
    if not len(candidates):
        return None, "no moving blobs found near the end of the track"

    seed_ref = np.array([
        [f, *cv2.perspectiveTransform(np.float64([[[x, y]]]), motion[int(f)][0])[0, 0]]
        for f, x, y in seed])

    descent, all_descent = fit_descent(seed_ref, candidates, MAX_DESCENT_GAP_S * fps)
    if len(descent) < MIN_LINE_POINTS:
        return None, f"descent not followed ({len(descent)} points)"

    down = np.polyfit(*all_descent[-20:, [0, 2]].T, 1)
    x_fit = np.polyfit(*all_descent[-20:, [0, 1]].T, 1)
    ascent = fit_ascent(candidates, descent[-1], x_fit, down, MIN_ASCENT_SPAN_S * fps)
    if ascent is None:
        return None, "no ascent after the bounce: the ball was not seen rising"

    up = np.polyfit(ascent[:, 0], ascent[:, 2], 1)
    frame = (up[1] - down[1]) / (down[0] - up[0])
    # The rise is curved, so a straight line meets the descent too early. A ball
    # seen still descending has not bounced yet, whatever the fit says.
    frame = max(frame, descent[-1, 0])
    if not descent[0, 0] <= frame <= ascent[-1, 0]:
        return None, f"descent and ascent do not meet sensibly (frame {frame:.0f})"
    flight_s = (frame - contact) / fps
    if not MIN_FLIGHT_S <= flight_s <= MAX_FLIGHT_S:
        return None, f"implausible flight time {flight_s:.2f} s"

    bounce_frame = int(round(frame))
    centre = (np.polyval(x_fit, frame), np.polyval(down, frame))
    radius = float(np.median(candidates[:, 3])) / 2

    # Back from stabilised pixels to the bounce frame's own pixels. The bounce
    # pixel is the bottom of the ball, where it touches the court.
    px = cv2.perspectiveTransform(np.float64([[[centre[0], centre[1] + radius]]]),
                                  np.linalg.inv(motion[bounce_frame][0]))[0, 0]
    return {"frame": bounce_frame, "px": [round(float(px[0]), 1), round(float(px[1]), 1)],
            "descent_points": int(len(descent)), "ascent_points": int(len(ascent)),
            "source": "frame differencing, ball bottom"}, None


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--track", required=True, type=Path)
    p.add_argument("--video", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    args = p.parse_args()

    if not args.track.is_file():
        raise SystemExit(f"Track file not found: {args.track}")
    cap = cv2.VideoCapture(str(args.video))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    if not fps:
        raise SystemExit(f"Could not read a frame rate from {args.video}")

    det = load_detections(args.track)
    contacts = find_contacts(det, fps)
    print(f"{len(contacts)} contact(s) found: {contacts}")

    serves = []
    for contact in contacts:
        bounce, reason = find_bounce(det, contact, args.video, fps)
        serves.append({"contact_frame": contact, "bounce": bounce, "no_bounce_reason": reason})
        if bounce:
            print(f"  contact {contact}: bounce at frame {bounce['frame']}, pixel "
                  f"{bounce['px']} ({bounce['descent_points']} descent, "
                  f"{bounce['ascent_points']} ascent points)")
        else:
            print(f"  contact {contact}: NO BOUNCE ({reason}). No speed for this serve.")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"fps": fps, "serves": serves}, indent=2))
    print(f"Events: {args.out}")


if __name__ == "__main__":
    main()
