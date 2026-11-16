"""
detect_court.py

Finds the court in a frame AUTOMATICALLY and writes the same calibration file
that calibrate_court.py produces from hand-clicked landmarks.

The idea, in four steps:

  1. FIND WHITE LINES. Court lines are thin, bright and colourless. A
     "top-hat" filter keeps bright things narrower than its window, which
     removes large bright areas such as the sky. A Hough transform then turns
     the surviving pixels into straight segments, and segments lying along
     the same line are merged.

     This finds far more than the court: fence rails, the net tape, the
     court next door. No attempt is made to tell them apart at this stage.

  2. GUESS AND CHECK. Take any two detected lines and suppose they are two of
     the court's cross lines (baselines, service lines). Take two more and
     suppose they are two of its long lines (sidelines, centre line). Four
     lines cross at four points, which is exactly enough to compute a
     homography. Every such supposition is tried: every choice of four
     detected lines, against every choice of four model lines.

     That is hundreds of thousands of guesses, but each one is cheap, and
     almost all are thrown out immediately because the camera they imply is
     impossible (a negative focal length, or a camera under the ground).

  3. SCORE. For each surviving guess the whole court model is drawn into the
     image. A correct guess puts every model line on top of white pixels. A
     wrong one puts most of them on bare court. The fence and the net tape
     can each be mistaken for one line, but they cannot explain all nine.

  4. REFINE. With the right guess known, each model line picks up the white
     pixels near it and a straight line is fitted through them. Landmarks are
     the crossings of those fitted lines, which is more precise than a click:
     each line is fitted through hundreds of pixels.

The approach follows Farin et al., "Robust camera calibration for sport
videos using court models" (2004), with the camera-plausibility pruning added.

Limits:
    * Needs white lines that are clearly brighter than the surface. The
      thresholds below were set on dusk footage of a blue hard court.
    * At least two cross lines and two long lines must be visible.
    * A court is symmetric, so "near" is taken to be the end closer to the
      camera. That matches calibrate_court.py's convention.

Usage (PowerShell):
    python pipeline/detect_court.py --video "TestRecs\\oncourt test 1\\240\\serve3.MOV" `
        --frame 1014 --out tracking\\serve3\\court_auto_f1014.json `
        --preview court_auto_check.jpg
"""

import argparse
import itertools
import json
import math
from pathlib import Path

import cv2
import numpy as np

from calibrate_court import (ALL_LANDMARKS, CENTRE_X, COURT_L, DOUBLES_MARGIN,
                             FAR_SERVICE_Y, MIN_POINTS, NET_Y, SERVICE_D, SINGLES_W,
                             draw_court, leave_one_out_rms, load_calibration,
                             read_frame, report)
from camera_pose import describe, solve_camera_pose, solve_pose_from_points

# --- Step 1: white-line pixels ------------------------------------------------
TOPHAT_KERNEL_PX = 21        # wider than any court line in a 1080p frame
TOPHAT_MIN = 35              # grey levels brighter than the surroundings
MIN_BRIGHTNESS = 120         # absolute grey level (0-255)
MAX_SATURATION = 90          # white has little colour (0-255 scale)

HOUGH_THRESHOLD = 60
HOUGH_MIN_LENGTH_PX = 60
HOUGH_MAX_GAP_PX = 12
MERGE_DISTANCE_PX = 4.0      # segment endpoints this close to a line join it
MIN_LINE_SUPPORT_PX = 120    # total length of white along a line to keep it
MAX_LINES = 14               # strongest lines passed to the search

# --- Step 2: plausibility of the camera a guess implies -------------------------
MIN_FOCAL_FRACTION, MAX_FOCAL_FRACTION = 0.4, 3.0     # of image width
MIN_CAMERA_HEIGHT_M, MAX_CAMERA_HEIGHT_M = 0.3, 12.0
MAX_CAMERA_DISTANCE_M = 60.0

# --- Step 3: scoring ------------------------------------------------------------
SAMPLE_SPACING_M = 0.25      # model lines are checked every 25 cm
HIT_DISTANCE_PX = 3.0
MISS_PENALTY = 0.6
MAX_SAMPLE_WEIGHT_PX = 20.0  # most image length one sample may stand for
MIN_HIT_LENGTH_PX = 1500     # length of painted line confirmed in the image
MIN_HIT_FRACTION = 0.6

# --- Acceptance: a detection is refused unless it passes all of these ----------
MIN_FINAL_HIT_FRACTION = 0.8     # of the model's visible line length on white
MAX_LEAVE_ONE_OUT_M = 0.20       # landmarks must predict each other this well
MAX_POSE_RMS_PX = 8.0

# --- Step 4: refinement ---------------------------------------------------------
REFINE_DISTANCE_PX = 5.0
REFINE_MIN_PIXELS = 40
REFINE_MIN_EXTENT_PX = 30

# The court model as real painted segments: (name, kind, constant, from, to).
# A "cross" line has constant y and runs in x; a "long" line the reverse.
LEFT, RIGHT = -DOUBLES_MARGIN, SINGLES_W + DOUBLES_MARGIN
MODEL_LINES = [
    ("near_baseline", "cross", 0.0, LEFT, RIGHT),
    ("near_service", "cross", SERVICE_D, 0.0, SINGLES_W),
    ("far_service", "cross", FAR_SERVICE_Y, 0.0, SINGLES_W),
    ("far_baseline", "cross", COURT_L, LEFT, RIGHT),
    ("left_doubles", "long", LEFT, 0.0, COURT_L),
    ("left_singles", "long", 0.0, 0.0, COURT_L),
    ("centre", "long", CENTRE_X, SERVICE_D, FAR_SERVICE_Y),
    ("right_singles", "long", SINGLES_W, 0.0, COURT_L),
    ("right_doubles", "long", RIGHT, 0.0, COURT_L),
]
CROSS_Y = [m[2] for m in MODEL_LINES if m[1] == "cross"]
LONG_X = [m[2] for m in MODEL_LINES if m[1] == "long"]


def line_mask(image):
    """Binary image of pixels that look like part of a painted white line."""
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    saturation = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)[..., 1]
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                       (TOPHAT_KERNEL_PX, TOPHAT_KERNEL_PX))
    # Top-hat = image minus its "opening". Opening erases bright features
    # narrower than the kernel, so the difference is exactly those features.
    tophat = cv2.morphologyEx(grey, cv2.MORPH_TOPHAT, kernel)
    mask = (tophat > TOPHAT_MIN) & (grey > MIN_BRIGHTNESS) & (saturation < MAX_SATURATION)
    return mask.astype(np.uint8) * 255


def _fit_line(points, weights=None):
    """Best straight line through points, as (a, b, c) with a*x + b*y + c = 0
    and a^2 + b^2 = 1, so that a*x + b*y + c is a signed distance in pixels.

    Total least squares: the line passes through the centroid along the
    direction of greatest spread. Unlike y = m*x + k, it works for vertical
    lines too.
    """
    points = np.asarray(points, dtype=float)
    weights = np.ones(len(points)) if weights is None else np.asarray(weights, float)
    centroid = (points * weights[:, None]).sum(0) / weights.sum()
    centred = (points - centroid) * np.sqrt(weights)[:, None]
    direction = np.linalg.svd(centred, full_matrices=False)[2][0]
    normal = np.array([-direction[1], direction[0]])
    return np.r_[normal, -normal @ centroid]


def detect_lines(mask):
    """Straight lines in the mask, strongest first.

    Returns a list of (line, support) where line is (a, b, c) and support is
    the length of white found along it, in pixels.
    """
    segments = cv2.HoughLinesP(mask, 1, np.pi / 360, HOUGH_THRESHOLD,
                               minLineLength=HOUGH_MIN_LENGTH_PX,
                               maxLineGap=HOUGH_MAX_GAP_PX)
    if segments is None:
        return []
    segments = segments.reshape(-1, 4).astype(float)
    lengths = np.hypot(segments[:, 2] - segments[:, 0], segments[:, 3] - segments[:, 1])

    # Hough reports one painted line as many overlapping segments. Starting
    # from the longest, each segment either joins an existing line (both its
    # ends lie on it) or starts a new one.
    clusters = []
    for i in np.argsort(-lengths):
        ends = segments[i].reshape(2, 2)
        for cluster in clusters:
            if np.all(np.abs(ends @ cluster["line"][:2] + cluster["line"][2])
                      < MERGE_DISTANCE_PX):
                cluster["points"].extend(ends)
                cluster["weights"].extend([lengths[i]] * 2)
                cluster["line"] = _fit_line(cluster["points"], cluster["weights"])
                break
        else:
            clusters.append({"line": _fit_line(ends), "points": list(ends),
                             "weights": [lengths[i]] * 2})

    lines = []
    for cluster in clusters:
        line = cluster["line"]
        # Support = how much of the line is covered, counting overlaps once.
        along = np.array([-line[1], line[0]])
        spans = np.sort((np.array(cluster["points"]) @ along).reshape(-1, 2), axis=1)
        covered, reach = 0.0, -np.inf
        for start, end in spans[np.argsort(spans[:, 0])]:
            covered += max(0.0, end - max(start, reach))
            reach = max(reach, end)
        if covered >= MIN_LINE_SUPPORT_PX:
            lines.append((line, covered))
    lines.sort(key=lambda item: -item[1])
    return lines[:MAX_LINES]


def _intersect(l1, l2):
    """Crossing point of two lines (a, b, c). Arrays broadcast; parallel
    lines give inf or nan."""
    p = np.cross(l1, l2)
    with np.errstate(divide="ignore", invalid="ignore"):
        return p[..., :2] / p[..., 2:3]


def _square_to_quad(quad):
    """Homographies taking the unit square to quadrilaterals, in closed form.

    quad: (B, 4, 2) corners in the order (0,0), (1,0), (1,1), (0,1).
    Solving 4 point pairs normally needs an 8x8 linear system. For the unit
    square the system collapses to these few lines (Heckbert, 1989), which
    lets hundreds of thousands be computed at once.
    """
    x, y = quad[..., 0], quad[..., 1]
    dx1, dx2 = x[:, 1] - x[:, 2], x[:, 3] - x[:, 2]
    dy1, dy2 = y[:, 1] - y[:, 2], y[:, 3] - y[:, 2]
    sx = x[:, 0] - x[:, 1] + x[:, 2] - x[:, 3]
    sy = y[:, 0] - y[:, 1] + y[:, 2] - y[:, 3]
    with np.errstate(divide="ignore", invalid="ignore"):
        den = dx1 * dy2 - dx2 * dy1
        g = (sx * dy2 - dx2 * sy) / den
        h = (dx1 * sy - sx * dy1) / den
    H = np.empty((len(quad), 3, 3))
    H[:, 0, 0] = x[:, 1] - x[:, 0] + g * x[:, 1]
    H[:, 0, 1] = x[:, 3] - x[:, 0] + h * x[:, 3]
    H[:, 0, 2] = x[:, 0]
    H[:, 1, 0] = y[:, 1] - y[:, 0] + g * y[:, 1]
    H[:, 1, 1] = y[:, 3] - y[:, 0] + h * y[:, 3]
    H[:, 1, 2] = y[:, 0]
    H[:, 2, 0], H[:, 2, 1], H[:, 2, 2] = g, h, 1.0
    return H


def _model_assignments():
    """Every way to name a quad's four sides as court lines.

    Returns (A, 3, 3) matrices taking court metres to the unit square, one
    per ordered pair of long lines (20) times ordered pair of cross lines
    (12). Ordered, because either detected line could be the nearer one.
    """
    out = []
    for xa, xb in itertools.permutations(LONG_X, 2):
        for ya, yb in itertools.permutations(CROSS_Y, 2):
            out.append([[1 / (xb - xa), 0, -xa / (xb - xa)],
                        [0, 1 / (yb - ya), -ya / (yb - ya)],
                        [0, 0, 1]])
    return np.array(out)


def _implied_camera(H, principal):
    """Focal length and camera position implied by court->image homographies.

    Same algebra as camera_pose.focal_from_homography, batched, then
    H = K [r1 r2 t] is unpicked to get the camera centre. Returns
    (focal, centre) with NaN focal where no real camera exists.
    """
    shift = np.array([[1, 0, -principal[0]], [0, 1, -principal[1]], [0, 0, 1.0]])
    G = shift @ H
    h1, h2, h3 = G[:, :, 0], G[:, :, 1], G[:, :, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        f_sq = -(h1[:, 0] * h2[:, 0] + h1[:, 1] * h2[:, 1]) / (h1[:, 2] * h2[:, 2])
        focal = np.sqrt(np.where(f_sq > 0, f_sq, np.nan))
        scale_k = np.stack([1 / focal, 1 / focal, np.ones_like(focal)], axis=1)
        r1, r2, t = h1 * scale_k, h2 * scale_k, h3 * scale_k
        # H is only known up to scale and sign. The scale makes r1 a unit
        # vector; the sign puts the court in front of the camera (t_z > 0).
        norm = np.linalg.norm(r1, axis=1) * np.sign(t[:, 2])
        r1, r2, t = r1 / norm[:, None], r2 / norm[:, None], t / norm[:, None]
        r3 = np.cross(r1, r2)
        # Camera centre = -R^T t, with R's columns r1, r2, r3.
        centre = -np.stack([(r1 * t).sum(1), (r2 * t).sum(1), (r3 * t).sum(1)], axis=1)
    return focal, centre


def _model_samples():
    """Points every 25 cm along every painted line.

    Returns (points, has_next): points is (N, 3) homogeneous court
    coordinates; has_next[i] is True when point i+1 is the next sample on
    the same line, which is what lets the scorer measure spacing in pixels.
    """
    points, has_next = [], []
    for _, kind, const, start, end in MODEL_LINES:
        along = np.arange(start, end + 1e-9, SAMPLE_SPACING_M)
        fixed = np.full_like(along, const)
        points.append(np.c_[along, fixed] if kind == "cross" else np.c_[fixed, along])
        has_next.extend([True] * (len(along) - 1) + [False])
    points = np.vstack(points)
    return np.c_[points, np.ones(len(points))], np.array(has_next)


def _score(H, distance, samples, has_next):
    """Length of court line confirmed, and contradicted, for a batch of guesses.

    distance: for every pixel, how far the nearest white-line pixel is (a
    "distance transform" of the mask, computed once). A model point is a hit
    if it projects within HIT_DISTANCE_PX of white.

    Hits are counted in PIXELS OF LINE, not in model points. Each sample
    stands for the image distance to the next sample on its line. Without
    this, a guess that squashes the whole court into a few pixels on top of
    a fence rail scores a hit for every one of its 550 points. Measured in
    pixels it scores almost nothing, because those points cover no length.

    Returns (hit_length, miss_length) in pixels.
    """
    h, w = distance.shape
    projected = samples @ H.transpose(0, 2, 1)             # (B, N, 3)
    with np.errstate(divide="ignore", invalid="ignore"):
        xy = projected[..., :2] / projected[..., 2:3]
    in_front = projected[..., 2] > 0
    xy = np.nan_to_num(xy, nan=-1e6, posinf=-1e6, neginf=-1e6)
    xi, yi = np.rint(xy[..., 0]), np.rint(xy[..., 1])
    inside = in_front & (xi >= 0) & (xi < w) & (yi >= 0) & (yi < h)
    hit = inside & (distance[np.clip(yi, 0, h - 1).astype(np.intp),
                             np.clip(xi, 0, w - 1).astype(np.intp)] <= HIT_DISTANCE_PX)

    spacing = np.zeros(xy.shape[:2])
    spacing[:, :-1] = np.linalg.norm(np.diff(xy, axis=1), axis=2)
    weight = np.minimum(spacing, MAX_SAMPLE_WEIGHT_PX) * has_next
    return (weight * hit).sum(1), (weight * (inside & ~hit)).sum(1)


def find_court_homography(mask, lines):
    """Best court->image homography, or None. Steps 2 and 3 of the method."""
    h, w = mask.shape
    principal = (w / 2, h / 2)
    line_array = np.array([line for line, _ in lines])
    n = len(line_array)
    if n < 4:
        return None

    # Every (cross pair, long pair) of detected lines whose four crossings
    # are usable: finite, and not absurdly far outside the image.
    pairs = list(itertools.combinations(range(n), 2))
    quads, quad_lines = [], []
    for (i, j), (k, l) in itertools.permutations(pairs, 2):
        if len({i, j, k, l}) < 4:
            continue
        corners = np.array([_intersect(line_array[i], line_array[k]),
                            _intersect(line_array[i], line_array[l]),
                            _intersect(line_array[j], line_array[l]),
                            _intersect(line_array[j], line_array[k])])
        if (np.all(np.isfinite(corners)) and np.all(corners > -2 * w)
                and np.all(corners < 3 * w)):
            quads.append(corners)
            quad_lines.append((i, j, k, l))
    if not quads:
        return None

    square_to_image = _square_to_quad(np.array(quads))          # (Q, 3, 3)
    court_to_square = _model_assignments()                      # (A, 3, 3)
    samples, has_next = _model_samples()
    # Distance from every pixel to the nearest white-line pixel.
    distance = cv2.distanceTransform(255 - mask, cv2.DIST_L2, 3)

    best = None
    for q in range(0, len(square_to_image), 200):               # in chunks, for memory
        H = (square_to_image[q:q + 200, None] @ court_to_square[None]).reshape(-1, 3, 3)
        focal, centre = _implied_camera(H, principal)
        with np.errstate(invalid="ignore"):
            plausible = ((focal > MIN_FOCAL_FRACTION * w) & (focal < MAX_FOCAL_FRACTION * w)
                         & (centre[:, 2] > MIN_CAMERA_HEIGHT_M)
                         & (centre[:, 2] < MAX_CAMERA_HEIGHT_M)
                         & (np.linalg.norm(centre, axis=1) < MAX_CAMERA_DISTANCE_M)
                         # "Near" is the end closer to the camera.
                         & (centre[:, 1] < NET_Y))
        if not plausible.any():
            continue
        H = H[plausible]
        hits, misses = _score(H, distance, samples, has_next)
        score = hits - MISS_PENALTY * misses
        ok = (hits >= MIN_HIT_LENGTH_PX) & (hits >= MIN_HIT_FRACTION * (hits + misses))
        if not ok.any():
            continue
        i = int(np.argmax(np.where(ok, score, -np.inf)))
        if best is None or score[i] > best[0]:
            best = (float(score[i]), H[i], int(hits[i]), int(misses[i]))
    return best


def refine_landmarks(mask, H):
    """Step 4: fit each model line to its own white pixels, then intersect.

    Returns ({landmark name: (x, y)}, {model line name: (a, b, c)}).
    """
    h, w = mask.shape
    ys, xs = np.nonzero(mask)
    white = np.c_[xs, ys].astype(float)

    fitted = {}
    for name, kind, const, start, end in MODEL_LINES:
        # The part of this line that is actually in the picture. A sideline
        # usually runs out of the bottom of the frame and on behind the
        # camera, so the model segment is sampled and only the samples that
        # project inside the image are kept.
        along_m = np.arange(start, end + 1e-9, SAMPLE_SPACING_M)
        fixed = np.full_like(along_m, const)
        court = np.c_[along_m, fixed] if kind == "cross" else np.c_[fixed, along_m]
        projected = np.c_[court, np.ones(len(court))] @ H.T
        with np.errstate(divide="ignore", invalid="ignore"):
            px = projected[:, :2] / projected[:, 2:3]
        visible = ((projected[:, 2] > 0) & (px[:, 0] >= 0) & (px[:, 0] < w)
                   & (px[:, 1] >= 0) & (px[:, 1] < h))
        if visible.sum() < 2:
            continue
        a, b = px[visible][0], px[visible][-1]
        if np.linalg.norm(b - a) < REFINE_MIN_EXTENT_PX:
            continue
        direction = (b - a) / np.linalg.norm(b - a)
        offset = white - a
        along = offset @ direction
        across = offset @ np.array([-direction[1], direction[0]])
        near = ((np.abs(across) < REFINE_DISTANCE_PX) & (along > -REFINE_DISTANCE_PX)
                & (along < np.linalg.norm(b - a) + REFINE_DISTANCE_PX))
        if near.sum() < REFINE_MIN_PIXELS or np.ptp(along[near]) < REFINE_MIN_EXTENT_PX:
            continue
        # fitLine with a Huber cost: a least-squares line that is not thrown
        # by a few stray pixels (the player's sock, a ball on the line).
        vx, vy, x0, y0 = cv2.fitLine(white[near].astype(np.float32), cv2.DIST_HUBER,
                                     0, 0.01, 0.01).ravel()
        normal = np.array([-vy, vx])
        fitted[name] = np.r_[normal, -normal @ [x0, y0]]

    by_constant = {(kind, const): name for name, kind, const, _, _ in MODEL_LINES}
    landmarks = {}
    for name, (x, y) in ALL_LANDMARKS.items():
        cross, long_ = by_constant.get(("cross", y)), by_constant.get(("long", x))
        if cross in fitted and long_ in fitted:
            px = _intersect(fitted[cross], fitted[long_])
            if np.all(np.isfinite(px)) and 0 <= px[0] < w and 0 <= px[1] < h:
                landmarks[name] = (round(float(px[0]), 2), round(float(px[1]), 2))
    return landmarks, fitted


def detect_court(image):
    """Landmarks of the court in an image, or a reason why not.

    Returns (landmarks, info). landmarks is None on failure and info["reason"]
    says why.
    """
    mask = line_mask(image)
    lines = detect_lines(mask)
    info = {"lines_found": len(lines)}
    if len(lines) < 4:
        return None, {**info, "reason": f"only {len(lines)} straight white lines found"}

    best = find_court_homography(mask, lines)
    if best is None:
        return None, {**info, "reason": "no arrangement of the detected lines "
                                        "matches a tennis court"}
    score, H, hits, misses = best
    info.update(score=score, line_px_confirmed=hits, line_px_missing=misses)

    # Refining twice lets lines that the first, rougher homography only
    # partly covered pick up all of their pixels.
    landmarks, fitted = refine_landmarks(mask, H)
    if len(landmarks) >= MIN_POINTS:
        world = np.float64([ALL_LANDMARKS[k] for k in landmarks])
        H2, _ = cv2.findHomography(world, np.float64(list(landmarks.values())), 0)
        if H2 is not None:
            landmarks, fitted = refine_landmarks(mask, H2)
    info["lines_fitted"] = sorted(fitted)

    # A court looks the same from either end, so the search may have named
    # the ends the wrong way round. The convention is that "near" is the
    # camera's end. If the solved camera sits beyond the net, every landmark
    # is renamed as its opposite number (the court turned through 180 deg).
    if len(landmarks) >= MIN_POINTS:
        world = [ALL_LANDMARKS[k] for k in landmarks]
        pose = solve_pose_from_points(list(landmarks.values()), world, image.shape[1::-1])
        if pose.centre[1] > NET_Y:
            by_position = {(round(x, 3), round(y, 3)): k
                           for k, (x, y) in ALL_LANDMARKS.items()}
            landmarks = {
                by_position[(round(SINGLES_W - ALL_LANDMARKS[k][0], 3),
                             round(COURT_L - ALL_LANDMARKS[k][1], 3))]: px
                for k, px in landmarks.items()}
            info["ends_swapped"] = True
    if len(landmarks) < MIN_POINTS:
        return None, {**info, "reason": f"only {len(landmarks)} landmarks could be "
                                        f"located (need {MIN_POINTS})"}

    # Final acceptance. The search always returns its best guess, and on a
    # frame with too little court in view the best guess can be a fence. A
    # real court passes all three checks comfortably (see the README table);
    # a false one fails at least one.
    pixels = np.float64(list(landmarks.values()))
    world = np.float64([ALL_LANDMARKS[k] for k in landmarks])
    H_final, _ = cv2.findHomography(world, pixels, 0)
    samples, has_next = _model_samples()
    distance = cv2.distanceTransform(255 - mask, cv2.DIST_L2, 3)
    hits, misses = _score(H_final[None], distance, samples, has_next)
    info["line_px_confirmed"], info["line_px_missing"] = float(hits[0]), float(misses[0])
    info["hit_fraction"] = float(hits[0] / max(hits[0] + misses[0], 1.0))
    info["leave_one_out_m"] = leave_one_out_rms(pixels, world)
    info["pose_rms_px"] = solve_pose_from_points(pixels, world, image.shape[1::-1]).rms_px
    if info["hit_fraction"] < MIN_FINAL_HIT_FRACTION:
        return None, {**info, "reason": f"only {info['hit_fraction']:.0%} of the "
                                        "best-matching court lies on white lines"}
    if info["leave_one_out_m"] > MAX_LEAVE_ONE_OUT_M:
        return None, {**info, "reason": "the best-matching court is not self-consistent "
                                        f"({info['leave_one_out_m']:.2f} m leave-one-out)"}
    if info["pose_rms_px"] > MAX_POSE_RMS_PX:
        return None, {**info, "reason": "no real camera reproduces the best-matching "
                                        f"court ({info['pose_rms_px']:.1f} px)"}
    return landmarks, info


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--video", required=True, type=Path)
    p.add_argument("--frame", required=True, type=int,
                   help="Frame to calibrate on. Pick one where the player does "
                        "not cover the far-court lines.")
    p.add_argument("--out", required=True, type=Path, help="Calibration JSON to write")
    p.add_argument("--preview", type=Path,
                   help="Save the frame with the court model drawn on it")
    p.add_argument("--compare", type=Path,
                   help="An existing calibration of the same frame (for example "
                        "hand-clicked) to compare landmark positions against")
    args = p.parse_args()

    image = read_frame(args.video, args.frame)
    landmarks, info = detect_court(image)
    if landmarks is None:
        raise SystemExit(f"Court not found: {info['reason']}. Use calibrate_court.py "
                         "to click the landmarks instead.")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "video": args.video.name,
        "frame": args.frame,
        "image_size": [image.shape[1], image.shape[0]],
        "source": "automatic (detect_court.py)",
        "points": {k: list(v) for k, v in landmarks.items()},
    }, indent=2))

    calib = load_calibration(args.out)
    print(f"Court found: {info['hit_fraction']:.0%} of the court model's "
          f"visible line length lies on white pixels; {len(info['lines_fitted'])} of "
          f"{len(MODEL_LINES)} court lines fitted.")
    report(calib)
    print(describe(solve_camera_pose(calib)))

    if args.compare:
        other = load_calibration(args.compare)
        shared = sorted(set(calib.points) & set(other.points))
        gaps = [math.dist(calib.points[k], other.points[k]) for k in shared]
        print(f"\nAgainst {args.compare} ({len(shared)} shared landmarks):")
        for k, gap in zip(shared, gaps):
            print(f"  {k:<22}{gap:6.1f} px")
        print(f"  median {np.median(gaps):.1f} px, worst {max(gaps):.1f} px")

    if args.preview:
        cv2.imwrite(str(args.preview), draw_court(image, calib))
        print(f"  preview: {args.preview}")


if __name__ == "__main__":
    main()
