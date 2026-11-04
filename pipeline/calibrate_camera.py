"""
calibrate_camera.py

Computes camera intrinsics (focal length, principal point, lens distortion)
from a VIDEO of a checkerboard -- not photos.

Why video, not photos:
    Phones frequently use a different sensor crop and field of view for
    high-frame-rate modes than for stills or standard video. Intrinsics
    calibrated from photos will NOT be valid for 240 fps footage. Record the
    board in the exact mode you'll shoot serves in, and calibrate from that.

IMPORTANT -- board geometry:
    --cols and --rows are INNER CORNERS, not squares. A printed board with
    10x7 squares has 9x6 inner corners, so pass --cols 9 --rows 6.
    This is the single most common reason detection finds nothing.

    --square_mm must be the square size you MEASURED on the printed sheet
    with a ruler, not the size you asked the printer for. Print scaling is
    rarely exact, and this number sets the real-world scale of everything
    downstream.

Usage:
    python pipeline/calibrate_camera.py --video calib_240fps.MOV \
        --cols 9 --rows 6 --square_mm 24.5 --out_dir .\\calibration
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def as_corner_array(pts):
    """Normalize corner output to (N, 1, 2).

    findChessboardCorners/cornerSubPix return (N,1,2) in some OpenCV versions
    and (N,2) in others, so pin the shape rather than assuming either.
    """
    return np.asarray(pts, dtype=np.float32).reshape(-1, 1, 2)


def find_corners(gray, pattern, fast=True):
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
    if fast:
        flags |= cv2.CALIB_CB_FAST_CHECK
    ok, corners = cv2.findChessboardCorners(gray, pattern, flags)
    if ok:
        corners = as_corner_array(corners)
    return ok, corners


def coverage_cell(corners, w, h, grid=4):
    """Which cell of a grid x grid split of the frame this board sits in.

    Distortion coefficients are only well-constrained if the board visits the
    edges and corners of the frame, not just the middle. Tracking coverage
    lets the script tell you if your captures are too centred.
    """
    cx = float(np.mean(corners[:, 0, 0]))
    cy = float(np.mean(corners[:, 0, 1]))
    return (min(int(cx / w * grid), grid - 1),
            min(int(cy / h * grid), grid - 1))


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--video", required=True, type=Path)
    p.add_argument("--cols", type=int, required=True,
                   help="INNER corners across (10 squares -> 9)")
    p.add_argument("--rows", type=int, required=True,
                   help="INNER corners down (7 squares -> 6)")
    p.add_argument("--square_mm", type=float, required=True,
                   help="Measured square size of the PRINTED board, in mm")
    p.add_argument("--out_dir", required=True, type=Path)
    p.add_argument("--frame_step", type=int, default=60,
                   help="Check every Nth frame (default 60). At 240 fps that "
                        "is 4 checks per second, which is ample -- the whole "
                        "clip is scanned regardless, so a larger step mostly "
                        "just saves time.")
    p.add_argument("--max_views", type=int, default=40,
                   help="Stop after this many accepted views (default 40)")
    p.add_argument("--min_spacing", type=float, default=120.0,
                   help="Minimum mean corner movement, in px, before another "
                        "view is kept (default 120). At 240 fps the board "
                        "barely moves between sampled frames, so a low value "
                        "collects near-duplicates.")
    p.add_argument("--save_debug", action="store_true",
                   help="Save accepted frames with corners drawn")
    p.add_argument("--start_sec", type=float, default=0.0,
                   help="Skip this many seconds at the start (e.g. while you "
                        "were still picking the board up)")
    p.add_argument("--trim_end_sec", type=float, default=0.0,
                   help="Ignore this many seconds at the end (e.g. while you "
                        "were putting the board down)")
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    pattern = (args.cols, args.rows)

    # Object points: the board's own coordinate frame, Z=0, in millimetres
    objp = np.zeros((args.rows * args.cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:args.cols, 0:args.rows].T.reshape(-1, 2)
    objp *= args.square_mm

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise SystemExit(f"Could not open {args.video}")

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Video: {w}x{h} @ {fps:.1f} fps, {total} frames")
    print(f"Pattern: {args.cols}x{args.rows} inner corners, "
          f"{args.square_mm} mm squares")

    duration = total / fps if fps else 0.0
    start_frame = int(args.start_sec * fps)
    end_frame = total - int(args.trim_end_sec * fps) if args.trim_end_sec else total
    if start_frame >= end_frame:
        raise SystemExit("start_sec/trim_end_sec leave no frames to search")
    if args.start_sec or args.trim_end_sec:
        print(f"Duration {duration:.1f}s -- searching "
              f"{args.start_sec:.1f}s to {end_frame / fps:.1f}s "
              f"(frames {start_frame}-{end_frame})")
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    print()

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # --- Pass 1: scan the ENTIRE range and collect every usable view ---
    # Taking the first N views found makes the result depend on where in the
    # clip you happened to start, and clusters them in a few seconds of
    # footage. Collect candidates across the whole video first, then choose.
    candidates = []          # (frame_idx, refined_corners, cell)
    last_corners = None
    frame_idx, checked = start_frame, 0

    print("Scanning for board views ...")
    while True:
        ok, frame = cap.read()
        if not ok or frame_idx >= end_frame:
            break
        if frame_idx % args.frame_step != 0:
            frame_idx += 1
            continue

        checked += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        found, corners = find_corners(gray, pattern)

        if found:
            # Cheap dedup: skip views nearly identical to the last one kept
            if last_corners is not None:
                move = float(np.mean(np.linalg.norm(
                    corners[:, 0, :] - last_corners[:, 0, :], axis=1)))
                if move < args.min_spacing:
                    frame_idx += 1
                    continue

            refined = as_corner_array(
                cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria))
            candidates.append((frame_idx, refined,
                               coverage_cell(refined, w, h)))
            last_corners = refined

        frame_idx += 1

    cap.release()
    print(f"  checked {checked} frames, found {len(candidates)} candidate views "
          f"across {len(set(c[2] for c in candidates))}/16 regions")

    # --- Pass 2: select a spatially diverse subset ---
    # Round-robin across occupied frame regions so the chosen views spread
    # over the image rather than piling up wherever the board spent longest.
    by_cell = {}
    for cand in candidates:
        by_cell.setdefault(cand[2], []).append(cand)

    selected = []
    while len(selected) < args.max_views and any(by_cell.values()):
        for cell in list(by_cell):
            if by_cell[cell] and len(selected) < args.max_views:
                selected.append(by_cell[cell].pop(0))
    selected.sort(key=lambda c: c[0])

    obj_points = [objp.copy() for _ in selected]
    img_points = [c[1] for c in selected]
    cells = set(c[2] for c in selected)
    accepted = len(selected)

    for i, (f_idx, _, cell) in enumerate(selected, 1):
        print(f"  view {i:2d}: frame {f_idx}, region {cell}")

    if args.save_debug and selected:
        (args.out_dir / "debug").mkdir(exist_ok=True)
        cap2 = cv2.VideoCapture(str(args.video))
        for i, (f_idx, corners, _) in enumerate(selected, 1):
            cap2.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
            ok, frame = cap2.read()
            if ok:
                cv2.drawChessboardCorners(frame, pattern, corners, True)
                cv2.imwrite(str(args.out_dir / "debug" /
                                f"view_{i:03d}_f{f_idx}.jpg"), frame)
        cap2.release()

    print(f"\nChecked {checked} frames, accepted {accepted} views")
    print(f"Frame coverage: {len(cells)}/16 regions occupied")

    if accepted < 10:
        raise SystemExit(
            f"\nOnly {accepted} usable views -- need at least 10, ideally 20+.\n"
            "Common causes:\n"
            "  * --cols/--rows given as SQUARES instead of INNER CORNERS\n"
            "  * board not fully visible in frame (every corner must be in shot)\n"
            "  * glare on glossy paper, or a screen instead of print\n"
            "  * board not flat, or too far away / motion-blurred")

    if len(cells) < 6:
        print("WARNING: board stayed in a small part of the frame. Distortion "
              "coefficients will be poorly constrained -- reshoot with the "
              "board near the edges and corners too.")

    print("\nCalibrating ...")
    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        obj_points, img_points, (w, h), None, None)

    # Per-view reprojection error. cv2.norm(...) gives sqrt(sum of squared
    # differences) across all points, so the per-point RMS divides by
    # sqrt(N) -- dividing by N understates the error by a factor of sqrt(N).
    per_view = []
    for i in range(len(obj_points)):
        proj, _ = cv2.projectPoints(obj_points[i], rvecs[i], tvecs[i], K, dist)
        err = cv2.norm(img_points[i], proj, cv2.NORM_L2) / np.sqrt(len(proj))
        per_view.append(err)
    per_view = np.array(per_view)

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    print(f"\n--- Results ---")
    print(f"RMS reprojection error: {rms:.4f} px")
    print(f"  per-view: mean {per_view.mean():.3f}, max {per_view.max():.3f} px")
    print(f"\nfx = {fx:.2f}   fy = {fy:.2f}")
    print(f"cx = {cx:.2f}   cy = {cy:.2f}   (frame centre is "
          f"{w/2:.1f}, {h/2:.1f})")
    print(f"distortion = {dist.ravel()}")

    if rms < 0.5:
        verdict = "excellent"
    elif rms < 1.0:
        verdict = "good -- usable"
    elif rms < 2.0:
        verdict = "marginal -- consider reshooting with sharper, more varied views"
    else:
        verdict = "poor -- do not use; check board flatness and focus"
    print(f"\nQuality: {verdict}")

    out = {
        "image_width": w,
        "image_height": h,
        "fps": fps,
        "camera_matrix": K.tolist(),
        "distortion_coefficients": dist.ravel().tolist(),
        "rms_reprojection_error": float(rms),
        "views_used": int(accepted),
        "square_mm": args.square_mm,
        "pattern_inner_corners": [args.cols, args.rows],
        "source_video": args.video.name,
    }
    json_path = args.out_dir / "intrinsics.json"
    json_path.write_text(json.dumps(out, indent=2))
    np.savez(args.out_dir / "intrinsics.npz", camera_matrix=K,
             distortion_coefficients=dist, image_size=(w, h))

    print(f"\nSaved: {json_path}")
    print(f"       {args.out_dir / 'intrinsics.npz'}")
    print("\nNOTE: these intrinsics are valid ONLY for "
          f"{w}x{h} at this capture mode. Recording serves in a different "
          "mode means recalibrating.")


if __name__ == "__main__":
    main()
