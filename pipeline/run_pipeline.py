"""
run_pipeline.py

One command from a video to a speed for every serve in it. Nothing is clicked
or marked by hand.

    video
      |  1. track_ball      detect the ball in every frame, Kalman-filter it
      v
    track.csv
      |  2. detect_events   find each contact (and its bounce, if visible)
      v
    events.json
      |  3. detect_court    find the court lines near each serve's bounce
      v
    court_serveN.json
      |  4. fit_trajectory  fit the 3D flight, read off the launch speed
      v
    serveN/trajectory_report.txt, plots, and serves_summary.txt for the clip
      |  5. serve_mechanics (optional, with --pose_weights)
      v
    serveN/mechanics_report.txt: jump, leg loading, reach, toss

Each stage writes a file, and a stage whose file already exists is skipped
unless --redo is given. Step 1 is the slow one (the detector runs on every
frame), so a rerun to try different fit settings takes seconds.

A serve that fails a stage is listed in the summary with the reason. It never
gets a guessed number.

Handheld or tripod:
    --camera handheld   The court is found again for every serve, on a frame
                        close to its bounce, and each flight frame is matched
                        back to that frame.
    --camera tripod     The court is found once and used for the whole clip.

Usage (PowerShell):
    python pipeline/run_pipeline.py --video "TestRecs\\oncourt test 1\\240\\serve3.MOV" `
        --weights "runs\\detect\\train-6\\weights\\best.pt" `
        --camera handheld --out_dir tracking\\serve3_auto
"""

import argparse
import csv
import json
from pathlib import Path

import cv2

import fit_trajectory
import serve_mechanics
from calibrate_court import (FAR_SERVICE_Y, NET_Y, SERVICE_D, draw_court,
                             load_calibration, read_frame)
from detect_court import detect_court
from detect_events import find_bounce, find_contacts_with_evidence, load_detections
from estimate_speed import MS_TO_KMH

# Frames tried for court detection, relative to a serve's bounce (or to its
# contact, when no bounce was found). Just after the bounce the server has
# landed and is roughly still, and the ball is no longer in front of the lines.
# Several are tried because the player can be standing on a line in any one.
COURT_FRAME_OFFSETS = (1, 12, 24, -24, 48)
COURT_FRAME_AFTER_CONTACT_S = 0.8


def stage_track(args, track_path):
    if track_path.is_file() and "track" not in args.redo:
        print(f"[1/4] Tracking: using existing {track_path}")
        return
    if args.weights is None:
        raise SystemExit(f"{track_path} does not exist yet, so --weights is needed "
                         "to run the detector.")
    # Imported here because it loads the detector's dependencies, which is
    # slow and pointless when the track already exists.
    import track_ball
    print(f"[1/4] Tracking: detecting on every frame of {args.video.name} ...")
    fps, size, detections = track_ball.detect_sequence(args.video, args.weights,
                                                       args.imgsz, args.conf)
    rows = track_ball.run_filter(detections)
    with track_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    if args.overlay_video:
        track_ball.write_overlay(args.video, rows, track_path.with_name("tracked.mp4"),
                                 size)
    print(f"      {len(rows)} frames -> {track_path}")


def stage_events(args, track_path, events_path, fps):
    if events_path.is_file() and "events" not in args.redo:
        print(f"[2/4] Events: using existing {events_path}")
        return json.loads(events_path.read_text())["serves"]
    print("[2/4] Events: finding contacts and bounces ...")
    detections = load_detections(track_path)
    serves = []
    for contact, evidence in find_contacts_with_evidence(detections, fps):
        bounce, reason = find_bounce(detections, contact, args.video, fps, args.camera)
        serves.append({"contact_frame": contact, "contact_evidence": evidence,
                       "bounce": bounce, "no_bounce_reason": reason})
    events_path.write_text(json.dumps({"fps": fps, "serves": serves}, indent=2))
    print(f"      {len(serves)} serve(s) -> {events_path}")
    return serves


def find_court_near(video, anchor_frame, frame_count, out_path, preview_path):
    """Detect the court on the best of a few frames around anchor_frame.

    "Best" is the lowest leave-one-out error: the frame whose landmarks
    predict each other most closely. Returns the calibration path, or None
    with the reason from the last attempt.
    """
    best, reason = None, "no frame could be read"
    for offset in COURT_FRAME_OFFSETS:
        frame = anchor_frame + offset
        if not 0 <= frame < frame_count:
            continue
        image = read_frame(video, frame)
        landmarks, info = detect_court(image)
        if landmarks is None:
            reason = info["reason"]
            continue
        if best is None or info["leave_one_out_m"] < best[2]["leave_one_out_m"]:
            best = (frame, landmarks, info, image)
    if best is None:
        return None, reason

    frame, landmarks, info, image = best
    out_path.write_text(json.dumps({
        "video": video.name, "frame": frame,
        "image_size": [image.shape[1], image.shape[0]],
        "source": "automatic (detect_court.py)",
        "points": {k: list(v) for k, v in landmarks.items()},
    }, indent=2))
    cv2.imwrite(str(preview_path), draw_court(image, load_calibration(out_path)))
    return out_path, (f"frame {frame}, {len(landmarks)} landmarks, "
                      f"{info['leave_one_out_m']:.3f} m leave-one-out")


def stage_court(args, serve, index, frame_count, fps, shared):
    """Calibration for one serve: supplied, shared (tripod) or detected."""
    if args.calibration:
        return args.calibration, "supplied with --calibration"
    if args.camera == "tripod" and shared.get("path"):
        return shared["path"], "shared across the clip (tripod)"

    out_path = args.out_dir / f"court_serve{index}.json"
    if out_path.is_file() and "court" not in args.redo:
        return out_path, "existing file"
    anchor = (serve["bounce"]["frame"] if serve["bounce"]
              else serve["contact_frame"] + int(COURT_FRAME_AFTER_CONTACT_S * fps))
    path, note = find_court_near(args.video, anchor, frame_count, out_path,
                                 args.out_dir / f"serve{index}" / "court_preview.jpg")
    if path is not None and args.camera == "tripod":
        shared["path"] = path
    return path, note


def stage_fit(args, index, calibration, events_path):
    argv = ["--track", str(args.out_dir / "track.csv"),
            "--calibration", str(calibration),
            "--camera", args.camera, "--video", str(args.video),
            "--events", str(events_path), "--serve", str(index),
            "--mc", str(args.mc), "--label", f"Serve {index + 1}",
            "--out_dir", str(args.out_dir / f"serve{index}")]
    return fit_trajectory.main(argv)


def stage_mechanics(args, index, calibration):
    """Optional: body mechanics for one serve. Returns a one-line outcome."""
    serve_dir = args.out_dir / f"serve{index}"
    argv = ["--video", str(args.video), "--camera", args.camera,
            "--track", str(args.out_dir / "track.csv"),
            "--calibration", str(calibration),
            "--trajectory", str(serve_dir / "trajectory.json"),
            "--pose_weights", str(args.pose_weights), "--out_dir", str(serve_dir)]
    try:
        serve_mechanics.main(argv)
    except SystemExit as failure:
        return " ".join(str(failure).split())
    return f"written to {serve_dir / 'mechanics_report.txt'}"


def summary_table(rows):
    """Plain-text table of every serve, including the ones that failed."""
    header = (f"{'Serve':<6}{'Contact':>8}  {'Launch speed':<19}{'Contact ht':<12}"
              f"{'Net clearance':<15}{'Landing':<28}Notes")
    lines = [header, "-" * len(header)]
    mechanics = [f"Serve {row['serve']} mechanics: {row['mechanics']}"
                 for row in rows if "mechanics" in row]
    for row in rows:
        if "error" in row:
            lines.append(f"{row['serve']:<6}{row['contact_frame']:>8}  "
                         f"NO RESULT: {row['error']}")
            continue
        s, sig = row["summary"], row["sigma"]
        depth = row["service_line_margin_m"]
        landing = (f"{abs(depth):.2f} m "
                   f"{'inside' if depth >= 0 else 'BEYOND'} service line")
        clearance = (f"{s['net_clearance_m']:.2f} m" if "net_clearance_m" in s else "-")
        notes = [] if row["anchored"] else ["no bounce seen: lower confidence"]
        if row.get("contact_evidence") == "sparse toss":
            notes.append("toss only partly seen; contact confirmed by the fit")
        lines.append(
            f"{row['serve']:<6}{row['contact_frame']:>8}  "
            f"{s['launch_speed_ms'] * MS_TO_KMH:>4.0f} +/- "
            f"{sig['launch_speed_ms'] * MS_TO_KMH:<3.0f} km/h   "
            f"{s['contact_height_m']:.2f} m      {clearance:<15}{landing:<28}"
            f"{'; '.join(notes)}")
    if mechanics:
        lines += [""] + mechanics
    return "\n".join(lines) + "\n"


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--video", required=True, type=Path)
    p.add_argument("--out_dir", required=True, type=Path)
    p.add_argument("--camera", required=True, choices=("handheld", "tripod"))
    p.add_argument("--weights", type=Path,
                   help="Detector weights. Only needed if track.csv does not exist")
    p.add_argument("--calibration", type=Path,
                   help="Use this court calibration instead of detecting the court")
    p.add_argument("--imgsz", type=int, default=1280)
    p.add_argument("--conf", type=float, default=0.15)
    p.add_argument("--mc", type=int, default=100,
                   help="Monte Carlo repeats per serve for the uncertainty")
    p.add_argument("--pose_weights", type=Path,
                   help="Pose model (for example weights\\yolov8s-pose.pt, downloaded "
                        "on first use). If given, serve mechanics are measured for "
                        "every serve that has a trajectory fit")
    p.add_argument("--overlay_video", action="store_true",
                   help="Also write tracked.mp4 when tracking runs")
    p.add_argument("--redo", nargs="*", default=[],
                   choices=("track", "events", "court"),
                   help="Stages to run again even though their output exists. "
                        "The fit always runs")
    args = p.parse_args()

    if not args.video.is_file():
        raise SystemExit(f"Video not found: {args.video}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(args.video))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if not fps:
        raise SystemExit(f"Could not read a frame rate from {args.video}")

    track_path = args.out_dir / "track.csv"
    events_path = args.out_dir / "events.json"
    stage_track(args, track_path)
    serves = stage_events(args, track_path, events_path, fps)
    if not serves:
        print("\nNo serves found in this clip: no contact followed a clean toss.\n"
              "See detect_events.py for what counts as a contact.")

    rows, shared = [], {}
    for index, serve in enumerate(serves):
        row = {"serve": index + 1, "contact_frame": serve["contact_frame"],
               "contact_evidence": serve.get("contact_evidence", "clean toss")}
        (args.out_dir / f"serve{index}").mkdir(exist_ok=True)
        print(f"\n[3/4] Serve {index + 1}: finding the court ...")
        calibration, note = stage_court(args, serve, index, frame_count, fps, shared)
        if calibration is None:
            row["error"] = f"court not found ({note})"
            print(f"      {row['error']}")
            rows.append(row)
            continue
        print(f"      {calibration} ({note})")
        row["calibration"] = str(calibration)

        print(f"[4/4] Serve {index + 1}: fitting the flight ...")
        try:
            results = stage_fit(args, index, calibration, events_path)
        except SystemExit as failure:
            # fit_trajectory stops with a message when it has no trustworthy
            # answer. That message is the result for this serve.
            row["error"] = " ".join(str(failure).split())
            print(f"      {row['error']}")
            rows.append(row)
            continue
        landing_y = results["summary"]["landing_y_m"]
        row.update(summary=results["summary"], sigma=results["sigma"],
                   anchored=results["anchored"],
                   service_line_margin_m=(FAR_SERVICE_Y - landing_y if landing_y > NET_Y
                                          else landing_y - SERVICE_D))
        if args.pose_weights:
            print(f"[5/5] Serve {index + 1}: body mechanics ...")
            row["mechanics"] = stage_mechanics(args, index, calibration)
            print(f"      {row['mechanics']}")
        rows.append(row)

    table = summary_table(rows)
    (args.out_dir / "serves_summary.txt").write_text(table, encoding="utf-8")
    (args.out_dir / "serves_summary.json").write_text(json.dumps(rows, indent=1),
                                                      encoding="utf-8")
    print("\n" + table)
    print(f"Summary: {args.out_dir / 'serves_summary.txt'}")
    print("Speeds are not validated against radar. See each serve's "
          "trajectory_report.txt for the full breakdown.")


if __name__ == "__main__":
    main()
