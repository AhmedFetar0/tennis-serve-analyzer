"""The whole measurement chain on clips that were never filmed.

Everything real that the pipeline has run on is one player, one court, one
evening, handheld. These tests guard against the pipeline quietly depending
on that. A virtual TRIPOD camera is placed somewhere new, a complete serve
(toss, strike, flight, bounce, rise) is rendered to a video file with the
court lines drawn in, and the stages are run on it exactly as run_pipeline.py
runs them:

    find the contact -> find the bounce -> find the court -> fit the flight

The only stage left out is the neural-network detector: the track is written
directly from the true ball positions, with noise, and with the ball "lost"
before the bounce as the real detector loses it. The launch speed that comes
out must match the speed that went in.
"""

import csv
import json

import cv2
import numpy as np
import pytest

import fit_trajectory
from ball_physics import BALL_RADIUS_M, G, simulate, time_to_height
from conftest import look_at_camera
from detect_court import detect_court
from detect_events import find_bounce, find_contacts_with_evidence, load_detections
from test_detect_court import render_court

FPS = 240.0
BALL_COLOUR = (190, 255, 235)       # BGR: bright optic yellow


def serve_positions(p0, v0, cl_side, cl_up, after_bounce_s=0.35):
    """True 3D ball position for every frame of a serve.

    Returns (positions (F, 3), contact_frame, bounce_frame). The toss is a
    vertical throw that peaks 1 m above the contact point. After the bounce
    the ball keeps 70% of its horizontal speed and rebounds at 75%.
    """
    p0, v0 = np.asarray(p0, float), np.asarray(v0, float)
    fall_s = np.sqrt(2 * 1.0 / G)                       # apex down to contact
    rise_s = np.sqrt(2 * (p0[2] + 1.0 - 1.4) / G)       # hand (1.4 m) up to apex
    toss_t = np.arange(-(rise_s + fall_s), 0, 1 / FPS)
    toss = np.tile(p0, (len(toss_t), 1))
    toss[:, 2] = p0[2] + 1.0 - 0.5 * G * (toss_t + fall_s) ** 2

    bounce_s, landing = time_to_height(p0, v0, BALL_RADIUS_M, cl_side=cl_side, cl_up=cl_up)
    flight_t = np.arange(0, bounce_s, 1 / FPS)
    flight = simulate(p0, v0, flight_t, cl_side=cl_side, cl_up=cl_up)[:, :3]

    rebound = landing[3:] * [0.7, 0.7, -0.75]
    after_t = np.arange(1 / FPS, after_bounce_s, 1 / FPS)
    after = simulate(landing[:3], rebound, after_t)[:, :3]
    return np.vstack([toss, flight, after]), len(toss), len(toss) + len(flight)


def render_clip(path, pose, positions):
    """Write the serve as a video seen by a fixed camera. Returns the ball's
    pixel position in every frame (NaN when it is out of view)."""
    background = render_court(pose, distractions=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), FPS,
                             pose.image_size)
    if not writer.isOpened():
        pytest.skip("this OpenCV build cannot write MJPG video")
    pixels = pose.project(positions)
    for point, px in zip(positions, pixels):
        frame = background.copy()
        if np.all(np.isfinite(px)):
            depth = np.linalg.norm(point - pose.centre)
            radius = max(2, int(round(pose.focal_px * BALL_RADIUS_M / depth)))
            cv2.circle(frame, (int(round(px[0])), int(round(px[1]))), radius,
                       BALL_COLOUR, -1, cv2.LINE_AA)
        writer.write(frame)
    writer.release()
    return pixels


def write_track(path, pixels, contact_frame, bounce_frame, rng, lost_fraction=0.25):
    """A track.csv as the tracker would write it: noisy positions, and
    nothing for the last part of the flight or after the bounce."""
    lost_from = contact_frame + int((bounce_frame - contact_frame) * (1 - lost_fraction))
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["frame", "raw_x", "raw_y", "raw_conf",
                                               "status", "mahalanobis", "filt_x",
                                               "filt_y", "vx", "vy", "speed_px"])
        writer.writeheader()
        for frame, px in enumerate(pixels):
            row = {"frame": frame, "status": "stale"}
            seen = frame < lost_from and np.all(np.isfinite(px)) and rng.random() < 0.9
            if seen:
                x, y = px + rng.normal(0, 0.6, 2)
                row.update(raw_x=f"{x:.2f}", raw_y=f"{y:.2f}", raw_conf="0.60",
                           status="updated")
            writer.writerow(row)


VIEWS = {
    # name: (camera position, point it looks at, focal length in px)
    "stand behind the baseline": ([5.5, -4.5, 2.2], [3.5, 12.0, 0.6], 1450.0),
    "stand behind, low and central": ([4.1, -7.0, 1.2], [4.1, 12.0, 1.0], 1500.0),
    "side-on at the service line": ([-9.0, 5.5, 1.5], [4.1, 9.0, 1.2], 1100.0),
}
SERVE = ([4.4, 0.3, 2.7], [-1.2, 41.0, -2.2], -0.12, -0.03)     # about 148 km/h


@pytest.mark.parametrize("view", VIEWS)
def test_tripod_clip_from_an_unfilmed_position(tmp_path, view):
    centre, target, focal = VIEWS[view]
    pose = look_at_camera(centre, target, focal_px=focal)
    p0, v0, cl_side, cl_up = SERVE
    true_speed = np.linalg.norm(v0)
    rng = np.random.default_rng(11)

    positions, contact_frame, bounce_frame = serve_positions(p0, v0, cl_side, cl_up)
    video = tmp_path / "serve.avi"
    pixels = render_clip(video, pose, positions)
    track = tmp_path / "track.csv"
    write_track(track, pixels, contact_frame, bounce_frame, rng)

    # 1. Contact, from the track alone.
    detections = load_detections(track)
    contacts = find_contacts_with_evidence(detections, FPS)
    assert len(contacts) == 1, f"contacts found: {contacts}"
    contact = contacts[0][0]
    assert abs(contact - contact_frame) <= 3

    # 2. Bounce, from the video. The track stops well before it.
    bounce, reason = find_bounce(detections, contact, video, FPS, camera="tripod")
    assert bounce is not None, f"no bounce: {reason}"
    assert abs(bounce["frame"] - bounce_frame) <= 3
    true_bounce_px = pose.project([[*positions[bounce_frame, :2], 0.0]])[0]
    assert np.linalg.norm(np.array(bounce["px"]) - true_bounce_px) < 6

    # 3. Court, from one frame of the video.
    cap = cv2.VideoCapture(str(video))
    ok, image = cap.read()
    cap.release()
    landmarks, info = detect_court(image)
    assert landmarks is not None, info.get("reason")
    court = tmp_path / "court.json"
    court.write_text(json.dumps({"video": video.name, "frame": 0,
                                 "image_size": list(pose.image_size),
                                 "points": {k: list(v) for k, v in landmarks.items()}}))
    events = tmp_path / "events.json"
    events.write_text(json.dumps({"fps": FPS, "serves": [
        {"contact_frame": contact, "bounce": bounce, "no_bounce_reason": None}]}))

    # 4. The fit, through its command-line entry point.
    results = fit_trajectory.main([
        "--track", str(track), "--calibration", str(court), "--camera", "tripod",
        "--video", str(video), "--events", str(events), "--serve", "0",
        "--mc", "0", "--out_dir", str(tmp_path / "out")])
    summary = results["summary"]
    assert summary["launch_speed_ms"] == pytest.approx(true_speed, rel=0.05)
    assert summary["contact_height_m"] == pytest.approx(p0[2], abs=0.12)
    assert (tmp_path / "out" / "trajectory_report.txt").is_file()
    assert (tmp_path / "out" / "trajectory_views.png").is_file()
