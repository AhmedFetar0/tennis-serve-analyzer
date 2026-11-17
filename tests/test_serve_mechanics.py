"""Serve mechanics geometry, tested without the pose model.

The pose model itself is not tested here (it is a downloaded network). What
is tested is everything done with its output: turning joint pixels into
heights, and turning heights over time into the reported figures. A stick
figure with known heights is projected through a virtual camera and must be
measured back correctly.
"""

import numpy as np
import pytest

from serve_mechanics import J, JOINTS, BodyPlane, measure, median_filter, server_ground_track


def test_body_plane_recovers_heights_of_points_on_it(behind_camera):
    ground = np.array([3.9, 0.6])
    plane = BodyPlane(behind_camera, ground)
    # Points directly above the plane's ground point lie on the plane.
    points = np.array([[3.9, 0.6, h] for h in (0.1, 1.0, 2.1, 3.5)])
    lifted = plane.lift(behind_camera.project(points))
    assert np.allclose(lifted, points, atol=1e-6)
    assert plane.distance()[0] == pytest.approx(
        np.linalg.norm(ground - behind_camera.centre[:2]))


def test_wrong_plane_distance_hurts_low_joints_most(behind_camera):
    # The documented error behaviour: a joint that is really 1 m nearer than
    # the plane is read too LOW, by more the further it is below the camera.
    plane = BodyPlane(behind_camera, [3.9, 1.0])
    truth = np.array([[3.9, 0.0, 0.1], [3.9, 0.0, 1.5]])      # ankle, shoulder
    lifted = plane.lift(behind_camera.project(truth))
    ankle_error, shoulder_error = lifted[:, 2] - truth[:, 2]
    assert ankle_error < -0.15
    assert abs(shoulder_error) < 0.05


def test_stance_position_comes_from_the_feet(behind_camera):
    times = np.linspace(-1.6, 0.3, 60)
    keypoints = np.full((len(times), 17, 2), np.nan)
    feet = {"left_ankle": [3.7, -0.1, 0.09], "right_ankle": [4.0, -0.1, 0.09]}
    for name, point in feet.items():
        keypoints[:, J[name]] = behind_camera.project([point])[0]
    ground, stance_xy, spread = server_ground_track(behind_camera, keypoints, times,
                                                    contact_xy=(3.9, 0.7))
    assert np.allclose(stance_xy, [3.85, -0.1], atol=1e-3)
    assert spread < 1e-3
    assert np.allclose(ground[0], stance_xy)            # starts at the stance
    at_contact = int(np.argmin(np.abs(times)))
    assert np.allclose(ground[at_contact], [3.9, 0.7], atol=0.03)   # ends at contact
    assert np.allclose(ground[-1], [3.9, 0.7])


def _serve_heights(times):
    """Joint heights for a made-up serve: stand, sink 0.15 m, jump 0.2 m."""
    load = -0.15 * np.exp(-((times + 0.35) / 0.12) ** 2)
    jump = 0.20 * np.exp(-(times / 0.10) ** 2)
    heights = {name: np.full(len(times), 1.0) for name in JOINTS}
    for name in ("left_hip", "right_hip"):
        heights[name] = 0.95 + load + jump
    for name in ("left_shoulder", "right_shoulder"):
        heights[name] = 1.45 + load + jump
    for name in ("left_ankle", "right_ankle"):
        heights[name] = 0.09 + jump
    heights["left_wrist"] = np.full(len(times), 1.0)
    heights["right_wrist"] = 1.0 + 1.1 * np.exp(-(times / 0.08) ** 2)   # reaches 2.1
    return heights


def test_measures_a_known_serve():
    times = np.linspace(-1.6, 0.35, 235)
    ball_times = np.linspace(-1.1, -0.01, 200)
    ball_heights = 2.6 + 0.5 * 9.81 * 0.46 ** 2 - 0.5 * 9.81 * (ball_times + 0.46) ** 2
    results, curves = measure(times, _serve_heights(times), ball_times, ball_heights,
                              contact_height_m=2.6)
    assert results["hip_drop_m"] == pytest.approx(0.15, abs=0.02)
    assert results["deepest_load_before_contact_s"] == pytest.approx(0.35, abs=0.03)
    assert results["jump_height_at_contact_m"] == pytest.approx(0.20, abs=0.02)
    assert results["hitting_side"] == "right"
    assert results["wrist_height_at_contact_m"] == pytest.approx(2.1, abs=0.03)
    assert results["wrist_to_ball_m"] == pytest.approx(0.5, abs=0.03)
    # A ball that takes 0.46 s to fall has dropped g t^2 / 2 = 1.04 m.
    assert results["toss_drop_before_contact_m"] == pytest.approx(1.04, abs=0.02)
    assert results["toss_peak_before_contact_s"] == pytest.approx(0.46, abs=0.01)
    assert set(curves) == {"hips", "ankles", "shoulders", "hitting_wrist"}


def test_missing_joints_give_no_number_not_a_wrong_one():
    times = np.linspace(-1.6, 0.35, 100)
    heights = _serve_heights(times)
    heights["left_wrist"][:] = np.nan
    heights["right_wrist"][:] = np.nan
    results, _ = measure(times, heights, np.array([]), np.array([]), 2.6)
    assert results["wrist_height_at_contact_m"] is None
    assert results["wrist_to_ball_m"] is None
    assert results["toss_peak_height_m"] is None
    assert results["hip_drop_m"] is not None          # legs were still seen


def test_median_filter_removes_a_glitch_but_keeps_a_peak():
    values = np.array([1.0, 1.0, 1.0, 9.0, 1.0, 1.0, 1.0])       # one bad frame
    assert np.allclose(median_filter(values), 1.0)
    peak = np.array([0.0, 1.0, 2.0, 3.0, 2.0, 1.0, 0.0])
    assert median_filter(peak)[3] >= 2.0
    with_gap = np.array([1.0, np.nan, 1.0, 1.0])
    assert np.all(np.isfinite(median_filter(with_gap)))
