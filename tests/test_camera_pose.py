"""Camera pose recovery, tested against virtual cameras whose pose is known."""

import numpy as np
import pytest

from camera_pose import solve_pose_from_points
from conftest import IMAGE_SIZE, look_at_camera, visible_landmarks


def test_recovers_an_exact_camera(behind_camera):
    _, pixels, world = visible_landmarks(behind_camera)
    assert len(pixels) >= 6
    pose = solve_pose_from_points(pixels, world, IMAGE_SIZE)
    assert pose.focal_px == pytest.approx(behind_camera.focal_px, rel=1e-4)
    assert np.allclose(pose.centre, behind_camera.centre, atol=1e-3)
    assert pose.rms_px < 1e-3


def test_tolerates_click_noise(behind_camera):
    # 2 px of error on every landmark, as from clicking by hand. Over many
    # noise draws the focal length must stay within a few percent.
    _, pixels, world = visible_landmarks(behind_camera)
    rng = np.random.default_rng(1)
    focal, height = [], []
    for _ in range(20):
        pose = solve_pose_from_points(pixels + rng.normal(0, 2.0, pixels.shape), world,
                                      IMAGE_SIZE)
        focal.append(pose.focal_px)
        height.append(pose.centre[2])
    assert np.std(focal) / behind_camera.focal_px < 0.03
    assert abs(np.mean(focal) / behind_camera.focal_px - 1) < 0.02
    assert abs(np.mean(height) - behind_camera.centre[2]) < 0.05


def test_side_view_also_works(side_camera):
    _, pixels, world = visible_landmarks(side_camera)
    pose = solve_pose_from_points(pixels, world, IMAGE_SIZE)
    assert pose.focal_px == pytest.approx(side_camera.focal_px, rel=1e-3)
    assert np.allclose(pose.centre, side_camera.centre, atol=1e-2)


def test_fixed_focal_length_is_respected(behind_camera):
    _, pixels, world = visible_landmarks(behind_camera)
    pose = solve_pose_from_points(pixels, world, IMAGE_SIZE, focal_px=1500.0)
    assert pose.focal_px == 1500.0 and pose.focal_sigma_px == 0.0


def test_project_and_ray_are_inverses(behind_camera):
    points = np.array([[4.0, 0.5, 2.6], [2.0, 12.0, 1.0], [7.0, 18.0, 0.0]])
    pixels = behind_camera.project(points)
    rays = behind_camera.ray(pixels)
    for point, ray in zip(points, rays):
        to_point = point - behind_camera.centre
        assert np.allclose(ray, to_point / np.linalg.norm(to_point), atol=1e-9)
    # Cutting each ray at the point's own height must return the point.
    for point, px in zip(points, pixels):
        if abs(point[2] - behind_camera.centre[2]) > 0.1:
            assert np.allclose(behind_camera.ray_at_height(px, point[2])[0], point,
                               atol=1e-6)


def test_points_behind_the_camera_have_no_image(behind_camera):
    assert np.all(np.isnan(behind_camera.project([[4.0, -20.0, 1.0]])))


def test_tilt_and_field_of_view():
    level = look_at_camera([0, -5, 1.5], [0, 10, 1.5], focal_px=960.0)
    assert level.tilt_deg() == pytest.approx(0.0, abs=1e-6)
    assert level.horizontal_fov_deg() == pytest.approx(90.0)
    down = look_at_camera([0, -5, 3.0], [0, 10, 0.0])
    assert down.tilt_deg() > 5
