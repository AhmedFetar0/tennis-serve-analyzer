"""Handheld compensation, tested on a made-up clip with known camera shake.

A textured still image is written to a short video with each frame shifted
and slightly rotated by a known amount, as a shaking hand would. The motion
estimate must undo that, and the stabilisation check must report both how
big the shake was and how little is left.
"""

import cv2
import numpy as np
import pytest

from estimate_speed import estimate_camera_motion, stabilisation_residual, to_reference

SIZE = (960, 540)


def _textured_scene(rng):
    """Blocks, lines and dots: plenty of corners for feature matching."""
    image = np.full((SIZE[1], SIZE[0], 3), 90, np.uint8)
    for _ in range(140):
        x, y = int(rng.integers(0, SIZE[0])), int(rng.integers(0, SIZE[1]))
        w, h = int(rng.integers(8, 60)), int(rng.integers(8, 60))
        colour = tuple(int(c) for c in rng.integers(0, 255, 3))
        cv2.rectangle(image, (x, y), (x + w, y + h), colour, -1)
    for _ in range(60):
        a = (int(rng.integers(0, SIZE[0])), int(rng.integers(0, SIZE[1])))
        b = (int(rng.integers(0, SIZE[0])), int(rng.integers(0, SIZE[1])))
        cv2.line(image, a, b, (255, 255, 255), 2)
    return image


@pytest.fixture
def shaky_clip(tmp_path):
    rng = np.random.default_rng(7)
    scene = _textured_scene(rng)
    path = tmp_path / "shaky.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 30.0, SIZE)
    if not writer.isOpened():
        pytest.skip("this OpenCV build cannot write MJPG video")
    shakes = []
    for i in range(12):
        # Frame 0 is the reference and is left alone.
        dx, dy, angle = (0, 0, 0) if i == 0 else (rng.uniform(-25, 25),
                                                  rng.uniform(-15, 15),
                                                  rng.uniform(-1.0, 1.0))
        M = cv2.getRotationMatrix2D((SIZE[0] / 2, SIZE[1] / 2), angle, 1.0)
        M[:, 2] += (dx, dy)
        writer.write(cv2.warpAffine(scene, M, SIZE, borderMode=cv2.BORDER_REFLECT))
        shakes.append(M)
    writer.release()
    return path, shakes


def test_motion_estimate_undoes_the_shake(shaky_clip):
    path, shakes = shaky_clip
    motion = estimate_camera_motion(path, 0, list(range(12)))
    # A point at a known place in the scene, as it appears in each shaken
    # frame, must map back to that place in the reference frame.
    point = np.array([400.0, 300.0])
    for frame in range(1, 12):
        seen_at = shakes[frame] @ np.r_[point, 1.0]
        assert np.linalg.norm(to_reference(seen_at, frame, motion) - point) < 1.0


def test_stabilisation_check_reports_drift_and_residual(shaky_clip):
    path, _ = shaky_clip
    frames = list(range(12))
    motion = estimate_camera_motion(path, 0, frames)
    result = stabilisation_residual(path, 0, motion, frames)
    assert result is not None
    assert result["drift_median_px"] > 5            # the shake was real
    assert result["residual_median_px"] < 1.0       # and it has been removed
    assert result["residual_median_px"] < result["drift_median_px"] / 5
