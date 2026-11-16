"""Shared test helpers.

The tests never touch real footage (it is private and not in the repo).
Instead they build a SYNTHETIC world: a virtual camera at a known position
looking at a virtual court, and a simulated serve with known speed and spin.
Because the truth is known exactly, a test can state how accurate each stage
is, which no amount of staring at real clips can do.

Run from the repo root:
    python -m pytest
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pipeline"))

from calibrate_court import COURT_LANDMARKS  # noqa: E402
from camera_pose import CameraPose  # noqa: E402

IMAGE_SIZE = (1920, 1080)


def look_at_camera(centre, target, focal_px=1500.0, image_size=IMAGE_SIZE):
    """A virtual camera at `centre` pointed at `target` (court coordinates,
    z up), held level (no roll). Returns a CameraPose with exact values."""
    centre = np.asarray(centre, dtype=float)
    forward = np.asarray(target, dtype=float) - centre
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, [0.0, 0.0, 1.0])     # level: perpendicular to "up"
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    R = np.vstack([right, down, forward])          # rows are the camera axes
    return CameraPose(focal_px, np.array([image_size[0] / 2, image_size[1] / 2]),
                      R, -R @ centre, 0.0, 0.0, image_size)


@pytest.fixture
def behind_camera():
    """Like serve3: behind the baseline on the centre line, at head height."""
    return look_at_camera([4.16, -6.1, 1.65], [3.2, 12.0, 1.2])


@pytest.fixture
def side_camera():
    """Like the planned Set B: outside the sideline, level with the service line."""
    return look_at_camera([-9.0, 5.5, 1.5], [4.1, 9.0, 1.2], focal_px=1100.0)


def visible_landmarks(pose, margin=0):
    """Court landmarks that fall inside the virtual camera's image."""
    names, pixels, world = [], [], []
    for name, xy in COURT_LANDMARKS.items():
        px = pose.project([[xy[0], xy[1], 0.0]])[0]
        if (np.all(np.isfinite(px)) and margin <= px[0] <= pose.image_size[0] - margin
                and margin <= px[1] <= pose.image_size[1] - margin):
            names.append(name)
            pixels.append(px)
            world.append(xy)
    return names, np.array(pixels), np.array(world)
