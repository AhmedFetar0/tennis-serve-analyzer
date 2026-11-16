"""Automatic court detection, tested on drawn images where the answer is known.

A virtual camera renders the court's painted lines onto a plain background,
with a fence and a neighbouring court added as distractions. The detector
must find the real court and place its landmarks to within a couple of
pixels, and must refuse an image that has no court in it.
"""

import cv2
import numpy as np
import pytest

from calibrate_court import ALL_LANDMARKS, COURT_L, SINGLES_W
from conftest import IMAGE_SIZE, look_at_camera
from detect_court import MODEL_LINES, detect_court, detect_lines, line_mask

COURT_BLUE, SURROUND_GREEN, WHITE = (150, 80, 50), (70, 150, 110), (235, 235, 235)


def _draw_segment(image, pose, a, b, thickness=4, colour=WHITE):
    """Draw a 3D segment as the camera sees it, in short pieces so that
    perspective is honoured along its length."""
    points = pose.project(np.linspace(a, b, 60))
    points = points[np.all(np.isfinite(points), axis=1)]
    if len(points) >= 2:
        cv2.polylines(image, [np.rint(points).astype(np.int32)], False, colour,
                      thickness, cv2.LINE_AA)


def render_court(pose, distractions=True):
    image = np.full((IMAGE_SIZE[1], IMAGE_SIZE[0], 3), SURROUND_GREEN, np.uint8)
    # Blue playing surface.
    corners = pose.project([[-1.37, 0, 0], [SINGLES_W + 1.37, 0, 0],
                            [SINGLES_W + 1.37, COURT_L, 0], [-1.37, COURT_L, 0]])
    if np.all(np.isfinite(corners)):
        cv2.fillPoly(image, [np.rint(corners).astype(np.int32)], COURT_BLUE)
    for _, kind, const, start, end in MODEL_LINES:
        a, b = ([start, const, 0], [end, const, 0]) if kind == "cross" \
            else ([const, start, 0], [const, end, 0])
        _draw_segment(image, pose, a, b)
    if distractions:
        # A fence behind the far baseline: long rails and posts, all white.
        for height in (1.5, 3.0):
            _draw_segment(image, pose, [-12, COURT_L + 6, height],
                          [20, COURT_L + 6, height], 3)
        for x in range(-12, 21, 3):
            _draw_segment(image, pose, [x, COURT_L + 6, 0], [x, COURT_L + 6, 3.0], 2)
        # The net tape: a white line that is NOT on the ground.
        _draw_segment(image, pose, [-2.3, COURT_L / 2, 1.0],
                      [SINGLES_W + 2.3, COURT_L / 2, 1.0], 5)
        # Part of the court next door.
        _draw_segment(image, pose, [-9, 0, 0], [-9, COURT_L, 0])
        _draw_segment(image, pose, [-7.6, 0, 0], [-7.6, COURT_L, 0])
    return image


CAMERAS = {
    "behind the baseline": ([4.16, -6.1, 1.65], [3.2, 12.0, 0.4]),
    "behind, off to one side": ([0.3, -4.0, 1.6], [5.0, 14.0, 0.2]),
    "raised, in the corner": ([-3.0, -5.0, 3.5], [5.0, 12.0, 0.0]),
}


@pytest.mark.parametrize("name", CAMERAS)
def test_finds_the_court_among_distractions(name):
    centre, target = CAMERAS[name]
    pose = look_at_camera(centre, target, focal_px=1400.0)
    landmarks, info = detect_court(render_court(pose))
    assert landmarks is not None, info.get("reason")
    assert len(landmarks) >= 8
    errors = [np.linalg.norm(np.array(px) - pose.project([[*ALL_LANDMARKS[k], 0.0]])[0])
              for k, px in landmarks.items()]
    # A correct match is accurate to a pixel or two. A wrong one (say, the
    # singles lines taken for the doubles lines) is off by tens of pixels.
    assert np.median(errors) < 2.0
    assert max(errors) < 6.0


def test_line_mask_keeps_lines_and_drops_surfaces():
    pose = look_at_camera(*CAMERAS["behind the baseline"], focal_px=1400.0)
    mask = line_mask(render_court(pose, distractions=False))
    assert 0.002 < (mask > 0).mean() < 0.05       # thin lines only
    lines = detect_lines(mask)
    assert 6 <= len(lines) <= 14


def test_refuses_an_image_with_no_court():
    # A fence and nothing else: plenty of straight white lines, but they do
    # not form a tennis court.
    image = np.full((IMAGE_SIZE[1], IMAGE_SIZE[0], 3), SURROUND_GREEN, np.uint8)
    for y in (300, 420, 540):
        cv2.line(image, (0, y), (IMAGE_SIZE[0], y + 15), WHITE, 3, cv2.LINE_AA)
    for x in range(100, IMAGE_SIZE[0], 220):
        cv2.line(image, (x, 300), (x, 560), WHITE, 3, cv2.LINE_AA)
    landmarks, info = detect_court(image)
    assert landmarks is None and "reason" in info


def test_refuses_a_blank_image():
    image = np.full((IMAGE_SIZE[1], IMAGE_SIZE[0], 3), SURROUND_GREEN, np.uint8)
    landmarks, info = detect_court(image)
    assert landmarks is None
