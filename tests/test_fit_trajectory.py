"""End-to-end test of the 3D trajectory fit on simulated serves.

Each test invents a serve with known launch speed and spin, films it with a
virtual camera, spoils the result the way real detection does (pixel noise,
missed frames, false detections), and checks that the fit gets the truth back.
"""

import numpy as np
import pytest

from ball_physics import BALL_RADIUS_M, simulate, time_to_height
from fit_trajectory import FitSettings, FlightObservations, derive, fit_flight

FPS = 240.0


def synthetic_serve(pose, p0, v0, cl_side=0.0, cl_up=0.0, noise_px=1.0, keep=0.7,
                    false_detections=0, seed=0, with_bounce=True, visible_fraction=0.7):
    """Observations of a simulated serve, as the tracker would report them.

    visible_fraction: the detector loses the ball well before the bounce in
    real clips, so only the first part of the flight is observed.
    """
    rng = np.random.default_rng(seed)
    bounce_time, landing = time_to_height(p0, v0, BALL_RADIUS_M, cl_side=cl_side,
                                          cl_up=cl_up)
    frames = np.arange(2, int(bounce_time * FPS * visible_fraction))
    frames = frames[rng.random(len(frames)) < keep]
    times = frames / FPS
    px = pose.project(simulate(p0, v0, times, cl_side=cl_side, cl_up=cl_up)[:, :3])
    px += rng.normal(0, noise_px, px.shape)
    for i in rng.choice(len(px), false_detections, replace=False):
        px[i] += rng.uniform(40, 120, 2) * rng.choice([-1, 1], 2)

    bounce_px = None
    if with_bounce:
        bounce_px = pose.project([[landing[0], landing[1], 0.0]])[0] + rng.normal(0, 1.0, 2)
        # The bounce frame is a whole number, so its time is off by up to
        # half a frame, as in real footage.
        bounce_time = round(bounce_time * FPS) / FPS
    return FlightObservations(frames, times, px,
                              bounce_time if with_bounce else None, bounce_px)


SERVES = {
    # name: (contact position, launch velocity, cl_side, cl_up)
    "slow slice": ([3.9, 0.5, 2.55], [0.3, 26.0, 0.2], -0.30, 0.0),
    "flat first serve": ([4.3, 0.3, 2.75], [-2.0, 47.0, -3.4], -0.05, -0.02),
    "kick serve": ([3.8, 0.2, 2.6], [1.5, 33.0, 2.2], 0.10, -0.20),
}


@pytest.mark.parametrize("name", SERVES)
def test_recovers_launch_speed_from_behind(behind_camera, name):
    p0, v0, cl_side, cl_up = SERVES[name]
    obs = synthetic_serve(behind_camera, p0, v0, cl_side, cl_up, false_detections=5)
    fit = fit_flight(behind_camera, obs)
    assert not fit.problems
    true_speed = np.linalg.norm(v0)
    assert np.linalg.norm(fit.v0) == pytest.approx(true_speed, rel=0.03)
    assert fit.p0[2] == pytest.approx(p0[2], abs=0.08)        # contact height
    assert fit.p0[1] == pytest.approx(p0[1], abs=0.6)         # distance: the weak axis
    assert fit.params[6] == pytest.approx(cl_side, abs=0.04)
    assert fit.params[7] == pytest.approx(cl_up, abs=0.06)
    # The planted false detections must be the ones rejected.
    assert 3 <= (~fit.inliers).sum() <= 9


def test_recovers_launch_speed_from_the_side(side_camera):
    p0, v0, cl_side, cl_up = SERVES["flat first serve"]
    obs = synthetic_serve(side_camera, p0, v0, cl_side, cl_up)
    fit = fit_flight(side_camera, obs)
    assert not fit.problems
    assert np.linalg.norm(fit.v0) == pytest.approx(np.linalg.norm(v0), rel=0.03)


def test_reported_uncertainty_is_honest(behind_camera):
    # Over many noise draws, the true speed should sit within the fit's own
    # error bar about as often as a 1-sigma bar promises. If the errors were
    # routinely several sigma, the error bars would be claiming too much.
    p0, v0, cl_side, cl_up = SERVES["slow slice"]
    true_speed = np.linalg.norm(v0)
    pulls = []
    for seed in range(12):
        obs = synthetic_serve(behind_camera, p0, v0, cl_side, cl_up, seed=seed)
        fit = fit_flight(behind_camera, obs)
        direction = fit.v0 / np.linalg.norm(fit.v0)
        sigma = np.sqrt(direction @ fit.cov[3:6, 3:6] @ direction)
        pulls.append((np.linalg.norm(fit.v0) - true_speed) / sigma)
    assert np.sqrt(np.mean(np.square(pulls))) < 2.0


def test_without_a_bounce_the_fit_is_rougher_but_usable(behind_camera):
    # No anchor: vertical spin is held at zero. With a serve that truly has
    # none, the launch speed should still come out, less precisely.
    p0, v0, cl_side, _ = SERVES["slow slice"]
    obs = synthetic_serve(behind_camera, p0, v0, cl_side, 0.0, with_bounce=False,
                          noise_px=0.5)
    fit = fit_flight(behind_camera, obs)
    assert not fit.anchored and not fit.fit_cl_up
    assert np.linalg.norm(fit.v0) == pytest.approx(np.linalg.norm(v0), rel=0.10)


def test_derived_quantities_match_the_simulation():
    p0, v0, cl_side, cl_up = SERVES["flat first serve"]
    params = np.r_[p0, v0, cl_side, cl_up]
    summary, path, times = derive(params, 0.55, fps=FPS)
    t_true, landing = time_to_height(p0, v0, BALL_RADIUS_M, cl_side=cl_side, cl_up=cl_up)
    assert summary["flight_time_s"] == pytest.approx(t_true)
    assert summary["landing_y_m"] == pytest.approx(landing[1], abs=1e-3)
    assert summary["launch_speed_ms"] == pytest.approx(np.linalg.norm(v0))
    assert summary["launch_elevation_deg"] < 0          # a fast serve is hit downward
    assert 0 < summary["net_clearance_m"] < 1.5
    assert summary["landing_speed_ms"] < summary["speed_at_net_ms"] < summary["launch_speed_ms"]


def test_nonsense_is_refused_not_reported(behind_camera):
    # Random pixels are not a serve. The fit must say so.
    rng = np.random.default_rng(3)
    frames = np.arange(2, 120)
    obs = FlightObservations(frames, frames / FPS,
                             rng.uniform([200, 100], [1700, 900], (len(frames), 2)))
    fit = fit_flight(behind_camera, obs, FitSettings())
    assert fit.problems
