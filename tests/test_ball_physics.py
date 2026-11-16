"""The flight model must be right before anything fitted with it can be."""

import numpy as np

from ball_physics import G, K_AERO, acceleration, simulate, spin_rpm_from_lift, time_to_height


def test_no_air_gives_the_textbook_parabola():
    # With drag and spin off, the exact answer is p0 + v0 t - g t^2 / 2.
    p0, v0 = np.array([1.0, 2.0, 2.5]), np.array([3.0, 30.0, 2.0])
    times = np.linspace(0, 0.8, 9)
    states = simulate(p0, v0, times, cd=0.0)
    exact = p0 + np.outer(times, v0) - np.outer(0.5 * G * times ** 2, [0, 0, 1])
    assert np.allclose(states[:, :3], exact, atol=1e-9)


def test_drag_without_gravity_matches_the_exact_solution(monkeypatch):
    # Pure drag in a straight line has a closed form: v(t) = v0 / (1 + k v0 t)
    # and distance = ln(1 + k v0 t) / k. Gravity is switched off to isolate it.
    import ball_physics
    monkeypatch.setattr(ball_physics, "G", 0.0)
    k, v0 = K_AERO * 0.55, 45.0
    times = np.array([0.1, 0.3, 0.6])
    states = simulate([0, 0, 2], [0, v0, 0], times, cd=0.55)
    assert np.allclose(states[:, 4], v0 / (1 + k * v0 * times), rtol=1e-8)
    assert np.allclose(states[:, 1], np.log(1 + k * v0 * times) / k, rtol=1e-8)


def test_step_size_is_fine_enough():
    # A 20x smaller step must not change the answer: the default is converged.
    args = ([4, 0, 2.6], [1.0, 45.0, -3.0], [0.5])
    coarse = simulate(*args, cl_side=-0.2, cl_up=-0.1)
    fine = simulate(*args, cl_side=-0.2, cl_up=-0.1, max_step=1 / 4800)
    assert np.abs(coarse - fine)[0, :3].max() < 1e-6     # under a micrometre


def test_drag_slows_a_serve_by_roughly_a_third():
    states = simulate([4, 0, 2.6], [0, 45.0, -2.0], [0.45])
    lost = 1 - np.linalg.norm(states[0, 3:]) / 45.0
    assert 0.2 < lost < 0.4


def test_spin_force_directions():
    plain = simulate([0, 0, 2.5], [0, 30.0, 0], [0.5])[0]
    right = simulate([0, 0, 2.5], [0, 30.0, 0], [0.5], cl_side=0.2)[0]
    left = simulate([0, 0, 2.5], [0, 30.0, 0], [0.5], cl_side=-0.2)[0]
    topspin = simulate([0, 0, 2.5], [0, 30.0, 0], [0.5], cl_up=-0.2)[0]
    # Travelling toward +y, "right" seen from above is +x.
    assert right[0] > 0.1 and left[0] < -0.1
    assert topspin[2] < plain[2] - 0.1          # topspin dips
    assert abs(topspin[0]) < 1e-9               # and does not curve sideways


def test_spin_force_is_perpendicular_to_velocity():
    # A force at right angles to the motion cannot change the speed, so the
    # spin part of the acceleration must have no component along v.
    v = np.array([3.0, 25.0, -4.0])
    magnus = acceleration(v, cd=0.0, cl_side=0.3, cl_up=-0.2) - acceleration(v, cd=0.0)
    assert abs(magnus @ v) < 1e-9


def test_batch_matches_one_at_a_time():
    p0 = np.array([[4, 0, 2.6], [3, 1, 2.4]], dtype=float)
    v0 = np.array([[1, 40, -2], [-2, 30, 1]], dtype=float)
    cl_side, cl_up = np.array([0.1, -0.3]), np.array([-0.05, 0.02])
    times = [0.2, 0.5]
    batch = simulate(p0, v0, times, 0.55, cl_side, cl_up)
    for i in range(2):
        single = simulate(p0[i], v0[i], times, 0.55, cl_side[i], cl_up[i])
        assert np.allclose(batch[:, i], single)


def test_time_to_height_finds_the_landing():
    t, state = time_to_height([4, 0, 2.6], [0, 30.0, 1.0], 0.0335)
    assert abs(state[2] - 0.0335) < 1e-3 and state[5] < 0
    assert 0.6 < t < 1.0
    # A ball launched steeply upward does not land within the time limit.
    assert time_to_height([0, 0, 2], [0, 5, 60.0], 0.0, max_time=1.0) == (None, None)


def test_spin_estimate_is_monotonic_and_saturates():
    assert spin_rpm_from_lift(0.0, 30) == 0.0
    assert spin_rpm_from_lift(0.1, 30) < spin_rpm_from_lift(0.2, 30)
    assert spin_rpm_from_lift(0.6, 30) is None
