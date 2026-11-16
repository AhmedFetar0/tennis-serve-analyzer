"""
ball_physics.py

The flight model for a tennis ball: gravity, air drag and spin (Magnus) force.
Used by fit_trajectory.py, which adjusts the launch conditions until the
simulated flight matches what the camera saw.

Forces on a ball in flight, as accelerations:

    gravity   g = 9.81 m/s^2 straight down. This is the only quantity in the
              whole model with a known size in metres, which is what lets a
              single camera recover real-world scale from a trajectory.

    drag      a = K * Cd * |v|^2, directed against the velocity.
              Cd (drag coefficient) is about 0.55 for a tennis ball. Drag is
              strong: a serve loses roughly a third of its speed before it
              bounces.

    Magnus    a = K * Cl * |v|^2, directed PERPENDICULAR to the velocity.
              A spinning ball drags air around with it and is pushed sideways
              to its path. Cl (lift coefficient) grows with spin rate.
              Topspin pushes the ball down, slice pushes it sideways.

    K = air density * cross-section area / (2 * mass)  =  0.0367 per metre

The spin force is split into two fixed coefficients rather than a spin
vector, because the split matches what can be observed:

    cl_side   sideways force, positive to the RIGHT of the direction of
              travel when seen from above. A right-hander's slice is negative.
    cl_up     force in the vertical plane of flight, positive up. Topspin is
              negative, backspin positive.

Both are treated as constant through the flight. Real spin decays only a few
percent over a serve, so this is a good approximation. A steady crosswind
produces nearly the same curve as cl_side, and the fit cannot tell the two
apart: cl_side is "sideways force from spin and wind together".

Units are SI throughout: metres, seconds, m/s. Court coordinates with z up.
"""

import math

import numpy as np

G = 9.81                    # m/s^2
BALL_RADIUS_M = 0.0335      # ITF: 6.54-6.86 cm diameter
BALL_MASS_KG = 0.0577       # ITF: 56.0-59.4 g
AIR_DENSITY = 1.2           # kg/m^3 at about 20 C near sea level

# rho * A / (2 m). Multiply by a coefficient and by speed squared to get an
# acceleration.
K_AERO = AIR_DENSITY * math.pi * BALL_RADIUS_M ** 2 / (2 * BALL_MASS_KG)

# Published wind-tunnel and trajectory measurements put a tennis ball's drag
# coefficient between about 0.5 and 0.6 (higher for a fluffed-up or spinning
# ball). The fit cannot measure it from one serve, so it is assumed, and the
# uncertainty analysis varies it.
DEFAULT_CD = 0.55
DEFAULT_CD_SIGMA = 0.05


def acceleration(v, cd=DEFAULT_CD, cl_side=0.0, cl_up=0.0):
    """Acceleration (m/s^2) of a ball moving with velocity v.

    v may be one velocity, shape (3,), or a batch, shape (M, 3). With a batch,
    each coefficient may be a single number or an array of M values, one per
    ball. The batch form lets the fitter fly many slightly different serves
    in one pass, which is how it measures the effect of each parameter.
    """
    v = np.asarray(v, dtype=float)
    vx, vy, vz = v[..., 0], v[..., 1], v[..., 2]
    horizontal_sq = vx * vx + vy * vy
    speed = np.sqrt(horizontal_sq + vz * vz)
    horizontal = np.sqrt(horizontal_sq)

    # Drag: opposite to v, size K Cd |v|^2. Written per component, which is
    # several times faster than whole-vector numpy calls on arrays this small,
    # and this function runs hundreds of thousands of times in a fit.
    drag = -K_AERO * cd * speed
    # Magnus force along two unit vectors perpendicular to v:
    #   side = ( vy, -vx, 0) / horizontal              horizontal, right of travel
    #   up   = (-vx vz, -vy vz, horizontal^2) / (horizontal * speed)
    #          lies in the vertical plane containing v
    # Each is multiplied by K Cl |v|^2. The tiny floor only matters for a ball
    # moving exactly vertically, where "sideways" has no meaning.
    h = np.maximum(horizontal, 1e-9)
    side = K_AERO * cl_side * speed * speed / h
    up = K_AERO * cl_up * speed / h
    out = np.empty_like(v)
    out[..., 0] = drag * vx + side * vy - up * vx * vz
    out[..., 1] = drag * vy - side * vx - up * vy * vz
    out[..., 2] = drag * vz + up * horizontal_sq - G
    return out


def simulate(p0, v0, times, cd=DEFAULT_CD, cl_side=0.0, cl_up=0.0, max_step=1 / 240):
    """Fly a ball from position p0 with velocity v0, both at time 0.

    times  ascending times (s, all >= 0) at which the state is wanted
    Returns [x, y, z, vx, vy, vz] at each time: shape (len(times), 6) for one
    ball, or (len(times), M, 6) when p0 and v0 are batches of shape (M, 3).

    Integration is classic 4th-order Runge-Kutta (RK4): each step samples the
    acceleration four times and combines them so that the error shrinks with
    the 4th power of the step size. The forces change smoothly, so steps of
    1/240 s are far more accurate than needed (tests/test_ball_physics.py
    checks this against exact solutions).

    The ball is flown straight through the court surface if times run past
    the bounce. Callers decide where the flight ends.
    """
    p = np.array(p0, dtype=float)
    v = np.array(v0, dtype=float)
    t = 0.0
    out = np.empty((len(times),) + p.shape[:-1] + (6,))
    for i, target in enumerate(times):
        while t < target - 1e-12:
            h = min(max_step, target - t)
            k1 = acceleration(v, cd, cl_side, cl_up)
            k2 = acceleration(v + 0.5 * h * k1, cd, cl_side, cl_up)
            k3 = acceleration(v + 0.5 * h * k2, cd, cl_side, cl_up)
            k4 = acceleration(v + h * k3, cd, cl_side, cl_up)
            # Position uses the RK4 velocity average, written out so that
            # only accelerations need evaluating.
            p = p + h * (v + h / 6 * (k1 + k2 + k3))
            v = v + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
            t += h
        out[i, ..., :3], out[i, ..., 3:] = p, v
    return out


def time_to_height(p0, v0, height, cd=DEFAULT_CD, cl_side=0.0, cl_up=0.0,
                   max_time=3.0, step=1 / 240):
    """Time at which a descending ball's centre reaches `height`, and its state
    there. Returns (None, None) if it does not within max_time.

    Used to find the landing point: a ball touches the court when its centre
    is one radius above it.
    """
    times = np.arange(0.0, max_time + step, step)
    states = simulate(p0, v0, times, cd, cl_side, cl_up)
    below = np.where((states[:, 2] <= height) & (states[:, 5] < 0))[0]
    if not len(below) or below[0] == 0:
        return None, None
    i = below[0]
    # Linear interpolation inside the one step that crosses the height.
    frac = (states[i - 1, 2] - height) / (states[i - 1, 2] - states[i, 2])
    return (float(times[i - 1] + frac * step),
            states[i - 1] + frac * (states[i] - states[i - 1]))


def spin_rpm_from_lift(cl, speed_ms):
    """Rough spin rate that would produce lift coefficient `cl` at this speed.

    Uses the empirical relation Cl = 1 / (2 + v / v_spin), where v_spin is
    the speed of the ball's surface (radius * angular velocity). It is a fit
    to wind-tunnel data, good to perhaps 20-30%. Returns None when cl is
    beyond what the relation can produce (it saturates at 0.5).

    Indicative only. The fitted coefficient also soaks up any crosswind.
    """
    cl = abs(cl)
    if cl < 1e-6:
        return 0.0
    if cl >= 0.5:
        return None
    surface_speed = speed_ms / (1 / cl - 2)
    return surface_speed / BALL_RADIUS_M * 60 / (2 * math.pi)
