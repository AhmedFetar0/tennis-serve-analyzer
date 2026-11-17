"""
fit_trajectory.py

Reconstructs the full 3D flight of a serve from one camera, and reads the
speed off the racket from it.

How one camera can give 3D:
    A single pixel only fixes a ray: the ball is somewhere along a line from
    the camera. But a ball in flight is not free to be anywhere. It obeys
    known physics (ball_physics.py), so its whole path is set by a handful of
    numbers: where it started, how fast, and how it was spinning.

    This script searches for the launch conditions whose simulated flight,
    viewed through the solved camera (camera_pose.py), lands on the detected
    pixels in every frame at once. Two things supply real-world scale:

      * the bounce, which is on the court, so its position is known in metres
      * gravity, which bends the path by a known amount per second

    The unknowns (8 numbers):
        contact position   x0, y0, z0       where the ball left the racket
        launch velocity    vx0, vy0, vz0    its speed is the headline result
        cl_side, cl_up     sideways and vertical spin force

What this replaces from estimate_speed.py:
    * The hand-marked ground point under the server. The contact position
      comes out of the fit.
    * The assumed 2.6 m contact height. It is measured.
    * The straight-line, no-gravity drag correction. Launch speed is a fitted
      quantity with a physically complete model behind it.
    * The manual --exclude_frames list. Detections that disagree with any
      possible flight are down-weighted automatically.

What it still needs: the court landmarks (calibration) and the contact and
bounce frames (detect_events.py).

If no bounce was detected:
    The fit still runs, but vertical spin can no longer be separated from
    distance (topspin makes a ball drop faster, and so does being closer to
    the camera). It is then held at zero and the result is labelled as lower
    confidence.

Uncertainty:
    Two parts are combined. The fit's own covariance covers random pixel
    noise. A Monte Carlo run (--mc) covers the inputs that a single fit takes
    as given: the clicked court landmarks, the image centre, the bounce pixel
    and frame, and the drag coefficient. Each is redrawn at random and the
    whole solve repeated; the spread of the answers is the uncertainty.
    Neither part covers a wrong physical model (gusting wind, for example),
    and nothing here has been compared with radar.

Usage (PowerShell):
    python pipeline/fit_trajectory.py --track tracking\\serve3_fixed\\track.csv `
        --calibration tracking\\serve3\\court_f1014.json `
        --camera handheld --video "TestRecs\\oncourt test 1\\240\\serve3.MOV" `
        --events tracking\\serve3_fixed\\events.json --serve 0 `
        --label "first serve" --out_dir tracking\\serve3_fixed\\fit_a
"""

import argparse
import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares

from ball_physics import (BALL_RADIUS_M, DEFAULT_CD, DEFAULT_CD_SIGMA, G, simulate,
                          spin_rpm_from_lift, time_to_height)
from calibrate_court import (CENTRE_X, COURT_L, ALL_LANDMARKS, COURT_LINES,
                             DOUBLES_MARGIN, FAR_SERVICE_Y, NET_Y, SERVICE_D, SINGLES_W,
                             draw_court, load_calibration, read_frame)
from camera_pose import CameraPose, describe, solve_pose_from_points
from estimate_speed import (MEASURED_STATUSES, MS_TO_KMH, MS_TO_MPH,
                            estimate_camera_motion, label, read_track,
                            stabilisation_residual, to_reference)

PARAM_NAMES = ("x0", "y0", "z0", "vx0", "vy0", "vz0", "cl_side", "cl_up")
# Finite-difference steps for the Jacobian: metres, m/s, and lift coefficient.
PARAM_STEPS = np.array([1e-4, 1e-4, 1e-4, 1e-4, 1e-4, 1e-4, 1e-5, 1e-5])
# Typical sizes of each parameter, so the optimiser takes balanced steps.
PARAM_SCALES = np.array([1.0, 1.0, 1.0, 10.0, 10.0, 10.0, 0.1, 0.1])

# The first frames after contact are skipped: the ball is still against the
# strings or hidden by the racket head, and the detector's box is unreliable.
SKIP_AFTER_CONTACT_FRAMES = 2
MIN_FLIGHT_POINTS = 20
MIN_FLIGHT_SPAN_S = 0.15
MAX_FLIGHT_S = 1.30             # same limit as detect_events.py

NET_HEIGHT_CENTRE_M = 0.914
NET_HEIGHT_POST_M = 1.07
NET_POST_OUTSIDE_M = 0.914      # posts stand this far outside the doubles line

# Without a bounce the vertical spin force cannot be measured and is taken as
# zero. This is how wrong that could be (1-sigma). The serves measured with a
# bounce so far gave cl_up between +0.01 and +0.04, and hiding their bounce
# made the launch speed read 4% to 16% high, so this term is not small.
UNANCHORED_CL_UP_SIGMA = 0.04

# A fit is refused, not reported, if it lands outside these. Wide enough for
# any real serve; narrow enough to catch a fit that has locked onto nonsense.
CONTACT_HEIGHT_LIMITS_M = (1.5, 3.6)
LAUNCH_SPEED_LIMITS_KMH = (30.0, 270.0)
MAX_MEDIAN_RESIDUAL_PX = 3.0
MIN_INLIER_FRACTION = 0.6


# --- Data -------------------------------------------------------------------

@dataclass
class FlightObservations:
    """What the camera saw of one serve, in the calibration frame's pixels."""
    frames: np.ndarray               # source frame numbers of each detection
    times: np.ndarray                # seconds since the contact frame
    px: np.ndarray                   # (N, 2) ball centre pixels
    bounce_time: float = None        # seconds since contact; None = no bounce seen
    bounce_px: np.ndarray = None     # pixel of the ball at the bounce
    bounce_is_bottom: bool = True    # bounce_px marks where the ball touches
                                     # the court (True) or its centre (False)

    @property
    def anchored(self):
        return self.bounce_time is not None


@dataclass
class FitSettings:
    cd: float = DEFAULT_CD
    px_sigma: float = None           # None = estimate from the residuals
    bounce_px_sigma: float = 2.0
    bounce_z_sigma: float = 0.03     # metres; covers +/- 1 frame of bounce timing
    outlier_scale: float = 3.0       # in sigmas; beyond this a detection loses weight
    fit_cl_up: bool = None           # None = only when a bounce anchors the fit
    cl_up_value: float = 0.0         # value cl_up is held at when it is not fitted


@dataclass
class TrajectoryFit:
    params: np.ndarray               # the 8 numbers in PARAM_NAMES order
    cov: np.ndarray                  # 8x8 covariance (zeros for a held parameter)
    cd: float
    anchored: bool
    fit_cl_up: bool
    px_sigma: float
    residuals_px: np.ndarray         # (N, 2) detection minus model, pixels
    inliers: np.ndarray              # (N,) bool
    problems: list = field(default_factory=list)

    @property
    def p0(self):
        return self.params[:3]

    @property
    def v0(self):
        return self.params[3:6]


# --- The fit ----------------------------------------------------------------

def _model_states(thetas, obs, cd):
    """Simulate a batch of parameter sets. Returns states at each detection
    time, shape (N, M, 6), and at the bounce, shape (M, 6) or None."""
    thetas = np.atleast_2d(thetas)
    times = obs.times
    if obs.anchored:
        times = np.r_[times, obs.bounce_time]
    order = np.argsort(times, kind="stable")
    states = np.empty((len(times), len(thetas), 6))
    states[order] = simulate(thetas[:, :3], thetas[:, 3:6], times[order], cd,
                             thetas[:, 6], thetas[:, 7])
    if obs.anchored:
        return states[:-1], states[-1]
    return states, None


def _residuals(thetas, pose, obs, settings, px_sigma):
    """Weighted mismatch between model and observations for a batch of
    parameter sets. Returns (M, R): one row of residuals per parameter set.

    Every residual is divided by its expected error, so the optimiser treats
    "2 px off in the image" and "3 cm above the court at the bounce" as
    equally serious.
    """
    thetas = np.atleast_2d(thetas)
    flight, bounce = _model_states(thetas, obs, settings.cd)
    n, m = flight.shape[:2]

    projected = pose.project(flight[:, :, :3].reshape(-1, 3)).reshape(n, m, 2)
    e = (projected - obs.px[:, None, :]) / px_sigma
    e = np.nan_to_num(e, nan=1e3)       # behind the camera: a huge miss
    # Robust weighting, applied to each detection as a whole. A normal
    # detection (a sigma or two off) passes almost unchanged. One that is far
    # off has its pull capped at outlier_scale, so a false detection cannot
    # drag the flight toward itself. This is what replaces --exclude_frames.
    distance_sq = (e ** 2).sum(axis=2, keepdims=True)
    e = e / np.sqrt(1 + distance_sq / settings.outlier_scale ** 2)
    parts = [e.transpose(1, 0, 2).reshape(m, -1)]

    if bounce is not None:
        # The anchor: at the bounce time the ball must be touching the court
        # (centre one radius up) and must appear at the detected bounce pixel.
        touch = bounce[:, :3].copy()
        if obs.bounce_is_bottom:
            touch[:, 2] = 0.0
        parts.append((pose.project(touch) - obs.bounce_px) / settings.bounce_px_sigma)
        parts.append(((bounce[:, 2] - BALL_RADIUS_M) / settings.bounce_z_sigma)[:, None])
    return np.nan_to_num(np.hstack(parts), nan=1e3)


def _solve(start, free, pose, obs, settings, px_sigma, max_nfev=200):
    """Least-squares over the free parameters, from one starting point."""
    free = np.asarray(free)

    def expand(x):
        theta = start.copy()
        theta[free] = x
        return theta

    def fun(x):
        return _residuals(expand(x), pose, obs, settings, px_sigma)[0]

    def jac(x):
        # Forward differences, all columns in one batched simulation: row 0
        # is the current parameters, row k+1 has free parameter k nudged.
        base = expand(x)
        batch = np.tile(base, (len(free) + 1, 1))
        batch[np.arange(1, len(free) + 1), free] += PARAM_STEPS[free]
        r = _residuals(batch, pose, obs, settings, px_sigma)
        return ((r[1:] - r[0]) / PARAM_STEPS[free][:, None]).T

    # A wild starting point can send the simulation to absurd speeds before
    # the optimiser recovers; the resulting overflow is harmless (it becomes
    # a huge residual) so the warnings are silenced. max_nfev stops a
    # hopeless fit, such as one on noise, from grinding on for minutes.
    with np.errstate(all="ignore"):
        fit = least_squares(fun, start[free], jac=jac, x_scale=PARAM_SCALES[free],
                            max_nfev=max_nfev)
    return expand(fit.x), fit


def _starting_points(pose, obs):
    """Rough guesses for the optimiser to start from.

    The contact pixel is found by running the first few detections back to
    time zero. Its ray is cut at a typical contact height to get a starting
    position. With a bounce, the starting velocity simply aims at it. Without
    one, two speeds in both directions along the court are tried.
    """
    early = slice(0, min(8, len(obs.times)))
    contact_px = [np.polyval(np.polyfit(obs.times[early], obs.px[early, i], 1), 0.0)
                  for i in (0, 1)]
    positions = []
    for height in (2.6, 2.2, 3.0):
        p = pose.ray_at_height(contact_px, height)[0]
        if np.all(np.isfinite(p)) and np.linalg.norm(p - pose.centre) < 60:
            positions.append(p)
    if not positions:
        # Camera above the contact point or looking steeply: fall back to
        # fixed distances along the ray.
        ray = pose.ray(contact_px)[0]
        positions = [pose.centre + d * ray for d in (5.0, 10.0, 20.0)]

    starts = []
    if obs.anchored:
        landing = pose.ray_at_height(obs.bounce_px, 0.0)[0]
        T = obs.bounce_time
        for p in positions:
            horizontal = (landing[:2] - p[:2]) / T * 1.15   # drag: launch > average
            vz = (BALL_RADIUS_M - p[2] + 0.5 * G * T ** 2) / T
            starts.append(np.r_[p, horizontal, vz, 0.0, 0.0])
    else:
        for p in positions:
            for speed in (22.0, 42.0):
                for direction in (1.0, -1.0):
                    starts.append(np.r_[p, 0.0, direction * speed, -1.0, 0.0, 0.0])
    return starts


def fit_flight(pose: CameraPose, obs: FlightObservations, settings: FitSettings = None,
               start=None) -> TrajectoryFit:
    """Fit the launch conditions of one serve to its detections.

    start  an 8-vector to begin from (used by the Monte Carlo, which starts
           each repeat at the main answer). Otherwise guesses are generated.
    """
    settings = settings or FitSettings()
    fit_cl_up = obs.anchored if settings.fit_cl_up is None else settings.fit_cl_up
    free = [i for i in range(8) if fit_cl_up or PARAM_NAMES[i] != "cl_up"]

    starts = [np.asarray(start, dtype=float)] if start is not None \
        else _starting_points(pose, obs)
    if not fit_cl_up:
        starts = [np.r_[s[:7], settings.cl_up_value] for s in starts]

    # Pass 1 uses a nominal pixel error. Pass 2 uses the error actually seen
    # in the residuals, so the detections and the bounce anchor are weighed
    # against each other by their real quality.
    px_sigma = settings.px_sigma or 2.0
    best = None
    for s in starts:
        # A short run from every start is enough to see which one is heading
        # for the right answer; only the best is then run to convergence.
        theta, fit = _solve(s, free, pose, obs, settings, px_sigma,
                            max_nfev=200 if len(starts) == 1 else 40)
        if best is None or fit.cost < best[1].cost:
            best = (theta, fit)
    theta, fit = best
    if len(starts) > 1:
        theta, fit = _solve(theta, free, pose, obs, settings, px_sigma)

    def raw_residuals(theta):
        flight, _ = _model_states(theta, obs, settings.cd)
        return obs.px - pose.project(flight[:, 0, :3])

    if settings.px_sigma is None:
        # 1.4826 * median(|error|) estimates a standard deviation while
        # ignoring outliers (it is exact for Gaussian noise).
        px_sigma = float(np.clip(1.4826 * np.nanmedian(np.abs(raw_residuals(theta))),
                                 0.5, 4.0))
        theta, fit = _solve(theta, free, pose, obs, settings, px_sigma)

    residuals = raw_residuals(theta)
    distance = np.linalg.norm(residuals, axis=1)
    inliers = distance < settings.outlier_scale * px_sigma * math.sqrt(2)

    # Covariance of the fitted parameters: (J^T J)^-1, valid because every
    # residual was already divided by its expected error.
    cov = np.zeros((8, 8))
    try:
        cov[np.ix_(free, free)] = np.linalg.inv(fit.jac.T @ fit.jac)
    except np.linalg.LinAlgError:
        cov[np.ix_(free, free)] = np.inf

    result = TrajectoryFit(theta, cov, settings.cd, obs.anchored, fit_cl_up, px_sigma,
                           residuals, inliers)
    result.problems = _sanity_problems(result)
    return result


def _sanity_problems(fit: TrajectoryFit):
    """Reasons not to believe a fit. An empty list means it passed."""
    problems = []
    speed_kmh = float(np.linalg.norm(fit.v0)) * MS_TO_KMH
    if not CONTACT_HEIGHT_LIMITS_M[0] <= fit.p0[2] <= CONTACT_HEIGHT_LIMITS_M[1]:
        problems.append(f"contact height {fit.p0[2]:.2f} m is not believable")
    if not LAUNCH_SPEED_LIMITS_KMH[0] <= speed_kmh <= LAUNCH_SPEED_LIMITS_KMH[1]:
        problems.append(f"launch speed {speed_kmh:.0f} km/h is not believable")
    median = float(np.nanmedian(np.linalg.norm(fit.residuals_px, axis=1)))
    if not median <= MAX_MEDIAN_RESIDUAL_PX:
        problems.append(f"median residual {median:.1f} px: the model does not "
                        "follow the detections")
    if fit.inliers.mean() < MIN_INLIER_FRACTION:
        problems.append(f"only {fit.inliers.mean():.0%} of detections agree with "
                        "the fitted flight")
    if not np.all(np.isfinite(fit.cov)):
        problems.append("the fit is degenerate (parameters cannot be separated)")
    return problems


# --- Quantities derived from a fit ----------------------------------------------

def net_height_at(x):
    """Net height at across-court position x. The net sags from 1.07 m at the
    posts to 0.914 m at the centre strap; a straight line between is close."""
    half_span = SINGLES_W / 2 + DOUBLES_MARGIN + NET_POST_OUTSIDE_M
    frac = min(abs(x - CENTRE_X) / half_span, 1.0)
    return NET_HEIGHT_CENTRE_M + frac * (NET_HEIGHT_POST_M - NET_HEIGHT_CENTRE_M)


def derive(params, cd, bounce_time=None, fps=240.0):
    """Headline quantities for one parameter set, as a dict of floats.

    Kept separate from the fit so the Monte Carlo can call it on every
    repeat and take the spread of each quantity directly.
    """
    p0, v0, cl_side, cl_up = params[:3], params[3:6], params[6], params[7]
    speed = float(np.linalg.norm(v0))

    if bounce_time is None:
        bounce_time, landing = time_to_height(p0, v0, BALL_RADIUS_M, cd, cl_side, cl_up)
        if bounce_time is None:
            raise ValueError("the fitted flight never comes down")
    times = np.arange(0.0, bounce_time, 1 / fps)
    times = np.r_[times, bounce_time]
    path = simulate(p0, v0, times, cd, cl_side, cl_up)
    landing = path[-1]

    out = {
        "launch_speed_ms": speed,
        "contact_x_m": float(p0[0]), "contact_y_m": float(p0[1]),
        "contact_height_m": float(p0[2]),
        # Elevation: angle of the launch above horizontal. Negative = hit
        # downward. Slow serves must be hit upward to clear the net.
        "launch_elevation_deg": math.degrees(math.asin(v0[2] / speed)),
        "flight_time_s": float(bounce_time),
        "landing_x_m": float(landing[0]), "landing_y_m": float(landing[1]),
        "landing_speed_ms": float(np.linalg.norm(landing[3:])),
        "average_speed_ms": float(np.linalg.norm(landing[:3] - p0) / bounce_time),
        "cl_side": float(cl_side), "cl_up": float(cl_up),
    }

    # Net clearance: height above the tape where the path crosses the net.
    direction = np.sign(landing[1] - p0[1]) or 1.0
    crossed = np.where((path[:, 1] - NET_Y) * direction >= 0)[0]
    if len(crossed) and crossed[0] > 0:
        i = crossed[0]
        frac = (NET_Y - path[i - 1, 1]) / (path[i, 1] - path[i - 1, 1])
        at_net = path[i - 1] + frac * (path[i] - path[i - 1])
        out["net_clearance_m"] = float(at_net[2] - BALL_RADIUS_M - net_height_at(at_net[0]))
        out["speed_at_net_ms"] = float(np.linalg.norm(at_net[3:]))

    # Sideways curve: greatest distance of the path from the straight line
    # joining contact and landing, seen from above. Signed, + to the right.
    chord = landing[:2] - p0[:2]
    chord_len = float(np.linalg.norm(chord))
    offset = ((path[:, 0] - p0[0]) * chord[1] - (path[:, 1] - p0[1]) * chord[0]) / chord_len
    out["max_curve_m"] = float(offset[np.argmax(np.abs(offset))])
    return out, path, times


def monte_carlo(calib, pose, obs, fit, settings, fps, n, rng, click_sigma_px,
                principal_sigma_px=15.0, frame_sigma=1.0, cd_sigma=DEFAULT_CD_SIGMA):
    """Spread of the derived quantities when the fit's fixed inputs are varied.

    Each repeat redraws, from its plausible error:
        court landmark pixels   -> a new camera pose and focal length
        the image centre        -> ditto (the pose assumes it is central)
        bounce pixel and time   -> a new anchor
        drag coefficient        -> a new flight model
    then refits the serve. Returns {quantity: standard deviation}, and the
    number of repeats that produced a valid fit.
    """
    names = list(calib.points)
    image_pts = np.float64([calib.points[k] for k in names])
    world = np.float64([ALL_LANDMARKS[k] for k in names])

    samples = []
    for _ in range(n):
        try:
            pose_i = solve_pose_from_points(
                image_pts + rng.normal(0, click_sigma_px, image_pts.shape), world,
                calib.image_size,
                principal=pose.principal + rng.normal(0, principal_sigma_px, 2),
                focal_guess=pose.focal_px)
        except SystemExit:
            continue
        obs_i = obs
        if obs.anchored:
            obs_i = FlightObservations(
                obs.frames, obs.times, obs.px,
                obs.bounce_time + rng.normal(0, frame_sigma) / fps,
                obs.bounce_px + rng.normal(0, settings.bounce_px_sigma, 2),
                obs.bounce_is_bottom)
        settings_i = FitSettings(max(settings.cd + rng.normal(0, cd_sigma), 0.3),
                                 fit.px_sigma, settings.bounce_px_sigma,
                                 settings.bounce_z_sigma, settings.outlier_scale,
                                 fit.fit_cl_up)
        fit_i = fit_flight(pose_i, obs_i, settings_i, start=fit.params)
        if fit_i.problems:
            continue
        try:
            samples.append(derive(fit_i.params, settings_i.cd, obs_i.bounce_time, fps)[0])
        except ValueError:
            continue

    if len(samples) < 5:
        return {}, len(samples)
    keys = set.intersection(*(set(s) for s in samples))
    return {k: float(np.std([s[k] for s in samples])) for k in keys}, len(samples)


def formal_sigma(fit: TrajectoryFit, bounce_time, fps):
    """Uncertainty of each derived quantity from pixel noise alone.

    Propagated numerically: nudge each fitted parameter, see how the derived
    quantity moves (that is the gradient g), then sigma^2 = g^T C g with C
    the parameter covariance.
    """
    base = derive(fit.params, fit.cd, bounce_time, fps)[0]
    grads = {k: np.zeros(8) for k in base}
    for i in range(8):
        if fit.cov[i, i] == 0:
            continue
        nudged = fit.params.copy()
        nudged[i] += PARAM_STEPS[i] * 10
        try:
            moved = derive(nudged, fit.cd, bounce_time, fps)[0]
        except ValueError:
            continue
        for k in base:
            if k in moved:
                grads[k][i] = (moved[k] - base[k]) / (PARAM_STEPS[i] * 10)
    return {k: float(math.sqrt(max(g @ fit.cov @ g, 0.0))) for k, g in grads.items()}


# --- Inputs -----------------------------------------------------------------

def flight_detections(track, contact_frame, last_frame, exclude=()):
    """Frames with a real detection during the flight, as a sorted list."""
    first = contact_frame + SKIP_AFTER_CONTACT_FRAMES
    return [f for f in range(first, last_frame + 1)
            if f in track and track[f]["status"] in MEASURED_STATUSES
            and f not in exclude]


def trim_to_continuous_flight(frames, px, fps):
    """Cut the detections where the track stops following the ball.

    Only needed without a bounce: with one, the flight's end is known. Uses
    the rule from detect_events.py: the flight ends at a gap over 0.05 s, or
    at a step far larger than the ball had been moving, which is the tracker
    hopping onto something else.
    """
    frames = np.asarray(frames)
    if len(frames) < 3:
        return frames, px
    dt = np.diff(frames)
    step = np.hypot(*np.diff(px, axis=0).T) / dt
    breaks = np.where((dt > 0.05 * fps) | (step > 6 * np.median(step) + 2))[0]
    if len(breaks):
        keep = breaks[0] + 1
        return frames[:keep], px[:keep]
    return frames, px


def load_events(args):
    """Contact frame, bounce frame and bounce pixel from --events or the
    command line. A missing bounce is allowed and returns None for both."""
    if args.events is None:
        if args.contact_frame is None:
            raise SystemExit("Pass --events, or at least --contact_frame.")
        if (args.bounce_frame is None) != (args.bounce_px is None):
            raise SystemExit("--bounce_frame and --bounce_px go together: the fit "
                             "needs both the time and the place of the bounce.")
        return args.contact_frame, args.bounce_frame, args.bounce_px, True, "picked by eye"

    if not args.events.is_file():
        raise SystemExit(f"Events file not found: {args.events}")
    serves = json.loads(args.events.read_text())["serves"]
    if not 0 <= args.serve < len(serves):
        raise SystemExit(f"--serve {args.serve}: {args.events} has {len(serves)} serve(s).")
    serve = serves[args.serve]
    bounce = serve["bounce"]
    if bounce is None:
        return serve["contact_frame"], None, None, True, \
            f"auto-detected contact; no bounce ({serve['no_bounce_reason']})"
    return (serve["contact_frame"], bounce["frame"], bounce["px"],
            "bottom" in bounce["source"], f"auto-detected ({bounce['source']})")


# --- Outputs ----------------------------------------------------------------

INK, MUTED, LINE = "#1f2328", "#59636e", "#8c959f"
PATH_BLUE, BOUNCE_ORANGE, COURT_FILL = "#1d4ed8", "#c2410c", "#eef1f4"


def lift_detections(pose, obs, path, times):
    """Place each detection in 3D for plotting: the point on its pixel ray
    closest to where the fitted ball is at that moment. Distance along the
    ray comes from the fit, so these show scatter across the line of sight
    only. They illustrate the fit; they are not independent measurements."""
    fitted = np.array([np.interp(obs.times, times, path[:, i]) for i in range(3)]).T
    rays = pose.ray(obs.px)
    along = ((fitted - pose.centre) * rays).sum(axis=1, keepdims=True)
    return pose.centre + along * rays


def write_views_plot(path_png, title, summary, path, lifted, inliers, anchored):
    """Side view and top view of the fitted flight, to scale."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (side, top) = plt.subplots(
        2, 1, figsize=(11, 7.6), dpi=150, sharex=True,
        gridspec_kw={"height_ratios": [1, 2.6], "hspace": 0.12})

    # Side view: down-court distance against height.
    side.axhline(0, color=LINE, lw=1.2)
    side.plot([NET_Y, NET_Y], [0, NET_HEIGHT_CENTRE_M], color=MUTED, lw=2)
    side.text(NET_Y, NET_HEIGHT_CENTRE_M + 0.15, "net", color=MUTED, fontsize=9,
              ha="center")
    side.scatter(lifted[inliers, 1], lifted[inliers, 2], s=7, color=LINE, zorder=2,
                 label="Detections, placed at the fitted distance")
    side.plot(path[:, 1], path[:, 2], color=PATH_BLUE, lw=2, zorder=3,
              label="Fitted flight")
    side.scatter(path[0, 1], path[0, 2], marker="s", s=50, color=INK, zorder=4)
    side.scatter(path[-1, 1], path[-1, 2], marker="X", s=90, color=BOUNCE_ORANGE,
                 edgecolors="white", linewidths=0.8, zorder=4)
    side.annotate(f"contact {path[0, 2]:.2f} m high\n"
                  f"{summary['launch_speed_ms'] * MS_TO_KMH:.0f} km/h",
                  (path[0, 1], path[0, 2]), xytext=(8, 6), textcoords="offset points",
                  fontsize=9, color=INK)
    side.annotate(f"bounce\n{summary['landing_speed_ms'] * MS_TO_KMH:.0f} km/h",
                  (path[-1, 1], path[-1, 2]), xytext=(8, 10), textcoords="offset points",
                  fontsize=9, color=INK)
    if "net_clearance_m" in summary:
        side.annotate(f"clears net by {summary['net_clearance_m']:.2f} m",
                      (NET_Y, NET_HEIGHT_CENTRE_M + summary["net_clearance_m"]),
                      xytext=(0, 14), textcoords="offset points", fontsize=9,
                      color=INK, ha="center")
    side.set_ylabel("Height (m)", color=MUTED)
    side.set_ylim(-0.2, max(3.6, path[:, 2].max() + 0.9))
    side.set_aspect("equal")
    side.legend(loc="upper right", frameon=False, fontsize=8.5)

    # Top view, turned so the serve runs left to right like the side view.
    top.add_patch(plt.Rectangle((0, -DOUBLES_MARGIN), COURT_L,
                                SINGLES_W + 2 * DOUBLES_MARGIN, color=COURT_FILL,
                                zorder=0))
    for a, b in COURT_LINES:
        top.plot([a[1], b[1]], [a[0], b[0]], color=LINE, lw=1.2, zorder=1)
    top.plot([NET_Y, NET_Y], [-DOUBLES_MARGIN - 0.9, SINGLES_W + DOUBLES_MARGIN + 0.9],
             color=MUTED, lw=2, zorder=1)
    top.plot([path[0, 1], path[-1, 1]], [path[0, 0], path[-1, 0]], color=INK, lw=1,
             ls=(0, (5, 3)), zorder=2, label="Straight line, contact to bounce")
    top.scatter(lifted[inliers, 1], lifted[inliers, 0], s=7, color=LINE, zorder=2)
    top.plot(path[:, 1], path[:, 0], color=PATH_BLUE, lw=2, zorder=3,
             label="Fitted flight (curve is spin and any crosswind)")
    top.scatter(path[0, 1], path[0, 0], marker="s", s=50, color=INK, zorder=4,
                label="Contact")
    top.scatter(path[-1, 1], path[-1, 0], marker="X", s=90, color=BOUNCE_ORANGE,
                edgecolors="white", linewidths=0.8, zorder=4,
                label="Bounce" if anchored else "Predicted landing (bounce not seen)")
    top.set_xlabel("Down court (m)", color=MUTED)
    top.set_ylabel("Across court (m)", color=MUTED)
    top.set_xlim(min(-3.0, path[:, 1].min() - 1), COURT_L + 3.0)
    top.set_ylim(SINGLES_W + DOUBLES_MARGIN + 1.5, -DOUBLES_MARGIN - 1.5)
    top.set_aspect("equal")
    top.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), frameon=False,
               fontsize=8.5, ncol=2)

    for ax in (side, top):
        ax.tick_params(colors=MUTED)
        for spine in ax.spines.values():
            spine.set_visible(False)
    side.set_title(f"{title}: fitted 3D flight (side view above, top view below, "
                   "both to scale)", color=INK, fontsize=11, pad=30)
    fig.savefig(path_png, bbox_inches="tight")
    plt.close(fig)


def write_overlay(path_jpg, video, calib, pose, obs, fit, path, summary, anchored):
    """The fitted flight drawn back onto the calibration frame, with the
    detections it was fitted to. If the curve does not run through the dots,
    the fit is wrong, whatever the numbers say."""
    image = draw_court(read_frame(video, calib.frame), calib, colour=(200, 200, 200))
    curve = pose.project(path[:, :3])
    curve = curve[np.all(np.isfinite(curve), axis=1)].astype(np.int32)
    cv2.polylines(image, [curve], False, (255, 120, 30), 2, cv2.LINE_AA)
    for (x, y), ok in zip(obs.px, fit.inliers):
        cv2.circle(image, (int(x), int(y)), 3, (0, 255, 0) if ok else (0, 0, 255), -1,
                   cv2.LINE_AA)
    end = pose.project(np.r_[path[-1, :2], 0.0])[0]
    if np.all(np.isfinite(end)):
        cv2.drawMarker(image, (int(end[0]), int(end[1])), (0, 80, 255),
                       cv2.MARKER_TILTED_CROSS, 18, 2)
        label(image, "bounce" if anchored else "predicted landing",
              (int(end[0]) - 90, int(end[1]) - 12), (0, 160, 255))
    label(image, f"launch {summary['launch_speed_ms'] * MS_TO_KMH:.0f} km/h, "
                 f"contact {summary['contact_height_m']:.2f} m high",
          (20, 40), (255, 255, 255))
    label(image, "blue: fitted 3D flight   green: detections used   "
                 "red: detections rejected", (20, 70), (255, 255, 255))
    cv2.imwrite(str(path_jpg), image)


# --- Main -------------------------------------------------------------------

def build_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--track", required=True, type=Path,
                   help="track.csv produced by track_ball.py")
    p.add_argument("--calibration", required=True, type=Path,
                   help="Court calibration JSON from calibrate_court.py")
    p.add_argument("--camera", required=True, choices=("handheld", "tripod"),
                   help="handheld compensates camera motion and requires --video")
    p.add_argument("--video", type=Path,
                   help="Source clip. Needed for handheld; also supplies fps and "
                        "the image overlay")
    p.add_argument("--fps", type=float, help="Frame rate, if there is no --video")
    p.add_argument("--events", type=Path,
                   help="events.json from detect_events.py; use with --serve")
    p.add_argument("--serve", type=int, default=0,
                   help="Which serve in --events to fit, from 0 (default 0)")
    p.add_argument("--contact_frame", type=int)
    p.add_argument("--bounce_frame", type=int)
    p.add_argument("--bounce_px", type=float, nargs=2, metavar=("X", "Y"),
                   help="Bottom of the ball at the bounce")
    p.add_argument("--no_anchor", action="store_true",
                   help="Ignore the bounce even if one is known. Used to test how "
                        "well the flight alone predicts it")
    p.add_argument("--focal_px", type=float,
                   help="Hold the focal length at this value (default: solve it "
                        "from the court)")
    p.add_argument("--cd", type=float, default=DEFAULT_CD,
                   help=f"Drag coefficient (default {DEFAULT_CD})")
    p.add_argument("--cd_sigma", type=float, default=DEFAULT_CD_SIGMA,
                   help=f"1-sigma uncertainty of it (default {DEFAULT_CD_SIGMA})")
    p.add_argument("--bounce_px_sigma", type=float, default=2.0)
    p.add_argument("--frame_sigma", type=float, default=1.0,
                   help="1-sigma error of the bounce frame (default 1.0)")
    p.add_argument("--mc", type=int, default=100,
                   help="Monte Carlo repeats for the uncertainty (default 100, "
                        "0 to skip)")
    p.add_argument("--seed", type=int, default=0,
                   help="Random seed, so a rerun gives the same uncertainty")
    p.add_argument("--contact_ground_px", type=float, nargs=2, metavar=("X", "Y"),
                   help="Optional hand-marked court point under the contact. NOT "
                        "used by the fit: reported as an independent comparison")
    p.add_argument("--contact_ground_frame", type=int)
    p.add_argument("--exclude_frames", type=int, nargs="*", default=[], metavar="N",
                   help="Frames to drop by hand. Normally unnecessary")
    p.add_argument("--label", default="Serve", help="Title for the outputs")
    p.add_argument("--out_dir", required=True, type=Path)
    return p


def main(argv=None):
    """Run one fit. argv lets run_pipeline.py call this exactly as the command
    line would. Returns the results that were written to trajectory.json."""
    args = build_parser().parse_args(argv)
    contact_frame, bounce_frame, bounce_px, bounce_is_bottom, event_source = \
        load_events(args)
    if args.no_anchor:
        last_known_bounce = bounce_frame
        bounce_frame = bounce_px = None
        event_source += "; bounce deliberately ignored (--no_anchor)"

    if args.camera == "handheld" and args.video is None:
        raise SystemExit("--camera handheld needs --video.")
    fps = args.fps
    if fps is None:
        if args.video is None:
            raise SystemExit("Pass --fps or --video: the frame rate sets the speed "
                             "directly and will not be guessed.")
        cap = cv2.VideoCapture(str(args.video))
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        if not fps:
            raise SystemExit(f"Could not read a frame rate from {args.video}")

    calib = load_calibration(args.calibration)
    names = list(calib.points)
    pose = solve_pose_from_points([calib.points[k] for k in names],
                                  [ALL_LANDMARKS[k] for k in names],
                                  calib.image_size, focal_px=args.focal_px)
    track = read_track(args.track)

    anchored = bounce_frame is not None
    last = bounce_frame - 1 if anchored else contact_frame + int(MAX_FLIGHT_S * fps)
    if args.no_anchor and last_known_bounce is not None:
        last = last_known_bounce - 1
    frames = flight_detections(track, contact_frame, last, set(args.exclude_frames))
    raw_px = np.array([[float(track[f]["raw_x"]), float(track[f]["raw_y"])]
                       for f in frames]).reshape(-1, 2)
    if not anchored:
        frames, raw_px = trim_to_continuous_flight(frames, raw_px, fps)
        frames = [int(f) for f in frames]

    span = (frames[-1] - frames[0]) / fps if len(frames) else 0.0
    if len(frames) < MIN_FLIGHT_POINTS or span < MIN_FLIGHT_SPAN_S:
        raise SystemExit(
            f"No trajectory fit: only {len(frames)} detections over {span:.2f} s "
            f"after contact frame {contact_frame} (need {MIN_FLIGHT_POINTS} over "
            f"{MIN_FLIGHT_SPAN_S} s). The detector did not follow this serve.")

    motion = None
    if args.camera == "handheld":
        print("Estimating camera motion ...")
        extra = [bounce_frame] if anchored else []
        if args.contact_ground_px:
            extra.append(args.contact_ground_frame or contact_frame)
        motion = estimate_camera_motion(args.video, calib.frame, frames + extra)

    stabilisation = None
    if motion:
        stabilisation = stabilisation_residual(args.video, calib.frame, motion, frames)

    px = np.array([to_reference(p, f, motion) for p, f in zip(raw_px, frames)])
    obs = FlightObservations(
        np.array(frames), (np.array(frames) - contact_frame) / fps, px,
        (bounce_frame - contact_frame) / fps if anchored else None,
        to_reference(bounce_px, bounce_frame, motion) if anchored else None,
        bounce_is_bottom)

    settings = FitSettings(cd=args.cd, bounce_px_sigma=args.bounce_px_sigma,
                           bounce_z_sigma=math.hypot(0.02, 5.0 * args.frame_sigma / fps))
    print(f"Fitting {len(frames)} detections "
          f"({'anchored on the bounce' if anchored else 'NO bounce anchor'}) ...")
    fit = fit_flight(pose, obs, settings)
    if fit.problems:
        raise SystemExit("No trajectory result: the fit failed its sanity checks.\n  - "
                         + "\n  - ".join(fit.problems))

    summary, path, times = derive(fit.params, fit.cd, obs.bounce_time, fps)
    sigma_fit = formal_sigma(fit, obs.bounce_time, fps)
    sigma_mc, mc_used = {}, 0
    if args.mc > 0:
        print(f"Monte Carlo over the inputs ({args.mc} repeats) ...")
        sigma_mc, mc_used = monte_carlo(
            calib, pose, obs, fit, settings, fps, args.mc,
            np.random.default_rng(args.seed), click_sigma_px=max(calib.rms_px, 1.5),
            frame_sigma=args.frame_sigma, cd_sigma=args.cd_sigma)
    sigma_spin = {}
    if not anchored:
        # Refit with the vertical spin force held one sigma either side of
        # zero. Half the difference is the uncertainty this assumption adds.
        shifted = []
        for value in (UNANCHORED_CL_UP_SIGMA, -UNANCHORED_CL_UP_SIGMA):
            other = fit_flight(pose, obs, FitSettings(
                settings.cd, fit.px_sigma, settings.bounce_px_sigma,
                settings.bounce_z_sigma, settings.outlier_scale, False, value),
                start=fit.params)
            try:
                shifted.append(derive(other.params, other.cd, None, fps)[0])
            except ValueError:
                pass
        if len(shifted) == 2:
            sigma_spin = {k: abs(shifted[0][k] - shifted[1][k]) / 2
                          for k in summary if k in shifted[0] and k in shifted[1]}
    sigma = {k: math.sqrt(sigma_fit.get(k, 0.0) ** 2 + sigma_mc.get(k, 0.0) ** 2
                          + sigma_spin.get(k, 0.0) ** 2) for k in summary}

    report = build_report(args, calib, pose, obs, fit, summary, sigma, sigma_fit,
                          sigma_mc, mc_used, event_source, contact_frame, bounce_frame,
                          fps, motion, stabilisation, sigma_spin)
    print(report)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "trajectory_report.txt").write_text(report, encoding="utf-8")
    results = {
        "label": args.label, "fps": fps, "contact_frame": contact_frame,
        "bounce_frame": bounce_frame, "anchored": anchored,
        "parameters": dict(zip(PARAM_NAMES, map(float, fit.params))),
        "cd": fit.cd, "summary": summary, "sigma": sigma,
        "sigma_fit_only": sigma_fit, "sigma_inputs_only": sigma_mc,
        "sigma_unknown_vertical_spin": sigma_spin,
        "pixel_noise_px": fit.px_sigma,
        "detections_used": int(fit.inliers.sum()),
        "detections_rejected": [int(f) for f in obs.frames[~fit.inliers]],
        "stabilisation": stabilisation,
        "camera": {"focal_px": pose.focal_px, "centre_m": pose.centre.tolist(),
                   "tilt_deg": pose.tilt_deg(), "rms_px": pose.rms_px},
        # One row per frame from contact to bounce: time, position, velocity.
        "path": [[round(float(t), 5)] + [round(float(v), 4) for v in row]
                 for t, row in zip(times, path)],
    }
    (args.out_dir / "trajectory.json").write_text(json.dumps(results, indent=1),
                                                  encoding="utf-8")

    lifted = lift_detections(pose, obs, path, times)
    plot_path = args.out_dir / "trajectory_views.png"
    write_views_plot(plot_path, args.label, summary, path, lifted, fit.inliers, anchored)
    print(f"Side and top views: {plot_path}")
    if args.video:
        overlay_path = args.out_dir / "trajectory_overlay.jpg"
        write_overlay(overlay_path, args.video, calib, pose, obs, fit, path, summary,
                      anchored)
        print(f"Image overlay: {overlay_path}")
    return results


def build_report(args, calib, pose, obs, fit, summary, sigma, sigma_fit, sigma_mc,
                 mc_used, event_source, contact_frame, bounce_frame, fps, motion,
                 stabilisation=None, sigma_spin=None):
    def speeds(key):
        v, s = summary[key], sigma[key]
        return (f"{v * MS_TO_KMH:.0f} +/- {s * MS_TO_KMH:.0f} km/h "
                f"({v:.1f} m/s, {v * MS_TO_MPH:.0f} mph)")

    def metres(key, places=2):
        return f"{summary[key]:.{places}f} +/- {sigma[key]:.{places}f} m"

    s = summary
    residual = np.linalg.norm(fit.residuals_px[fit.inliers], axis=1)
    rejected = [int(f) for f in obs.frames[~fit.inliers]]

    lines = [f"=== {args.label}: 3D trajectory fit ===", ""]
    if not fit.anchored:
        lines += ["LOWER CONFIDENCE: no bounce was used. Vertical spin force is assumed",
                  "zero, and the landing point below is a prediction, not a measurement.", ""]
    lines += [
        f"LAUNCH SPEED (off the racket): {speeds('launch_speed_ms')}",
        "  This is the figure a radar gun reports. Not yet validated against one.",
        "",
        f"  speed at the net:      {speeds('speed_at_net_ms')}" if "speed_at_net_ms" in s else "",
        f"  speed at the bounce:   {speeds('landing_speed_ms')}",
        f"  average over flight:   {speeds('average_speed_ms')}",
        "",
        "Contact",
        f"  height:                {metres('contact_height_m')}",
        f"  position:              x = {metres('contact_x_m')} across, "
        f"y = {metres('contact_y_m')} down court",
        f"                         (y = 0 is the near baseline, y = {COURT_L} the far one)",
        f"  launch angle:          {s['launch_elevation_deg']:+.1f} +/- "
        f"{sigma['launch_elevation_deg']:.1f} deg from horizontal (+ is upward)",
        "",
        "Flight",
        f"  time to bounce:        {s['flight_time_s']:.3f} s",
    ]
    if "net_clearance_m" in s:
        lines.append(f"  net clearance:         {metres('net_clearance_m')}")
    depth = FAR_SERVICE_Y - s["landing_y_m"] if s["landing_y_m"] > NET_Y \
        else s["landing_y_m"] - SERVICE_D
    lines += [
        f"  {'bounce:' if fit.anchored else 'predicted landing:':<23}"
        f"x = {metres('landing_x_m')}, y = {metres('landing_y_m')}",
        f"                         {abs(depth):.2f} m "
        f"{'inside' if depth >= 0 else 'BEYOND'} the service line, "
        f"{abs(s['landing_x_m'] - CENTRE_X):.2f} m from the centre line",
        f"  sideways curve:        {abs(s['max_curve_m']):.2f} m to the "
        f"{'right' if s['max_curve_m'] > 0 else 'left'} of a straight line",
        "",
        "Spin (as force coefficients; see ball_physics.py)",
        f"  sideways  cl_side:     {s['cl_side']:+.3f} +/- {sigma['cl_side']:.3f}"
        f"   (- curves left, + curves right; includes any crosswind)",
    ]
    if fit.fit_cl_up:
        lines.append(f"  vertical  cl_up:       {s['cl_up']:+.3f} +/- {sigma['cl_up']:.3f}"
                     f"   (- is topspin, + is backspin)")
    else:
        lines.append("  vertical  cl_up:       held at 0 (cannot be measured without a bounce)")
    total_cl = math.hypot(s["cl_side"], s["cl_up"])
    rpm = spin_rpm_from_lift(total_cl, s["launch_speed_ms"])
    lines.append("  equivalent spin:       "
                 + (f"roughly {rpm:.0f} rpm if it is all spin and no wind (indicative only)"
                    if rpm is not None else "beyond the range of the spin formula"))

    lines += [
        "",
        "Fit quality",
        f"  detections:            {int(fit.inliers.sum())} used of {len(obs.frames)} "
        f"(frames {int(obs.frames[0])}-{int(obs.frames[-1])})",
        f"  rejected as outliers:  {rejected if rejected else 'none'}",
        f"  residual:              median {np.median(residual):.2f} px, "
        f"95% under {np.percentile(residual, 95):.2f} px",
        f"  pixel noise used:      {fit.px_sigma:.2f} px per axis",
        "",
        "Uncertainty of the launch speed (1-sigma)",
        f"  from pixel noise:      +/- {sigma_fit['launch_speed_ms'] * MS_TO_KMH:.1f} km/h",
    ]
    if sigma_mc:
        lines.append(f"  from the inputs:       +/- "
                     f"{sigma_mc['launch_speed_ms'] * MS_TO_KMH:.1f} km/h "
                     f"({mc_used} Monte Carlo repeats: landmarks, image centre, "
                     "bounce, drag)")
    else:
        lines.append("  from the inputs:       not evaluated (--mc 0)")
    if sigma_spin:
        lines.append(f"  from unknown topspin:  +/- "
                     f"{sigma_spin['launch_speed_ms'] * MS_TO_KMH:.1f} km/h "
                     f"(vertical spin force assumed 0 +/- {UNANCHORED_CL_UP_SIGMA})")

    if args.contact_ground_px:
        ground_frame = args.contact_ground_frame or contact_frame
        ground = calib.to_court(to_reference(args.contact_ground_px, ground_frame,
                                             motion))[0][0]
        lines += [
            "",
            "Independent check (not used by the fit)",
            f"  hand-marked ground point under the server: ({ground[0]:.2f}, "
            f"{ground[1]:.2f}) m",
            f"  fitted contact position:                   ({s['contact_x_m']:.2f}, "
            f"{s['contact_y_m']:.2f}) m",
            f"  difference: {math.hypot(ground[0] - s['contact_x_m'], ground[1] - s['contact_y_m']):.2f} m",
        ]

    motion_note = "none (tripod)"
    if motion:
        motion_note = (f"ORB homography to frame {calib.frame}, min "
                       f"{min(i for _, i in motion.values())} inliers")
    if stabilisation:
        st = stabilisation
        lines += [
            "",
            "Handheld check (static background points, "
            f"{st['frames_checked']} flight frames)",
            f"  camera drift during the flight:  median {st['drift_median_px']:.1f} px, "
            f"up to {st['drift_max_px']:.1f} px",
            f"  left after compensation:         median {st['residual_median_px']:.2f} px, "
            f"90% under {st['residual_p90_px']:.2f} px",
            "  The second line is the error handheld filming adds to each ball position.",
        ]
    lines += [
        "",
        "Camera (solved from the court landmarks)",
        describe(pose),
        "",
        "Inputs",
        f"  calibration:           {calib.path} ({len(calib.points)} landmarks)",
        f"  events:                {event_source}",
        f"  contact / bounce frame: {contact_frame} / {bounce_frame}, {fps:.2f} fps",
        f"  motion compensation:   {motion_note}",
        f"  drag coefficient:      {fit.cd} +/- {args.cd_sigma} (assumed)",
        "",
        "Caveats",
        "  * Not validated against radar.",
        "  * The uncertainty covers the listed inputs and pixel noise. It does not",
        "    cover a wrong model: gusting wind, or spin that changes in flight.",
        "  * Contact position is the fitted flight run back to the contact frame,",
        "    so it inherits that frame's +/- 1 frame (about 0.1 m along the path).",
    ]
    return "\n".join(line for line in lines if line is not None) + "\n"


if __name__ == "__main__":
    main()
