"""
camera_pose.py

Recovers WHERE THE CAMERA IS and HOW IT IS POINTED from a court calibration,
so that pixels can be related to full 3D positions, not just to the court
surface.

Why this exists:
    calibrate_court.py gives a homography. A homography maps the image to ONE
    plane (the court surface) and says nothing about points off that plane.
    A ball in flight is off the plane for the whole serve. To reason about it
    we need the full pinhole camera model:

        pixel = K [R | t] X

        X  a 3D point in court coordinates (metres)
        R  3x3 rotation, court axes -> camera axes        } "extrinsics",
        t  translation, so that X_cam = R X + t           } the camera pose
        K  the "intrinsics": focal length and principal point

    With K, R and t known, two things become possible that the homography
    cannot do:
        project(X)  where a 3D point appears in the image
        ray(pixel)  the 3D line of all points that appear at that pixel

    fit_trajectory.py uses project() to compare a simulated 3D ball flight
    against the detected pixels.

Where the focal length comes from:
    The usual route is a checkerboard calibration (calibrate_camera.py). That
    has only been done in portrait, and the phone's stabilisation crops the
    sensor by an amount we cannot see. Instead the focal length is solved
    from the court itself, at the same time as the pose. This works because
    the court is viewed at a steep slant: a wrong focal length cannot
    reproduce how quickly the far court shrinks relative to the near court.
    On serve3, moving f by 6% from its best value more than triples the
    landmark error, so f is pinned to within about 1%.

    The caveat is that this only works for a strongly slanted view of a large
    plane. A camera looking straight down at the court could not separate
    focal length from height.

Assumptions (all reported by --check so they can be judged):
    * Principal point at the image centre. Phone cameras are within a few
      tens of pixels of this.
    * Square pixels, no skew.
    * No lens distortion. The checkerboard run measured small coefficients
      for this lens, and video frames arrive already corrected.

Coordinates:
    Court: origin at the near-left singles corner, x across, y down the
    court, z UP from the court surface (right-handed). Same x, y as
    calibrate_court.py.
    Camera (OpenCV convention): x right, y DOWN, z forward out of the lens.

Usage (PowerShell):
    python pipeline/camera_pose.py --calibration tracking\\serve3\\court_f1014.json
"""

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares

from calibrate_court import ALL_LANDMARKS, CourtCalibration, load_calibration

# A phone main camera in video mode has a horizontal field of view somewhere
# between about 40 and 80 degrees. Starting guesses are spread across that
# range as a fraction of image width, in case the closed-form estimate fails.
FALLBACK_FOCAL_FRACTIONS = (0.6, 0.75, 0.9, 1.1, 1.4)

# A pose is refused if the camera comes out below the court or implausibly
# high. This catches mirrored solutions and swapped landmarks.
MIN_CAMERA_HEIGHT_M = 0.2
MAX_CAMERA_HEIGHT_M = 15.0


@dataclass
class CameraPose:
    """A pinhole camera located in court coordinates.

    Built by solve_camera_pose(). Holds everything needed to go between
    3D court positions and pixels of the calibration frame.
    """
    focal_px: float          # focal length in pixels (same in x and y)
    principal: np.ndarray    # (cx, cy), the pixel the optical axis hits
    R: np.ndarray            # 3x3, court axes -> camera axes
    t: np.ndarray            # 3-vector, X_cam = R @ X_court + t
    rms_px: float            # landmark reprojection error of the solved pose
    focal_sigma_px: float    # 1-sigma uncertainty of the focal length
    image_size: tuple        # (width, height)

    @property
    def centre(self):
        """Camera position in court coordinates (metres).

        From X_cam = R X + t, the camera centre is the point with X_cam = 0,
        so X = -R^T t.
        """
        return -self.R.T @ self.t

    @property
    def K(self):
        return np.array([[self.focal_px, 0, self.principal[0]],
                         [0, self.focal_px, self.principal[1]],
                         [0, 0, 1.0]])

    def project(self, xyz):
        """3D court points (N, 3) -> pixels (N, 2).

        Points behind the camera have no image and come back as NaN.
        """
        cam = np.asarray(xyz, dtype=float).reshape(-1, 3) @ self.R.T + self.t
        with np.errstate(divide="ignore", invalid="ignore"):
            px = self.focal_px * cam[:, :2] / cam[:, 2:3] + self.principal
        px[cam[:, 2] <= 0] = np.nan
        return px

    def ray(self, px):
        """Pixels (N, 2) -> unit direction vectors (N, 3) in court coordinates.

        Every 3D point that appears at a pixel lies on the line
        centre + s * direction, for s > 0. One camera can never tell you s:
        that is the depth ambiguity the trajectory fit has to resolve.
        """
        px = np.asarray(px, dtype=float).reshape(-1, 2)
        cam = np.c_[(px - self.principal) / self.focal_px, np.ones(len(px))]
        world = cam @ self.R            # R^T applied to each row
        return world / np.linalg.norm(world, axis=1, keepdims=True)

    def ray_at_height(self, px, z):
        """Where the ray through each pixel crosses the horizontal plane at
        height z. Returns (N, 3); NaN where the ray never reaches it."""
        d = self.ray(px)
        c = self.centre
        with np.errstate(divide="ignore", invalid="ignore"):
            s = (z - c[2]) / d[:, 2]
        s[s <= 0] = np.nan
        return c + s[:, None] * d

    def horizontal_fov_deg(self):
        return math.degrees(2 * math.atan(self.image_size[0] / (2 * self.focal_px)))

    def tilt_deg(self):
        """Angle of the optical axis below horizontal (positive = looking down)."""
        forward = self.R[2]             # camera z axis, in court coordinates
        return math.degrees(math.asin(-forward[2]))


def _project(params, obj, principal):
    """Pinhole projection for the solver. params = [f, rvec(3), tvec(3)].

    rvec is a Rodrigues rotation vector: its direction is the rotation axis
    and its length the angle. It has exactly 3 numbers for the 3 degrees of
    freedom of a rotation, which is what an optimiser needs. A 3x3 matrix
    has 9 numbers and would have to be kept orthogonal by force.
    """
    R = cv2.Rodrigues(np.float64(params[1:4]))[0]
    cam = obj @ R.T + params[4:7]
    return params[0] * cam[:, :2] / cam[:, 2:3] + principal


def focal_from_homography(H_court_to_image, principal):
    """Closed-form focal length estimates from a plane homography.

    For points on the court plane (z = 0) the camera model collapses to
    H = K [r1 r2 t], where r1 and r2 are the first two columns of R. Those
    columns are perpendicular and the same length, because R is a rotation.
    Writing r = K^-1 h for the first two columns of H gives two equations in
    which f is the only unknown:

        r1 . r2 = 0       (perpendicular)
        |r1| = |r2|       (equal length)

    Returns the estimates that came out real and positive. They are used
    only as a starting point for the least-squares refinement.
    """
    shift = np.array([[1, 0, -principal[0]], [0, 1, -principal[1]], [0, 0, 1.0]])
    G = shift @ H_court_to_image
    h1, h2 = G[:, 0], G[:, 1]
    estimates = []
    with np.errstate(divide="ignore", invalid="ignore"):
        perpendicular = -(h1[0] * h2[0] + h1[1] * h2[1]) / (h1[2] * h2[2])
        equal_length = ((h2[0] ** 2 + h2[1] ** 2 - h1[0] ** 2 - h1[1] ** 2)
                        / (h1[2] ** 2 - h2[2] ** 2))
    for f_squared in (perpendicular, equal_length):
        if np.isfinite(f_squared) and f_squared > 0:
            estimates.append(math.sqrt(f_squared))
    return estimates


def solve_pose_from_points(image_pts, world_xy, image_size, principal=None,
                           focal_px=None, focal_guess=None):
    """Camera pose from pixel <-> court-plane correspondences.

    image_pts  (N, 2) pixels
    world_xy   (N, 2) court metres (all on the court surface, z = 0)
    focal_px   pass a value to hold the focal length fixed (for example from
               a checkerboard calibration); leave None to solve for it.
    focal_guess  a single starting value for the solve, when a good one is
               already known. Skips the multi-start search (used by the
               Monte Carlo, which re-solves the same camera many times).

    Method: minimise the landmark reprojection error over [f, rotation,
    translation] with Levenberg-Marquardt, starting from a closed-form pose.
    Several starting focal lengths are tried and the best result kept,
    because a planar scene has two mirror-image pose solutions and a bad
    start can land in the wrong one.
    """
    image_pts = np.asarray(image_pts, dtype=float)
    obj = np.c_[np.asarray(world_xy, dtype=float), np.zeros(len(image_pts))]
    if principal is None:
        principal = np.array([image_size[0] / 2, image_size[1] / 2])
    principal = np.asarray(principal, dtype=float)

    if focal_px is not None:
        starts = [focal_px]
    elif focal_guess is not None:
        starts = [focal_guess]
    else:
        H, _ = cv2.findHomography(obj[:, :2], image_pts, 0)
        starts = focal_from_homography(H, principal) if H is not None else []
        starts += [frac * image_size[0] for frac in FALLBACK_FOCAL_FRACTIONS]

    best = None
    for f0 in starts:
        K0 = np.array([[f0, 0, principal[0]], [0, f0, principal[1]], [0, 0, 1.0]])
        # IPPE is OpenCV's solver for planar targets. It returns both mirror
        # solutions; both are refined and the lower-error one kept.
        count, rvecs, tvecs, _ = cv2.solvePnPGeneric(obj, image_pts, K0, None,
                                                     flags=cv2.SOLVEPNP_IPPE)
        for i in range(count):
            x0 = np.r_[f0, rvecs[i].ravel(), tvecs[i].ravel()]
            if focal_px is None:
                fit = least_squares(
                    lambda p: (_project(p, obj, principal) - image_pts).ravel(), x0)
                params = fit.x
            else:
                fit = least_squares(
                    lambda p: (_project(np.r_[focal_px, p], obj, principal)
                               - image_pts).ravel(), x0[1:])
                params = np.r_[focal_px, fit.x]
            cost = float((fit.fun ** 2).sum())
            # The optimiser is free to wander into a mirror world: negative
            # focal length with the court behind the camera reproduces the
            # same pixels. Only physical solutions are kept.
            depth = (obj @ cv2.Rodrigues(np.float64(params[1:4]))[0].T + params[4:7])[:, 2]
            if params[0] <= 0 or np.any(depth <= 0):
                continue
            if best is None or cost < best[0]:
                best = (cost, params, fit)
    if best is None:
        raise SystemExit("Camera pose could not be solved from these landmarks.")

    cost, params, fit = best
    n = len(image_pts)
    rms = math.sqrt(cost / n)

    # Uncertainty of f from the fit itself: covariance = s^2 (J^T J)^-1, with
    # s^2 the residual variance. This says how far f can move before the
    # landmark errors grow noticeably. Zero when f was supplied.
    focal_sigma = 0.0
    if focal_px is None:
        dof = max(2 * n - len(params), 1)
        try:
            cov = np.linalg.inv(fit.jac.T @ fit.jac) * cost / dof
            focal_sigma = float(math.sqrt(cov[0, 0]))
        except np.linalg.LinAlgError:
            focal_sigma = float("nan")

    R = cv2.Rodrigues(np.float64(params[1:4]))[0]
    pose = CameraPose(float(params[0]), principal, R, np.array(params[4:7]),
                      rms, focal_sigma, tuple(image_size))

    height = pose.centre[2]
    if not MIN_CAMERA_HEIGHT_M <= height <= MAX_CAMERA_HEIGHT_M:
        raise SystemExit(
            f"Camera pose is not believable: camera height {height:.2f} m. "
            "Check the landmark names in the calibration file.")
    return pose


def solve_camera_pose(calib: CourtCalibration, focal_px=None, principal=None):
    """Camera pose for the calibration frame of a court calibration."""
    names = list(calib.points)
    return solve_pose_from_points(
        [calib.points[n] for n in names], [ALL_LANDMARKS[n] for n in names],
        calib.image_size, principal=principal, focal_px=focal_px)


def describe(pose: CameraPose):
    c = pose.centre
    return (
        f"  focal length:      {pose.focal_px:.0f} px  (+/- {pose.focal_sigma_px:.0f} px, "
        f"horizontal field of view {pose.horizontal_fov_deg():.1f} deg)\n"
        f"  camera position:   x = {c[0]:.2f} m across, y = {c[1]:.2f} m down court, "
        f"{c[2]:.2f} m above the court\n"
        f"  tilt:              {pose.tilt_deg():.1f} deg below horizontal\n"
        f"  landmark error:    {pose.rms_px:.2f} px RMS")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--calibration", required=True, type=Path,
                   help="Court calibration JSON from calibrate_court.py")
    p.add_argument("--focal_px", type=float,
                   help="Hold the focal length at this value instead of solving "
                        "for it")
    args = p.parse_args()

    calib = load_calibration(args.calibration)
    pose = solve_camera_pose(calib, focal_px=args.focal_px)
    print(f"Camera pose from {args.calibration} (frame {calib.frame}, "
          f"{len(calib.points)} landmarks)")
    print(describe(pose))
    print("\n  A negative y means the camera is behind the near baseline.")


if __name__ == "__main__":
    main()
