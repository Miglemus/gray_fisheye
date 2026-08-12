"""The camera axis of the FoV sweep: one knob, one exact mapping, zero fitted coefficients.

The sweep needs to change the FIELD OF VIEW and nothing else -- not the resolution, not
the gaussians, not the poses, not the family of lens.  The construction that does that:

    equidistant fisheye,  r(theta) = f * theta,   k1 = k2 = k3 = k4 = 0

`OPENCV_FISHEYE` with all four radial coefficients set to zero IS the equidistant
mapping, exactly.  Then, for a given output half-size R (pixels) and a target half-field
`theta_max`,

    f = R / theta_max

and the whole sweep is a sweep of one scalar.  Three properties make this the right
knob, and each of them kills an alternative:

1. **The inversion is exact and monotone everywhere.**  gray inverts r(theta) by
   bisection (`cuda/core/opencv_fisheye.cuh`) and 3dgrut folds the polynomial at its
   first stationary point.  With non-zero k's the two disagree about where the lens
   stops being invertible, and the disagreement lands *at the rim*, which is the part
   of the field the sweep is about.  With k = 0 there is nothing to invert.
2. **The lens family never changes across the sweep.**  Re-fitting a real rttpf lens at
   each field angle would change the mapping shape and the distortion at once.
3. **The pinhole control is available in the same construction** for the narrow half of
   the sweep: `r(theta) = f_p * tan(theta)`, i.e. `f_p = R / tan(theta_max)`.  Comparing
   the two at the *same* FoV, same resolution, same gaussians isolates "cost of a
   non-rectilinear projection" from "cost of a wider field".

WHAT THIS CONSTRUCTION DOES **NOT** CONTROL (say it in the paper, do not hide it):

* **Angular resolution falls as the field widens.**  At fixed pixel count a 200 deg
  frame samples ~11x fewer pixels per steradian than a 60 deg one.  That is what a real
  wide lens does, but it means "FPS at fixed resolution" and "FPS at fixed angular
  resolution" are different curves.  This module computes both scales so the second can
  be reported as a derived column.
* **Scene content in frustum grows with the field.**  More gaussians are visible at
  200 deg than at 60 deg.  Both engines see exactly the same content at each sweep
  point, so the RT/raster *ratio* is controlled; the per-engine *absolute* curve is not.
  `visible_fraction` below is the column that makes that explicit.

HARD LIMIT, MEASURED IN THE SOURCE, NOT ASSUMED:
`cuda/core/opencv_fisheye.cuh:31` sets `FISHEYE_MAX_THETA = 1.5707963f` and returns the
invalid bearing (0,0,0) beyond it; `rtpf.cuh:120` and `tpf.cuh:96` carry the same
constant.  **gray cannot render past 180 deg of field with any camera model it has
today.**  The sweep therefore stops at `GRAY_MAX_FOV_DEG` unless an uncapped equidistant
model is added first (see scripts/perf/README.md, "The 180 deg wall").
"""

from __future__ import annotations

import math
from typing import Dict, List, Sequence

#: gray's fisheye raygen returns an invalid bearing beyond theta = pi/2.
GRAY_MAX_THETA_RAD = 1.5707963
#: ... so this is the widest field gray can trace today.  Kept just under the cap: the
#: outermost pixel ring must stay strictly inside it or it renders as background.
GRAY_MAX_FOV_DEG = 175.0

#: The pre-registered sweep points.  Two regimes on purpose:
#:   * 60-120 deg: both a rectilinear and a fisheye camera exist, so the pinhole control
#:     is available and the distortion cost can be separated from the field cost.
#:   * 140-175 deg: fisheye only; a rectilinear camera is not merely slow there, it is
#:     undefined (tan(theta) -> inf).
SWEEP_FOV_DEG: List[float] = [60.0, 80.0, 100.0, 120.0, 140.0, 160.0, 175.0]
#: FoVs at which the rectilinear control is meaningful (tan(theta_max) stays sane).
PINHOLE_CONTROL_FOV_DEG: List[float] = [60.0, 80.0, 100.0, 120.0]
#: Beyond this, a rectilinear frame's corner magnification is absurd and the comparison
#: stops being about renderers.
PINHOLE_MAX_FOV_DEG = 120.0


def equidistant_intrinsics(width: int, height: int, fov_deg: float) -> List[float]:
    """OPENCV_FISHEYE params [fx, fy, cx, cy, k1, k2, k3, k4] for an exact r = f*theta lens.

    `fov_deg` is the FULL field across the INSCRIBED circle, i.e. across the shorter
    image axis.  Using the shorter axis (rather than the diagonal) means the image
    circle is inscribed in the frame and every traced pixel is inside the field: the
    corners fall outside the circle and are background in every sweep point, so the
    background fraction is constant and cannot masquerade as a speed trend.
    """
    if fov_deg <= 0 or fov_deg >= 360:
        raise ValueError(f"fov_deg out of range: {fov_deg}")
    theta_max = math.radians(fov_deg) / 2.0
    radius = min(width, height) / 2.0
    f = radius / theta_max
    return [f, f, width / 2.0, height / 2.0, 0.0, 0.0, 0.0, 0.0]


def pinhole_fov_y(width: int, height: int, fov_deg: float) -> Dict[str, float]:
    """fov_x / fov_y of the rectilinear control at the same inscribed-circle field.

    Same convention as `equidistant_intrinsics`: `fov_deg` is the field across the
    shorter axis, so `f = R / tan(theta_max)` with `R = min(w, h) / 2`.
    """
    if fov_deg >= 180.0:
        raise ValueError("a rectilinear camera cannot reach 180 deg")
    theta_max = math.radians(fov_deg) / 2.0
    radius = min(width, height) / 2.0
    f = radius / math.tan(theta_max)
    return {
        "focal": f,
        "fov_x": 2.0 * math.atan((width / 2.0) / f),
        "fov_y": 2.0 * math.atan((height / 2.0) / f),
    }


def theta_max_rad(fov_deg: float) -> float:
    return math.radians(fov_deg) / 2.0


def solid_angle_sr(fov_deg: float) -> float:
    """Solid angle of the inscribed cone, 2*pi*(1 - cos theta_max)."""
    return 2.0 * math.pi * (1.0 - math.cos(theta_max_rad(fov_deg)))


def pixels_per_steradian(width: int, height: int, fov_deg: float) -> float:
    """Angular sampling density of the inscribed disk -- the variable the sweep cannot hold fixed."""
    radius = min(width, height) / 2.0
    disk_pixels = math.pi * radius * radius
    return disk_pixels / solid_angle_sr(fov_deg)


def check_gray_can_trace(fov_deg: float) -> None:
    if theta_max_rad(fov_deg) >= GRAY_MAX_THETA_RAD:
        raise ValueError(
            f"fov {fov_deg} deg needs theta_max = {math.degrees(theta_max_rad(fov_deg)):.1f} deg, "
            f"but gray's fisheye raygen caps theta at 90 deg "
            f"(cuda/core/opencv_fisheye.cuh:31, rtpf.cuh:120, tpf.cuh:96). "
            f"Max renderable field is {GRAY_MAX_FOV_DEG} deg."
        )


def sweep_plan(width: int, height: int,
               fovs: Sequence[float] = tuple(SWEEP_FOV_DEG)) -> List[Dict[str, object]]:
    """The full list of (camera model, intrinsics) points, with the derived columns."""
    plan: List[Dict[str, object]] = []
    for fov in fovs:
        check_gray_can_trace(fov)
        plan.append({
            "arm": "fisheye_equidistant",
            "fov_deg": float(fov),
            "camera_model": "opencv_fisheye",
            "intrinsics": equidistant_intrinsics(width, height, fov),
            "theta_max_deg": math.degrees(theta_max_rad(fov)),
            "solid_angle_sr": solid_angle_sr(fov),
            "pixels_per_sr": pixels_per_steradian(width, height, fov),
        })
        if fov <= PINHOLE_MAX_FOV_DEG:
            ph = pinhole_fov_y(width, height, fov)
            plan.append({
                "arm": "pinhole_control",
                "fov_deg": float(fov),
                "camera_model": "pinhole",
                "focal": ph["focal"],
                "fov_x": ph["fov_x"],
                "fov_y": ph["fov_y"],
                "theta_max_deg": math.degrees(theta_max_rad(fov)),
                "solid_angle_sr": solid_angle_sr(fov),
                "pixels_per_sr": pixels_per_steradian(width, height, fov),
            })
    return plan


if __name__ == "__main__":  # pragma: no cover - a printable summary, no GPU
    import argparse
    ap = argparse.ArgumentParser(description="Print the FoV sweep camera plan (no GPU).")
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--height", type=int, default=1024)
    a = ap.parse_args()
    print(f"# FoV sweep camera plan at {a.width}x{a.height}\n")
    print(f"{'arm':22s} {'FoV':>7s} {'theta_max':>10s} {'f (px)':>10s} "
          f"{'sr':>8s} {'px/sr':>12s}")
    for p in sweep_plan(a.width, a.height):
        f = p["intrinsics"][0] if "intrinsics" in p else p["focal"]
        print(f"{p['arm']:22s} {p['fov_deg']:6.1f}d {p['theta_max_deg']:9.1f}d "
              f"{f:10.2f} {p['solid_angle_sr']:8.3f} {p['pixels_per_sr']:12.1f}")
    print(f"\ngray fisheye raygen caps theta at 90 deg -> max field {GRAY_MAX_FOV_DEG} deg.")
