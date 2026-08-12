"""Does z(theta) still mean anything when it is the ONLY thing being learned?

The subtractive rung `z_only` scores nearly as well as the full model, but score is not the
question here. The mean-over-depth part of the non-central shift, z(theta) sin(theta)
E[1/t | theta], has exactly the form of a central radial correction, so with `radial`
removed there is nothing stopping `z` from being pulled into a second job it was never
meant to do. If that happens the learned profile stops being a statement about the lens.

The test is shape, not amplitude: a physical z(theta) should be the SAME curve the full
model finds, up to a scale. A z(theta) doing double duty should not correlate with it.
"""

import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fields import bspline_eval, load_weights  # noqa: E402
from subtractive import run_dir  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
SCENES = ["atrium", "classroom", "forest", "library", "reception", "tunnel", "workshop"]


def profile(scene, variant, samples):
    run = run_dir(scene, variant)
    if not run or not os.path.exists(f"{run}/gaussians_15000.safetensors"):
        return None
    z = bspline_eval(load_weights(run)["z_weights"], samples)[0]
    return z - z[0]  # * the gauge z(0) = 0 the forward pass imposes


def main():
    samples = np.linspace(0.0, 1.0, 91)  # * 0..90 deg in 1-deg steps, so index 85 IS 85 deg
    out = {"theta_deg": (samples * 90.0).tolist(), "scenes": {}}
    print(f"{'scene':11s} {'|z|max full':>12s} {'|z|max z_only':>14s} {'ratio':>7s} "
          f"{'r@85deg':>9s} {'corr':>7s}")
    rows = []
    for scene in SCENES:
        full = profile(scene, "noncentral", samples)
        alone = profile(scene, "z_only", samples)
        if full is None or alone is None:
            print(f"  -- {scene}: missing")
            continue
        # * Pearson correlation of the two curves over theta -- shape agreement, scale-free.
        corr = float(np.corrcoef(full, alone)[0, 1])
        af, aa = float(np.abs(full).max()), float(np.abs(alone).max())
        # * Also at a fixed field angle: `max |z|` can land at different theta on the two
        # * curves, so it is not by itself a like-for-like amplitude ratio.
        r85 = float(alone[85] / full[85])
        out["scenes"][scene] = {
            "full": [float(f"{v:.6g}") for v in full],
            "z_only": [float(f"{v:.6g}") for v in alone],
            "corr": corr,
            "amp_full": af,
            "amp_z_only": aa,
            "ratio_at_85deg": r85,
        }
        rows.append((corr, aa / af, r85))
        print(f"{scene:11s} {af:12.5f} {aa:14.5f} {aa/af:7.2f} {r85:9.2f} {corr:7.3f}")

    if rows:
        c = [r[0] for r in rows]
        print(f"\ncorrelation: median {np.median(c):.3f}, min {min(c):.3f}, "
              f"{sum(1 for v in c if v > 0.9)}/{len(c)} above 0.9")
        print(f"amplitude ratio z_only/full: on max|z| median "
              f"{np.median([r[1] for r in rows]):.2f} (range {min(r[1] for r in rows):.2f}"
              f"-{max(r[1] for r in rows):.2f}); at 85 deg median "
              f"{np.median([r[2] for r in rows]):.2f} (range {min(r[2] for r in rows):.2f}"
              f"-{max(r[2] for r in rows):.2f})")
        # * A correlation near 1 between two smooth monotone curves is nearly free -- it is
        # * the amplitude that carries the information. Read the two together.

    with open(os.path.join(HERE, "z_shape.json"), "w") as handle:
        json.dump(out, handle)
    print("wrote z_shape.json")


if __name__ == "__main__":
    main()
