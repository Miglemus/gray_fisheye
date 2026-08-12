#!/usr/bin/env python3
"""Attest a gray run's existing `fps.csv`: read the run's own files, write the sidecar.

    python3 scripts/perf/stamp_gray_run.py --run <gray run dir> --gpu 1 [--method gray]

Convenience wrapper over `stamp_provenance.py` for the ONE case where re-measuring is not
an option: a shared pipeline (`dataset/fullcircle_code/queue_perf.sh`) whose `fps.csv` the
comparison viewer already consumes.  It fills the schema from what the run directory
already records -- `config.json`, `num_gaussians.csv`, `cameras.json`, the rendered test
views -- and copies the *current* `measure_fps.py` protocol into the timing block.

It writes `tier = "attested"`, which `collect_fps.py` REJECTS by default.  That is the
point: the sidecar makes the number's conditions legible, it does not make it publishable.
Publishable means `scripts/perf/bench_fps.py`.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import stamp_provenance  # noqa: E402


def infer(run: Path):
    """(n_gaussians, width, height, n_views) from what the run already wrote."""
    n_gauss = None
    ng = run / "num_gaussians.csv"
    if ng.exists():
        rows = [r for r in ng.read_text().splitlines() if r.strip()]
        if len(rows) > 1:
            # header is comma-separated, rows are space-separated (train.py:215 vs :503)
            n_gauss = int(rows[-1].split()[-1])

    width = height = None
    cams = run / "cameras.json"
    if cams.exists():
        data = json.loads(cams.read_text())
        if data:
            width = int(data[0]["image_width"])
            height = int(data[0]["image_height"])
    cfg = json.loads((run / "config.json").read_text())
    ds = int(cfg.get("downsampling") or 1)
    if width and ds > 1 and not (run / "test").exists():
        pass  # cameras.json already carries the loaded (downsampled) size

    # renders live at test/<iteration>/<camera model>/renders/*.png; count ONLY renders,
    # never the sibling gt/ (same count today, but that is a coincidence to not rely on)
    n_views = None
    for pat in ("test/*/*/renders/*.png", "test/*/renders/*.png", "test/renders/*.png"):
        hits = glob.glob(str(run / pat))
        if hits:
            n_views = len(hits)
            break
    return n_gauss, width, height, n_views


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--gpu", type=int, required=True)
    ap.add_argument("--method", default="gray")
    ap.add_argument("--views", type=int, default=None, help="override the inferred count")
    a = ap.parse_args(argv)

    run = Path(a.run)
    fps_csv = run / "fps.csv"
    if not fps_csv.exists():
        print(f"no {fps_csv}; nothing to attest", file=sys.stderr)
        return 2
    n_gauss, w, h, n_views = infer(run)
    n_views = a.views or n_views
    missing = [k for k, v in (("n_gaussians", n_gauss), ("width", w),
                              ("height", h), ("views", n_views)) if not v]
    if missing:
        print(f"cannot attest {run}: could not infer {', '.join(missing)} from the run "
              f"directory. Re-measure with scripts/perf/bench_fps.py instead.",
              file=sys.stderr)
        return 3

    return stamp_provenance.main([
        "--run", str(run), "--method", a.method, "--gpu", str(a.gpu),
        "--fps-from", str(fps_csv),
        "--n-gaussians", str(n_gauss), "--width", str(w), "--height", str(h),
        "--views", str(n_views), "--context", "test",
        "--repeats", "1", "--aggregation", "single pass", "--warmup-passes", "1",
        "--sync", "cuda events + torch.cuda.synchronize() around the whole loop",
        "--timed-region",
        "measure_fps.py: loop over test views, raytracer(cam, skip_copy=True), no_grad",
        "--excludes", "scene load", "safetensors load", "BVH build", "warmup pass",
        "readback (skip_copy=True)",
        "--repo", str(run),
    ])


if __name__ == "__main__":
    raise SystemExit(main())
