#!/usr/bin/env python3
"""Stamp a provenance sidecar next to a value another program produced.

    python3 scripts/perf/stamp_provenance.py --run <dir> --method gray \
        --fps-from <dir>/fps.csv --gpu 1 --n-gaussians 130246 \
        --width 1440 --height 1440 --views 106 --context test \
        --repeats 1 --aggregation "single pass" --warmup-passes 1 \
        --sync "cuda events + synchronize around the loop" \
        --timed-region "loop over test views, render(), no image saving"

Pure standard library, so it runs inside DirectFisheye-GS', 3dgrut's, SPaGS' and
3DGEER's virtualenvs as well as gray's.

READ THIS BEFORE USING IT.  The sidecar it writes is `tier = "attested"`, not
`"measured"`, and `collect_fps.py` REJECTS it by default.  That is deliberate: the card
identity and the free VRAM it records are real, but they are observed *after* the timed
region, so they cannot prove the card was exclusive while the clock was running.  Its
legitimate uses are:

  * attaching the conditions to a legacy number so a future reader knows what it is;
  * upgrading a shared pipeline (`dataset/fullcircle_code/queue_perf.sh`) whose value
    files feed the viewer, without changing what that pipeline writes.

It is NOT a substitute for re-measuring with `bench_fps.py`.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import provenance  # noqa: E402


def read_value(path: Path) -> float:
    txt = path.read_text().strip()
    if path.suffix == ".csv":
        return float(txt.splitlines()[0].split(",")[0])
    data = json.loads(txt)
    if isinstance(data, dict):
        for key in ("fps", "1.0", "1"):
            if key in data:
                return float(data[key])
        for v in data.values():
            if isinstance(v, (int, float)):
                return float(v)
    raise ValueError(f"cannot read an FPS value out of {path}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="run directory the sidecar belongs to")
    ap.add_argument("--method", required=True)
    ap.add_argument("--gpu", type=int, required=True, help="physical nvidia-smi index")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--fps", type=float)
    g.add_argument("--fps-from", help="fps.csv or fps.json to read the value from")
    ap.add_argument("--n-gaussians", type=int, required=True)
    ap.add_argument("--width", type=int, required=True)
    ap.add_argument("--height", type=int, required=True)
    ap.add_argument("--views", type=int, required=True)
    ap.add_argument("--context", default="test")
    ap.add_argument("--repeats", type=int, required=True)
    ap.add_argument("--aggregation", required=True)
    ap.add_argument("--warmup-passes", type=int, required=True)
    ap.add_argument("--sync", required=True)
    ap.add_argument("--timed-region", required=True)
    ap.add_argument("--excludes", nargs="*", default=["scene load", "BVH build", "warmup"])
    ap.add_argument("--repo", default=None, help="repo root, for the git stamp")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)

    fps = a.fps if a.fps is not None else read_value(Path(a.fps_from))
    gpu = provenance.gpu_info(a.gpu)
    foreign = provenance.compute_apps(a.gpu, exclude_pid=os.getpid())

    doc = provenance.build(
        tier="attested",
        harness="scripts/perf/stamp_provenance.py",
        method=a.method,
        run_path=a.run,
        gpu_index=a.gpu,
        n_gaussians=a.n_gaussians,
        width=a.width, height=a.height,
        n_views=a.views, context=a.context,
        fps=fps,
        timing={
            "repeats": a.repeats,
            "aggregation": a.aggregation,
            "warmup_passes": a.warmup_passes,
            "sync": a.sync,
            "timed_region": a.timed_region,
            "timed_region_excludes": list(a.excludes),
            "per_repeat_fps": None,
            "spread_pct": None,
        },
        gpu_start=gpu,
        foreign_start=foreign,
        extra={"git": provenance.git_info(a.repo or a.run),
               "value_source": a.fps_from or "--fps",
               "warning": "observed AFTER the timed region; exclusivity is not proven"},
    )
    out = Path(a.out) if a.out else provenance.sidecar_path(a.run)
    provenance.write(doc, out)
    print(f"stamped (tier=attested) {out}  fps={fps:.2f}  card={gpu['name']} {gpu['uuid']}")
    problems = provenance.validate(doc, strict_tier="attested")
    for p in problems:
        print(f"  WARNING: {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
