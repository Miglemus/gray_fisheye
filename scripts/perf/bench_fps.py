"""Measure gray's render FPS so that the number is CITABLE.

    python scripts/perf/bench_fps.py -m <run_dir> --gpu 1 --out tmp/perf/<name>.json

What "citable" means here, and what each rule prevents:

  exclusive card          `nvidia-smi --query-compute-apps` before AND after the timed
                          region; the run aborts if any foreign CUDA context is on the
                          card.  A pueue group does NOT reserve a GPU -- it serialises
                          its own tasks only, and this project has already lost four runs
                          to OOM and an unknown number of FPS values to contention that
                          way.
  strictly sequential     one process, one card, one checkpoint.  Nothing is timed while
                          anything else of ours runs.
  explicit warmup         `--warmup-passes` full passes over every view, outside the
                          clock: OptiX pipeline compile, BVH upload, the base-bearing
                          cache and the pose-independent ray cache all populate on the
                          first frame.
  explicit synchronisation `torch.cuda.synchronize()` immediately before `start.record()`
                          and immediately after `end.record()`, every repeat.  Without
                          it the clock measures launch queueing.
  median of N passes      `--repeats 3` (median, not best-of).  A best-of, which is what
                          SPaGS reports, is a different estimator: it is the *floor* of
                          the contention distribution, and it is not comparable with
                          anyone else's single pass.
  provenance sidecar      written on EVERY measurement (scripts/perf/provenance.py):
                          GPU name/uuid/driver, timestamp, free VRAM at start and end,
                          resolution, gaussian count, view count, and the text of what is
                          inside the timed region.

It deliberately does NOT write `fps.csv`.  `measure_fps.py` at the repo root does, and
running it twice to A/B anything leaves the last arm in the canonical file the FullCircle
and myscenes tables read.  This harness only writes where `--out` points.

The FoV sweep (`scripts/perf/fov_sweep.py`) drives this same timing core through
`bench_one()`, so the sweep and the plain measurement cannot drift apart.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_HERE))

import provenance  # noqa: E402

#: Verbatim description of the clocked region.  It is written into every sidecar, because
#: "what is inside the clock" is the single field that makes two methods comparable or not.
TIMED_REGION = (
    "for cam in views: raytracer(cam, skip_copy=True) under torch.no_grad(); "
    "cuda events around the whole loop; torch.cuda.synchronize() before start.record() "
    "and after end.record()"
)
TIMED_REGION_EXCLUDES = [
    "COLMAP scene load and image decode",
    "safetensors load and Raytracer construction",
    "BVH / OptiX acceleration-structure build",
    "warmup passes (pipeline compile, base-bearing cache, pose-independent ray cache)",
    "device->host readback (skip_copy=True keeps the framebuffer clone out too)",
    "metric computation and PNG encoding",
]


# --------------------------------------------------------------------------------------
# exclusivity
# --------------------------------------------------------------------------------------

def resolve_smi_index(requested: Optional[int]) -> int:
    """Map what the caller asked for onto the nvidia-smi index that actually ran.

    `CUDA_VISIBLE_DEVICES` renumbers devices for torch but not for nvidia-smi, and this
    machine flips the two orders unless CUDA_DEVICE_ORDER=PCI_BUS_ID.  We therefore take
    the *physical* index from CUDA_VISIBLE_DEVICES when it names exactly one device, and
    refuse to guess when it names several.
    """
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible not in (None, "", "<unset>"):
        parts = [p for p in visible.split(",") if p.strip() != ""]
        if len(parts) != 1:
            raise SystemExit(
                f"CUDA_VISIBLE_DEVICES={visible!r} exposes {len(parts)} devices. "
                "Pin exactly one card, or the sidecar cannot say which one ran."
            )
        if not parts[0].strip().isdigit():
            raise SystemExit(f"CUDA_VISIBLE_DEVICES={visible!r} is not a physical index")
        return int(parts[0])
    if requested is None:
        raise SystemExit("pass --gpu <physical index> or set CUDA_VISIBLE_DEVICES")
    return int(requested)


def assert_exclusive(smi_index: int, allow_shared: bool, when: str) -> List[Dict[str, Any]]:
    foreign = provenance.compute_apps(smi_index, exclude_pid=os.getpid())
    if foreign and not allow_shared:
        lines = "\n".join(
            f"    pid {p['pid']:>8}  {p['used_mib']:>6} MiB  {p['name']}" for p in foreign)
        raise SystemExit(
            f"REFUSING TO TIME: GPU {smi_index} is not exclusive ({when}).\n{lines}\n"
            "A pueue group does not reserve a card. Wait, or pass --allow-shared to write "
            "a sidecar that collect_fps.py will REJECT."
        )
    return foreign


# --------------------------------------------------------------------------------------
# timing core
# --------------------------------------------------------------------------------------

def timed_passes(render_one: Callable[[Any], None], views: Sequence[Any],
                 repeats: int, warmup_passes: int) -> Dict[str, Any]:
    """Warm up, then time `repeats` full passes over `views`.  Returns the timing block.

    Kept free of gray imports so the FoV sweep and any future engine wrapper can reuse
    the exact same clock.
    """
    import torch

    if len(views) == 0:
        raise SystemExit("no views to render")
    for _ in range(max(0, warmup_passes)):
        for cam in views:
            render_one(cam)
    torch.cuda.synchronize()

    per_repeat: List[float] = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        start.record()
        for cam in views:
            render_one(cam)
        end.record()
        torch.cuda.synchronize()
        secs = start.elapsed_time(end) / 1000.0
        per_repeat.append(len(views) / secs)

    med = statistics.median(per_repeat)
    spread = 100.0 * (max(per_repeat) - min(per_repeat)) / med if med > 0 else float("inf")
    return {
        "repeats": repeats,
        "aggregation": "median",
        "warmup_passes": warmup_passes,
        "sync": "torch.cuda.synchronize() before start.record() and after end.record(), every repeat",
        "clock": "torch.cuda.Event(enable_timing=True)",
        "timed_region": TIMED_REGION,
        "timed_region_excludes": list(TIMED_REGION_EXCLUDES),
        "per_repeat_fps": [round(x, 4) for x in per_repeat],
        "fps_median": med,
        "fps_min": min(per_repeat),
        "fps_max": max(per_repeat),
        "spread_pct": spread,
    }


# --------------------------------------------------------------------------------------
# gray driver
# --------------------------------------------------------------------------------------

def load_gray(run_dir: str, context: str, width: Optional[int], height: Optional[int]):
    """Load a gray run: (raytracer, views, n_gaussians, (render_w, render_h))."""
    import json as _json

    import tyro
    from gray.config import Config
    from gray.raytracer import Raytracer
    from gray.scene import SceneInfo
    from gray.utils import search_for_max_iteration

    cfg_path = os.path.join(run_dir, "config.json")
    cfg = tyro.cli(Config, args=[], default=Config(**_json.load(open(cfg_path))))
    iteration = search_for_max_iteration(run_dir)
    ckpt = os.path.join(run_dir, f"gaussians_{iteration:05d}.safetensors")

    scene = SceneInfo.from_colmap(cfg)
    cams = scene.test_cameras if context == "test" else scene.train_cameras
    if not cams:
        raise SystemExit(f"{run_dir}: no {context} cameras")
    cam0 = cams[0]
    rw = int(width or cam0.image_width)
    rh = int(height or cam0.image_height)
    raytracer = Raytracer.from_safetensors(cfg, ckpt, rw, rh, inference_only=True)
    # * the LIVE count, from the CUDA-side gaussian block -- not the checkpoint's row
    # * count, which can differ if anything prunes on load.
    n_gauss = int(raytracer.cuda_module.get_gaussians().mean.shape[0])
    return raytracer, cams, n_gauss, (rw, rh), iteration, ckpt


#: `PipelineWrapper::~PipelineWrapper` aborts when a second `Raytracer` exists in the same
#: process (`IMPLEMENTATION.md`, GOTCHAS; it is why `pyproject.toml` sets
#: `addopts = "--forked"`). So a process gets exactly ONE. The FoV sweep therefore loads
#: once and varies only the cameras -- which is also what its protocol demands: same BVH,
#: same gaussians, same acceleration structure at every field angle.
_LOADED: Optional["GrayBench"] = None


class GrayBench:
    """One loaded gray run, measurable many times.  Construct at most one per process."""

    def __init__(self, *, run_dir: str, smi_index: int, context: str,
                 width: Optional[int], height: Optional[int], allow_shared: bool):
        global _LOADED
        if _LOADED is not None:
            raise SystemExit(
                "a second Raytracer in one process aborts at teardown "
                "(PipelineWrapper::~PipelineWrapper). Use one GrayBench per process; the "
                "FoV sweep varies cameras on a single load by design.")
        self.run_dir = run_dir
        self.smi_index = smi_index
        self.context = context
        self.allow_shared = allow_shared
        self.gpu_start = provenance.gpu_info(smi_index)
        self.foreign_start = assert_exclusive(smi_index, allow_shared, "before load")
        (self.raytracer, self.cams, self.n_gaussians, (self.rw, self.rh),
         self.iteration, self.ckpt) = load_gray(run_dir, context, width, height)
        _LOADED = self

    def measure(self, *, method: str, repeats: int, warmup_passes: int,
                label: Optional[str] = None,
                camera_patch: Optional[Callable[[Sequence[Any]], Sequence[Any]]] = None,
                extra: Optional[Dict[str, Any]] = None,
                post_hook: Optional[Callable[[Any, Sequence[Any]], Dict[str, Any]]] = None,
                ) -> Dict[str, Any]:
        import torch

        views = list(camera_patch(self.cams)) if camera_patch else list(self.cams)

        def render_one(cam):
            with torch.no_grad():
                self.raytracer(cam, skip_copy=True)

        # re-probe right before the clock: a foreign job may have started during the load
        # or during the previous sweep point
        assert_exclusive(self.smi_index, self.allow_shared, "before timing")
        timing = timed_passes(render_one, views, repeats, warmup_passes)
        foreign_end = assert_exclusive(self.smi_index, self.allow_shared, "after timing")
        gpu_end = provenance.gpu_info(self.smi_index)

        ex = {
            "iteration": self.iteration,
            "checkpoint": os.path.abspath(self.ckpt),
            "git": provenance.git_info(str(_ROOT)),
            "camera_model": getattr(views[0], "model", "pinhole"),
            "label": label,
        }
        if extra:
            ex.update(extra)
        # anything expensive that is NOT a timing goes here: after the clock, never inside
        if post_hook is not None:
            ex.update(post_hook(self.raytracer, views))

        return provenance.build(
            tier="measured",
            harness="scripts/perf/bench_fps.py",
            method=method,
            run_path=self.run_dir,
            gpu_index=self.smi_index,
            n_gaussians=self.n_gaussians,
            width=self.rw, height=self.rh,
            n_views=len(views), context=self.context,
            fps=timing["fps_median"],
            timing=timing,
            gpu_start=self.gpu_start, gpu_end=gpu_end,
            foreign_start=self.foreign_start, foreign_end=foreign_end,
            extra=ex,
        )


def bench_one(*, run_dir: str, smi_index: int, repeats: int, warmup_passes: int,
              context: str, width: Optional[int], height: Optional[int],
              allow_shared: bool, method: str, label: Optional[str] = None,
              camera_patch: Optional[Callable[[Sequence[Any]], Sequence[Any]]] = None,
              extra: Optional[Dict[str, Any]] = None,
              post_hook: Optional[Callable[[Any, Sequence[Any]], Dict[str, Any]]] = None,
              ) -> Dict[str, Any]:
    """One fully-provenanced measurement, loading the run.  One call per process."""
    bench = GrayBench(run_dir=run_dir, smi_index=smi_index, context=context,
                      width=width, height=height, allow_shared=allow_shared)
    return bench.measure(method=method, repeats=repeats, warmup_passes=warmup_passes,
                         label=label, camera_patch=camera_patch, extra=extra,
                         post_hook=post_hook)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-m", "--model-path", required=True, help="gray run directory")
    ap.add_argument("--out", required=True, help="where to write the sidecar JSON")
    ap.add_argument("--gpu", type=int, default=None,
                    help="physical (nvidia-smi) index; inferred from CUDA_VISIBLE_DEVICES")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--warmup-passes", type=int, default=1)
    ap.add_argument("--context", choices=["test", "train"], default="test")
    ap.add_argument("--width", type=int, default=None)
    ap.add_argument("--height", type=int, default=None)
    ap.add_argument("--method", default="gray")
    ap.add_argument("--label", default=None)
    ap.add_argument("--allow-shared", action="store_true",
                    help="do not abort on a foreign CUDA context (the row will be REJECTED)")
    args = ap.parse_args(argv)

    smi_index = resolve_smi_index(args.gpu)
    doc = bench_one(
        run_dir=args.model_path, smi_index=smi_index, repeats=args.repeats,
        warmup_passes=args.warmup_passes, context=args.context,
        width=args.width, height=args.height, allow_shared=args.allow_shared,
        method=args.method, label=args.label,
    )
    provenance.write(doc, args.out)

    problems = provenance.validate(doc)
    t = doc["timing"]
    print(f"{doc['method']}  {doc['run_path']}")
    print(f"  card      : {doc['gpu']['name']} ({doc['gpu']['uuid']}) "
          f"cc {doc['gpu']['compute_capability']} driver {doc['gpu']['driver_version']}")
    print(f"  render    : {doc['resolution']['width']}x{doc['resolution']['height']}, "
          f"{doc['views']['n']} {doc['views']['context']} views, {doc['n_gaussians']} gaussians")
    print(f"  passes    : {t['per_repeat_fps']}  spread {t['spread_pct']:.2f} %")
    print(f"  FPS       : {t['fps_median']:.2f}  (median of {t['repeats']})")
    print(f"  sidecar   : {args.out}")
    if problems:
        print("  NOT CITABLE:")
        for p in problems:
            print(f"    - {p}")
        return 1
    print("  CITABLE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
