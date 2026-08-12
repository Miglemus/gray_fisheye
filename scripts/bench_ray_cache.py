#!/usr/bin/env python
"""A/B the pose-independent ray cache on one checkpoint: FPS, and proof the pixels agree.

Why not just run `measure_fps.py` twice
--------------------------------------
Three reasons, and the first one has already cost this project a table:

1. **`measure_fps.py` writes `fps.csv` into the model directory.** Running it twice on a
   published run leaves whichever arm ran LAST in the canonical file. The FullCircle and
   myscenes FPS columns are read from exactly that file. This script writes nothing into the
   run directory; `--out` is explicit and lands where you point it.
2. **Two invocations are two processes minutes apart**, so any change in what else holds the
   card lands entirely on one arm. Here both arms run in one process, interleaved ABBA,
   `--repeats` times, off one BVH.
3. **Speed without bit-exactness is meaningless.** The cache is only allowed to exist because
   it changes nothing; this script re-renders one view under each arm and compares the two
   images with `torch.equal` before reporting any timing. If they differ it refuses to print
   an FPS at all.

What it measures
----------------
`--camera_opt` rungs synthesise their rays in Python and copy two [H,W,3] buffers into the
framebuffer per image. `CameraModel.forward()` caches the pose-independent half of that
synthesis per (camera, resolution); `apply_pose()` -- one 3x3 GEMM, a renormalisation and the
origin offset -- necessarily stays per image, because that half depends on the pose. So this
measures the part of the FPS tax the cache can remove, not the whole tax. The rest needs the
synthesis to move into CUDA.

`GRAY_NO_RAY_CACHE=1` is the same switch as an environment variable, for a whole run.

Usage (GPU: queue it, never run it bare)
----------------------------------------
    pueue add --group gpu1 --print-task-id -- "cd $PWD && \\
      PATH=/workspace/gray/.venv/bin:\\$PATH CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 \\
      python scripts/bench_ray_cache.py -m tmp/final/workshop_noncentral --repeats 3 \\
        --out tmp/p2_ray_cache/workshop.json"
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-m", "--model-path", required=True)
    parser.add_argument("-x", "--context", default="test", choices=["test", "train"])
    parser.add_argument("--repeats", type=int, default=3, help="ABBA blocks; 3 = 6 timed passes")
    parser.add_argument("--max-views", type=int, default=0, help="0 = every view of the split")
    parser.add_argument("--out", help="write the result as JSON here (never into the run dir)")
    args, unknown = parser.parse_known_args()

    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    import torch
    import tyro

    from gray.prelude import Config, Raytracer, SceneInfo, search_for_max_iteration

    model_dir = Path(args.model_path)
    cfg = tyro.cli(
        Config,
        args=unknown,
        default=Config(**json.loads((model_dir / "config.json").read_text())),
    )
    cfg.model_path = str(model_dir)
    if cfg.camera_opt == "off":
        raise SystemExit(
            f"{model_dir} was trained with --camera_opt off: it uses the native raygen and "
            "has no Python ray synthesis to cache. Compare its fps.csv against a rung run."
        )

    iteration = search_for_max_iteration(str(model_dir))
    checkpoint = str(model_dir / f"gaussians_{iteration:05d}.safetensors")

    scene = SceneInfo.from_colmap(cfg)
    cameras = scene.test_cameras if args.context == "test" else scene.train_cameras
    if not cameras:
        # * A held-out split can be empty (`--llffhold 0`); `cameras[0]` would raise an
        # * IndexError far from the config that caused it.
        raise SystemExit(f"split '{args.context}' is empty for {model_dir}; try -x train")
    if args.max_views:
        cameras = cameras[: args.max_views]
    reference = cameras[0]
    raytracer = Raytracer.from_safetensors(
        cfg, checkpoint, reference.image_width, reference.image_height, inference_only=True
    )
    model = raytracer.camera_model
    assert model is not None, "camera model is off; checked above, re-check"

    def set_arm(enabled: bool):
        model.ray_cache_enabled = enabled
        model.invalidate_ray_cache()

    # ------------------------------------------------------------------ correctness first
    with torch.no_grad():
        set_arm(False)
        plain = raytracer(reference).clone()
        set_arm(True)
        raytracer(reference)  # * populates the cache
        cached = raytracer(reference).clone()  # * this one reads it
    identical = bool(torch.equal(plain, cached))
    max_abs = float((plain - cached).abs().max())
    if not identical:
        raise SystemExit(
            f"REFUSING TO TIME: cached and un-cached renders differ by {max_abs:.3e}. "
            "The cache is wrong; fix it before measuring how fast it is."
        )

    # ------------------------------------------------------------------------- the timing
    def timed_pass() -> float:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        start.record()
        with torch.no_grad():
            for camera in cameras:
                raytracer(camera, skip_copy=True)
        end.record()
        torch.cuda.synchronize()
        return len(cameras) / (start.elapsed_time(end) / 1000.0)

    # * Warm up both arms before any measurement: the first pass pays the cache fill, the
    # * BVH's first traversal and cuDNN-style lazy init, none of which belong in the number.
    for enabled in (False, True):
        set_arm(enabled)
        timed_pass()

    samples = {"cache": [], "nocache": []}
    for _ in range(max(args.repeats, 1)):
        # * ABBA: any monotone drift (thermal, or another job arriving) cancels to first order.
        for enabled in (True, False, False, True):
            set_arm(enabled)
            # * The cache-on arm must not pay its fill inside the timed pass either.
            if enabled:
                with torch.no_grad():
                    raytracer(reference, skip_copy=True)
            samples["cache" if enabled else "nocache"].append(timed_pass())

    def mean(values):
        return sum(values) / len(values)

    result = {
        "model_path": str(model_dir),
        "iteration": iteration,
        "rung": cfg.camera_opt,
        "context": args.context,
        "views": len(cameras),
        "render_size": [raytracer.render_width, raytracer.render_height],
        "num_gaussians": int(raytracer.cuda_module.get_gaussians().mean.shape[0]),
        "bit_exact": identical,
        "max_abs_pixel_difference": max_abs,
        "fps_cache": samples["cache"],
        "fps_nocache": samples["nocache"],
        "fps_cache_mean": mean(samples["cache"]),
        "fps_nocache_mean": mean(samples["nocache"]),
        "speedup": mean(samples["cache"]) / mean(samples["nocache"]),
    }
    print(json.dumps(result, indent=2))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(result, indent=2))
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
