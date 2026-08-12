"""FPS(FoV): the ray-tracing / rasterisation cost ratio as a function of field of view.

    python scripts/perf/fov_sweep.py --plan -m <off run>          # no GPU, prints the plan
    python scripts/perf/fov_sweep.py -m <off run> --gpu 1 \
        --out tmp/perf/fov_sweep_gray                              # GPU, MUST be queued

WHY THIS IS THE EXPERIMENT
--------------------------
Every published speed comparison in this line of work -- GRay's own 248 / 253 / 68 FPS on
an RTX 4090, 3DGUT, 3DGEER, Radiant Foam, GRTX -- is measured on PINHOLE data (MipNeRF360,
Tanks & Temples).  A rasteriser's cost grows with distortion (more tiles touched per
primitive, sigma-point re-estimation, degenerate footprints at the rim); a ray tracer's
does not, because it never projects a primitive at all.  **Nobody has published the
RT/raster ratio as a function of field of view.**  That is the natural sequel to GRay and
it is unclaimed.

WHAT IS HELD FIXED (by construction, not by hope)
-------------------------------------------------
1. **The gaussians.**  ONE checkpoint, transferred to the other engine with
   `convert/to_3dgs.py` or `convert/to_3dgrt.py`.  Verified on 2026-08-12: the round-trip
   through `convert/safetensors_ply_conversion.py` is BIT-EXACT on all seven gaussian
   tensors (`mean`, `opacity`, `rotation`, `scale`, `sh_coeffs_dc`, `sh_coeffs_rest`,
   `current_sh_degree`) for a 156 206-gaussian FullCircle run -- see `test_perf.py`.
   ** But read the two caveats in README.md, "The transfer is lossless -- of WHAT". **
2. **The output resolution.**  Every point renders the same W x H.
3. **The poses.**  The same camera centres and rotations at every field angle.
4. **The scene.**  One scene per sweep; never averaged across scenes.
5. **The card.**  One physical GPU, exclusive, all points back to back in one process.

WHAT CANNOT BE HELD FIXED (report these as columns, do not hide them)
--------------------------------------------------------------------
* **Content in frustum grows with the field.**  Both engines see the *same* content at
  each point, so the RT/raster ratio is controlled; the per-engine absolute curve mixes
  "wider field" with "more primitives".  `visible_fraction` is the column that makes it
  explicit, and the per-engine curve should also be reported normalised to its own 60 deg
  point.
* **Angular sampling density falls ~11x from 60 to 200 deg** at fixed pixel count.
* **Image agreement between engines.**  gray traces one exact ray per pixel; a rasteriser
  approximates the projection, and that approximation is exactly what degrades with
  field angle.  A speed ratio without a fidelity column is meaningless -- the rasteriser
  can always be fast by being wrong.  Use `--save-renders` and score the two engines
  against each other per sweep point.
* **The RT-core generation.**  See README.md, "The most attackable variable".

Timing is `bench_fps.bench_one`, the same core as the plain measurement, so the sweep and
the single-run number cannot drift apart, and every point writes its own provenance
sidecar.
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_HERE))

import fovmath  # noqa: E402
import provenance  # noqa: E402


# --------------------------------------------------------------------------------------
# camera synthesis
# --------------------------------------------------------------------------------------

def patch_cameras(cams, point: Dict[str, Any], width: int, height: int, n_views: Optional[int]):
    """Return copies of `cams` re-labelled with the sweep point's synthetic intrinsics.

    `image_width/height` are set to the RENDER size on purpose: `upload_camera_intrinsics`
    rescales the intrinsics by `render_width / cam_info.image_width`, so making the two
    equal is what keeps the focal we computed the focal that is actually traced.

    `uid` is offset per sweep point so the base-bearing and ray caches, which key on
    (uid, model, fov_y, image size, intrinsics), cannot serve one field angle's bearings
    to another.  The key already covers the intrinsics since the 2026-08-07 fix, but a
    distinct uid makes the invariant visible instead of implicit.
    """
    import numpy as np

    selected = list(cams)[: n_views] if n_views else list(cams)
    uid_base = 100000 + int(round(point["fov_deg"] * 10)) + (500000 if point["arm"] == "pinhole_control" else 0)
    out = []
    for i, cam in enumerate(selected):
        c = copy.deepcopy(cam)
        c.uid = uid_base + i
        c.image_width = width
        c.image_height = height
        c.is_test = True
        if point["camera_model"] == "pinhole":
            c.model = "pinhole"
            c.intrinsics = None
            c.fov_y = float(point["fov_y"])
            c.fov_x = float(point["fov_x"])
        else:
            c.model = point["camera_model"]
            c.intrinsics = np.asarray(point["intrinsics"], dtype=np.float64)
            # fov_* are logging-only for a fisheye, but they go into the cache key, so
            # keep them consistent with the synthetic lens.
            c.fov_y = float(math.radians(point["fov_deg"]))
            c.fov_x = float(math.radians(point["fov_deg"]))
        out.append(c)
    return out


def visible_fraction(raytracer, cams, fov_deg: float) -> float:
    """Fraction of gaussian CENTRES inside the cone of half-angle fov/2, averaged over views.

    A cheap, exact content column: it is pure geometry, it runs outside the timed region,
    and it is what separates "the renderer got slower" from "there was more to render".
    Centres only -- a gaussian whose centre is outside the cone can still touch the rim --
    so read it as a trend, not as a primitive count.

    Convention: `CameraInfo.R` is camera-to-world in COLMAP/OpenCV axes (+z forward), so
    `(p - origin) @ R` = `R^T (p - origin)` is the OpenCV camera frame and `local[:, 2]` is
    forward depth. Deliberately NOT the Blender-flipped matrix
    `rotation_c2w_blender_cuda()` that the raygen receives (`gray/camera.py:156`, which
    negates the whole matrix and re-negates column 0) -- that one would put the scene
    behind the camera here.
    """
    import torch

    means = raytracer.cuda_module.get_gaussians().mean.detach()
    theta_max = math.radians(fov_deg) / 2.0
    fracs = []
    for cam in cams:
        origin = torch.as_tensor(cam.origin, dtype=means.dtype, device=means.device)
        R = torch.as_tensor(cam.R, dtype=means.dtype, device=means.device)
        local = (means - origin) @ R
        z = local[:, 2]
        r = torch.linalg.norm(local, dim=1).clamp_min(1e-12)
        cos = (z / r).clamp(-1.0, 1.0)
        fracs.append(float((cos > math.cos(theta_max)).float().mean()))
    return sum(fracs) / len(fracs)


# --------------------------------------------------------------------------------------
# the sweep
# --------------------------------------------------------------------------------------

def print_plan(width: int, height: int, fovs: List[float], run_dir: str,
               n_views: Optional[int]) -> None:
    plan = fovmath.sweep_plan(width, height, fovs)
    print(f"# FoV sweep plan -- {run_dir}")
    print(f"#   render {width}x{height}, {n_views or 'all'} views, "
          f"{len(plan)} points ({sum(1 for p in plan if p['arm'] == 'fisheye_equidistant')} "
          f"fisheye + {sum(1 for p in plan if p['arm'] == 'pinhole_control')} rectilinear control)\n")
    print(f"{'arm':22s} {'FoV':>7s} {'theta_max':>10s} {'f (px)':>10s} {'sr':>8s} {'px/sr':>12s}")
    for p in plan:
        f = p["intrinsics"][0] if "intrinsics" in p else p["focal"]
        print(f"{p['arm']:22s} {p['fov_deg']:6.1f}d {p['theta_max_deg']:9.1f}d "
              f"{f:10.2f} {p['solid_angle_sr']:8.3f} {p['pixels_per_sr']:12.1f}")
    print(f"\ngray's fisheye raygen caps theta at 90 deg (cuda/core/opencv_fisheye.cuh:31), "
          f"so the sweep stops at {fovmath.GRAY_MAX_FOV_DEG} deg.")
    print("Queue it, never run it bare: see scripts/perf/queue_fov_sweep.sh")


def run_sweep(args) -> int:
    import bench_fps

    smi_index = bench_fps.resolve_smi_index(args.gpu)
    plan = fovmath.sweep_plan(args.width, args.height, args.fovs)

    cfg_path = os.path.join(args.model_path, "config.json")
    rung = json.loads(Path(cfg_path).read_text()).get("camera_opt", "off")
    if rung != "off" and not args.allow_camera_opt:
        raise SystemExit(
            f"{args.model_path} was trained with --camera_opt {rung}. The learned residual "
            "is a correction for ONE real lens; under synthetic intrinsics it is noise, and "
            "it also puts the whole ray synthesis on the Python path, which is the FPS tax "
            "this sweep is not about. Use an `off` run, or pass --allow-camera-opt.")

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    results: List[Dict[str, Any]] = []

    # ONE load for the whole sweep. Two reasons, and both are load-bearing:
    #   * a second `Raytracer` in one process aborts at teardown
    #     (`PipelineWrapper::~PipelineWrapper`, IMPLEMENTATION.md GOTCHAS);
    #   * the protocol requires the SAME acceleration structure at every field angle. A
    #     per-point reload would rebuild the BVH and quietly add a second variable.
    bench = bench_fps.GrayBench(
        run_dir=args.model_path, smi_index=smi_index, context="test",
        width=args.width, height=args.height, allow_shared=args.allow_shared)
    print(f"loaded {args.model_path}: {bench.n_gaussians} gaussians, "
          f"{len(bench.cams)} test poses, rendering at {bench.rw}x{bench.rh}")

    for i, point in enumerate(plan):
        tag = f"{point['arm']}_{int(point['fov_deg']):03d}deg"
        print(f"\n=== [{i + 1}/{len(plan)}] {tag} ===")
        doc = bench.measure(
            method=f"gray-rt/{point['arm']}",
            repeats=args.repeats, warmup_passes=args.warmup_passes, label=tag,
            camera_patch=lambda cams, p=point: patch_cameras(
                cams, p, args.width, args.height, args.n_views),
            extra={"sweep": {k: v for k, v in point.items()},
                   "sweep_id": tag, "engine": "gray (OptiX ray tracing)"},
            # the content column: computed AFTER the clock, never inside it
            post_hook=lambda rt, views, p=point: {
                "visible_fraction": visible_fraction(rt, views, p["fov_deg"])},
        )
        provenance.write(doc, outdir / f"{tag}.perf.json")
        results.append(doc)
        t = doc["timing"]
        print(f"  {doc['value']['fps']:8.2f} FPS  spread {t['spread_pct']:.2f} %  "
              f"({doc['n_gaussians']} gaussians, "
              f"visible {doc['extra']['visible_fraction'] * 100:.1f} %)")

    summary = {
        "schema": "gray.perf.fov_sweep/1",
        "run_path": os.path.abspath(args.model_path),
        "resolution": {"width": args.width, "height": args.height},
        "n_views": args.n_views,
        "gray_max_fov_deg": fovmath.GRAY_MAX_FOV_DEG,
        "points": [
            {"sweep_id": d["extra"]["sweep_id"],
             "arm": d["extra"]["sweep"]["arm"],
             "fov_deg": d["extra"]["sweep"]["fov_deg"],
             "fps": d["value"]["fps"],
             "spread_pct": d["timing"]["spread_pct"],
             "n_gaussians": d["n_gaussians"],
             "visible_fraction": d["extra"].get("visible_fraction"),
             "pixels_per_sr": d["extra"]["sweep"]["pixels_per_sr"]}
            for d in results],
    }
    (outdir / "sweep_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {outdir}/sweep_summary.json ({len(results)} points)")
    print("Now run the raster arm (see README.md) and join on (scene, fov_deg, resolution, "
          "n_gaussians). NEVER join across cards.")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-m", "--model-path", required=True,
                   help="a gray run trained with --camera_opt off")
    ap.add_argument("--out", default="tmp/perf/fov_sweep_gray")
    ap.add_argument("--gpu", type=int, default=None)
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--height", type=int, default=1024)
    ap.add_argument("--n-views", type=int, default=30,
                    help="how many of the run's test poses to reuse (0 = all)")
    ap.add_argument("--fovs", type=float, nargs="*", default=fovmath.SWEEP_FOV_DEG)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--warmup-passes", type=int, default=1)
    ap.add_argument("--allow-shared", action="store_true")
    ap.add_argument("--allow-camera-opt", action="store_true")
    ap.add_argument("--plan", action="store_true",
                    help="print the camera plan and exit (no GPU, no torch import)")
    a = ap.parse_args(argv)
    a.n_views = a.n_views or None

    if a.plan:
        print_plan(a.width, a.height, a.fovs, a.model_path, a.n_views)
        return 0
    return run_sweep(a)


if __name__ == "__main__":
    raise SystemExit(main())
