"""W3.2 -- mask-radius robustness sweep (r = 0.85 / 0.95 / 1.00) over EVERY method of the
four shared metric stores.

Why: the camera-model gain is a PERIPHERAL effect, and the canonical evaluation mask cuts
the periphery off (r = 0.95 keeps 44 % of the frame on myscenes, 47.9 % on immervision).
"The ranking is stable under three mask radii" is a much stronger claim than picking one.
r = 0.95 stays canonical; this is the robustness table around it.

NO TRAINING, NO RE-RENDERING. Every number is recomputed from PNGs already on disk, with
one shared pass per radius: same mask, same masked PSNR/SSIM, same zero-then-bbox-crop
LPIPS for every method. Never a repo's self-reported metric.

    python scripts/analysis/mask_radius_sweep.py --track myscenes_rttpf
    python scripts/analysis/mask_radius_sweep.py --track all          # spawns subprocesses
        -> tmp/w3_radial/sweep_<track>.json     (one file per track, side stems only)

ONE TRACK PER PROCESS, AND THAT IS NOT OPTIONAL
---------------------------------------------------------------------------------------
`masked_eval_fujinon.py` and `masked_eval_immervision.py` both reconfigure the SHARED
`masked_eval` module (`me.SCENES`, `me.METHODS`, `me.canonical_gt`, `me.build_mask`) --
fujinon at import time, immervision inside `install()`. Two tracks in one interpreter
would silently score one scene with another's GT resolver. `--track all` therefore
re-execs this file once per track instead of looping in-process.

*** THE STORE-OVERWRITE TRAP (already paid once, the 3dgrut-nofix columns vanished) ***
`masked_eval_fullcircle.py` without `--out` rewrites the canonical table with only the
methods it evaluated. This script NEVER calls any evaluator's `main()`; it imports their
`eval_*` functions and writes its own JSON under `tmp/w3_radial/`. Nothing it does can
touch `dataset/*/masked_metrics.json`.

*** r = 1.00 IS NOT A VALID CROSS-METHOD COMPARISON ON A FULL-FRAME CAPTURE ***
DFGS and SPaGS render EXACT BLACK outside the disk they trained on. On a circular-frame
capture (myscenes, FullCircle) the ground truth is black there too and nothing happens; on
a full-frame capture (workshop_fujinon) the ground truth carries real content, so widening
the mask hands those methods a ~10 dB penalty that measures their output convention, not
their reconstruction. The `annotation` field of every track flags this. Report the row,
do not read a ranking off it.

*** THE THREE RADII ARE NOT ALWAYS THREE DIFFERENT MASKS ***
`geometric_valid_mask_*` also drops any pixel where the 100-iteration inversion of the
distortion polynomial fails (`err_sq < 1e-5`). Where that criterion binds first, the radius
knob does nothing: on `workshop_immervision` (rttpf) the three radii differ by SEVEN pixels
out of 1.56 M. `valid_fraction` is reported per radius per track for exactly this reason --
check it before reading a robustness row, because a flat row can mean "robust" or "same
mask three times".

TWO DELIBERATE DEVIATIONS FROM THE CANONICAL EVALUATORS
---------------------------------------------------------------------------------------
1. The LPIPS bbox cache is repaired. Every canonical evaluator caches the lens-disk
   bounding box on `mask.shape` (`masked_eval.py:118`, `masked_eval_ocv.py:120`), so all
   four 1368x912 myscenes scenes reuse whichever scene ran first. That is a real bug worth
   ~4e-4 LPIPS (see radial_eval.py's docstring), and here it would be fatal: the bbox is
   the thing the radius sweep MOVES. This script keys the cache on the mask object, so
   each (scene, radius) gets its own bbox. Consequence: the r=0.95 LPIPS column reproduces
   the stores' PSNR and SSIM exactly but differs from their LPIPS by ~1e-4 to 4e-4. The
   `store_delta` block reports that difference per entry rather than hiding it.
2. Masks are rebuilt geometrically at every radius. FullCircle's evaluators read the
   shipped `valid_mask_cam<uid>.png`, which exists only at r=0.95; a sweep must regenerate
   from the track's own COLMAP intrinsics. `mask_rebuild_check` verifies that the rebuild
   at r=0.95 reproduces the shipped PNG (reported as the number of differing pixels, which
   must be 0).
"""

import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np
import torch
from PIL import Image

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for path in ("/workspace/gray", "/workspace/gray/scripts",
             "/workspace/dataset/myscenes_ocv_code", "/workspace/dataset/fullcircle_code"):
    if path not in sys.path:
        sys.path.append(path)

RADII = [0.85, 0.95, 1.00]
OUTDIR = os.path.join(REPO, "tmp", "w3_radial")

TRACKS = [
    "myscenes_rttpf", "fujinon", "immervision_rttpf", "immervision_ocv",
    "myscenes_ocv", "fullcircle_ocv", "fullcircle_rttpf",
]

# * Which store each track's r=0.95 column must reproduce, and the frame geometry that
# * decides whether r=1.00 is interpretable at all.
STORE = {
    "myscenes_rttpf": "/workspace/dataset/fisheye_baselines/masked_metrics.json",
    "fujinon": "/workspace/dataset/fisheye_baselines/masked_metrics.json",
    "immervision_rttpf": "/workspace/dataset/fisheye_baselines/masked_metrics.json",
    "immervision_ocv": "/workspace/dataset/fisheye_baselines_ocv/masked_metrics.json",
    "myscenes_ocv": "/workspace/dataset/fisheye_baselines_ocv/masked_metrics.json",
    "fullcircle_ocv": "/workspace/dataset/fullcircle_baselines/fullcircle_masked_metrics.json",
    "fullcircle_rttpf": "/workspace/dataset/fullcircle_tracks/rttpf_masked_metrics.json",
}
FRAME = {  # circular = black surround in the GT; full = real content out to the frame edge
    "myscenes_rttpf": "circular", "myscenes_ocv": "circular",
    "fullcircle_ocv": "circular", "fullcircle_rttpf": "circular",
    "fujinon": "full", "immervision_rttpf": "full", "immervision_ocv": "full",
}
ANNOTATION_FULL = (
    "r=1.00 IS NOT A VALID CROSS-METHOD ROW HERE. This is a full-frame capture: the ground "
    "truth carries real content outside each method's training disk, and DFGS/SPaGS render "
    "exact black there, so widening the mask scores their output convention. Report, do not "
    "rank."
)
ANNOTATION_CIRC = (
    "r=1.00 is interpretable here (circular frame: the ground truth is black outside the "
    "lens disk too), but it is still outside the canonical protocol and adds pixels no "
    "method was scored on before."
)


# --------------------------------------------------------------------------------------
# shared helpers
# --------------------------------------------------------------------------------------


def make_frame_lpips(module, lpips_getter, extra_key=False):
    """A `frame_lpips` whose bbox cache is keyed on the MASK, not on its shape.

    The canonical ones key on `mask.shape`, which collides across scenes of equal size and
    -- fatally for this script -- across radii of the same scene.
    """
    cache = {}

    def frame_lpips(render, gt, mask, *_ignored):
        key = id(mask)
        if key not in cache:
            ys, xs = torch.where(mask)
            cache[key] = (ys.min().item(), ys.max().item() + 1,
                          xs.min().item(), xs.max().item() + 1)
        y0, y1, x0, x1 = cache[key]
        m = mask.unsqueeze(0).float()
        r = (render * m)[:, y0:y1, x0:x1].unsqueeze(0)
        g = (gt * m)[:, y0:y1, x0:x1].unsqueeze(0)
        with torch.no_grad():
            return float(lpips_getter()(r, g).item())

    return frame_lpips


def colmap_masks(sparse_dir, height, width, radius, device, names=None):
    """{camera_id: bool [H,W]} rebuilt from a COLMAP model at an arbitrary radius."""
    import pycolmap

    from gray.fisheye_mask import (
        geometric_valid_mask_opencv_fisheye,
        geometric_valid_mask_rad_tan_thin_prism_fisheye,
        geometric_valid_mask_thin_prism_fisheye,
    )

    builders = {
        "OPENCV_FISHEYE": geometric_valid_mask_opencv_fisheye,
        "THIN_PRISM_FISHEYE": geometric_valid_mask_thin_prism_fisheye,
        "RAD_TAN_THIN_PRISM_FISHEYE": geometric_valid_mask_rad_tan_thin_prism_fisheye,
    }
    rec = pycolmap.Reconstruction(sparse_dir)
    out, models = {}, {}
    for cam_id, cam in rec.cameras.items():
        name = cam.model.name
        if name not in builders:
            raise SystemExit(f"{sparse_dir} camera {cam_id} is {name}; not a fisheye model")
        p = list(cam.params)
        sx, sy = width / cam.width, height / cam.height
        intr = torch.tensor([p[0] * sx, p[1] * sy, p[2] * sx, p[3] * sy] + p[4:],
                            dtype=torch.float32, device=device)
        out[cam_id] = builders[name](intr, height, width, device, radius)
        models[cam_id] = name
    return out, models


def valid_fraction(masks):
    values = [float(m.float().mean()) for m in masks]
    return float(np.mean(values)) if values else float("nan")


# --------------------------------------------------------------------------------------
# tracks driven through scripts/masked_eval.py (myscenes rttpf, fujinon, immervision)
# --------------------------------------------------------------------------------------


def run_masked_eval_track(track, device):
    import masked_eval as me

    me.DEVICE = device
    scenes, note = None, ""
    if track == "myscenes_rttpf":
        scenes = ["atrium", "tunnel", "library", "reception"]
    elif track == "fujinon":
        import masked_eval_fujinon  # noqa: F401  (patches `me` at import -- see docstring)

        scenes = ["workshop_fujinon"]
        note = "masked_eval_fujinon patched masked_eval at import"
    elif track.startswith("immervision"):
        import masked_eval_immervision as mi

        spec = mi.install(track.split("_", 1)[1])
        scenes = [spec["key"]]
        note = f"masked_eval_immervision.install({track.split('_', 1)[1]})"
    me.frame_lpips = make_frame_lpips(me, me.lpips_fn)

    results = {}
    for radius in RADII:
        me.RADIUS_SCALE = radius
        mask_cache = {}
        for scene in scenes:
            for method in me.METHODS:
                t0 = time.time()
                res = me.eval_method(method, scene, mask_cache)
                if isinstance(res, tuple) or res is None:
                    reason = res[1] if isinstance(res, tuple) else "missing"
                    print(f"  SKIP {method:18s} {scene:22s} r={radius}: {reason}", flush=True)
                    continue
                results.setdefault(scene, {}).setdefault(method, {})[f"{radius:.2f}"] = {
                    "PSNR": res["psnr"], "SSIM": res["ssim"], "LPIPS": res["lpips"],
                    "n": res["n"],
                }
                print(f"  {method:18s} {scene:22s} r={radius:.2f} PSNR {res['psnr']:6.3f} "
                      f"SSIM {res['ssim']:.4f} LPIPS {res['lpips']:.4f} ({time.time()-t0:.0f}s)",
                      flush=True)
        for (scene, height, width), mask in mask_cache.items():
            results.setdefault(scene, {}).setdefault("_geometry", {})[f"{radius:.2f}"] = {
                "valid_fraction": float(mask.float().mean()),
                "resolution": [height, width],
            }
    return results, note


# --------------------------------------------------------------------------------------
# myscenes -ocv track
# --------------------------------------------------------------------------------------


def run_myscenes_ocv(device):
    import masked_eval_ocv as mo

    mo.DEVICE = device
    mo.frame_lpips = make_frame_lpips(mo, mo.lpips_fn)
    store = json.load(open(STORE["myscenes_ocv"]))
    tags = [t for t in store if t != "workshop_immervision_ocv"]

    results = {}
    for radius in RADII:
        mo.RADIUS_SCALE = radius
        mask_cache = {}
        for tag in tags:
            for method in mo.METHODS:
                if method not in store[tag]:
                    continue
                t0 = time.time()
                res, err = mo.eval_method(method, tag, mask_cache)
                if res is None:
                    print(f"  SKIP {method:18s} {tag:22s} r={radius}: {err}", flush=True)
                    continue
                results.setdefault(tag, {}).setdefault(method, {})[f"{radius:.2f}"] = {
                    "PSNR": res["psnr"], "SSIM": res["ssim"], "LPIPS": res["lpips"],
                    "n": res["n"],
                }
                print(f"  {method:18s} {tag:22s} r={radius:.2f} PSNR {res['psnr']:6.3f} "
                      f"SSIM {res['ssim']:.4f} LPIPS {res['lpips']:.4f} ({time.time()-t0:.0f}s)",
                      flush=True)
        for (tag, height, width), mask in mask_cache.items():
            results.setdefault(tag, {}).setdefault("_geometry", {})[f"{radius:.2f}"] = {
                "valid_fraction": float(mask.float().mean()),
                "resolution": [height, width],
            }
    return results, "masked_eval_ocv"


# --------------------------------------------------------------------------------------
# FullCircle (both stores) -- masks rebuilt from COLMAP, shipped PNGs used only to check
# --------------------------------------------------------------------------------------


def pin_verify_gt_to_canonical_radius(module):
    """Make the run's GT sanity check use the r=0.95 disk whatever radius is being swept.

    `verify_gt` exists to catch a mis-ordered index -> COLMAP-name mapping, and it does that
    by diffing a run's own saved GT against the canonical image. Some runs (SPaGS) save
    their GT ALREADY MULTIPLIED BY THE r=0.95 MASK. Handing it the r=1.00 mask therefore
    makes it diff real content against zeros in the 0.95-1.00 annulus and reject a
    perfectly well-paired run -- the check would silently delete SPaGS from the widest
    radius, which is precisely the row the sweep exists to look at. Pinning the check to
    r=0.95 keeps it doing its one job.
    """
    original = module.verify_gt
    holder = {}

    def verify_gt(run_dir, views, scene, masks=None, max_checks=6):
        return original(run_dir, views, scene,
                        holder.get("masks") if masks is not None else None, max_checks)

    module.verify_gt = verify_gt
    return holder  # * caller writes holder["masks"] = <the r=0.95 masks of the scene>


def _shipped_mask_check(path, rebuilt, device):
    if not os.path.exists(path):
        return None
    arr = np.asarray(Image.open(path).convert("L")) > 127
    shipped = torch.from_numpy(arr).to(device)
    if shipped.shape != rebuilt.shape:
        return -1
    return int((shipped != rebuilt).sum())


def run_fullcircle_ocv(device):
    import masked_eval_fullcircle as mf

    mf.DEVICE = device
    mf.frame_lpips = make_frame_lpips(mf, mf.lpips_fn)
    store = json.load(open(STORE["fullcircle_ocv"]))
    methods = sorted({k.rsplit("_", 1)[0] for scene in store for k in store[scene]})
    verify_masks = pin_verify_gt_to_canonical_radius(mf)

    results, checks = {}, {}
    for scene in mf.SCENES:
        names, cam_of = mf.scene_test_views(scene)
        probe = os.path.join(mf.BASE, scene, "images_4", names[0])
        H, W = np.asarray(Image.open(probe)).shape[:2]
        per_radius = {}
        for radius in RADII:
            built, models = colmap_masks(os.path.join(mf.BASE, scene, "sparse", "0"),
                                         H, W, radius, device)
            per_radius[radius] = {n: built[cam_of[n]] for n in names}
            if abs(radius - 0.95) < 1e-9:
                for cam_id, mask in built.items():
                    checks[f"{scene}/cam{cam_id}"] = _shipped_mask_check(
                        os.path.join(mf.BASE, scene, f"valid_mask_cam{cam_id}.png"), mask, device)
            results.setdefault(scene, {}).setdefault("_geometry", {})[f"{radius:.2f}"] = {
                "valid_fraction": valid_fraction(list(built.values())),
                "resolution": [H, W], "camera_models": sorted(set(models.values())),
            }
        verify_masks["masks"] = per_radius[0.95]
        for radius in RADII:
            masks = per_radius[radius]
            for method in methods:
                t0 = time.time()
                res = mf.eval_run(method, scene, "masked", "", masks)
                if res is None:
                    print(f"  SKIP {method:14s} {scene:8s} r={radius}", flush=True)
                    continue
                results[scene].setdefault(method, {})[f"{radius:.2f}"] = {
                    "PSNR": res["PSNR"], "SSIM": res["SSIM"], "LPIPS": res["LPIPS"],
                    "n": res["n"],
                }
                print(f"  {method:14s} {scene:8s} r={radius:.2f} PSNR {res['PSNR']:6.3f} "
                      f"SSIM {res['SSIM']:.4f} LPIPS {res['LPIPS']:.4f} ({time.time()-t0:.0f}s)",
                      flush=True)
    return results, {"mask_rebuild_check_differing_pixels_at_0.95": checks}


def run_fullcircle_rttpf(device):
    import masked_eval_fullcircle as mf
    import masked_eval_rttpf as mr

    mf.DEVICE = device
    patched = make_frame_lpips(mf, mf.lpips_fn)
    mf.frame_lpips = patched
    mr.frame_lpips = patched  # * imported by name at module load -- must be patched too
    store = json.load(open(STORE["fullcircle_rttpf"]))
    verify_masks = pin_verify_gt_to_canonical_radius(mr)

    results, checks = {}, {}
    for variant in store:
        for scene in mr.SCENES:
            if scene not in store[variant]:
                continue
            names, cam_of = mr.scene_test_views(variant, scene)
            sdir = mr.scene_dir(variant, scene)
            probe = os.path.join(sdir, "images_4", names[0])
            H, W = np.asarray(Image.open(probe)).shape[:2]
            node = results.setdefault(variant, {}).setdefault(scene, {})
            per_radius = {}
            for radius in RADII:
                built, models = colmap_masks(os.path.join(sdir, "sparse", "0"),
                                             H, W, radius, device)
                per_radius[radius] = {n: built[cam_of[n]] for n in names}
                if abs(radius - 0.95) < 1e-9:
                    for cam_id, mask in built.items():
                        checks[f"{variant}/{scene}/cam{cam_id}"] = _shipped_mask_check(
                            os.path.join(sdir, f"valid_mask_cam{cam_id}.png"), mask, device)
                node.setdefault("_geometry", {})[f"{radius:.2f}"] = {
                    "valid_fraction": valid_fraction(list(built.values())),
                    "resolution": [H, W], "camera_models": sorted(set(models.values())),
                }
            verify_masks["masks"] = per_radius[0.95]
            for radius in RADII:
                masks = per_radius[radius]
                for method in store[variant][scene]:
                    t0 = time.time()
                    res = mr.eval_run(method, variant, scene, "", masks, names)
                    if res is None:
                        print(f"  SKIP {method:14s} {variant}/{scene} r={radius}", flush=True)
                        continue
                    node.setdefault(method, {})[f"{radius:.2f}"] = {
                        "PSNR": res["PSNR"], "SSIM": res["SSIM"], "LPIPS": res["LPIPS"],
                        "n": res["n"],
                    }
                    print(f"  {method:14s} {variant:12s} {scene:8s} r={radius:.2f} "
                          f"PSNR {res['PSNR']:6.3f} SSIM {res['SSIM']:.4f} "
                          f"LPIPS {res['LPIPS']:.4f} ({time.time()-t0:.0f}s)", flush=True)
    return results, {"mask_rebuild_check_differing_pixels_at_0.95": checks}


# --------------------------------------------------------------------------------------


def _flatten(results, track):
    """{'<scene>/<method>': {radius: metrics}} regardless of the track's nesting depth."""
    flat = {}
    if track == "fullcircle_rttpf":
        for variant, scenes in results.items():
            for scene, methods in scenes.items():
                for method, radii in methods.items():
                    flat[f"{variant}/{scene}/{method}"] = radii
    else:
        for scene, methods in results.items():
            for method, radii in methods.items():
                flat[f"{scene}/{method}"] = radii
    return flat


def store_deltas(track, results):
    """r=0.95 recomputation minus the canonical store, per entry.

    PSNR and SSIM must agree to ~1e-6 -- this is the check that the sweep reproduces the
    canonical protocol before it is trusted at the other two radii. LPIPS is EXPECTED to
    differ by ~1e-4 because the store carries the shape-keyed bbox cache bug (see the
    module docstring) and this script does not.
    """
    try:
        store = json.load(open(STORE[track]))
    except (OSError, ValueError):
        return {}

    out = {}
    for key, radii in _flatten(results, track).items():
        parts = key.split("/")
        method = parts[-1]
        if method == "_geometry" or "0.95" not in radii:
            continue
        node = store
        for part in parts[:-1]:
            node = node.get(part, {}) if isinstance(node, dict) else {}
        if not isinstance(node, dict):
            continue
        ref = node.get(method) or node.get(f"{method}_masked")
        if not isinstance(ref, dict):
            continue
        mine, got = radii["0.95"], {}
        for upper, lower in (("PSNR", "psnr"), ("SSIM", "ssim"), ("LPIPS", "lpips")):
            value = ref.get(upper, ref.get(lower))
            if value is not None:
                got[upper] = round(mine[upper] - value, 8)
        if got:
            out[key] = got
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--track", required=True, choices=TRACKS + ["all"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--outdir", default=OUTDIR)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    if args.track == "all":
        for track in TRACKS:
            print(f"\n########## {track}", flush=True)
            subprocess.run([sys.executable, os.path.abspath(__file__), "--track", track,
                            "--device", args.device, "--outdir", args.outdir], check=True)
        return

    extra = {}
    if args.track in ("myscenes_rttpf", "fujinon", "immervision_rttpf", "immervision_ocv"):
        results, note = run_masked_eval_track(args.track, args.device)
        extra["note"] = note
    elif args.track == "myscenes_ocv":
        results, note = run_myscenes_ocv(args.device)
        extra["note"] = note
    elif args.track == "fullcircle_ocv":
        results, extra = run_fullcircle_ocv(args.device)
    elif args.track == "fullcircle_rttpf":
        results, extra = run_fullcircle_rttpf(args.device)

    payload = {
        "track": args.track,
        "radii": RADII,
        "canonical_radius": 0.95,
        "frame_type": FRAME[args.track],
        "annotation_r100": ANNOTATION_FULL if FRAME[args.track] == "full" else ANNOTATION_CIRC,
        "store_compared_against": STORE[args.track],
        "store_delta_at_0.95": store_deltas(args.track, results),
        "lpips_note": ("this script repairs the shape-keyed bbox cache of the canonical "
                       "evaluators, so its LPIPS differs from the store by ~1e-4; PSNR and "
                       "SSIM must match to ~1e-6"),
        "results": results,
    }
    payload.update({k: v for k, v in extra.items() if k != "note"})
    if "note" in extra:
        payload["note"] = extra["note"]

    out = os.path.join(args.outdir, f"sweep_{args.track}.json")
    with open(out, "w") as handle:
        json.dump(payload, handle, indent=1)
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
