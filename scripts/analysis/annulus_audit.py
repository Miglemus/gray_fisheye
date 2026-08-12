"""W3.2 companion -- WHY the r=1.00 row is not a cross-method comparison, measured.

    python scripts/analysis/annulus_audit.py            # CPU only, no GPU, ~2 min
        -> tmp/w3_radial/annulus_audit.json

The mask-radius sweep widens the scored disk from 85.5 deg (r=0.95) to 90 deg (r=1.00).
Everything that makes that annulus non-comparable between methods is a property of what
each repo WRITES there, not of the mask. Two independent effects, both measured here on a
sample of views per (track, scene, method):

  1. THE RENDER. DFGS and SPaGS emit exact black outside the disk they trained on.
     `render_zero_fraction_annulus` = fraction of annulus pixels where the render is
     exactly (0,0,0). A method at ~1.0 is being scored on its output convention.
  2. THE GROUND TRUTH. `scripts/masked_eval.py` and `masked_eval_ocv.py` pair each method
     with ITS OWN saved `gt/`, and some repos save that GT already multiplied by their
     r=0.95 mask. Those methods are then scored against ZEROS in the annulus while gray is
     scored against real content -- a bias in the OPPOSITE direction, which no choice of
     mask can repair. `gt_zero_fraction_annulus` and `gt_mean_intensity_annulus_255` show
     it directly. The FullCircle evaluators read the canonical dataset image for every
     method and are immune by construction; the audit confirms that rather than assuming.

`*_inner_255` is the control, measured inside the r=0.85 disk where every convention
agrees. `canonical_mean_intensity_annulus_255` (per scene, from the dataset images) is the
number that decides whether any of this matters: if the capture is black in the annulus,
widening the mask is harmless; if it carries real content, every method that blacks it out
is penalised for a convention.
"""

import glob
import json
import os
import subprocess
import sys

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
for path in (HERE, "/workspace/gray", "/workspace/gray/scripts",
             "/workspace/dataset/myscenes_ocv_code", "/workspace/dataset/fullcircle_code"):
    if path not in sys.path:
        sys.path.append(path)

DEVICE = "cpu"
SAMPLE = 4  # * views per entry; these are frame-level conventions, not per-view noise


def profile(paths, annulus, inner):
    """Zero fraction and mean intensity of a list of PNGs over two pixel sets."""
    zero_a, mean_a, zero_i, mean_i = [], [], [], []
    step = max(1, len(paths) // SAMPLE)
    for path in paths[::step][:SAMPLE]:
        image = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32)
        if image.shape[:2] != annulus.shape:
            return None
        zero_a.append(float((image[annulus].sum(-1) == 0).mean()))
        mean_a.append(float(image[annulus].mean()))
        zero_i.append(float((image[inner].sum(-1) == 0).mean()))
        mean_i.append(float(image[inner].mean()))
    if not zero_a:
        return None
    return {
        "zero_fraction_annulus": round(float(np.mean(zero_a)), 5),
        "mean_intensity_annulus_255": round(float(np.mean(mean_a)), 3),
        "zero_fraction_inner": round(float(np.mean(zero_i)), 5),
        "mean_intensity_inner_255": round(float(np.mean(mean_i)), 3),
        "views_sampled": len(zero_a),
    }


def entry_for(renders, gts, annulus, inner):
    render_stats = profile(renders, annulus, inner)
    if render_stats is None:
        return None
    out = {
        "render_zero_fraction_annulus": render_stats["zero_fraction_annulus"],
        "render_mean_intensity_annulus_255": render_stats["mean_intensity_annulus_255"],
        "render_zero_fraction_inner": render_stats["zero_fraction_inner"],
        "views_sampled": render_stats["views_sampled"],
        "annulus_fraction_of_frame": round(float(annulus.mean()), 5),
    }
    gt_stats = profile(gts, annulus, inner) if gts else None
    if gt_stats:
        out.update({
            "gt_zero_fraction_annulus": gt_stats["zero_fraction_annulus"],
            "gt_mean_intensity_annulus_255": gt_stats["mean_intensity_annulus_255"],
            "gt_mean_intensity_inner_255": gt_stats["mean_intensity_inner_255"],
        })
    return out


def canonical_reference(image_dir, annulus, inner):
    files = sorted(p for p in glob.glob(os.path.join(image_dir, "*.png"))
                   if not p.endswith("_mask.png"))
    stats = profile(files, annulus, inner) if files else None
    if stats is None:
        return {}
    return {
        "canonical_mean_intensity_annulus_255": stats["mean_intensity_annulus_255"],
        "canonical_zero_fraction_annulus": stats["zero_fraction_annulus"],
        "canonical_mean_intensity_inner_255": stats["mean_intensity_inner_255"],
    }


def audit_myscenes_rttpf(scenes, images_dir=None):
    import masked_eval as me

    me.DEVICE = DEVICE
    out = {}
    for scene in scenes:
        annulus = inner = None
        for method, spec in me.METHODS.items():
            run = me.spags_run(scene) if spec["run"] is None else spec["run"].format(scene=scene)
            if run is None or not os.path.isdir(run):
                continue
            renders = sorted(glob.glob(os.path.join(run, spec["renders"])))
            if not renders:
                continue
            if annulus is None:
                probe = np.asarray(Image.open(renders[0]).convert("RGB"))
                height, width = probe.shape[:2]
                me.RADIUS_SCALE = 0.95
                inner = me.build_mask(scene, height, width).cpu().numpy()
                me.RADIUS_SCALE = 1.00
                annulus = me.build_mask(scene, height, width).cpu().numpy() & ~inner
                out.setdefault(scene, {})["_canonical"] = canonical_reference(
                    images_dir or f"/workspace/dataset/fisheye_baselines/{scene}/images_4",
                    annulus, inner)
            gts = sorted(glob.glob(os.path.join(run, spec["gts"]))) if spec["gts"] else []
            if len(gts) != len(renders):
                gts = []
            item = entry_for(renders, gts, annulus, inner)
            if item is None:
                continue
            item["gt_source"] = "run's own gt/" if gts else "canonical images_4 (via views.json)"
            out.setdefault(scene, {})[method] = item
            print(f"  masked_eval    {scene:22s} {method:18s} "
                  f"render0={item['render_zero_fraction_annulus']:.3f} "
                  f"gt0={item.get('gt_zero_fraction_annulus', float('nan')):.3f} "
                  f"gtI={item.get('gt_mean_intensity_annulus_255', float('nan')):7.2f}",
                  flush=True)
    return out


def audit_ocv(tags):
    import masked_eval_ocv as mo

    mo.DEVICE = DEVICE
    out = {}
    for tag in tags:
        annulus = inner = None
        for method, spec in mo.METHODS.items():
            run = mo.resolve_run(spec, tag)
            if run is None or not os.path.isdir(run):
                continue
            renders = sorted(glob.glob(os.path.join(run, spec["renders"])))
            if not renders:
                continue
            parents = {os.path.dirname(r) for r in renders}
            if len(parents) > 1:
                newest = max(parents, key=os.path.getmtime)
                renders = [r for r in renders if os.path.dirname(r) == newest]
            if annulus is None:
                probe = np.asarray(Image.open(renders[0]).convert("RGB"))
                height, width = probe.shape[:2]
                mo.RADIUS_SCALE = 0.95
                inner = mo.build_mask(tag, height, width).cpu().numpy()
                mo.RADIUS_SCALE = 1.00
                annulus = mo.build_mask(tag, height, width).cpu().numpy() & ~inner
                out.setdefault(tag, {})["_canonical"] = canonical_reference(
                    f"/workspace/dataset/fisheye_baselines_ocv/{tag}/images_4", annulus, inner)
            gts = sorted(glob.glob(os.path.join(run, spec["gts"])))
            if len(gts) != len(renders):
                gts = []
            item = entry_for(renders, gts, annulus, inner)
            if item is None:
                continue
            item["gt_source"] = "run's own gt/" if gts else "canonical images_4 (via views.json)"
            out.setdefault(tag, {})[method] = item
            print(f"  myscenes_ocv   {tag:22s} {method:18s} "
                  f"render0={item['render_zero_fraction_annulus']:.3f} "
                  f"gt0={item.get('gt_zero_fraction_annulus', float('nan')):.3f} "
                  f"gtI={item.get('gt_mean_intensity_annulus_255', float('nan')):7.2f}",
                  flush=True)
    return out


def audit_fullcircle(scenes):
    """GT is the canonical dataset image for every method here -- only the render varies."""
    import mask_radius_sweep as sweep
    import masked_eval_fullcircle as mf

    mf.DEVICE = DEVICE
    out = {}
    for scene in scenes:
        names, cam_of = mf.scene_test_views(scene)
        probe = os.path.join(mf.BASE, scene, "images_4", names[0])
        height, width = np.asarray(Image.open(probe)).shape[:2]
        sparse = os.path.join(mf.BASE, scene, "sparse", "0")
        inner, _ = sweep.colmap_masks(sparse, height, width, 0.95, DEVICE)
        wide, _ = sweep.colmap_masks(sparse, height, width, 1.00, DEVICE)
        for method in ("gray", "DFGS", "3dgrut", "SPaGS", "SPaGS-fe", "SPaGS-panofe"):
            run_dir, renders = None, []
            for candidate in mf.run_dirs(method, scene, "masked", ""):
                hits = sorted(glob.glob(os.path.join(candidate, "renders", "*.png")))
                if hits:
                    run_dir, renders = candidate, hits
                    break
            if not renders:
                continue
            views, _ = mf.resolve_views(method, scene, run_dir, len(renders))
            uids = [cam_of[views[os.path.basename(r)]] for r in renders]
            uid = max(set(uids), key=uids.count)
            keep = [i for i, u in enumerate(uids) if u == uid]
            inner_np = inner[uid].cpu().numpy()
            annulus = (wide[uid] & ~inner[uid]).cpu().numpy()
            canonical = [os.path.join(mf.BASE, scene, "images_4",
                                      views[os.path.basename(renders[i])]) for i in keep]
            item = entry_for([renders[i] for i in keep], canonical, annulus, inner_np)
            if item is None:
                continue
            item["gt_source"] = "canonical dataset image (identical for every method)"
            item["camera_uid"] = uid
            out.setdefault(scene, {})[method] = item
            out[scene].setdefault("_canonical", {}).update({
                "canonical_mean_intensity_annulus_255": item["gt_mean_intensity_annulus_255"],
                "canonical_zero_fraction_annulus": item.get("gt_zero_fraction_annulus"),
            })
            print(f"  fullcircle     {scene:10s} {method:18s} "
                  f"render0={item['render_zero_fraction_annulus']:.3f} "
                  f"content={item['gt_mean_intensity_annulus_255']:7.2f}", flush=True)
    return out


def audit_fullcircle_rttpf(variant, scenes):
    """Same as the ocv track but under fullcircle_tracks/<variant>."""
    import mask_radius_sweep as sweep
    import masked_eval_fullcircle as mf
    import masked_eval_rttpf as mr

    mf.DEVICE = DEVICE
    out = {}
    for scene in scenes:
        names, cam_of = mr.scene_test_views(variant, scene)
        sdir = mr.scene_dir(variant, scene)
        probe = os.path.join(sdir, "images_4", names[0])
        height, width = np.asarray(Image.open(probe)).shape[:2]
        sparse = os.path.join(sdir, "sparse", "0")
        inner, _ = sweep.colmap_masks(sparse, height, width, 0.95, DEVICE)
        wide, _ = sweep.colmap_masks(sparse, height, width, 1.00, DEVICE)
        for method in ("gray", "DFGS", "3dgrut", "SPaGS-fe", "SPaGS-panofe",
                       "gray-nc", "gray-nc-off"):
            run_dir, renders = None, []
            for candidate in mr.run_dirs(method, variant, scene, ""):
                hits = sorted(glob.glob(os.path.join(candidate, "renders", "*.png")))
                if hits:
                    run_dir, renders = candidate, hits
                    break
            if not renders:
                continue
            views_json = os.path.join(run_dir, "views.json")
            views = (json.load(open(views_json)) if os.path.exists(views_json)
                     else {f"{i:05d}.png": n for i, n in enumerate(names)})
            uids = [cam_of[views[os.path.basename(r)]] for r in renders]
            uid = max(set(uids), key=uids.count)
            keep = [i for i, u in enumerate(uids) if u == uid]
            inner_np = inner[uid].cpu().numpy()
            annulus = (wide[uid] & ~inner[uid]).cpu().numpy()
            canonical = [os.path.join(sdir, "images_4", views[os.path.basename(renders[i])])
                         for i in keep]
            item = entry_for([renders[i] for i in keep], canonical, annulus, inner_np)
            if item is None:
                continue
            item["gt_source"] = "canonical dataset image (identical for every method)"
            item["camera_uid"] = uid
            out.setdefault(f"{variant}/{scene}", {})[method] = item
            print(f"  fc_rttpf       {variant}/{scene:10s} {method:14s} "
                  f"render0={item['render_zero_fraction_annulus']:.3f} "
                  f"content={item['gt_mean_intensity_annulus_255']:7.2f}", flush=True)
    return out


TRACK_FN = {
    "fullcircle_ocv": lambda: audit_fullcircle(["room1", "lounge", "dark"]),
    "fullcircle_rttpf": lambda: audit_fullcircle_rttpf("refit_rttpf", ["room1", "dark"]),
    "myscenes_ocv": lambda: audit_ocv(["tunnel_warmstart", "workshop_warmstart"]),
    "myscenes_rttpf": lambda: audit_myscenes_rttpf(["atrium", "tunnel"]),
    "fujinon": lambda: audit_patched_masked_eval("fujinon", ["workshop_fujinon"]),
    "immervision_rttpf": lambda: audit_patched_masked_eval(
        "immervision_rttpf", ["workshop_immervision"]),
    "immervision_ocv": lambda: audit_patched_masked_eval(
        "immervision_ocv", ["workshop_immervision_ocv"]),
}


def audit_patched_masked_eval(track, scenes):
    """fujinon / immervision -- both reconfigure `masked_eval` globally, so ONE PER PROCESS."""
    import masked_eval as me  # noqa: F401  (imported for the side effect of the patchers)

    if track == "fujinon":
        import masked_eval_fujinon  # noqa: F401
        images = "/workspace/dataset/fisheye_baselines/workshop_fujinon/images"
    else:
        import masked_eval_immervision as mi

        spec = mi.install(track.split("_", 1)[1])
        images = os.path.join(spec["shared"], "images")
    return audit_myscenes_rttpf(scenes, images_dir=images)


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--track", default="all", choices=sorted(TRACK_FN) + ["all"])
    args = ap.parse_args()

    out = os.path.join(REPO, "tmp", "w3_radial", "annulus_audit.json")
    if args.track == "all":
        for track in TRACK_FN:
            subprocess.run([sys.executable, os.path.abspath(__file__), "--track", track],
                           check=False)
        return

    payload = {"what": "how comparable the r=0.95 -> r=1.00 annulus is between methods",
               "sample_views_per_entry": SAMPLE, "tracks": {}}
    if os.path.exists(out):
        try:
            payload = json.load(open(out))
        except ValueError:
            pass
    payload.setdefault("tracks", {})[args.track] = TRACK_FN[args.track]()
    with open(out, "w") as handle:
        json.dump(payload, handle, indent=1)
    print("wrote", out)


if __name__ == "__main__":
    main()
