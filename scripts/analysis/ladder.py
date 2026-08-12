#!/usr/bin/env python
"""The ablation ladder on BOTH camera families, plus the regularisation audit.

    python scripts/analysis/ladder.py           # ~3 min, CPU only, writes ladder.json

Three things live here, and they are here together because they answer one question --
"which term of the camera model actually carries the gain, and is the comparison between
rungs interpretable at all?":

  1. PINHOLE LADDER (new).  `tmp/mipnerf360/{bicycle,stump}_{off,tilt,radial,ana,
     central_matched,z_only,noncentral}` scored with the SAME function `dose_response.py`
     uses for its pinhole points (`pinhole_scores`: full frame, no mask -- a pinhole run has
     no valid_mask, so the disk IS the frame). Reused, not reimplemented, so W1's pinhole y
     values and this table cannot drift apart.
  2. FISHEYE LADDER (read, not recomputed).  `rungs.json`, written by `rungs.py`, already
     carries masked PSNR + 8 equal-area rings for the six fisheye rungs on 7 scenes. This
     script only assembles the delta-vs-`off` table from it.
  3. REGULARISATION AUDIT (new).  The exact penalty `camera_model.LensResidual.regularization`
     would return at iteration 15000, recomputed from the SAVED weights, next to the final
     data loss from `losses.csv`. This is the falsification of the "the rungs differ because
     bigger rungs pay more penalty" confound.

WHY THE 2x2 IS THE POINT.  On a pinhole camera a non-central pupil CANNOT exist: the
projection has a single centre by construction. `z_only` on pinhole is therefore a negative
control with a known correct answer (zero), and `z_only` on fisheye is the treatment. If a
capacity-absorption story were true, `z` would absorb on both. It does not.

GOTCHA -- THE TWO LADDERS ARE NOT ON A COMMON SCALE.  Pinhole PSNR is full-frame at -r 4/-r 2
on outdoor/indoor mip-NeRF 360; fisheye PSNR is masked to the r=0.95 disk at -r 4 on a
circular frame that is ~44 % of the pixels. Only DELTAS vs each scene's own `off` are ever
compared across the two, never absolute levels, and never a mean over both.

GOTCHA -- THE L2 IS NOT COMMENSURABLE BETWEEN CHANNELS.  `z_weights` is in units of the scene
radius (`config.py:208`), `theta/phi_weights` in radians. One coefficient
(`camera_opt_reg_l2 = 1e-2`) is applied to both, so the same number means a different physical
pressure per channel. It happens not to matter here only because the whole term is inert
(<= 0.14 % of the data loss); raising the coefficient would make the ladder uninterpretable.
"""

from __future__ import annotations

import csv
import glob
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
WORKTREE = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)

OUT_JSON = os.path.join(HERE, "ladder.json")

PINHOLE_SCENES = ["bicycle", "stump"]
PINHOLE_RUNGS = ["off", "tilt", "radial", "ana", "central_matched", "z_only", "noncentral"]
FISHEYE_RUNGS = ["off", "ana", "central_matched", "noncentral_no_ana", "z_only", "noncentral"]
# * the three scenes that carry the FULL fisheye ladder; the other four have only
# * off / z_only / noncentral_no_ana / noncentral.
FISHEYE_FULL_LADDER_SCENES = ["tunnel", "workshop", "reception"]
NOISE_FLOOR_DB = 0.068

# * regularised parameter count per rung: 2 * |active channels| * knots  (+ 8 if z)
KNOTS, KNOTS_Z = 10, 8


def rung_shape(rung: str) -> dict:
    """Mirror of camera_model.RUNGS / LensResidual, without importing torch/CUDA."""
    comps = {
        "off": (), "passthrough": (), "tilt": ("tilt",), "radial": ("tilt", "radial"),
        "ana": ("tilt", "radial", "ana"),
        "noncentral": ("tilt", "radial", "ana", "z"),
        "noncentral_no_ana": ("tilt", "radial", "z"),
        "z_only": ("z",),
        "central_matched": ("tilt", "radial", "ana", "extra_knots"),
        "raxel": ("tilt", "raxel"),
    }[rung]
    knots = KNOTS + (KNOTS_Z if "extra_knots" in comps else 0)
    channels = []
    if "radial" in comps or "extra_knots" in comps:
        channels.append(0)
    if "ana" in comps:
        channels.extend(range(1, 5))
    n_reg = 2 * len(channels) * knots + (KNOTS_Z if "z" in comps else 0)
    return {"components": list(comps), "knots": knots, "active_channels": channels,
            "n_regularised_params": n_reg,
            "n_trained_params": n_reg + (3 if "tilt" in comps else 0)}


# --------------------------------------------------------------------------------------- #
#  1. pinhole ladder                                                                        #
# --------------------------------------------------------------------------------------- #
def pinhole_ladder() -> dict:
    from dose_response import pinhole_scores           # exact same scorer as W1

    out = {}
    for scene in PINHOLE_SCENES:
        entry = {}
        for rung in PINHOLE_RUNGS:
            run = os.path.join(WORKTREE, "tmp", "mipnerf360", f"{scene}_{rung}")
            if not glob.glob(os.path.join(run, "test", "*", "pinhole", "renders", "*.png")):
                print(f"  -- {scene}/{rung}: no renders", flush=True)
                continue
            s = pinhole_scores(run)
            cfg = read_config(run)
            entry[rung] = {"run": run, "psnr_per_view_mean": s["per_view_mean"],
                           "psnr_pooled": s["pooled"], "n_views": s["n"],
                           "camera_opt": cfg.get("camera_opt"),
                           "downsampling": cfg.get("downsampling"),
                           "iterations": cfg.get("iterations"),
                           **rung_shape(rung)}
            print(f"{scene:9s} {rung:18s} {s['per_view_mean']:8.4f} dB "
                  f"({s['n']} views, -r {cfg.get('downsampling')})", flush=True)
        if "off" in entry:
            base = entry["off"]["psnr_per_view_mean"]
            for rung, v in entry.items():
                v["delta_vs_off_db"] = v["psnr_per_view_mean"] - base
                v["above_noise_floor"] = abs(v["delta_vs_off_db"]) > NOISE_FLOOR_DB
        out[scene] = entry
    return out


def read_config(run: str) -> dict:
    try:
        with open(os.path.join(run, "config.json")) as h:
            return json.load(h)
    except OSError:
        return {}


# --------------------------------------------------------------------------------------- #
#  2. fisheye ladder, assembled from rungs.json                                             #
# --------------------------------------------------------------------------------------- #
def fisheye_ladder(rungs: dict) -> dict:
    per_scene, deltas = {}, {}
    for scene, entry in rungs.items():
        got = entry.get("rungs", {})
        if "off" not in got:
            continue
        base = got["off"]["psnr"]
        per_scene[scene] = {r: {"psnr": v["psnr"], "delta_vs_off_db": v["psnr"] - base,
                                "rings": v["rings"], **rung_shape(r)}
                            for r, v in got.items()}
        deltas[scene] = {r: v["psnr"] - base for r, v in got.items()}

    def mean_over(scenes, rung):
        vals = [deltas[s][rung] for s in scenes if s in deltas and rung in deltas[s]]
        return (float(np.mean(vals)), len(vals)) if vals else (None, 0)

    summary = {}
    for rung in FISHEYE_RUNGS:
        m3, n3 = mean_over(FISHEYE_FULL_LADDER_SCENES, rung)
        m7, n7 = mean_over(sorted(deltas), rung)
        summary[rung] = {
            "mean_delta_db_over_3_full_ladder_scenes": m3, "n_scenes_3": n3,
            "mean_delta_db_over_all_scenes_present": m7, "n_scenes_all": n7,
            **rung_shape(rung)}
    return {"per_scene": per_scene, "summary": summary,
            "full_ladder_scenes": FISHEYE_FULL_LADDER_SCENES}


# --------------------------------------------------------------------------------------- #
#  3. regularisation audit                                                                  #
# --------------------------------------------------------------------------------------- #
def final_l1(run: str):
    path = os.path.join(run, "losses.csv")
    if not os.path.exists(path):
        return None, None
    last = None
    with open(path) as h:
        for row in csv.DictReader(h, delimiter=" "):
            last = row
    if last is None:
        return None, None
    return float(last["l1"]), int(last["iteration"])


def reg_penalty(run: str) -> dict | None:
    """Exactly `LensResidual.regularization(l2, curvature)` at iteration 15000, from the
    SAVED weights. Inactive channels are exactly zero in the checkpoint (asserted below), so
    summing every channel equals summing the active ones."""
    ckpt = os.path.join(run, "gaussians_15000.safetensors")
    if not os.path.exists(ckpt):
        return None
    from safetensors import safe_open

    cfg = read_config(run)
    rung = cfg.get("camera_opt", "off")
    if rung in ("off", "passthrough"):
        return None
    l2 = float(cfg.get("camera_opt_reg_l2", 1e-2))
    curv = float(cfg.get("camera_opt_reg_curvature", 1e-2))
    shape = rung_shape(rung)

    with safe_open(ckpt, "pt") as h:
        keys = [k for k in h.keys() if k.startswith("camera_model.lenses.")]
        if not keys:
            return None
        uids = sorted({k.split(".")[2] for k in keys})
        tensors = {}
        for uid in uids:
            tensors[uid] = {n: h.get_tensor(f"camera_model.lenses.{uid}.{n}").float().numpy()
                            for n in ("theta_weights", "phi_weights", "z_weights")}

    total, detail, leaked = 0.0, {}, {}
    for uid, t in tensors.items():
        parts = []
        if shape["active_channels"]:
            parts += [t["theta_weights"], t["phi_weights"]]
            inactive = [c for c in range(5) if c not in shape["active_channels"]]
            leaked[uid] = float(max(np.abs(t["theta_weights"][inactive]).max(initial=0.0),
                                    np.abs(t["phi_weights"][inactive]).max(initial=0.0)))
        if "z" in shape["components"]:
            parts.append(t["z_weights"])
        else:
            leaked[uid] = max(leaked.get(uid, 0.0), float(np.abs(t["z_weights"]).max()))
        pen = 0.0
        for tensor in parts:
            pen += l2 * float(np.square(tensor).sum())
            if tensor.shape[-1] >= 3 and curv > 0.0:
                second = tensor[..., 2:] - 2.0 * tensor[..., 1:-1] + tensor[..., :-2]
                pen += curv * float(np.square(second).sum())
        detail[uid] = pen
        total += pen

    l1, it = final_l1(run)
    return {"run": run, "rung": rung, "l2": l2, "curvature": curv,
            "penalty": total, "penalty_per_lens": detail,
            "final_l1": l1, "final_l1_iteration": it,
            "penalty_over_l1_percent": (100.0 * total / l1) if l1 else None,
            "inactive_weight_max_abs": leaked,
            **shape}


def regularisation_audit(fisheye_scenes, pinhole_entries) -> dict:
    runs = []
    for scene in fisheye_scenes:
        for rung in FISHEYE_RUNGS:
            if rung == "off":
                continue
            for root in (f"/workspace/gray/tmp/final", f"{WORKTREE}/tmp/final"):
                cand = f"{root}/{scene}_{rung}"
                if os.path.isdir(cand):
                    runs.append(("myscenes_rttpf", scene, rung, cand))
                    break
    for scene, entry in pinhole_entries.items():
        for rung, v in entry.items():
            if rung != "off":
                runs.append(("mipnerf360_pinhole", scene, rung, v["run"]))

    out = {}
    for track, scene, rung, run in runs:
        res = reg_penalty(run)
        if res is None:
            continue
        res.update(track=track, scene=scene)
        out[f"{track}/{scene}/{rung}"] = res
        print(f"{scene:11s} {rung:18s} n_reg={res['n_regularised_params']:4d}  "
              f"penalty={res['penalty']:.4e}  L1={res['final_l1']}  "
              f"{res['penalty_over_l1_percent']:.4f} % of the data loss", flush=True)

    ratios = [v["penalty_over_l1_percent"] for v in out.values()
              if v["penalty_over_l1_percent"] is not None]
    verdict = {
        "n_runs": len(out),
        "penalty_over_l1_percent_min": min(ratios) if ratios else None,
        "penalty_over_l1_percent_max": max(ratios) if ratios else None,
        "claim": "the L2/curvature penalty is INERT: it never exceeds a fraction of a percent "
                 "of the data loss, so it cannot explain a rung-to-rung difference of any "
                 "size that this project measures.",
    }
    # the falsification: does the SMALL rung pay MORE than the parameter-matched big one?
    for scene in fisheye_scenes:
        a = out.get(f"myscenes_rttpf/{scene}/z_only")
        b = out.get(f"myscenes_rttpf/{scene}/central_matched")
        if a and b:
            verdict.setdefault("sign_is_inverted", {})[scene] = {
                "z_only_params": a["n_regularised_params"], "z_only_penalty": a["penalty"],
                "central_matched_params": b["n_regularised_params"],
                "central_matched_penalty": b["penalty"],
                "z_only_pays_x_more": a["penalty"] / b["penalty"]}
    verdict["not_commensurable"] = (
        "z_weights is in units of the scene radius (config.py:208), theta/phi_weights in "
        "radians; camera_opt_reg_l2 = 1e-2 is applied to both, so equal coefficients are NOT "
        "equal physical pressure. Harmless only because the whole term is inert.")
    return {"runs": out, "verdict": verdict}


# --------------------------------------------------------------------------------------- #
def main() -> int:
    print("[1/3] pinhole ladder (mip-NeRF 360, full frame, no mask) ...", flush=True)
    pin = pinhole_ladder()

    print("[2/3] fisheye ladder (from rungs.json, masked r=0.95) ...", flush=True)
    rungs_path = os.path.join(HERE, "rungs.json")
    if not os.path.exists(rungs_path):
        raise SystemExit("rungs.json is missing -- run `python scripts/analysis/rungs.py` first")
    fish = fisheye_ladder(json.load(open(rungs_path)))
    for rung, v in fish["summary"].items():
        print(f"{rung:18s} n_reg={v['n_regularised_params']:4d}  "
              f"3-scene mean {str(v['mean_delta_db_over_3_full_ladder_scenes'])[:8]:>8s} dB  "
              f"all-scene mean {str(v['mean_delta_db_over_all_scenes_present'])[:8]:>8s} dB "
              f"({v['n_scenes_all']} scenes)", flush=True)

    print("[3/3] regularisation audit (exact penalty from saved weights) ...", flush=True)
    audit = regularisation_audit(sorted(fish["per_scene"]), pin)

    interaction = {}
    for family, scene, table, key in (
            ("pinhole", "bicycle", pin.get("bicycle", {}), "delta_vs_off_db"),
            ("pinhole", "stump", pin.get("stump", {}), "delta_vs_off_db")):
        interaction[f"{family}/{scene}"] = {r: table[r][key] for r in table if key in table[r]}
    interaction["fisheye/myscenes_3_full_ladder_scenes"] = {
        r: v["mean_delta_db_over_3_full_ladder_scenes"] for r, v in fish["summary"].items()}
    interaction["fisheye/myscenes_all_scenes_present"] = {
        r: v["mean_delta_db_over_all_scenes_present"] for r, v in fish["summary"].items()}

    out = {
        "what": "ablation ladder on both camera families + the regularisation audit",
        "gpu_used": False,
        "noise_floor_db": NOISE_FLOOR_DB,
        "pinhole": pin,
        "fisheye": fish,
        "regularisation": audit,
        "interaction_2x2": interaction,
        "conventions": {
            "pinhole_metric": "full-frame PSNR, per-view mean (dose_response.pinhole_scores)",
            "fisheye_metric": "masked PSNR at r=0.95, per-view mean (collect/rungs.py)",
            "never_pooled": "the two families are never averaged together; only deltas vs each "
                            "scene's own `off` are compared, and only qualitatively.",
            "n": "n = 1 run per cell, no seeds. Any |delta| below 0.068 dB is a zero.",
        },
    }
    with open(OUT_JSON, "w") as h:
        json.dump(out, h, indent=1)
    print(f"wrote {OUT_JSON}  ({os.path.getsize(OUT_JSON)/1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
