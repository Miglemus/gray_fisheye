#!/usr/bin/env python
"""Assemble EVERY Phase-1 measurement into one payload: `report_data.json`.

    python scripts/analysis/pack.py            # ~10 s, CPU only
    python scripts/analysis/pack.py --strict   # fail instead of warning on a missing input

This file is the single join point of Phase 1. Everything downstream -- the four figures, the
report page, any table anyone writes later -- reads `report_data.json` and NOTHING else. If a
number is not in here it is not reproducible, and that is the whole point: the paper's central
contribution is the measurement, so an unreproducible measurement pipeline would be a
contradiction.

WHAT IT EATS (all of it already on disk; this script computes no metric of its own)

  A1  sfm_residual.json           62 scene x camera tracks, SfM reprojection residual in urad
  A2  plate_scale.json            S(theta) = dr/dtheta per camera, evaluated-disk edge, urad/px
  W1  dose_response.json          the paired gains, the estimators, the fits, the prereg
  W2  ranking_flip.json           the six method-pair regressions + LODO + the prereg
  W3  tmp/w3_radial/*.json        rings (PSNR/SSIM/LPIPS), mask-radius sweep, non-regression
  --  ladder.json                 the ablation ladder on both families + the reg audit
  --  calib_consistency*.json     E_shared / E_calib in px (legacy) and in urad
  --  optics/fields/rings/rungs/crops.json   the per-scene report-page payloads

WHAT IT WRITES

  report_data.json
    provenance   sha256 + mtime + size + producing command of every input file
    a1 a2 w1 w2 w3 ladder calibration        the inputs, embedded verbatim
    report_page                              the legacy per-scene payload pack.py used to build
    figures                                  the pre-joined arrays each figure draws
    headline                                 the numbers the written report quotes
    checks                                   cross-file agreements re-verified at pack time

THREE THINGS THAT WILL BITE WHOEVER EDITS THIS
  1. The inputs were written at DIFFERENT TIMES against checkpoints that moved. pueue tasks
     1539-1545 retrained 7 of the 9 FullCircle `noncentral` runs on 2026-08-10 19:30-20:40,
     AFTER calib_consistency.json (13:16) and plate_scale.json (16:58) were written. Their
     FullCircle learned-residual numbers therefore describe checkpoint instance A while
     dose_response.json describes instance B. The differences are below the 0.068 dB noise
     floor, but they are not zero. `provenance` records every mtime so this stays visible.
  2. NOTHING here is averaged across lens groups, frame geometries or resolutions. Several
     embedded files carry a deliberately-named `*do_not_publish*` pooled key; they are copied
     through unchanged, name included.
  3. `report_page.psnr.gray_published` for workshop is the published baseline, which was
     trained WITHOUT vignetting_comp at batch_size 1. It is not a valid paired control and W1
     does not use it; `gray_matched` is the control.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
WORKTREE = os.path.abspath(os.path.join(HERE, "..", ".."))
W3 = os.path.join(WORKTREE, "tmp", "w3_radial")
OUT = os.path.join(HERE, "report_data.json")

SCENES = ["atrium", "classroom", "forest", "library", "reception", "tunnel", "workshop"]
SWEEP_TRACKS = ["myscenes_rttpf", "myscenes_ocv", "fullcircle_rttpf", "fullcircle_ocv",
                "fujinon", "immervision_rttpf", "immervision_ocv"]

# name -> (path, the command that produces it, required?)
INPUTS = {
    "sfm_residual":        (f"{HERE}/sfm_residual.json", "python scripts/analysis/sfm_residual.py", True),
    "plate_scale":         (f"{HERE}/plate_scale.json", "python scripts/analysis/plate_scale.py", True),
    "dose_response":       (f"{HERE}/dose_response.json", "python scripts/analysis/dose_response.py", True),
    "dose_response_prereg": (f"{HERE}/dose_response_prereg.md", "written by hand before fitting", False),
    "ranking_flip":        (f"{HERE}/ranking_flip.json", "python scripts/analysis/ranking_flip.py", True),
    "ladder":              (f"{HERE}/ladder.json", "python scripts/analysis/ladder.py", True),
    "calib_consistency":   (f"{HERE}/calib_consistency.json", "python scripts/analysis/calib_consistency.py", False),
    "calib_consistency_urad": (f"{HERE}/calib_consistency_urad.json", "python scripts/analysis/calib_consistency_urad.py", False),
    "residual_expressible": (f"{HERE}/residual_expressible.json", "python scripts/analysis/residual_expressible.py", False),
    "optics":              (f"{HERE}/optics.json", "python scripts/analysis/optics.py", False),
    "fields":              (f"{HERE}/fields.json", "python scripts/analysis/fields.py", True),
    "rings":               (f"{HERE}/rings.json", "python scripts/analysis/collect.py", True),
    "rungs":               (f"{HERE}/rungs.json", "python scripts/analysis/rungs.py", True),
    "crops":               (f"{HERE}/crops.json", "python scripts/analysis/crops.py workshop reception tunnel", True),
    "subtractive":         (f"{HERE}/subtractive.json", "python scripts/analysis/subtractive.py", False),
    "z_shape":             (f"{HERE}/z_shape.json", "python scripts/analysis/z_shape.py", False),
    "rings_all":           (f"{W3}/rings_all.json", "python scripts/analysis/rings_all.py", True),
    "rings_all_3":         (f"{W3}/rings_all_3.json", "python scripts/analysis/rings_all.py --rings 3", False),
    "rings_stats":         (f"{W3}/rings_stats.json", "python scripts/analysis/rings_stats.py", True),
    "annulus_audit":       (f"{W3}/annulus_audit.json", "python scripts/analysis/annulus_audit.py", False),
    "ocv_mask_convention": (f"{W3}/ocv_mask_convention.json", "python scripts/analysis/annulus_audit.py", False),
    "nonregression":       (f"{W3}/nonregression_2026-08-11.json", "python scripts/radial_eval.py --runs /workspace/gray/out/tunnel_fisheye_baseline", False),
    "full_metrics":        ("/workspace/gray/tmp/final/full_metrics.json", "written by the 7-scene eval pass", False),
}
for _t in SWEEP_TRACKS:
    INPUTS[f"sweep_{_t}"] = (f"{W3}/sweep_{_t}.json",
                             f"python scripts/analysis/mask_radius_sweep.py --track {_t}  [GPU: via pueue]",
                             False)


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_inputs(strict: bool):
    data, prov, missing = {}, {}, []
    for name, (path, cmd, required) in INPUTS.items():
        if not os.path.exists(path):
            missing.append((name, path, cmd, required))
            prov[name] = {"path": path, "present": False, "produced_by": cmd,
                          "required": required}
            continue
        prov[name] = {
            "path": path, "present": True, "produced_by": cmd, "required": required,
            "bytes": os.path.getsize(path), "sha256": sha256(path),
            "mtime_utc": datetime.fromtimestamp(os.path.getmtime(path),
                                                timezone.utc).isoformat(timespec="seconds"),
        }
        if path.endswith(".json"):
            with open(path) as handle:
                data[name] = json.load(handle)
        else:
            with open(path) as handle:
                data[name] = handle.read()
    hard = [m for m in missing if m[3]]
    for name, path, cmd, required in missing:
        tag = "MISSING (required)" if required else "missing (optional)"
        print(f"  !! {tag}: {name} -> {path}\n     produce it with: {cmd}")
    if hard and strict:
        raise SystemExit(f"{len(hard)} required input(s) missing; refusing to pack")
    return data, prov


def rnd(values, digits=6):
    return [None if v is None else float(f"%.{digits}g" % v) for v in values]


# --------------------------------------------------------------------------------------- #
#  the legacy per-scene report page payload (what pack.py used to build, now guarded)        #
# --------------------------------------------------------------------------------------- #
def report_page(data):
    optics, fields = data.get("optics", {}), data.get("fields", {})
    rings, rungs, crops = data.get("rings", {}), data.get("rungs", {}), data.get("crops", {})
    if not (optics and fields and rings):
        return {"available": False,
                "why": "needs optics.json + fields.json + rings.json"}

    page = {
        "available": True,
        "theta_deg": fields["theta_deg"],
        "bin_centres_deg": optics["bin_centres_deg"],
        "bin_edges_deg": optics["bin_edges_deg"],
        "ring_edges": rings["scenes"]["tunnel"]["ring_edges"],
        "num_rings": rings.get("num_rings"),
        "scenes": {},
        "metrics": data.get("full_metrics", {}),
        # * dev ladder: tunnel, -r 8, 7500 iterations, unfreeze at 20 %. NEVER mixed with the
        # * r4/15k numbers -- different resolution and schedule, and the plan is explicit that
        # * the whole of tmp/ladder_r8 sits inside the noise.
        "dev_ladder": {"off": 27.40, "passthrough": 27.46, "tilt": 27.39, "radial": 27.51,
                       "ana": 27.39, "noncentral": 27.55, "central_matched": 27.39,
                       "raxel": 27.59},
        "dev_ladder_warning": "tmp/ladder_r8, -r 8 / 7500 it: the entire ladder is inside the "
                              "noise there. Illustrative only, never a result.",
        "crops": crops,
    }
    for scene in SCENES:
        o, f = optics["scenes"][scene], fields["scenes"][scene]
        r = rings["scenes"][scene]["methods"]
        if "gray" not in r or "gray-non-central" not in r:
            continue
        baseline_key = "gray-workshopfix" if scene == "workshop" else "gray"
        baseline_key = baseline_key if baseline_key in r else "gray"
        entry = {
            "focal_px": o["focal_px"],
            "camera_extent": o["camera_extent"],
            "depth": [o["depth_p05"], o["depth_p50"], o["depth_p95"]],
            "n_obs": o["n_observations"],
            "n_views": r["gray"]["views"],
            "profile_z": rnd(o["profile"]["z"]),
            "profile_dtheta": rnd(o["profile"]["dtheta"]),
            "profile_dphi": rnd(o["profile"]["dphi"]),
            "caustic_x": rnd(o["caustic_x"]),
            "caustic_z": rnd(o["caustic_z"]),
            "r_px": f["r_px"],
            "drdtheta_px": f["drdtheta_px"],
            "dtheta_channels": f["noncentral"]["dtheta_channels"],
            "dphi_channels": f["noncentral"]["dphi_channels"],
            "z_profile": f["noncentral"]["z"],
            "omega": f["noncentral"]["omega"],
            "irreducible_px": rnd(o["irreducible_px"], 4),
            "mean_shift_px": rnd(o["mean_shift_px"], 4),
            "depth_hist": o["depth_hist"],
            "depth_hist_log_edges": rnd(o["depth_hist_log_edges"], 5),
            "psnr": {
                "gray_published": r["gray"]["psnr"],
                "gray_matched": r[baseline_key]["psnr"],
                "noncentral": r["gray-non-central"]["psnr"],
            },
            "rings": {
                "gray_matched": rnd(r[baseline_key]["rings"], 5),
                "noncentral": rnd(r["gray-non-central"]["rings"], 5),
            },
            "per_view": {
                "gray_matched": rnd(r[baseline_key]["per_view"], 5),
                "noncentral": rnd(r["gray-non-central"]["per_view"], 5),
            },
        }
        if "SPaGS" in r:
            entry["psnr"]["spags"] = r["SPaGS"]["psnr"]
            entry["rings"]["spags"] = rnd(r["SPaGS"]["rings"], 5)
        if scene in rungs:
            entry["rungs"] = {name: {"psnr": v["psnr"], "rings": rnd(v["rings"], 5)}
                              for name, v in rungs[scene]["rungs"].items()}
        page["scenes"][scene] = entry
    return page


# --------------------------------------------------------------------------------------- #
#  pre-joined figure payloads: every figure draws from these arrays, nothing else            #
# --------------------------------------------------------------------------------------- #
def figure_payloads(data):
    figs = {}

    # ---- W1: the two dose-response panels + the estimator panels ----------------------- #
    if "dose_response" in data:
        dr = data["dose_response"]
        figs["dose_response"] = {
            "renders": ["figures/dose_response.svg", "figures/dose_response.pdf"],
            "drawn_by": "dose_response.figure_dose_response(payload, path)",
            "payload_key": "w1",
        }
        figs["estimator_agreement"] = {
            "renders": ["figures/estimator_agreement.svg", "figures/estimator_agreement.pdf"],
            "drawn_by": "dose_response.figure_estimator_agreement(payload, path)",
            "payload_key": "w1",
        }
        figs["estimator_lens_level"] = {
            "renders": ["figures/estimator_lens_level.svg", "figures/estimator_lens_level.pdf"],
            "drawn_by": "dose_response.figure_lens_level(payload, path)",
            "payload_key": "w1",
        }
        figs["dose_response"]["points"] = [
            {k: p.get(k) for k in ("track", "scene", "family", "kind", "lens", "primary",
                                   "y_db", "y_db_pooled", "E_sfm_urad", "E_sfm_rms_urad",
                                   "E_sfm_sys_urad", "E_learned_urad", "E_sfm_eval_px",
                                   "eval_resolution", "colmap_model")}
            for p in dr["points"]]

    # ---- W2 --------------------------------------------------------------------------- #
    if "ranking_flip" in data:
        figs["ranking_flip"] = {
            "renders": ["figures/ranking_flip.svg", "figures/ranking_flip.pdf"],
            "drawn_by": "ranking_flip.make_figure(payload, path)",
            "payload_key": "w2",
        }

    # ---- W3 rings --------------------------------------------------------------------- #
    if "rings_all" in data:
        groups = {}
        for key, entry in data["rings_all"].items():
            ds = entry["dataset"]
            g = groups.setdefault(ds, {"scenes": [], "off": {}, "noncentral": {}})
            g["scenes"].append(key)
            for arm in ("off", "noncentral"):
                for metric in ("psnr", "ssim", "lpips"):
                    g[arm].setdefault(metric, []).append(
                        entry[arm]["metrics"][metric]["rings"])
        figs["rings"] = {
            "renders": ["figures/rings.svg", "figures/rings.pdf"],
            "drawn_by": "make_rings_figure.draw(payload, outdir)",
            "payload_key": "w3.rings_all",
            "groups": {k: {"scenes": v["scenes"],
                           "n_rings": len(v["off"]["psnr"][0])} for k, v in groups.items()},
        }

    # ---- ladder ----------------------------------------------------------------------- #
    if "ladder" in data:
        figs["ladder"] = {
            "renders": ["figures/ladder.svg", "figures/ladder.pdf"],
            "drawn_by": "make_figures.figure_ladder(payload, path)",
            "payload_key": "ladder",
        }

    # ---- learned field ---------------------------------------------------------------- #
    if "fields" in data:
        figs["field"] = {
            "renders": ["figures/field.svg", "figures/field.pdf"],
            "drawn_by": "make_figures.figure_field(payload, path)",
            "payload_key": "report_page",
        }

    # ---- crops ------------------------------------------------------------------------ #
    if "crops" in data:
        figs["crops"] = {
            "renders": ["figures/crops.svg", "figures/crops.pdf"],
            "drawn_by": "make_figures.figure_crops(payload, path)",
            "payload_key": "report_page.crops",
            "scenes": sorted(data["crops"]),
        }
    return figs


# --------------------------------------------------------------------------------------- #
#  cross-file checks re-run at pack time                                                     #
# --------------------------------------------------------------------------------------- #
def checks(data):
    out = {}

    # 1. rungs.json vs subtractive.json -- two invocations of the same scoring path.
    if "rungs" in data and "subtractive" in data:
        worst, n = 0.0, 0
        for scene, entry in data["rungs"].items():
            for rung, v in entry["rungs"].items():
                other = data["subtractive"].get(scene, {}).get(rung)
                if other:
                    worst = max(worst, abs(v["psnr"] - other["psnr"]))
                    n += 1
        out["rungs_vs_subtractive"] = {
            "n_compared": n, "max_abs_psnr_diff_db": worst, "pass": worst < 1e-9,
            "note": "plumbing check only: both files call collect.load / collect.ring_map, so "
                    "agreement proves the two invocations saw the same runs, not that the "
                    "metric is right.",
        }

    # 2. W1's myscenes gains vs the ring-decomposed gains of W3 (different scorers).
    if "dose_response" in data and "rings_all" in data:
        w1 = {p["scene"]: p["y_db"] for p in data["dose_response"]["points"]
              if p["track"] == "myscenes_rttpf"}
        rows = []
        for key, entry in data["rings_all"].items():
            ds, scene = key.split("/", 1)
            if ds != "myscenes" or scene not in w1:
                continue
            w3 = (entry["noncentral"]["metrics"]["psnr"]["disk_per_view_mean"]
                  - entry["off"]["metrics"]["psnr"]["disk_per_view_mean"])
            rows.append({"scene": scene, "w1_gain_db": w1[scene], "w3_gain_db": w3,
                         "abs_diff": abs(w1[scene] - w3)})
        if rows:
            out["w1_gain_vs_w3_rings"] = {
                "rows": rows, "max_abs_diff_db": max(r["abs_diff"] for r in rows),
                "pass": max(r["abs_diff"] for r in rows) < 1e-6,
                "note": "W1 scores through radial_eval, W3 through its own ring pass; both "
                        "must land on the same disk per-view mean.",
            }

    # 3. the non-regression anchor: 28.537 on out/tunnel_fisheye_baseline.
    if "nonregression" in data:
        v = data["nonregression"]["tunnel_fisheye_baseline"]["disk_per_view_mean"]
        out["radial_eval_nonregression"] = {
            "measured": v, "canonical": 28.53749677113124, "diff_db": v - 28.53749677113124,
            "pass": abs(v - 28.53749677113124) < 1e-5,
        }

    # 4. E_sfm in the two files that carry it (A1 native, W1 joined).
    if "sfm_residual" in data and "dose_response" in data:
        worst = 0.0
        for p in data["dose_response"]["points"]:
            t = data["sfm_residual"]["tracks"].get(p["sfm_key"])
            if t and p.get("E_sfm_urad") is not None:
                worst = max(worst, abs(t["residual_urad_theta_le_85_5"]["median"]
                                       - p["E_sfm_urad"]))
        out["E_sfm_A1_vs_W1"] = {"max_abs_diff_urad": worst, "pass": worst < 1e-9}

    # 5. x used by W2 is the same A1 column.
    if "sfm_residual" in data and "ranking_flip" in data:
        worst, n = 0.0, 0
        pairs = data["ranking_flip"].get("pairs_primary", {})
        for res in pairs.values():
            for p in res.get("points", []):
                t = data["sfm_residual"]["tracks"].get(p["track"])
                if t:
                    worst = max(worst, abs(t["residual_urad_theta_le_85_5"]["median"] - p["x"]))
                    n += 1
        if n:
            out["E_sfm_A1_vs_W2"] = {"n_compared": n, "max_abs_diff_urad": worst,
                                     "pass": worst < 1e-9}

    # 6. the regularisation audit's own falsification.
    if "ladder" in data:
        v = data["ladder"]["regularisation"]["verdict"]
        out["regularisation_is_inert"] = {
            "max_penalty_over_l1_percent": v.get("penalty_over_l1_percent_max"),
            "pass": (v.get("penalty_over_l1_percent_max") or 1e9) < 1.0,
            "sign_is_inverted": v.get("sign_is_inverted"),
        }
    return out


# --------------------------------------------------------------------------------------- #
#  headline: exactly the numbers the written report quotes                                   #
# --------------------------------------------------------------------------------------- #
def headline(data):
    h = {"noise_floor_db": 0.068}
    if "dose_response" in data:
        dr = data["dose_response"]
        h["w1"] = {
            "per_track_gain": {k: {"n": v["gain"].get("n"),
                                   "mean_db": v["gain"].get("mean_db"),
                                   "median_db": v["gain"].get("median_db"),
                                   "ci95_db": v["gain"].get("mean_ci95_db"),
                                   "wilcoxon_p": v["gain"].get("wilcoxon_p"),
                                   "n_positive": v["gain"].get("n_positive"),
                                   "n_negative": v["gain"].get("n_negative"),
                                   "family": v.get("family"),
                                   "scenes": v.get("scenes")}
                               for k, v in dr["per_track"].items() if "gain" in v},
            "association_primary_axis": dr["associations"].get("E_sfm_urad"),
            "association_within_track": dr.get("associations_within_track", {})
                                          .get("POOLED_WITHIN_TRACK_exploratory", {})
                                          .get("E_sfm_urad"),
            "fit_on_E_sfm": {k: dr["fits"]["E_sfm_urad"].get(k)
                             for k in ("A_rad^-2", "A_ci95", "rmse_db", "r2",
                                       "threshold_urad_at_noise_floor",
                                       "threshold_ci95_urad")},
            "threshold_on_E_calib": dr.get("threshold_on_E_calib"),
            "e_calib": dr.get("e_calib"),
            "e_shared_bugfixed": dr.get("e_shared_bugfixed"),
            "falsification_conditions": dr.get("prereg_falsification"),
            "prereg_22_only": dr.get("prereg_22_only"),
            "seed_repeat_noise_floor": dr.get("fullcircle_seed_repeat"),
        }
    if "ranking_flip" in data:
        rf = data["ranking_flip"]
        h["w2"] = {k: {"n_tracks": v["n_tracks"], "n_clusters": v["n_clusters"],
                       "rho": v["spearman_rho"],
                       "p_permutation": v["p_permutation_track_level"],
                       "p_holm": v["p_holm_adjusted"],
                       "ci95_cluster_bootstrap": v.get("bootstrap_cluster_ci"),
                       "cluster_aggregate": v.get("cluster_aggregate")}
                   for k, v in rf.get("pairs_primary", {}).items()}
        h["w2_verdicts"] = rf.get("verdicts")
    if "rings_stats" in data:
        h["w3_rings"] = data["rings_stats"].get("blocks")
    if "ladder" in data:
        h["ladder"] = data["ladder"]["interaction_2x2"]
        h["regularisation"] = data["ladder"]["regularisation"]["verdict"]
    if "calib_consistency_urad" in data:
        h["e_shared_urad"] = data["calib_consistency_urad"]
    return h


def mask_radius_table():
    """The r = 0.85 / 0.95 / 1.00 robustness table, as data instead of prose.

    `w3_tables.py` already renders this to Markdown; the *counting* logic (`material_flips`,
    `kendall_tau`, `r100_verdict`, the per-metric noise floors) is imported from it rather
    than rewritten, so the JSON and the .md cannot disagree.

    A flip is only counted when the two methods are separated by more than the run-to-run
    noise floor at BOTH radii -- raw Kendall tau punishes a 0.001 dB tie exactly as hard as a
    real reordering, and several methods in these stores sit inside the noise of each other.
    """
    import itertools

    try:
        import w3_tables as W
    except Exception as exc:                                    # pragma: no cover
        return {"available": False, "why": f"cannot import w3_tables: {exc}"}
    rows = W.load_rows()
    if not rows:
        return {"available": False, "why": "no tmp/w3_radial/sweep_*.json on disk"}
    audit = W.load_audit()

    per_scene, flips, totals = [], [], {}
    verdicts = {}
    for track in sorted({r["track"] for r in rows}):
        ok, why = W.r100_verdict(track, audit)
        verdicts[track] = {"r100_is_a_ranking": ok, "why": why,
                           "frame_type": next(r["frame_type"] for r in rows
                                              if r["track"] == track)}
        entries = [r for r in rows if r["track"] == track]
        for scene in sorted({e["scene"] for e in entries}):
            here = [e for e in entries if e["scene"] == scene]
            frac = here[0]["valid_fraction"]
            identical = (frac.get("0.85") is not None
                         and abs(frac["0.85"] - frac["1.00"]) < 1e-6)
            for metric in ("PSNR", "SSIM", "LPIPS"):
                values, orders = {}, {}
                for radius in W.RADII:
                    scored = [(e["method"], e["metrics"].get(radius, {}).get(metric))
                              for e in here]
                    scored = [(m, v) for m, v in scored if v is not None]
                    values[radius] = dict(scored)
                    orders[radius] = [m for m, _ in sorted(scored, key=lambda kv: kv[1],
                                                           reverse=W.HIGHER[metric])]
                sep_lo, flip_lo = W.material_flips(values["0.95"], values["0.85"],
                                                   W.HIGHER[metric], W.NOISE[metric])
                sep_hi, flip_hi = W.material_flips(values["0.95"], values["1.00"],
                                                   W.HIGHER[metric], W.NOISE[metric])
                per_scene.append({
                    "track": track, "scene": scene, "metric": metric,
                    "n_methods": len(orders["0.95"]),
                    "valid_fraction": frac, "identical_masks": identical,
                    "scores": values,
                    "tau_0.85_vs_0.95": W.kendall_tau(orders["0.95"], orders["0.85"]),
                    "tau_1.00_vs_0.95": W.kendall_tau(orders["0.95"], orders["1.00"]),
                    "separated_pairs_0.85": sep_lo, "material_flips_0.85": len(flip_lo),
                    "separated_pairs_1.00": sep_hi, "material_flips_1.00": len(flip_hi),
                    "r100_is_a_ranking": verdicts[track]["r100_is_a_ranking"],
                })
                for radius, group in (("0.85", flip_lo), ("1.00", flip_hi)):
                    for x, y, gap95, gap_other in group:
                        flips.append({"track": track, "scene": scene, "metric": metric,
                                      "radius": radius, "method_a": x, "method_b": y,
                                      "gap_at_0.95": gap95, "gap_at_radius": gap_other})
                if not identical and len(orders["0.95"]) >= 3:
                    b = totals.setdefault(metric, {"separated_0.85": 0, "flips_0.85": 0,
                                                   "separated_1.00": 0, "flips_1.00": 0,
                                                   "n_rows": 0})
                    b["separated_0.85"] += sep_lo
                    b["flips_0.85"] += len(flip_lo)
                    b["n_rows"] += 1
                    if verdicts[track]["r100_is_a_ranking"] is True:
                        b["separated_1.00"] += sep_hi
                        b["flips_1.00"] += len(flip_hi)
    for metric, b in totals.items():
        b["survival_rate_0.85"] = (1.0 - b["flips_0.85"] / b["separated_0.85"]
                                   if b["separated_0.85"] else None)
        b["survival_rate_1.00"] = (1.0 - b["flips_1.00"] / b["separated_1.00"]
                                   if b["separated_1.00"] else None)
    grand = {
        "separated_pairs_0.85": sum(b["separated_0.85"] for b in totals.values()),
        "material_flips_0.85": sum(b["flips_0.85"] for b in totals.values()),
    }
    grand["survival_rate_0.85"] = (1.0 - grand["material_flips_0.85"]
                                   / grand["separated_pairs_0.85"]
                                   if grand["separated_pairs_0.85"] else None)
    return {"available": True, "noise_floors": W.NOISE, "radii": list(W.RADII),
            "r100_verdict_per_track": verdicts, "per_scene": per_scene,
            "material_flips": flips, "per_metric_totals": totals, "all_metrics": grand,
            "warning": "r = 1.00 is an ANNOTATED row, not a ranking, on every track whose "
                       "verdict says so: several methods render exact black in the "
                       "0.95->1.00 annulus, which the evaluator scores as a perfect match or "
                       "as total error depending on which GT it pairs them with. Rows where "
                       "the three radii are the SAME mask (polynomial inversion binds before "
                       "the theta cut) are marked identical_masks: their stability is "
                       "arithmetic, not evidence."}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--strict", action="store_true",
                    help="fail if a required input is missing instead of packing without it")
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()

    print("[1/4] reading inputs ...", flush=True)
    data, prov = load_inputs(args.strict)

    print("[2/4] assembling ...", flush=True)
    try:
        commit = subprocess.check_output(["git", "-C", WORKTREE, "rev-parse", "--short", "HEAD"],
                                         text=True).strip()
    except Exception:
        commit = None

    payload = {
        "what": "Phase 1 -- every measurement behind 'the camera model dominates', in one file",
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "worktree": WORKTREE,
        "commit": commit,
        "gpu_used_by_this_script": False,
        "rules": [
            "Never average across lens groups, frame geometries or resolutions: the plan "
            "forbids it and the regime is what drives every result here.",
            "Never compare methods through their self-reported metrics: every cross-method "
            "number in this file comes from one shared masked eval pass over raw renders.",
            "E_learned is never an x axis (circular). E_shared is NOT training-free either.",
        ],
        "provenance": prov,
        "a1_sfm_residual": data.get("sfm_residual"),
        "a2_plate_scale": data.get("plate_scale"),
        "w1": data.get("dose_response"),
        "w1_preregistration": data.get("dose_response_prereg"),
        "w2": data.get("ranking_flip"),
        "w3": {
            "rings_all": data.get("rings_all"),
            "rings_all_3": data.get("rings_all_3"),
            "rings_stats": data.get("rings_stats"),
            "annulus_audit": data.get("annulus_audit"),
            "ocv_mask_convention": data.get("ocv_mask_convention"),
            "nonregression": data.get("nonregression"),
            "mask_radius_sweep": {t: data.get(f"sweep_{t}") for t in SWEEP_TRACKS
                                  if data.get(f"sweep_{t}")},
            "mask_radius_robustness": mask_radius_table(),
        },
        "ladder": data.get("ladder"),
        "calibration": {
            "calib_consistency_px_LEGACY": data.get("calib_consistency"),
            "calib_consistency_urad": data.get("calib_consistency_urad"),
            "residual_expressible": data.get("residual_expressible"),
            "warning": "the *_px columns of calib_consistency.json are legacy: they are a peak "
                       "taken at the edge and converted through a local derivative whose radial "
                       "index list is the 12-parameter THIN_PRISM layout, while every camera it "
                       "reads has 16 parameters. Use the urad file, or W1's E_calib.",
        },
        "report_page": report_page(data),
        "figures": figure_payloads(data),
        "headline": headline(data),
        "checks": checks(data),
    }

    print("[3/4] writing ...", flush=True)
    with open(args.out, "w") as handle:
        json.dump(payload, handle, separators=(",", ":"))
    print(f"wrote {args.out}  ({os.path.getsize(args.out)/1024/1024:.2f} MB)")

    print("[4/4] checks:")
    ok = True
    for name, res in payload["checks"].items():
        if isinstance(res, dict) and "pass" in res:
            ok &= bool(res["pass"])
            print(f"   {'PASS' if res['pass'] else 'FAIL'}  {name}: "
                  + ", ".join(f"{k}={v}" for k, v in res.items()
                              if k in ("max_abs_psnr_diff_db", "max_abs_diff_db",
                                       "max_abs_diff_urad", "diff_db", "n_compared",
                                       "max_penalty_over_l1_percent")))
        else:
            print(f"   ----  {name}")
    missing = [k for k, v in prov.items() if not v["present"]]
    if missing:
        print(f"   note: {len(missing)} input(s) absent: {', '.join(sorted(missing))}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
