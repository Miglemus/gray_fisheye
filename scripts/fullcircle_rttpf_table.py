#!/usr/bin/env python
"""FullCircle `refit_rttpf` results table, including gray + the non-central camera model.

Every number comes from the ONE shared masked eval pass
(`dataset/fullcircle_code/masked_eval_rttpf.py` -> `rttpf_masked_metrics.json`): same
r=0.95 per-camera disk masks, same masked PSNR/SSIM, same LPIPS handling, GT read from the
dataset images. Nothing here reads a repo's self-reported metric.

Perf columns (#gaussians / FPS / train time) come from each run's own sidecars and are
only shown for the methods that have them; FPS is only comparable when every run was timed
on the same card (see dataset/fullcircle_code/queue_perf.sh).

usage: fullcircle_rttpf_table.py [--variant refit_rttpf] [--out <stem>]
"""

import argparse
import csv
import glob
import json
import os

METRICS = "/workspace/dataset/fullcircle_tracks/rttpf_masked_metrics.json"
SCENES = ["room1", "room2", "room3", "flat1", "flat2", "lab", "lounge", "dark", "persons"]

# key in the metrics json -> (display label, run-dir root, layout)
METHODS = [
    ("gray", "gray", "/workspace/gray/worktrees/fullcircle-erp/out/fullcircle_rttpf", "gray"),
    ("gray-nc", "gray + non-central camera (ours)",
     "/workspace/gray/worktrees/noncentral-camera/out/fullcircle_rttpf", "gray"),
    ("DFGS", "DirectFisheye-GS",
     "/workspace/DirectFisheye-GS/worktrees/rttpf/out/fullcircle", "dfgs"),
    ("3dgrut", "3DGUT", "/workspace/3dgrut/worktrees/rttpf/out", "grut"),
    ("SPaGS-fe", "SPaGS (fisheye port)",
     "/workspace/SPaGS-rttpf/nerficg/output/SPaGS", "spags"),
]
# Control rungs: shown as extra rows, never averaged into the headline table.
CONTROLS = [("gray-nc-off", "gray-nc worktree, --camera_opt off (control)")]


def _num(path, key=None):
    try:
        if path.endswith(".json"):
            return float(json.load(open(path))[key])
        with open(path) as f:
            last = [ln for ln in f.read().split("\n") if ln.strip()][-1]
        return float(last.split(",")[-1].split()[-1])
    except (OSError, ValueError, IndexError, KeyError):
        return None


def _spags_gaussians(run_dir):
    """nerficg keeps the final gaussian count inside training_time.json, not a sidecar."""
    return _num(f"{run_dir}/training_time.json", "n_gaussians")


def _hms(path):
    try:
        with open(path) as f:
            tok = [ln for ln in f.read().split("\n") if ln.strip()][-1].split()[-1]
        parts = [int(x) for x in tok.split(":")]
        while len(parts) < 3:
            parts.insert(0, 0)
        return parts[0] * 3600 + parts[1] * 60 + parts[2]
    except (OSError, ValueError, IndexError):
        return None


def perf(key, root, layout, scene, variant):
    """{n_gaussians, fps, train_s} from the run's own sidecars, or empty."""
    name = f"{scene}_{variant}"
    if layout == "gray":
        d = os.path.join(root, name if key != "gray-nc-off" else f"{name}_off")
        return {"n_gaussians": _num(f"{d}/num_gaussians.csv"), "fps": _num(f"{d}/fps.csv"),
                "train_s": _hms(f"{d}/time.csv")}
    if layout == "dfgs":
        d = os.path.join(root, name)
        return {"n_gaussians": _num(f"{d}/gaussians_end.json", "count"),
                "fps": _num(f"{d}/fps.json", "fps"),
                "train_s": _num(f"{d}/training_time.json", "training_time")}
    if layout == "grut":
        # * Sidecar depth varies: measure_fps.py re-derives its output dir from the
        # * checkpoint's config, which sometimes nests one level deeper than the renders.
        # * Search the subtree and take the newest, exactly as the viewer's read_perf does,
        # * rather than guessing the depth (guessing silently produced a `--` column).
        out = {}
        for key, fname, field in (("n_gaussians", "gaussians_end.json", "count"),
                                  ("fps", "fps.json", "fps"),
                                  ("train_s", "training_time.json", "training_time")):
            hits = sorted(glob.glob(f"{root}/fullcircle_{name}/**/{fname}", recursive=True),
                          key=os.path.getmtime)
            out[key] = _num(hits[-1], field) if hits else None
        return out
    if layout == "spags":
        # * nerficg writes fps.csv (a bare number) rather than fps.json, and reports the
        # * gaussian count inside results.json instead of a sidecar of its own.
        for d in sorted(glob.glob(f"{root}/fc_{scene}_rttpf_*"), reverse=True):
            if os.path.exists(f"{d}/fps.csv") or os.path.exists(f"{d}/fps.json"):
                return {"n_gaussians": _spags_gaussians(d),
                        "fps": _num(f"{d}/fps.json", "fps") or _num(f"{d}/fps.csv"),
                        "train_s": _num(f"{d}/training_time.json", "training_time")}
    return {"n_gaussians": None, "fps": None, "train_s": None}


def mean(values):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def fmt(value, digits=3):
    return "--" if value is None else f"{value:.{digits}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="refit_rttpf")
    ap.add_argument("--out", default="/workspace/dataset/fullcircle_baselines/"
                                     "fullcircle_rttpf_results")
    args = ap.parse_args()

    raw = json.load(open(METRICS)).get(args.variant, {})
    rows = METHODS + [(k, label, METHODS[1][2], "gray") for k, label in CONTROLS]

    got = {key: {s: raw.get(s, {}).get(key) for s in SCENES} for key, *_ in rows}
    present = [r for r in rows if any(got[r[0]].values())]
    missing = [f"{r[0]}:{s}" for r in present for s in SCENES if not got[r[0]][s]]

    lines = [f"# FullCircle — {args.variant} — shared masked eval (r=0.95, per-camera disk)", ""]
    if missing:
        lines += [f"> incomplete: {', '.join(missing)}", ""]

    # ---- per-scene PSNR, plus the delta that is the point of the table
    lines += ["## PSNR per scene", "",
              "| method | " + " | ".join(SCENES) + " | mean |",
              "|---" * (len(SCENES) + 2) + "|"]
    for key, label, *_ in present:
        cells = [fmt(got[key][s]["PSNR"], 2) if got[key][s] else "--" for s in SCENES]
        m = mean([got[key][s]["PSNR"] for s in SCENES if got[key][s]])
        lines.append(f"| {label} | " + " | ".join(cells) + f" | **{fmt(m, 3)}** |")
    if got.get("gray") and got.get("gray-nc"):
        d = [(got["gray-nc"][s]["PSNR"] - got["gray"][s]["PSNR"])
             if got["gray-nc"][s] and got["gray"][s] else None for s in SCENES]
        lines.append("| *ours - gray* | " + " | ".join(
            "--" if x is None else f"{x:+.2f}" for x in d)
            + f" | **{'--' if mean(d) is None else f'{mean(d):+.3f}'}** |")
    lines.append("")

    # ---- the full metric set, means over the scenes each method actually has
    lines += ["## Aggregate (mean over the 9 scenes)", "",
              "| method | PSNR | SSIM | LPIPS | #gauss | FPS | train |",
              "|---|---|---|---|---|---|---|"]
    for key, label, root, layout in present:
        entries = [got[key][s] for s in SCENES if got[key][s]]
        p = [perf(key, root, layout, s, args.variant) for s in SCENES]
        train = mean([x["train_s"] for x in p])
        lines.append(
            f"| {label} | {fmt(mean([e['PSNR'] for e in entries]))} "
            f"| {fmt(mean([e['SSIM'] for e in entries]), 4)} "
            f"| {fmt(mean([e['LPIPS'] for e in entries]), 4)} "
            f"| {fmt(mean([x['n_gaussians'] for x in p]), 0)} "
            f"| {fmt(mean([x['fps'] for x in p]), 1)} "
            f"| {'--' if train is None else f'{train / 60:.1f} min'} |")
    lines += ["", "FPS is only comparable across rows when every run was timed on the same "
                  "card (queue_perf.sh pins gpu1). `--` = the sidecar does not exist.", ""]

    with open(args.out + ".md", "w") as f:
        f.write("\n".join(lines) + "\n")
    with open(args.out + ".csv", "w") as f:
        w = csv.writer(f)
        w.writerow(["variant", "scene", "method", "psnr", "ssim", "lpips", "n",
                    "n_gaussians", "fps", "train_s"])
        for key, label, root, layout in present:
            for s in SCENES:
                e = got[key][s]
                if not e:
                    continue
                pf = perf(key, root, layout, s, args.variant)
                w.writerow([args.variant, s, key, f"{e['PSNR']:.3f}", f"{e['SSIM']:.4f}",
                            f"{e['LPIPS']:.4f}", e["n"], pf["n_gaussians"], pf["fps"],
                            pf["train_s"]])
    print("\n".join(lines))
    print(f"\nwrote {args.out}.md + .csv")


if __name__ == "__main__":
    main()
