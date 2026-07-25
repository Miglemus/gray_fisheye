#!/usr/bin/env python
"""Consolidate FullCircle baseline results into one CSV + markdown table.

Sources:
- gray: out/fullcircle/<scene>_{masked,control}/results.json  (masked-PSNR protocol,
  golden tripod test split via test.txt, r=0.95 disk masks)
- SPaGS: /workspace/nerficg-native/output/SPaGS/<scene>_fullcircle_*/masked_eval_static.json
  (static protocol: NOT-capturer masked PSNR/SSIM, mask-multiplied LPIPS, %8 pano split)

NOTE: gray and SPaGS numbers are NOT directly comparable (different views, image
spaces and masks); masked-vs-control deltas within a method are the comparison.
"""

import csv
import glob
import json
import os

GRAY_OUT = "/workspace/gray/worktrees/person-masks/out/fullcircle"
SPAGS_OUT = "/workspace/nerficg-native/output/SPaGS"
SCENES = ["room1", "room2", "room3", "flat1", "flat2", "lab", "lounge", "dark", "persons"]
DEST = "/workspace/dataset/fullcircle_baselines/fullcircle_results"


def gray_metrics(scene, variant):
    p = os.path.join(GRAY_OUT, f"{scene}_{variant}", "results.json")
    if not os.path.exists(p):
        return None
    r = json.load(open(p))
    it = max(r, key=int)
    m = r[it]["opencv_fisheye"]
    return {"psnr": m["PSNR"], "ssim": m["SSIM"], "lpips": m["LPIPS"], "iters": it}


def spags_metrics(scene, variant):
    tag = "personmask" if variant == "masked" else "control"
    dirs = sorted(glob.glob(os.path.join(SPAGS_OUT, f"{scene}_fullcircle_{tag}_*")))
    for d in reversed(dirs):
        p = os.path.join(d, "masked_eval_static.json")
        if os.path.exists(p):
            s = json.load(open(p))["summary"]
            return {"psnr": s["psnr"], "ssim": s["ssim"], "lpips": s["lpips"],
                    "n": s["n"], "run": os.path.basename(d)}
    return None


def main():
    rows = []
    for scene in SCENES:
        row = {"scene": scene}
        for variant in ("masked", "control"):
            g = gray_metrics(scene, variant)
            s = spags_metrics(scene, variant)
            for k in ("psnr", "ssim", "lpips"):
                row[f"gray_{variant}_{k}"] = round(g[k], 3) if g else ""
                row[f"spags_{variant}_{k}"] = round(s[k], 3) if s else ""
        g_m, g_c = gray_metrics(scene, "masked"), gray_metrics(scene, "control")
        s_m, s_c = spags_metrics(scene, "masked"), spags_metrics(scene, "control")
        row["gray_delta_psnr"] = round(g_m["psnr"] - g_c["psnr"], 2) if g_m and g_c else ""
        row["spags_delta_psnr"] = round(s_m["psnr"] - s_c["psnr"], 2) if s_m and s_c else ""
        rows.append(row)

    os.makedirs(os.path.dirname(DEST), exist_ok=True)
    with open(DEST + ".csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    lines = ["| scene | gray masked | gray control | Δgray | SPaGS masked (static) | SPaGS control | ΔSPaGS |",
             "|---|---|---|---|---|---|---|"]
    fmt = lambda r, p: (f"{r[p+'_psnr']} / {r[p+'_ssim']} / {r[p+'_lpips']}"
                        if r[p + "_psnr"] != "" else "—")
    for r in rows:
        lines.append(f"| {r['scene']} | {fmt(r,'gray_masked')} | {fmt(r,'gray_control')} | "
                     f"{r['gray_delta_psnr'] or '—'} | {fmt(r,'spags_masked')} | "
                     f"{fmt(r,'spags_control')} | {r['spags_delta_psnr'] or '—'} |")
    with open(DEST + ".md", "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nwrote {DEST}.csv / .md")


if __name__ == "__main__":
    main()
