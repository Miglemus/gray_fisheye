"""figures/rings.{svg,pdf} -- the three metrics per equal-area ring, `off` vs `noncentral`.

    python scripts/analysis/make_rings_figure.py            # reads tmp/w3_radial/rings_all.json

Row 1 shows the ABSOLUTE ring profile of each dataset (dashed = `off`, solid =
`noncentral`); row 2 shows the paired delta with a +-1 standard-error band over scenes.

Row 1's SSIM and LPIPS are the DISK convention (the metric map averaged over valid pixels
only), not the `frame` convention of the canonical cross-method tables, which averages the
map over the whole frame and so lets the black surround score ~1.0. On myscenes that is the
difference between SSIM 0.893 (here) and 0.951 (the store). The rings decompose the disk
convention; never read a store number off this panel.

THE THREE DATASETS ARE NEVER POOLED. myscenes (7 circular-frame fisheye scenes at -r 4),
FullCircle refit_rttpf (9 scenes, dual-lens rig, -r 4) and workshop_immervision (1
full-frame panomorph at native resolution) are different lenses, different frame
geometries and different noise regimes; a mean across them would destroy exactly the
contrast the figure exists to show. Each dataset is one colour and one panel line.

THE OUTER RING IS HATCHED IN THE SSIM AND LPIPS PANELS ON PURPOSE. Both metrics have
spatial support (11x11 window / ~212 px receptive field) and are computed on mask-zeroed
images, so at the rim they partly score black-against-black agreement. PSNR has no support
and is clean everywhere. Read the periphery off the PSNR panel.
"""

import json
import os

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# * W4: the figure now draws from `report_data.json` (key w3.rings_all), which is the single
# * packed payload; `rings_all.json` stays as the fallback so the script still runs standalone
# * before a pack. Same arrays either way -- pack.py embeds the file verbatim.
REPORT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "report_data.json")
SOURCE = os.path.join(REPO, "tmp", "w3_radial", "rings_all.json")

GROUPS = [
    ("myscenes", "myscenes (7 scenes, rttpf, circular, -r 4)", "#1f77b4"),
    ("fullcircle", "FullCircle refit_rttpf (9 scenes, rig, -r 4)", "#d62728"),
    ("others", "workshop_immervision (1 scene, panomorph, native)", "#2ca02c"),
]
METRICS = [
    ("psnr", "masked PSNR (dB)", "delta PSNR (dB)", False),
    ("ssim", "masked SSIM (disk convention)", "delta SSIM", True),
    ("lpips", "masked LPIPS, disk convention (lower better)", "delta LPIPS", True),
]


def collect(data, group, metric):
    """(absolute off, absolute noncentral, delta) stacked over the group's scenes."""
    off, nc = [], []
    for key, entry in data.items():
        if entry["dataset"] != group:
            continue
        off.append(entry["off"]["metrics"][metric]["rings"])
        nc.append(entry["noncentral"]["metrics"][metric]["rings"])
    if not off:
        return None
    off, nc = np.array(off, dtype=float), np.array(nc, dtype=float)
    return off, nc, nc - off


def draw(data, outdir=None):
    """`data` is the {dataset/scene: {...}} map of rings_all.json / report_data w3.rings_all."""
    n_rings = len(next(iter(data.values()))["off"]["metrics"]["psnr"]["rings"])
    x = np.arange(n_rings)

    fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.2))
    for col, (metric, ylabel_abs, ylabel_delta, contaminated) in enumerate(METRICS):
        top, bottom = axes[0][col], axes[1][col]
        for group, _label, colour in GROUPS:
            got = collect(data, group, metric)
            if got is None:
                continue
            off, nc, delta = got
            top.plot(x, off.mean(0), color=colour, linestyle="--", marker="o", ms=3.5, lw=1.3)
            top.plot(x, nc.mean(0), color=colour, linestyle="-", marker="o", ms=3.5, lw=1.8)
            mean = delta.mean(0)
            bottom.plot(x, mean, color=colour, marker="o", ms=4, lw=1.8)
            if delta.shape[0] > 1:
                err = delta.std(0, ddof=1) / np.sqrt(delta.shape[0])
                bottom.fill_between(x, mean - err, mean + err, color=colour, alpha=0.18, lw=0)
        bottom.axhline(0.0, color="0.35", lw=0.8)
        top.set_title(ylabel_abs, fontsize=10)
        top.set_ylabel(ylabel_abs, fontsize=9)
        bottom.set_ylabel(ylabel_delta + "  (noncentral - off)", fontsize=9)
        for ax in (top, bottom):
            ax.set_xticks(x)
            ax.set_xticklabels([f"{k}" for k in x], fontsize=8)
            ax.grid(alpha=0.25, lw=0.5)
            ax.tick_params(labelsize=8)
            if contaminated:
                ax.axvspan(n_rings - 1.5, n_rings - 0.5, facecolor="0.55", alpha=0.16,
                           lw=0, hatch="///", edgecolor="0.4")
        bottom.set_xlabel("equal-area ring (0 = centre)", fontsize=9)

    handles = [Line2D([], [], color=c, lw=2, label=lab) for _, lab, c in GROUPS]
    handles += [
        Line2D([], [], color="0.25", lw=1.3, ls="--", label="off"),
        Line2D([], [], color="0.25", lw=1.8, ls="-", label="noncentral"),
        matplotlib.patches.Patch(facecolor="0.5", alpha=0.3, hatch="///", edgecolor="0.4",
                                 label="outer ring: SSIM/LPIPS support reaches past the rim"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=3, fontsize=8.5,
               frameon=False, bbox_to_anchor=(0.5, 1.005))
    fig.suptitle("", y=1.0)
    fig.tight_layout(rect=(0, 0, 1, 0.90))

    outdir = outdir or os.path.join(REPO, "figures")
    os.makedirs(outdir, exist_ok=True)
    written = []
    for ext in ("svg", "pdf"):
        path = os.path.join(outdir, f"rings.{ext}")
        fig.savefig(path, bbox_inches="tight")
        written.append(path)
        print(f"wrote {path}")
    plt.close(fig)
    return written


def main():
    if os.path.exists(REPORT):
        draw(json.load(open(REPORT))["w3"]["rings_all"])
    else:
        draw(json.load(open(SOURCE)))


if __name__ == "__main__":
    main()
