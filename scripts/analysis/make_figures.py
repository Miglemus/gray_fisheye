#!/usr/bin/env python
"""Draw EVERY Phase-1 figure from `report_data.json`, and from nothing else.

    python scripts/analysis/make_figures.py                 # all figures, svg + pdf
    python scripts/analysis/make_figures.py --only rings ladder
    python scripts/analysis/make_figures.py --report /path/to/report_data.json

The contract this file exists to enforce: a figure may read the packed payload and no other
file. Not a run directory, not a COLMAP model, not another analysis JSON. Delete `figures/`,
run this, and every panel comes back -- that is the test, and `run_phase1.py --verify` runs it.

The drawing code for the three W1 panels and for W2 lives in `dose_response.py` and
`ranking_flip.py`; it is imported and handed the embedded payload rather than copied, so a
figure cannot drift from the statistics it illustrates. What changed for W4 is only where the
data comes from (the pack, not each script's own JSON) and that both svg and pdf are written.

FOUR THINGS TO KNOW BEFORE READING THE OUTPUT
  * `dose_response.svg` panel B (E_learned) is drawn in red-flag styling on purpose: it is
    CIRCULAR, the optimiser's own output, and exists only to show the shape of the law.
  * `rings.svg` row 1 is the DISK convention (metric map averaged over valid pixels only),
    not the `frame` convention of the cross-method stores. On myscenes that is SSIM 0.893
    against the store's 0.951; the black surround scores ~1.0 in the store convention.
  * `ladder.svg` never puts a pinhole dB next to a fisheye dB. Only each rung's delta against
    its OWN scene's `off` is shown, and the two families sit in two panels.
  * `field.svg` is in MICRORADIANS, never pixels. Converting an angle to pixels needs the
    local plate scale S(theta) = dr/dtheta, which is not fx on a fisheye (0.633 fx at the
    evaluated edge on these lenses) and which the report page only carries in its
    polynomial-only form. A2's `plate_scale.json` has the true one; the figure sidesteps the
    whole question by staying in angle.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
WORKTREE = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)

REPORT = os.path.join(HERE, "report_data.json")
FIGDIR = os.path.join(WORKTREE, "figures")
NOISE_FLOOR_DB = 0.068


def _mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return matplotlib, plt


def both(stem):
    return [os.path.join(FIGDIR, f"{stem}.svg"), os.path.join(FIGDIR, f"{stem}.pdf")]


# --------------------------------------------------------------------------------------- #
#  W1 -- three panels, drawn by dose_response.py from the embedded payload                   #
# --------------------------------------------------------------------------------------- #
def figure_dose_response(report):
    import dose_response as dr
    out = []
    for path in both("dose_response"):
        dr.figure_dose_response(report["w1"], path)
        out.append(path)
    return out


def figure_estimator_agreement(report):
    import dose_response as dr
    out = []
    for path in both("estimator_agreement"):
        dr.figure_estimator_agreement(report["w1"], path)
        out.append(path)
    return out


def figure_estimator_lens_level(report):
    import dose_response as dr
    out = []
    for path in both("estimator_lens_level"):
        dr.figure_lens_level(report["w1"], path)
        out.append(path)
    return out


# --------------------------------------------------------------------------------------- #
#  W2 -- six method-pair panels, drawn by ranking_flip.py                                    #
# --------------------------------------------------------------------------------------- #
def figure_ranking_flip(report):
    import ranking_flip as rf
    out = []
    for path in both("ranking_flip"):
        rf.make_figure(report["w2"], path)
        out.append(path)
    return out


# --------------------------------------------------------------------------------------- #
#  W3 -- rings, drawn by make_rings_figure.py                                                #
# --------------------------------------------------------------------------------------- #
def figure_rings(report):
    import make_rings_figure as mrf
    return mrf.draw(report["w3"]["rings_all"], FIGDIR)


# --------------------------------------------------------------------------------------- #
#  the ablation ladder, both camera families, + the regularisation audit                     #
# --------------------------------------------------------------------------------------- #
def figure_ladder(report):
    _, plt = _mpl()
    lad = report["ladder"]
    order = ["off", "tilt", "radial", "ana", "central_matched", "noncentral_no_ana",
             "z_only", "noncentral"]
    label = {"off": "off", "tilt": "tilt", "radial": "radial", "ana": "ana",
             "central_matched": "central\nmatched", "noncentral_no_ana": "noncentral\nno ana",
             "z_only": "z_only", "noncentral": "noncentral"}
    colour = {"z_only": "#c9432a", "noncentral": "#1f6feb", "noncentral_no_ana": "#5b8fd6",
              "central_matched": "#6f6f6f", "ana": "#9c9c9c", "radial": "#b9b9b9",
              "tilt": "#d0d0d0", "off": "#000000"}

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.4),
                             gridspec_kw={"width_ratios": [1.35, 1.0, 0.95]})

    # ---- A. fisheye ladder, per scene + mean ---------------------------------------- #
    ax = axes[0]
    per_scene = lad["fisheye"]["per_scene"]
    rungs = [r for r in order if any(r in v for v in per_scene.values())]
    xs = np.arange(len(rungs))
    for scene, entry in sorted(per_scene.items()):
        ys = [entry[r]["delta_vs_off_db"] if r in entry else np.nan for r in rungs]
        ax.plot(xs, ys, marker="o", ms=3.5, lw=0.9, alpha=0.45, color="#666666")
        ax.annotate(scene, (xs[-1], ys[-1]), xytext=(4, 0), textcoords="offset points",
                    fontsize=6.2, color="#666666", va="center")
    m3 = [lad["fisheye"]["summary"][r]["mean_delta_db_over_3_full_ladder_scenes"] for r in rungs]
    m7 = [lad["fisheye"]["summary"][r]["mean_delta_db_over_all_scenes_present"] for r in rungs]
    ax.plot(xs, [np.nan if v is None else v for v in m3], marker="s", ms=6, lw=2.2,
            color="#c9432a", label="mean over the 3 full-ladder scenes")
    ax.plot(xs, [np.nan if v is None else v for v in m7], marker="^", ms=6, lw=1.4,
            ls="--", color="#1f6feb", label="mean over every scene where the rung exists")
    ax.set_xticks(xs)
    ax.set_xticklabels([label[r] for r in rungs], fontsize=7.2)
    ax.set_title("A. fisheye — myscenes, rttpf, masked r=0.95, -r 4", fontsize=9.5, loc="left")
    ax.set_ylabel("PSNR delta vs the same scene's `off`  [dB]", fontsize=8.5)
    ax.margins(x=0.10, y=0.16)

    # ---- B. pinhole ladder ------------------------------------------------------------ #
    ax = axes[1]
    pin = lad["pinhole"]
    rungs_p = [r for r in order if any(r in v for v in pin.values())]
    xs = np.arange(len(rungs_p))
    for scene, entry in sorted(pin.items()):
        ys = [entry[r]["delta_vs_off_db"] if r in entry else np.nan for r in rungs_p]
        ax.plot(xs, ys, marker="o", ms=5, lw=1.6, label=f"{scene} (pinhole)")
    ax.set_xticks(xs)
    ax.set_xticklabels([label[r] for r in rungs_p], fontsize=7.2)
    ax.set_title("B. pinhole — mip-NeRF 360, full frame", fontsize=9.5, loc="left")
    ax.set_ylabel("PSNR delta vs `off`  [dB]", fontsize=8.5)
    ax.margins(x=0.10, y=0.22)

    for ax in axes[:2]:
        ax.axhspan(-NOISE_FLOOR_DB, NOISE_FLOOR_DB, color="#b0453a", alpha=0.12, lw=0,
                   zorder=0)
        ax.axhline(0.0, color="#999999", lw=0.7)
        ax.grid(alpha=0.2, lw=0.5)
        ax.tick_params(labelsize=8)
        ax.legend(fontsize=7, loc="upper left", framealpha=0.9)

    # ---- C. the 2x2 interaction + the regularisation audit ---------------------------- #
    ax = axes[2]
    cells = [("pinhole\n(bicycle)", "z_only", pin.get("bicycle", {}).get("z_only", {}).get("delta_vs_off_db")),
             ("pinhole\n(bicycle)", "central_matched", pin.get("bicycle", {}).get("central_matched", {}).get("delta_vs_off_db")),
             ("fisheye\n(3 scenes)", "z_only", lad["fisheye"]["summary"]["z_only"]["mean_delta_db_over_3_full_ladder_scenes"]),
             ("fisheye\n(3 scenes)", "central_matched", lad["fisheye"]["summary"]["central_matched"]["mean_delta_db_over_3_full_ladder_scenes"])]
    fams = ["pinhole\n(bicycle)", "fisheye\n(3 scenes)"]
    width = 0.36
    for i, rung in enumerate(("z_only", "central_matched")):
        vals = [v for f, r, v in cells if r == rung]
        ax.bar(np.arange(2) + (i - 0.5) * width, vals, width,
               color=colour[rung], label=f"{rung} ({'8' if rung == 'z_only' else '180'} params)")
        for x, v in zip(np.arange(2) + (i - 0.5) * width, vals):
            ax.annotate(f"{v:+.3f}", (x, v), ha="center",
                        va="bottom" if v >= 0 else "top",
                        xytext=(0, 3 if v >= 0 else -10), textcoords="offset points",
                        fontsize=7.5)
    ax.axhspan(-NOISE_FLOOR_DB, NOISE_FLOOR_DB, color="#b0453a", alpha=0.12, lw=0, zorder=0)
    ax.axhline(0.0, color="#999999", lw=0.7)
    ax.set_xticks(np.arange(2))
    ax.set_xticklabels(fams, fontsize=8.5)
    ax.set_ylabel("PSNR delta vs `off`  [dB]", fontsize=8.5)
    ax.legend(fontsize=7.5, loc="lower left", framealpha=0.9)  # * upper left hides a bar label
    ax.grid(alpha=0.2, lw=0.5, axis="y")
    ax.tick_params(labelsize=8)
    ax.margins(y=0.18)
    reg = lad["regularisation"]["verdict"]
    ax.set_title("C. the 2x2 — the order INVERTS between families", fontsize=9.5, loc="left")

    fig.suptitle("Ablation ladder: which term of the camera model carries the gain, and is the "
                 "comparison interpretable?   A pinhole camera cannot have a non-central "
                 "pupil,\nso `z_only` on pinhole is a negative control whose correct answer is "
                 "zero. The two families are NEVER averaged: only deltas against each scene's "
                 "own `off`.", fontsize=9.6, y=0.995)
    fig.text(0.012, 0.045,
             "shaded band = the measured 0.068 dB run-to-run noise floor;  n = 1 run per cell, "
             "no seeds, so any |delta| inside the band is a zero.\n"
             "regularisation audit (exact penalty recomputed from the saved weights): the "
             "L2 + curvature term is "
             f"{reg['penalty_over_l1_percent_min']:.4f}–{reg['penalty_over_l1_percent_max']:.3f} % "
             f"of the data loss over {reg['n_runs']} runs, and `z_only` (8 regularised params) "
             f"pays {reg['sign_is_inverted']['workshop']['z_only_pays_x_more']:.1f}x MORE than "
             "`central_matched` (180) on workshop.\n"
             "The 'bigger rungs are penalised more' confound is therefore falsified twice over: "
             "the term is inert, and its sign is inverted.",
             fontsize=7.4, va="bottom", color="#444444", linespacing=1.5)
    fig.tight_layout(rect=(0, 0.155, 1, 0.885))
    out = []
    for path in both("ladder"):
        fig.savefig(path, bbox_inches="tight")
        out.append(path)
    plt.close(fig)
    return out


# --------------------------------------------------------------------------------------- #
#  the learned residual field, in microradians                                               #
# --------------------------------------------------------------------------------------- #
def figure_field(report):
    _, plt = _mpl()
    page = report["report_page"]
    theta = np.asarray(page["theta_deg"], float)
    scenes = sorted(page["scenes"])
    cycle = plt.get_cmap("tab10").colors          # * distinct hues; viridis buries `workshop`
    colours = {s: cycle[i % len(cycle)] for i, s in enumerate(scenes)}

    fig, axes = plt.subplots(1, 3, figsize=(14.4, 4.4))
    edge = 85.5

    ax = axes[0]
    for s in scenes:
        ch = np.asarray(page["scenes"][s]["dtheta_channels"], float)
        ax.plot(theta, ch[0] * 1e6, lw=1.7, color=colours[s], label=s)
    ax.set_title("A. radial channel  k=0  of the learned $\\Delta\\theta$", fontsize=9.5,
                 loc="left")
    ax.set_ylabel("$\\Delta\\theta$  [µrad]", fontsize=9)

    ax = axes[1]
    for s in scenes:
        ch = np.asarray(page["scenes"][s]["dtheta_channels"], float)
        amp = np.sqrt(ch[1] ** 2 + ch[2] ** 2 + ch[3] ** 2 + ch[4] ** 2)
        ax.plot(theta, amp * 1e6, lw=1.7, color=colours[s])
    ax.set_title("B. anamorphic channels  k=1,2  (RMS over the four harmonics)",
                 fontsize=9.5, loc="left")
    ax.set_ylabel("azimuthal modulation of $\\Delta\\theta$  [µrad]", fontsize=9)

    ax = axes[2]
    for s in scenes:
        z = np.asarray(page["scenes"][s]["z_profile"], float)
        ax.plot(theta, z, lw=1.7, color=colours[s])
    ax.set_title("C. the non-central profile  z($\\theta$)", fontsize=9.5, loc="left")
    ax.set_ylabel("z  [scene radii]   (gauge: z(0) = 0)", fontsize=9)

    for ax in axes:
        ax.axvline(edge, color="#b0453a", ls=":", lw=1.1)
        ax.axhline(0.0, color="#999999", lw=0.6)
        ax.set_xlabel("field angle $\\theta$  [deg]", fontsize=9)
        ax.grid(alpha=0.2, lw=0.5)
        ax.tick_params(labelsize=8)
    axes[0].legend(fontsize=7, ncol=2, loc="upper left")
    axes[2].annotate("evaluated edge  85.5° = 0.95 · 90°", (edge, 0.02), xycoords=("data",
                     "axes fraction"), xytext=(-6, 0), textcoords="offset points",
                     fontsize=7, color="#b0453a", ha="right")

    fig.suptitle("The learned camera-model residual — 7 myscenes scenes, ONE physical lens, "
                 "seven independent COLMAP fits — in ANGLE, never in pixels.\n"
                 "The spline is parameterised on $\\theta/(\\pi/2)$, so it is defined out to "
                 "90° on every scene; only $\\theta \\leq 85.5°$ is ever evaluated, and "
                 "everything past the dotted line is extrapolation the loss never saw.",
                 fontsize=9.6, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    out = []
    for path in both("field"):
        fig.savefig(path, bbox_inches="tight")
        out.append(path)
    plt.close(fig)
    return out


# --------------------------------------------------------------------------------------- #
#  the qualitative crops                                                                     #
# --------------------------------------------------------------------------------------- #
def figure_crops(report):
    from PIL import Image
    _, plt = _mpl()
    crops = report["report_page"].get("crops") or {}
    if not crops:
        return []
    scenes = sorted(crops)
    panels = [("gt", "ground truth"), ("base", "off"), ("nc", "noncentral"),
              ("err_base", "squared error, off"), ("err_nc", "squared error, noncentral")]

    # * the panels are square crops, so the axes grid has to be square too or matplotlib
    # * pads each cell with half a screen of white.
    fig, axes = plt.subplots(len(scenes), len(panels),
                             figsize=(1.95 * len(panels), 1.95 * len(scenes) + 1.1),
                             gridspec_kw={"wspace": 0.06, "hspace": 0.16})
    axes = np.atleast_2d(axes)
    for row, scene in enumerate(scenes):
        entry = crops[scene]
        for col, (key, title) in enumerate(panels):
            ax = axes[row][col]
            raw = base64.b64decode(entry["panels"][key].split(",", 1)[1])
            ax.imshow(np.asarray(Image.open(io.BytesIO(raw)).convert("RGB")))
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0:
                ax.set_title(title, fontsize=9)
            if col == 0:
                ax.set_ylabel(f"{scene}\n{entry['view']}", fontsize=8)
        axes[row][-1].annotate(
            f"+{entry['gain_db']:.2f} dB on this view,  local SSE −{entry['err_drop_pct']:.0f} %,"
            f"  patch at $\\rho$ = {entry['rho']:.2f}",
            (1.0, -0.06), xycoords="axes fraction", ha="right", va="top", fontsize=7.5,
            color="#444444")

    fig.suptitle("Where the camera model changes the picture.\n"
                 "View and window are chosen MECHANICALLY — best-gain test view, then the\n"
                 "224 px window with the largest drop in summed squared error over the outer\n"
                 "disk (ρ > 0.55). Nothing here can be cherry-picked by eye. The two error\n"
                 "panels of a row share one colour scale; display gamma only on the RGB ones.",
                 fontsize=9.2, y=0.998)
    fig.subplots_adjust(top=1.0 - 1.05 / fig.get_size_inches()[1], bottom=0.035,
                        left=0.06, right=0.99)
    out = []
    for path in both("crops"):
        fig.savefig(path, dpi=150, bbox_inches="tight")
        out.append(path)
    plt.close(fig)
    return out


FIGURES = {
    "dose_response": figure_dose_response,
    "estimator_agreement": figure_estimator_agreement,
    "estimator_lens_level": figure_estimator_lens_level,
    "ranking_flip": figure_ranking_flip,
    "rings": figure_rings,
    "ladder": figure_ladder,
    "field": figure_field,
    "crops": figure_crops,
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", default=REPORT)
    ap.add_argument("--only", nargs="*", choices=sorted(FIGURES), default=None)
    args = ap.parse_args()

    if not os.path.exists(args.report):
        raise SystemExit(f"{args.report} is missing -- run `python scripts/analysis/pack.py`")
    report = json.load(open(args.report))
    os.makedirs(FIGDIR, exist_ok=True)

    written, failed = [], []
    for name in (args.only or list(FIGURES)):
        try:
            paths = FIGURES[name](report)
        except Exception as exc:                                # keep going: one bad panel
            failed.append((name, repr(exc)))                    # must not cost the other seven
            print(f"  !! {name}: {exc!r}")
            continue
        written += paths
        print(f"  ok  {name}: " + ", ".join(os.path.relpath(p, WORKTREE) for p in paths))
    print(f"\n{len(written)} file(s) written to {FIGDIR}")
    if failed:
        print(f"{len(failed)} figure(s) FAILED: " + ", ".join(n for n, _ in failed))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
