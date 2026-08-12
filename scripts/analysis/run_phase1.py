#!/usr/bin/env python
"""THE single command that reproduces Phase 1 end to end.

    python scripts/analysis/run_phase1.py              # everything missing, then pack, then figures
    python scripts/analysis/run_phase1.py --verify     # + delete figures/ first and prove they come back
    python scripts/analysis/run_phase1.py --force pack figures
    python scripts/analysis/run_phase1.py --force all  # recompute every CPU stage (~35 min)
    python scripts/analysis/run_phase1.py --list       # what each stage costs and what it writes

WHAT IT DOES, IN ORDER
  1. runs every stage whose OUTPUT IS MISSING (or every stage named in `--force`);
  2. `pack.py`  -> scripts/analysis/report_data.json, the single joined payload;
  3. `make_figures.py` -> figures/*.{svg,pdf}, drawn from that payload and nothing else.

By default it only computes what is absent. That is deliberate: several stages read hundreds
of rendered PNGs, and re-running them changes nothing as long as the runs on disk have not
changed. `--force` is how you say the runs DID change.

THE GPU RULE. Three W3 stages need a GPU (LPIPS is a VGG forward pass) and this script will
NEVER launch them. If their output is missing it prints the exact `pueue add` line and stops
that stage, because the house rule is absolute: every GPU task goes through the queue, no
exceptions, and no task belonging to another session is ever touched. Everything else here is
pure CPU and runs with CUDA_VISIBLE_DEVICES="" so that a GPU cannot be taken by accident.

TWO STAGES ARE NOT RE-RUNNABLE FROM ZERO HERE ON PURPOSE
  * Phase 1 retrains NOTHING. Every stage reads finished runs; if a run is missing the stage
    says so and skips its rows rather than training anything.
  * `dose_response.py` and `ranking_flip.py` carry timestamped PRE-REGISTRATIONS
    (`dose_response_prereg.md`, `preregistration_ranking_flip_*.md`) written before any fit.
    Re-running them is fine -- both are deterministic, seeded 20260810 -- but the
    pre-registration files must never be regenerated after the fact.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
WORKTREE = os.path.abspath(os.path.join(HERE, "..", ".."))
W3 = os.path.join(WORKTREE, "tmp", "w3_radial")
FIGDIR = os.path.join(WORKTREE, "figures")
PY = sys.executable

# stage -> (output that proves it ran, argv, ~cost, needs GPU, one-line what)
STAGES: dict[str, tuple] = {
    # ---- A1 / A2: the training-free camera statistics ------------------------------- #
    "sfm_residual": (f"{HERE}/sfm_residual.json", [PY, f"{HERE}/sfm_residual.py", "--workers", "5"],
                     "~4 min", False,
                     "SfM reprojection residual of 62 tracks, in urad (A1)"),
    "plate_scale": (f"{HERE}/plate_scale.json", [PY, f"{HERE}/plate_scale.py"],
                    "~45 s", False,
                    "S(theta) = dr/dtheta per camera, evaluated-disk edge, urad/px (A2)"),
    "calib_consistency": (f"{HERE}/calib_consistency.json", [PY, f"{HERE}/calib_consistency.py"],
                          "~30 s", False, "E_shared / calibration disagreement, legacy px"),
    "calib_consistency_urad": (f"{HERE}/calib_consistency_urad.json",
                               [PY, f"{HERE}/calib_consistency_urad.py"], "~30 s", False,
                               "the same, natively in urad (use this one)"),
    "residual_expressible": (f"{HERE}/residual_expressible.json",
                             [PY, f"{HERE}/residual_expressible.py"], "~20 s", False,
                             "how much of the learned residual COLMAP's own k1..k4 could express"),
    # ---- W1 / W2 -------------------------------------------------------------------- #
    "dose_response": (f"{HERE}/dose_response.json",
                      [PY, f"{HERE}/dose_response.py", "--workers", "5"], "~7 min", False,
                      "W1: paired gains, the three estimators, the fits (PRE-REGISTERED)"),
    "ranking_flip": (f"{HERE}/ranking_flip.json", [PY, f"{HERE}/ranking_flip.py"],
                     "~4 min", False,
                     "W2: six method-pair regressions + LODO (PRE-REGISTERED)"),
    # ---- W3: the GPU ones ----------------------------------------------------------- #
    "rings_all": (f"{W3}/rings_all.json", [PY, f"{HERE}/rings_all.py"], "~25 min", True,
                  "W3: PSNR/SSIM/LPIPS per equal-area ring, off vs noncentral"),
    "rings_stats": (f"{W3}/rings_stats.json", [PY, f"{HERE}/rings_stats.py"], "~10 s", False,
                    "W3: the paired rim-minus-centre statistics on rings_all"),
    "mask_radius_sweep": (f"{W3}/sweep_myscenes_rttpf.json",
                          [PY, f"{HERE}/mask_radius_sweep.py", "--track", "all"],
                          "~3 h", True,
                          "W3: every method re-evaluated at r = 0.85 / 0.95 / 1.00"),
    "annulus_audit": (f"{W3}/annulus_audit.json", [PY, f"{HERE}/annulus_audit.py"],
                      "~10 min", False,
                      "W3: who renders black in the 0.95->1.00 annulus, and against which GT"),
    # ---- the per-scene / report-page producers -------------------------------------- #
    "optics": (f"{HERE}/optics.json", [PY, f"{HERE}/optics.py"], "~6 min", False,
               "learned profiles, caustics, depth histograms per scene"),
    "fields": (f"{HERE}/fields.json", [PY, f"{HERE}/fields.py"], "~10 s", False,
               "all five azimuthal channels of the learned residual, from the checkpoints"),
    "rings": (f"{HERE}/rings.json", [PY, f"{HERE}/collect.py"], "~4 min", False,
              "per-view and 8-ring masked PSNR for gray / noncentral / SPaGS"),
    "rungs": (f"{HERE}/rungs.json", [PY, f"{HERE}/rungs.py"], "~6 min", False,
              "the six-rung fisheye ablation ladder, per scene, with rings"),
    "subtractive": (f"{HERE}/subtractive.json", [PY, f"{HERE}/subtractive.py"], "~6 min", False,
                    "the same ladder through a second entry point (cross-checked in pack)"),
    "crops": (f"{HERE}/crops.json",
              [PY, f"{HERE}/crops.py", "workshop", "reception", "tunnel"], "~3 min", False,
              "the mechanically-chosen qualitative crops, base64 WEBP"),
    "ladder": (f"{HERE}/ladder.json", [PY, f"{HERE}/ladder.py"], "~2 min", False,
               "pinhole ladder + the 2x2 interaction + the regularisation audit"),
    # ---- the join and the figures ---------------------------------------------------- #
    "pack": (f"{HERE}/report_data.json", [PY, f"{HERE}/pack.py"], "~15 s", False,
             "THE join: every measurement above into one report_data.json"),
    "figures": (f"{FIGDIR}/dose_response.svg", [PY, f"{HERE}/make_figures.py"], "~40 s", False,
                "every figure, svg + pdf, from report_data.json alone"),
}
ALWAYS = ["pack", "figures"]          # cheap, and the whole point: always re-run
GPU_HINT = ("pueue add --print-task-id -- 'cd {wt} && CUDA_DEVICE_ORDER=PCI_BUS_ID "
            "CUDA_VISIBLE_DEVICES=<free gpu> {cmd}'")


def run(name: str, argv: list[str]) -> float:
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = ""            # CPU stages cannot touch a GPU, by construction
    env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    env["MPLBACKEND"] = "Agg"
    t0 = time.time()
    print(f"\n=== {name}  ({' '.join(os.path.basename(a) for a in argv[1:2])}) "
          + "=" * max(0, 60 - len(name)), flush=True)
    proc = subprocess.run(argv, cwd=WORKTREE, env=env)
    dt = time.time() - t0
    if proc.returncode != 0:
        raise SystemExit(f"stage `{name}` failed with exit code {proc.returncode}")
    print(f"--- {name} done in {dt:.0f} s", flush=True)
    return dt


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", nargs="*", default=[],
                    help="stage names to recompute even if their output exists, or `all`")
    ap.add_argument("--only", nargs="*", default=None,
                    help="run just these stages (still followed by pack + figures)")
    ap.add_argument("--verify", action="store_true",
                    help="delete figures/ before drawing, to prove they regenerate")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    if args.list:
        print(f"{'stage':22s}{'cost':>8s}  {'gpu':>4s}  output / what it is")
        for name, (out, _argv, cost, gpu, what) in STAGES.items():
            mark = "yes" if gpu else "-"
            here = "present" if os.path.exists(out) else "MISSING"
            print(f"{name:22s}{cost:>8s}  {mark:>4s}  [{here}] {os.path.relpath(out, WORKTREE)}"
                  f"\n{'':38s}{what}")
        return 0

    force = set(args.force)
    if "all" in force:
        force = set(STAGES)
    todo = list(args.only) if args.only else [n for n in STAGES if n not in ALWAYS]
    todo += [n for n in ALWAYS if n not in todo]

    if args.verify and os.path.isdir(FIGDIR):
        removed = sorted(f for f in os.listdir(FIGDIR) if f.endswith((".svg", ".pdf", ".png")))
        for f in removed:
            os.remove(os.path.join(FIGDIR, f))
        print(f"--verify: removed {len(removed)} file(s) from {FIGDIR}")

    skipped, blocked, ran = [], [], []
    for name in todo:
        out, argv, cost, gpu, what = STAGES[name]
        needed = name in force or not os.path.exists(out) or name in ALWAYS
        if not needed:
            skipped.append(name)
            continue
        if gpu:
            blocked.append((name, out, argv, cost))
            continue
        run(name, argv)
        ran.append(name)

    print("\n" + "=" * 78)
    print(f"ran     : {', '.join(ran) if ran else '-'}")
    print(f"skipped : {', '.join(skipped) if skipped else '-'}   (output already on disk; "
          "use --force <stage> to redo)")
    for name, out, argv, cost in blocked:
        cmd = " ".join(argv).replace(PY, "python")
        print(f"\nBLOCKED : `{name}` needs a GPU ({cost}) and its output is missing:\n"
              f"          {out}\n"
              "          this script never launches a GPU task. Queue it yourself:\n"
              "          " + GPU_HINT.format(wt=WORKTREE, cmd=cmd) + "\n"
              "          then `pueue wait <id>` and re-run run_phase1.py.")
    if blocked:
        print("\n(report_data.json was still packed, with those sections absent; "
              "`provenance` records exactly which.)")

    figs = sorted(f for f in os.listdir(FIGDIR) if f.endswith((".svg", ".pdf"))) \
        if os.path.isdir(FIGDIR) else []
    print(f"\nfigures ({len(figs)}): " + ", ".join(figs))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
