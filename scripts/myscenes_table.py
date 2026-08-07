"""7-scene myscenes rttpf comparison table.

Produces the number this whole branch exists to move: the mean masked PSNR over the seven
myscenes, against SPaGS.

Both sides come from the *same* metric definition. The gray-side runs are scored by
`radial_eval.evaluate`, whose `disk_per_view_mean` was validated to reproduce the canonical
masked PSNR exactly (28.537 on out/tunnel_fisheye_baseline, i.e. gray's published tunnel
score). The baseline side is read from the canonical shared-mask aggregate that the
comparison viewer publishes, never from any repo's self-reported metric.

usage:
  python scripts/myscenes_table.py --runs-root /workspace/gray/tmp/r4 --suffix _noncentral
  python scripts/myscenes_table.py --runs-root ... --suffix _noncentral --compare-to SPaGS
"""

import argparse
import json
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from radial_eval import evaluate  # noqa: E402

SCENES = ["atrium", "classroom", "forest", "library", "reception", "tunnel", "workshop"]
CANONICAL = "/workspace/fisheye-baseline-viewer/data/perview_metrics.json"
MASK_PRESET = "0.95"


def reference_scores(method):
    "Per-scene masked PSNR from the canonical shared-mask aggregate."
    with open(CANONICAL) as handle:
        data = json.load(handle)
    scores = {}
    for scene in SCENES:
        try:
            scores[scene] = statistics.mean(data[scene][method][MASK_PRESET]["psnr"])
        except (KeyError, TypeError):
            scores[scene] = None
    return scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-root", required=True)
    parser.add_argument("--suffix", default="_noncentral", help="run dir is <root>/<scene><suffix>")
    parser.add_argument("--compare-to", default="SPaGS")
    parser.add_argument("--baseline-method", default="gray", help="canonical published gray row")
    parser.add_argument("--rings", type=int, default=6)
    args = parser.parse_args()

    published = reference_scores(args.baseline_method)
    rival = reference_scores(args.compare_to)

    print(
        f"{'scene':12s}{'gray(pub)':>11s}{'ours':>11s}{'d(ours)':>10s}"
        f"{args.compare_to:>11s}{'ours-rival':>12s}"
    )
    print("-" * 67)
    ours, missing = {}, []
    for scene in SCENES:
        run = os.path.join(args.runs_root, f"{scene}{args.suffix}")
        try:
            ours[scene] = evaluate(run, args.rings)["disk_per_view_mean"]
        except SystemExit:
            missing.append(scene)
            print(f"{scene:12s}{published[scene]:11.3f}{'--':>11s}{'':>10s}{rival[scene]:11.3f}")
            continue
        print(
            f"{scene:12s}{published[scene]:11.3f}{ours[scene]:11.3f}"
            f"{ours[scene] - published[scene]:+10.3f}{rival[scene]:11.3f}"
            f"{ours[scene] - rival[scene]:+12.3f}"
        )

    if missing:
        print(f"\nINCOMPLETE: {len(missing)} scene(s) missing ({', '.join(missing)}); "
              "the mean below is NOT the 7-scene number.")

    scored = [s for s in SCENES if s in ours]
    print("-" * 67)
    print(
        f"{'MEAN(' + str(len(scored)) + ')':12s}"
        f"{statistics.mean(published[s] for s in scored):11.3f}"
        f"{statistics.mean(ours[s] for s in scored):11.3f}"
        f"{statistics.mean(ours[s] - published[s] for s in scored):+10.3f}"
        f"{statistics.mean(rival[s] for s in scored):11.3f}"
        f"{statistics.mean(ours[s] - rival[s] for s in scored):+12.3f}"
    )
    if len(scored) == len(SCENES):
        target = statistics.mean(rival[s] for s in SCENES)
        got = statistics.mean(ours[s] for s in SCENES)
        verdict = "BEATS" if got > target else "does not beat"
        print(f"\n7-scene mean {got:.3f} {verdict} {args.compare_to} {target:.3f} "
              f"({got - target:+.3f} dB)")


if __name__ == "__main__":
    main()
