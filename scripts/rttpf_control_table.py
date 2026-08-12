"""Paired scene x rung table for the re-calibration control.

The question it answers: how much of the `noncentral` gain survives once the *baseline's own*
camera model is allowed to move? Every number is recomputed from the rendered PNGs by
`radial_eval.evaluate` under the shared masked protocol (`disk_per_view_mean`, the
convention the published myscenes rows use), never read from a run's `psnr.csv` -- the two
disagreeing is the signal that catches a broken render path, so they must stay independent.

usage:
  python scripts/rttpf_control_table.py --root tmp/r8_control --rungs off rttpf noncentral
  python scripts/rttpf_control_table.py --root tmp/final --pattern '{scene}_{rung}' \
      --rungs off rttpf --scenes tunnel workshop
"""

import argparse
import json
import math
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from radial_eval import evaluate  # noqa: E402

SCENES = ["atrium", "classroom", "forest", "library", "reception", "tunnel", "workshop"]


def live_psnr(run_dir):
    "Test PSNR from the training loop itself -- only ever used as a cross-check."
    path = os.path.join(run_dir, "psnr.csv")
    if not os.path.exists(path):
        return None
    with open(path) as handle:
        rows = [line.split() for line in handle if line.strip()][1:]
    return float(rows[-1][2]) if rows else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--pattern", default="{scene}_{rung}")
    parser.add_argument("--rungs", nargs="+", default=["off", "rttpf", "noncentral"])
    parser.add_argument("--scenes", nargs="+", default=SCENES)
    parser.add_argument("--rings", type=int, default=6)
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args()

    table = {}
    for scene in args.scenes:
        for rung in args.rungs:
            run = os.path.join(args.root, args.pattern.format(scene=scene, rung=rung))
            try:
                result = evaluate(run, args.rings)
            except (FileNotFoundError, ValueError, IndexError) as error:
                print(f"  -- {scene}/{rung}: {type(error).__name__} ({run})", flush=True)
                continue
            result["live_psnr"] = live_psnr(run)
            table.setdefault(scene, {})[rung] = result

    reference = args.rungs[0]
    header = f"{'scene':11s}" + "".join(f"{rung[:12]:>13s}" for rung in args.rungs)
    header += "".join(f"{('d ' + rung)[:12]:>13s}" for rung in args.rungs[1:])
    print("\n" + header)
    print("-" * len(header))
    columns = {rung: [] for rung in args.rungs}
    deltas = {rung: [] for rung in args.rungs[1:]}
    for scene in args.scenes:
        entries = table.get(scene, {})
        if reference not in entries:
            continue
        line = f"{scene:11s}"
        for rung in args.rungs:
            value = entries.get(rung)
            line += f"{value['disk_per_view_mean']:13.3f}" if value else f"{'--':>13s}"
            if value:
                columns[rung].append(value["disk_per_view_mean"])
        for rung in args.rungs[1:]:
            value = entries.get(rung)
            if value:
                delta = value["disk_per_view_mean"] - entries[reference]["disk_per_view_mean"]
                deltas[rung].append(delta)
                line += f"{delta:+13.3f}"
            else:
                line += f"{'--':>13s}"
        print(line)

    complete = [s for s in args.scenes if all(r in table.get(s, {}) for r in args.rungs)]
    print("-" * len(header))
    if complete:
        line = f"{'mean(' + str(len(complete)) + ')':11s}"
        for rung in args.rungs:
            values = [table[s][rung]["disk_per_view_mean"] for s in complete]
            line += f"{statistics.mean(values):13.3f}"
        for rung in args.rungs[1:]:
            paired = [
                table[s][rung]["disk_per_view_mean"] - table[s][reference]["disk_per_view_mean"]
                for s in complete
            ]
            line += f"{statistics.mean(paired):+13.3f}"
        print(line)
        for rung in args.rungs[1:]:
            paired = [
                table[s][rung]["disk_per_view_mean"] - table[s][reference]["disk_per_view_mean"]
                for s in complete
            ]
            if len(paired) > 1:
                spread = statistics.stdev(paired)
                print(
                    f"  {rung} - {reference}: mean {statistics.mean(paired):+.3f} dB, "
                    f"std {spread:.3f}, stderr {spread / math.sqrt(len(paired)):.3f}, "
                    f"positive on {sum(v > 0 for v in paired)}/{len(paired)}"
                )

    # * The live-vs-rerendered cross-check. A disagreement above ~0.1 dB means the PNGs and
    # * the training loop are not looking at the same camera (IMPLEMENTATION.md GOTCHAS).
    worst = None
    for scene, entries in table.items():
        for rung, value in entries.items():
            if value["live_psnr"] is None:
                continue
            gap = abs(value["live_psnr"] - value["disk_per_view_mean"])
            if worst is None or gap > worst[0]:
                worst = (gap, scene, rung, value["live_psnr"], value["disk_per_view_mean"])
    if worst:
        print(
            f"\nlive vs re-rendered, worst: {worst[1]}/{worst[2]} "
            f"{worst[3]:.3f} vs {worst[4]:.3f} ({worst[0]:.3f} dB)"
            + ("  <-- INVESTIGATE" if worst[0] > 0.15 else "")
        )

    if args.json_out:
        with open(args.json_out, "w") as handle:
            json.dump(table, handle, indent=1)
        print(f"wrote {args.json_out}")


if __name__ == "__main__":
    main()
