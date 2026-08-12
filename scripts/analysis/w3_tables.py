"""W3 tables in markdown, from the two JSONs W3 produced. No GPU, no re-evaluation.

    python scripts/analysis/w3_tables.py
        tmp/w3_radial/rings_all.json          -> tmp/w3_radial/rings_table.md
        tmp/w3_radial/sweep_<track>.json      -> tmp/w3_radial/mask_radius_robustness.md
                                              -> tmp/w3_radial/mask_radius_sweep.csv

WHAT THE ROBUSTNESS TABLE ACTUALLY TESTS, AND WHAT IT CANNOT
---------------------------------------------------------------------------------------
For each (track, scene) it ranks the methods at r = 0.85, 0.95 and 1.00 and reports
Kendall's tau between the r=0.95 ranking (canonical) and each of the other two. tau = 1
means the sweep did not reorder anything.

Three ways a tau of 1.0 can be vacuous, all of them checked and reported:
  * `identical_masks` -- the three radii are the SAME mask (the polynomial-inversion
    criterion binds before the theta cutoff). Then stability is arithmetic, not evidence.
    Detected from `valid_fraction`.
  * `n_methods < 3` -- Kendall's tau on two methods can only be +-1, and on one is
    undefined. Those rows are reported as `n/a`.
  * r=1.00 scores a 0.95-1.00 annulus in which several methods write EXACT BLACK. Whether
    that inflates or penalises them depends on the evaluator's GT source, and both
    directions occur across the four stores -- so the verdict is taken from
    `annulus_audit.json`, per track, and those rows are excluded from the tau summary.
    See the verdict table at the top of the generated file.
"""

import csv
import glob
import itertools
import json
import os

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
W3 = os.path.join(REPO, "tmp", "w3_radial")
RADII = ["0.85", "0.95", "1.00"]
HIGHER = {"PSNR": True, "SSIM": True, "LPIPS": False}

GROUP_LABEL = {
    "myscenes": "myscenes (7 scenes, rttpf, circular frame, -r 4)",
    "myscenes_alt": "myscenes, ALTERNATIVE `off` run (not part of the myscenes mean)",
    "fullcircle": "FullCircle refit_rttpf (9 scenes, dual-lens rig, -r 4)",
    "others": "workshop_immervision (1 scene, panomorph, native resolution)",
}


# --------------------------------------------------------------------------------------
# ring table
# --------------------------------------------------------------------------------------


def ring_table(lines):
    path = os.path.join(W3, "rings_all.json")
    if not os.path.exists(path):
        lines.append("_rings_all.json missing_\n")
        return
    data = json.load(open(path))
    n_rings = len(next(iter(data.values()))["off"]["metrics"]["psnr"]["rings"])

    lines.append("# W3.1 -- ring-resolved PSNR / SSIM / LPIPS, `off` vs `noncentral`\n")
    lines.append("Equal-area annuli in pixel radius, ring 0 = centre. Every number is "
                 "recomputed from the rendered PNGs by `scripts/radial_eval.py`; no run was "
                 "retrained or re-rendered.\n")
    lines.append("> The OUTERMOST ring of the SSIM and LPIPS rows is contaminated: both "
                 "metrics have spatial support and are computed on mask-zeroed images, so at "
                 "the rim they partly score black against black. PSNR has no support and is "
                 "clean everywhere. Quote PSNR at the periphery.\n")
    lines.append("> LPIPS' deepest receptive field (~212 px) is wider than a ring (~70 px on "
                 "myscenes), so its ring-to-ring structure is not resolved. A flat LPIPS "
                 "profile against a sloped PSNR profile is information; a single LPIPS ring "
                 "step is not.\n")

    header = "| dataset / scene | " + " | ".join(f"ring{k}" for k in range(n_rings)) + \
             " | disk(pool) | disk(view) | n |"
    for metric in ("psnr", "ssim", "lpips"):
        digits = 3 if metric == "psnr" else 4
        lines.append(f"\n## delta {metric.upper()} (noncentral - off)\n")
        lines.append(header)
        lines.append("|" + "---|" * (n_rings + 4))
        by_group = {}
        for key, entry in data.items():
            off = entry["off"]["metrics"][metric]
            nc = entry["noncentral"]["metrics"][metric]
            delta = [b - a for a, b in zip(off["rings"], nc["rings"])]
            row = delta + [nc["disk_pooled"] - off["disk_pooled"],
                           nc["disk_per_view_mean"] - off["disk_per_view_mean"]]
            lines.append(f"| {key} | " + " | ".join(f"{v:+.{digits}f}" for v in row) +
                         f" | {entry['off']['views']} |")
            by_group.setdefault(entry["dataset"], []).append(row)
        for group, rows in by_group.items():
            if len(rows) < 2:
                continue
            mean = [sum(col) / len(col) for col in zip(*rows)]
            lines.append(f"| **{group} mean ({len(rows)} scenes)** | " +
                         " | ".join(f"**{v:+.{digits}f}**" for v in mean) + " | |")

    lines.append("\n## absolute disk values (frame convention = the canonical shared pass)\n")
    lines.append("| dataset / scene | PSNR off | PSNR nc | SSIM off | SSIM nc | "
                 "LPIPS off | LPIPS nc | valid frac |")
    lines.append("|" + "---|" * 8)
    for key, entry in data.items():
        get = lambda side, m, f: entry[side]["metrics"][m][f]  # noqa: E731
        lines.append(
            f"| {key} | {get('off','psnr','disk_pooled'):.3f} | "
            f"{get('noncentral','psnr','disk_pooled'):.3f} | "
            f"{get('off','ssim','frame_per_view_mean'):.4f} | "
            f"{get('noncentral','ssim','frame_per_view_mean'):.4f} | "
            f"{get('off','lpips','frame_per_view_mean'):.4f} | "
            f"{get('noncentral','lpips','frame_per_view_mean'):.4f} | "
            f"{entry['off']['valid_fraction']:.4f} |")
    lines.append("")
    lines.append("Dataset labels: " + "; ".join(f"`{k}` = {v}" for k, v in GROUP_LABEL.items()))
    lines.append("")


# --------------------------------------------------------------------------------------
# mask-radius robustness
# --------------------------------------------------------------------------------------


def kendall_tau(order_a, order_b):
    """Kendall tau-b over the methods present in both rankings."""
    shared = [m for m in order_a if m in order_b]
    if len(shared) < 3:
        return None
    rank_a = {m: i for i, m in enumerate(order_a)}
    rank_b = {m: i for i, m in enumerate(order_b)}
    concordant = discordant = 0
    for x, y in itertools.combinations(shared, 2):
        sign = (rank_a[x] - rank_a[y]) * (rank_b[x] - rank_b[y])
        concordant += sign > 0
        discordant += sign < 0
    total = concordant + discordant
    return (concordant - discordant) / total if total else None


def flatten(payload):
    """{'scene': {'method': {radius: metrics}, '_geometry': {...}}} for any track depth."""
    results = payload["results"]
    if payload["track"] == "fullcircle_rttpf":
        return {f"{variant}/{scene}": methods
                for variant, scenes in results.items() for scene, methods in scenes.items()}
    return results


def load_audit():
    """{track: {method: {'render_black': bool, 'gt_black': bool, 'annulus': float}}}.

    From `annulus_audit.json`. These are frame-level output conventions, so a method that
    blacks the annulus out on one scene of a track does it on all of them; the audit
    samples 2-3 scenes per track and 4 views per entry.
    """
    path = os.path.join(W3, "annulus_audit.json")
    if not os.path.exists(path):
        return {}
    data = json.load(open(path)).get("tracks", {})
    out = {}
    for track, scenes in data.items():
        for methods in scenes.values():
            for method, entry in methods.items():
                if method == "_canonical":
                    continue
                render = entry.get("render_zero_fraction_annulus")
                gt = entry.get("gt_zero_fraction_annulus")
                node = out.setdefault(track, {}).setdefault(
                    method, {"render_black": False, "gt_black": False,
                             "annulus": entry.get("annulus_fraction_of_frame", 0.0)})
                node["render_black"] |= bool(render is not None and render == render
                                             and render > 0.5)
                node["gt_black"] |= bool(gt is not None and gt == gt and gt > 0.5)
    return out


def r100_verdict(track, audit):
    """(is r=1.00 a valid cross-method ranking here, one-line reason)."""
    methods = audit.get(track, {})
    if not methods:
        return None, "no annulus audit for this track"
    annulus = max((m["annulus"] for m in methods.values()), default=0.0)
    if annulus < 1e-4:
        return None, ("the r=0.95->1.00 annulus is EMPTY (the polynomial-inversion criterion "
                      "binds first): the three radii are the same mask, so there is nothing "
                      "to compare")
    render_black = sorted(m for m, v in methods.items() if v["render_black"])
    gt_black = sorted(m for m, v in methods.items() if v["gt_black"])
    if not render_black:
        return True, (f"every method writes content in the annulus ({annulus:.1%} of the "
                      "frame); the widened row is comparable")
    if set(render_black) <= set(gt_black):
        return False, (f"INFLATED for {', '.join(render_black)}: they render exact black in "
                       f"the annulus AND the evaluator pairs them with their own saved GT, "
                       f"which is black there too -- {annulus:.1%} of the frame scores as a "
                       "perfect match for them and as real error for everyone else")
    return False, (f"PENALISES {', '.join(sorted(set(render_black) - set(gt_black)))}: they "
                   f"render exact black in the annulus while the GT carries real content "
                   f"over {annulus:.1%} of the frame")


def robustness(lines, rows, audit):
    lines.append("# W3.2 -- mask-radius robustness, r = 0.85 / 0.95 / 1.00\n")
    lines.append("One shared eval pass per radius over the raw renders of every method in "
                 "the four metric stores: same mask, same masked PSNR/SSIM, same "
                 "zero-then-bbox-crop LPIPS. **r = 0.95 stays canonical**; the other two are "
                 "the robustness check. Nothing was retrained.\n")
    lines.append("> **r = 1.00 IS AN ANNOTATED ROW, NOT A RANKING** -- and the direction of "
                 "the bias is not the one a full-frame/circular-frame split predicts. It "
                 "depends on whether the evaluator pairs each method with its OWN saved GT "
                 "(then a method that renders black in the annulus is also scored against a "
                 "black GT, i.e. INFLATED) or with the canonical dataset image (then the same "
                 "method is PENALISED). Both happen in these four stores. The per-track "
                 "verdict below is measured, not assumed -- see "
                 "`scripts/analysis/annulus_audit.py`.\n")
    lines.append("> **A flat row can mean two things.** `geometric_valid_mask_*` also drops "
                 "pixels where the distortion polynomial fails to invert, and where that "
                 "binds first the radius knob does nothing. Check `valid frac`: if the three "
                 "are equal, the three radii ARE the same mask and stability is arithmetic.\n")

    lines.append("\n## r=1.00 verdict per track (measured on the 0.95->1.00 annulus)\n")
    lines.append("| track | frame | annulus (% of frame) | r=1.00 usable as a ranking | why |")
    lines.append("|" + "---|" * 5)
    for track in sorted({r["track"] for r in rows}):
        frame = [r for r in rows if r["track"] == track][0]["frame_type"]
        verdict, why = r100_verdict(track, audit)
        annulus = max((m["annulus"] for m in audit.get(track, {}).values()), default=float("nan"))
        label = {True: "yes", False: "**NO**", None: "n/a"}[verdict]
        lines.append(f"| {track} | {frame} | {annulus * 100:.2f} | {label} | {why} |")

    for track in sorted({r["track"] for r in rows}):
        entries = [r for r in rows if r["track"] == track]
        frame = entries[0]["frame_type"]
        _, why = r100_verdict(track, audit)
        lines.append(f"\n## {track}  ({frame}-frame)\n")
        lines.append(f"r=1.00: {why}\n")
        lines.append("| scene | method | " +
                     " | ".join(f"PSNR {r}" for r in RADII) + " | " +
                     " | ".join(f"SSIM {r}" for r in RADII) + " | " +
                     " | ".join(f"LPIPS {r}" for r in RADII) + " |")
        lines.append("|" + "---|" * 11)
        for entry in entries:
            cells = []
            for metric in ("PSNR", "SSIM", "LPIPS"):
                digits = 3 if metric == "PSNR" else 4
                for radius in RADII:
                    value = entry["metrics"].get(radius, {}).get(metric)
                    cells.append("--" if value is None else f"{value:.{digits}f}")
            lines.append(f"| {entry['scene']} | {entry['method']} | " + " | ".join(cells) + " |")


def material_flips(scores_a, scores_b, higher, threshold):
    """(#pairs separated at BOTH radii, #of those that swapped order).

    Plain Kendall tau counts a swap between two methods that are 0.005 dB apart exactly as
    hard as a swap between two that are 2 dB apart. On these stores several methods sit
    inside the +-0.06 dB run-to-run noise of each other (`3dgrut` vs `3dgrut-oldbase`
    differ by 0.001 dB on tunnel), so raw tau reads as instability where there is only a
    tie. This restricts the count to pairs whose gap exceeds `threshold` at BOTH radii --
    a swap there is a claim the sweep actually changed.
    """
    shared = [m for m in scores_a if m in scores_b]
    sign = 1.0 if higher else -1.0
    separated, which = 0, []
    for x, y in itertools.combinations(shared, 2):
        gap_a, gap_b = scores_a[x] - scores_a[y], scores_b[x] - scores_b[y]
        if abs(gap_a) <= threshold or abs(gap_b) <= threshold:
            continue
        separated += 1
        if (sign * gap_a) * (sign * gap_b) < 0:
            which.append((x, y, gap_a, gap_b))
    return separated, which


# * one run-to-run noise floor per metric, from IMPLEMENTATION.md's +-0.06 dB and the
# * corresponding SSIM/LPIPS scatter of a repeated run.
NOISE = {"PSNR": 0.06, "SSIM": 0.001, "LPIPS": 0.002}


def rank_table(lines, rows, audit):
    lines.append("\n# Ranking stability (Kendall tau vs the canonical r = 0.95 ranking)\n")
    lines.append("tau = +1 means the sweep reordered nothing. `tau(1.00)` is reported for "
                 "completeness on every track but is only interpretable where the verdict "
                 "table above says so.\n")
    lines.append("`flips` counts only pairs of methods separated by more than the run-to-run "
                 f"noise floor ({NOISE['PSNR']} dB / {NOISE['SSIM']} SSIM / {NOISE['LPIPS']} "
                 "LPIPS) at BOTH radii, and how many of those swapped. That is the number "
                 "that supports or refutes a robustness claim; raw tau punishes ties as "
                 "hard as real reorderings.\n")
    lines.append("| track | scene | metric | n methods | valid frac 0.85 / 0.95 / 1.00 | "
                 "tau(0.85) | flips(0.85) | tau(1.00) | flips(1.00) | note |")
    lines.append("|" + "---|" * 10)
    summary, detail = {}, []
    for track in sorted({r["track"] for r in rows}):
        entries = [r for r in rows if r["track"] == track]
        frame = entries[0]["frame_type"]
        r100_ok, _ = r100_verdict(track, audit)
        for scene in sorted({e["scene"] for e in entries}):
            here = [e for e in entries if e["scene"] == scene]
            fractions = here[0]["valid_fraction"]
            identical = (fractions.get("0.85") is not None
                         and abs(fractions["0.85"] - fractions["1.00"]) < 1e-6)
            for metric in ("PSNR", "SSIM", "LPIPS"):
                orders, values = {}, {}
                for radius in RADII:
                    scored = [(e["method"], e["metrics"].get(radius, {}).get(metric))
                              for e in here]
                    scored = [(m, v) for m, v in scored if v is not None]
                    values[radius] = dict(scored)
                    orders[radius] = [m for m, _ in sorted(
                        scored, key=lambda kv: kv[1], reverse=HIGHER[metric])]
                tau_low = kendall_tau(orders["0.95"], orders["0.85"])
                tau_high = kendall_tau(orders["0.95"], orders["1.00"])
                sep_low, which_low = material_flips(values["0.95"], values["0.85"],
                                                    HIGHER[metric], NOISE[metric])
                sep_high, which_high = material_flips(values["0.95"], values["1.00"],
                                                      HIGHER[metric], NOISE[metric])
                flip_low, flip_high = len(which_low), len(which_high)
                for radius, group in (("0.85", which_low), ("1.00", which_high)):
                    for x, y, gap95, gap_other in group:
                        detail.append((track, scene, metric, radius, x, y, gap95, gap_other))
                notes = []
                if identical:
                    notes.append("IDENTICAL MASKS -- tau is vacuous")
                if r100_ok is False:
                    notes.append("r=1.00 not a ranking (annulus convention)")
                elif r100_ok is None:
                    notes.append("r=1.00 annulus empty or unaudited")
                lines.append(
                    f"| {track} | {scene} | {metric} | {len(orders['0.95'])} | "
                    f"{fractions.get('0.85', float('nan')):.4f} / "
                    f"{fractions.get('0.95', float('nan')):.4f} / "
                    f"{fractions.get('1.00', float('nan')):.4f} | "
                    f"{'n/a' if tau_low is None else f'{tau_low:+.3f}'} | "
                    f"{flip_low}/{sep_low} | "
                    f"{'n/a' if tau_high is None else f'{tau_high:+.3f}'} | "
                    f"{flip_high}/{sep_high} | "
                    f"{'; '.join(notes)} |")
                if not identical and len(orders["0.95"]) >= 3:
                    bucket = summary.setdefault(
                        (track, metric),
                        {"low": [], "high": [], "sep_low": 0, "flip_low": 0,
                         "sep_high": 0, "flip_high": 0})
                    if tau_low is not None:
                        bucket["low"].append(tau_low)
                    bucket["sep_low"] += sep_low
                    bucket["flip_low"] += flip_low
                    if r100_ok is True:
                        if tau_high is not None:
                            bucket["high"].append(tau_high)
                        bucket["sep_high"] += sep_high
                        bucket["flip_high"] += flip_high

    lines.append("\n## summary over the rows where tau is meaningful\n")
    lines.append("| track | metric | n scenes | mean tau(0.85 vs 0.95) | "
                 "mean tau(1.00 vs 0.95) | perfect tau=1 at 0.85 | "
                 "material flips 0.85 | material flips 1.00 |")
    lines.append("|" + "---|" * 8)
    for (track, metric), bucket in sorted(summary.items()):
        low, high = bucket["low"], bucket["high"]
        mean = lambda v: f"{sum(v)/len(v):+.3f}" if v else "n/a"  # noqa: E731
        perfect = sum(1 for t in low if t > 0.999)
        high_cell = (f"{bucket['flip_high']}/{bucket['sep_high']}"
                     if bucket["sep_high"] else "n/a")
        lines.append(f"| {track} | {metric} | {len(low)} | {mean(low)} | {mean(high)} "
                     f"| {perfect}/{len(low)} | {bucket['flip_low']}/{bucket['sep_low']} "
                     f"| {high_cell} |")
    lines.append("\n## every material flip, listed\n")
    lines.append("A pair of methods whose gap exceeds the noise floor at BOTH radii and whose "
                 "order nevertheless changed. `gap@0.95` is the canonical gap, `gap@r` the one "
                 "at the swept radius; a flip whose two gaps are both small is a near-tie that "
                 "cleared the threshold, not a reordering worth reporting.\n")
    lines.append("| track | scene | metric | radius | method A | method B | gap@0.95 | gap@r |")
    lines.append("|" + "---|" * 8)
    for track, scene, metric, radius, x, y, gap95, gap_other in detail:
        digits = 3 if metric == "PSNR" else 4
        lines.append(f"| {track} | {scene} | {metric} | {radius} | {x} | {y} | "
                     f"{gap95:+.{digits}f} | {gap_other:+.{digits}f} |")
    if not detail:
        lines.append("| _none_ | | | | | | | |")

    lines.append("\nThe r=1.00 column is left EMPTY for every track whose verdict row says "
                 "the widened annulus is not a ranking; those taus exist in the per-scene "
                 "table above but averaging them would launder an output convention into a "
                 "robustness claim. Tracks are never pooled with each other.\n")


def load_rows():
    rows = []
    for path in sorted(glob.glob(os.path.join(W3, "sweep_*.json"))):
        payload = json.load(open(path))
        for scene, methods in flatten(payload).items():
            geometry = methods.get("_geometry", {})
            fractions = {r: geometry.get(r, {}).get("valid_fraction") for r in RADII}
            for method, radii in methods.items():
                if method == "_geometry":
                    continue
                rows.append({
                    "track": payload["track"], "frame_type": payload["frame_type"],
                    "scene": scene, "method": method, "metrics": radii,
                    "valid_fraction": {k: (v if v is not None else float("nan"))
                                       for k, v in fractions.items()},
                })
    return rows


def main():
    lines = []
    ring_table(lines)
    with open(os.path.join(W3, "rings_table.md"), "w") as handle:
        handle.write("\n".join(lines) + "\n")
    print("wrote", os.path.join(W3, "rings_table.md"))

    rows = load_rows()
    if not rows:
        print("no sweep_*.json yet -- skipping the robustness table")
        return
    lines = []
    audit = load_audit()
    robustness(lines, rows, audit)
    rank_table(lines, rows, audit)
    with open(os.path.join(W3, "mask_radius_robustness.md"), "w") as handle:
        handle.write("\n".join(lines) + "\n")
    print("wrote", os.path.join(W3, "mask_radius_robustness.md"))

    with open(os.path.join(W3, "mask_radius_sweep.csv"), "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["track", "frame_type", "scene", "method", "radius",
                         "psnr", "ssim", "lpips", "n", "valid_fraction"])
        for row in rows:
            for radius in RADII:
                metrics = row["metrics"].get(radius)
                if not metrics:
                    continue
                writer.writerow([row["track"], row["frame_type"], row["scene"], row["method"],
                                 radius, f"{metrics['PSNR']:.4f}", f"{metrics['SSIM']:.5f}",
                                 f"{metrics['LPIPS']:.5f}", metrics["n"],
                                 f"{row['valid_fraction'][radius]:.5f}"])
    print("wrote", os.path.join(W3, "mask_radius_sweep.csv"))


if __name__ == "__main__":
    main()
