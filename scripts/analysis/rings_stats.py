"""Paired statistics on the ring profiles -- does the gain really land at the rim?

    python scripts/analysis/rings_stats.py            # CPU, <1 s
        reads  tmp/w3_radial/rings_all.json           (and rings_all_3.json if present)
        writes tmp/w3_radial/rings_stats.json
               tmp/w3_radial/rings_stats.md

WHAT IS TESTED, AND WHY THIS SHAPE
---------------------------------------------------------------------------------------
The claim under test is "the camera-model gain is PERIPHERAL", not "the camera model
helps". The second is the disk delta and is reported elsewhere. The first is a statement
about the SHAPE of the per-ring delta, so the statistic is

    D_scene = delta(outer ring) - delta(ring 0)                   [one number per scene]

with delta = noncentral - off, computed inside the scene so that every scene-level
nuisance (content, exposure, how hard the scene is) cancels twice. A one-sample t-test on
D over scenes is then the paired test of "rim > centre". n is the number of SCENES, never
the number of views or rings: the views of one scene are not independent draws of the
quantity we are testing.

Reported alongside, because with n = 7 and n = 9 a t-test alone is thin:
  * a BCa-free percentile bootstrap CI over scenes (10 000 resamples, fixed seed);
  * the sign count (how many scenes have D > 0), i.e. an exact binomial sign test;
  * the mean over scenes of Spearman(ring index, delta), which uses the whole profile
    rather than only its two endpoints and so does not care about one noisy ring.

DIRECTION CONVENTIONS. PSNR and SSIM are higher-is-better, LPIPS is lower-is-better. This
script does NOT flip LPIPS' sign: a NEGATIVE LPIPS delta is an improvement, so a negative
D means the improvement is larger at the rim. The sign is stated per row in the markdown.

THE OUTER RING OF SSIM AND LPIPS IS CONTAMINATED, AND THE BIAS IS CONSERVATIVE HERE.
Both metrics have spatial support and are computed on mask-zeroed images, so at the rim
they partly score black against black. That contamination is IDENTICAL in `off` and in
`noncentral` -- it is a property of the mask, not of the reconstruction -- so it pulls the
outer-ring DELTA toward zero. A significant positive D for SSIM is therefore a lower
bound, not an artefact. The same is not true of the absolute ring values, which are
inflated; quote PSNR for those.

WHY A 3-RING CONTROL EXISTS. LPIPS' deepest receptive field (relu5_3, ~212 px) is about
three times the width of one ring on a 419 px disk radius, so a 6-ring LPIPS profile is
blurred beyond ring resolution by construction. `rings_all_3.json` re-runs everything at
3 rings (~140 px each). If LPIPS' flatness were only the blur, it should sharpen there.
The `n_rings=3` block reports whether it does.
"""

import json
import math
import os

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
W3 = os.path.join(REPO, "tmp", "w3_radial")
METRICS = ("psnr", "ssim", "lpips")
# * `myscenes_alt` is a second `off` run for tunnel kept for provenance (see W3 notes); it
# * would double-count tunnel, so it is never part of a dataset mean.
EXCLUDE_DATASETS = ("myscenes_alt",)
BOOTSTRAP = 10000
SEED = 20260811


def spearman(x, y):
    """Spearman rho without scipy -- ties are impossible here (x is 0..n-1)."""
    def rank(v):
        order = np.argsort(np.asarray(v, dtype=float), kind="mergesort")
        out = np.empty(len(v), dtype=float)
        out[order] = np.arange(len(v), dtype=float)
        # * average ranks over ties so a flat profile does not produce a fake +1
        values = np.asarray(v, dtype=float)[order]
        i = 0
        while i < len(values):
            j = i
            while j + 1 < len(values) and values[j + 1] == values[i]:
                j += 1
            if j > i:
                out[order[i : j + 1]] = np.mean(np.arange(i, j + 1))
            i = j + 1
        return out

    rx, ry = rank(x), rank(y)
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    denom = math.sqrt(float((rx**2).sum()) * float((ry**2).sum()))
    return float((rx * ry).sum() / denom) if denom else float("nan")


def t_test(values):
    """One-sample two-sided t-test, p from the Student CDF via the incomplete beta."""
    values = np.asarray(values, dtype=float)
    n = len(values)
    if n < 2:
        return float("nan"), float("nan"), float("nan")
    mean = float(values.mean())
    sd = float(values.std(ddof=1))
    if sd == 0:
        return mean, float("inf"), 0.0
    t = mean / (sd / math.sqrt(n))
    df = n - 1
    x = df / (df + t * t)
    p = _betainc(df / 2.0, 0.5, x)
    return mean, t, float(min(1.0, max(0.0, p)))


def _betainc(a, b, x):
    """Regularised incomplete beta I_x(a, b) by continued fraction (Lentz)."""
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    lbeta = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
    front = math.exp(math.log(x) * a + math.log(1 - x) * b - lbeta) / a
    if x >= (a + 1) / (a + b + 2):
        return 1.0 - _betainc(b, a, 1 - x)
    f, c, d = 1.0, 1.0, 0.0
    for i in range(0, 300):
        m = i // 2
        if i == 0:
            numerator = 1.0
        elif i % 2 == 0:
            numerator = (m * (b - m) * x) / ((a + 2 * m - 1) * (a + 2 * m))
        else:
            numerator = -((a + m) * (a + b + m) * x) / ((a + 2 * m) * (a + 2 * m + 1))
        d = 1.0 + numerator * d
        d = 1e-30 if abs(d) < 1e-30 else d
        d = 1.0 / d
        c = 1.0 + numerator / c
        c = 1e-30 if abs(c) < 1e-30 else c
        f *= c * d
        if abs(1.0 - c * d) < 1e-12:
            break
    return front * (f - 1.0)


def bootstrap_ci(values, alpha=0.05):
    values = np.asarray(values, dtype=float)
    if len(values) < 2:
        return [float("nan"), float("nan")]
    rng = np.random.default_rng(SEED)
    draws = rng.integers(0, len(values), size=(BOOTSTRAP, len(values)))
    means = values[draws].mean(axis=1)
    return [float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))]


def collect(path):
    data = json.load(open(path))
    by_dataset = {}
    for key, entry in data.items():
        if entry["dataset"] in EXCLUDE_DATASETS:
            continue
        node = by_dataset.setdefault(entry["dataset"], {})
        for metric in METRICS:
            off = entry["off"]["metrics"][metric]
            nc = entry["noncentral"]["metrics"][metric]
            node.setdefault(metric, []).append({
                "scene": entry["scene"],
                "delta_rings": [b - a for a, b in zip(off["rings"], nc["rings"])],
                "delta_disk": nc["disk_pooled"] - off["disk_pooled"],
                "delta_disk_view": nc["disk_per_view_mean"] - off["disk_per_view_mean"],
                "delta_frame": (nc.get("frame_per_view_mean", float("nan"))
                                - off.get("frame_per_view_mean", float("nan"))),
                "ring_edge_fraction_5px": entry["off"]["ring_edge_fraction_5px"],
            })
    return by_dataset


def analyse(by_dataset, n_rings):
    out = {}
    for dataset, metrics in by_dataset.items():
        for metric, scenes in metrics.items():
            profile = np.array([s["delta_rings"] for s in scenes], dtype=float)
            rim_minus_centre = profile[:, -1] - profile[:, 0]
            mean, t, p = t_test(rim_minus_centre)
            rhos = [spearman(list(range(n_rings)), row) for row in profile]
            out[f"{dataset}/{metric}"] = {
                "dataset": dataset,
                "metric": metric,
                "n_scenes": len(scenes),
                "scenes": [s["scene"] for s in scenes],
                "mean_delta_per_ring": [float(v) for v in profile.mean(axis=0)],
                "sem_delta_per_ring": [
                    float(profile[:, k].std(ddof=1) / math.sqrt(len(scenes)))
                    if len(scenes) > 1 else float("nan")
                    for k in range(n_rings)
                ],
                "mean_delta_disk_pooled": float(np.mean([s["delta_disk"] for s in scenes])),
                "mean_delta_disk_per_view": float(
                    np.mean([s["delta_disk_view"] for s in scenes])),
                "mean_delta_frame": float(np.mean([s["delta_frame"] for s in scenes])),
                "rim_minus_centre": {
                    "per_scene": [float(v) for v in rim_minus_centre],
                    "mean": mean,
                    "t": t,
                    "df": len(scenes) - 1,
                    "p_two_sided": p,
                    "bootstrap_ci95": bootstrap_ci(rim_minus_centre),
                    "n_positive": int((rim_minus_centre > 0).sum()),
                },
                "mean_spearman_ring_vs_delta": float(np.mean(rhos)),
                "spearman_per_scene": [float(v) for v in rhos],
            }
    return out


def markdown(blocks):
    lines = ["# W3.1 -- is the gain peripheral? paired statistics on the ring profile\n"]
    lines.append("Every number is `noncentral - off` computed **within** a scene, then "
                 "aggregated over scenes; `n` is the number of scenes. Datasets are never "
                 "pooled. LPIPS is NOT sign-flipped: negative is better, so a negative "
                 "`rim - centre` means the improvement is larger at the rim.\n")
    for n_rings, stats in blocks:
        lines.append(f"\n## {n_rings} equal-area rings\n")
        lines.append("| dataset | metric | n | " +
                     " | ".join(f"ring{k}" for k in range(n_rings)) +
                     " | disk(pool) | rim - centre | t (df) | p | bootstrap CI95 | "
                     "scenes with rim>centre | mean Spearman |")
        lines.append("|" + "---|" * (n_rings + 10))
        for key in sorted(stats):
            s = stats[key]
            digits = 3 if s["metric"] == "psnr" else 4
            rim = s["rim_minus_centre"]
            single = s["n_scenes"] < 2
            # * n = 1 has no sampling distribution: print the observation, refuse the test.
            cells = (["--"] * 4 if single else [
                f"{rim['t']:+.2f} ({rim['df']})",
                f"{rim['p_two_sided']:.3f}",
                f"[{rim['bootstrap_ci95'][0]:+.{digits}f}, "
                f"{rim['bootstrap_ci95'][1]:+.{digits}f}]",
                f"{rim['n_positive']}/{s['n_scenes']}",
            ])
            lines.append(
                f"| {s['dataset']} | {s['metric'].upper()} | {s['n_scenes']} | " +
                " | ".join(f"{v:+.{digits}f}" for v in s["mean_delta_per_ring"]) +
                f" | {s['mean_delta_disk_pooled']:+.{digits}f}"
                f" | **{rim['mean']:+.{digits}f}** | " + " | ".join(cells) +
                f" | {s['mean_spearman_ring_vs_delta']:+.3f} |")
    lines.append("\n`n = 1` datasets (workshop_immervision) have no t, no CI and no sign "
                 "count: one scene is an observation. They are listed for their profile only.\n")
    return "\n".join(lines) + "\n"


def main():
    blocks = []
    for name, n_rings in (("rings_all.json", 6), ("rings_all_3.json", 3)):
        path = os.path.join(W3, name)
        if not os.path.exists(path):
            continue
        blocks.append((n_rings, analyse(collect(path), n_rings)))

    payload = {
        "source": [name for name, _ in
                   (("rings_all.json", 6), ("rings_all_3.json", 3))
                   if os.path.exists(os.path.join(W3, name))],
        "bootstrap_resamples": BOOTSTRAP,
        "seed": SEED,
        "excluded_datasets": list(EXCLUDE_DATASETS),
        "blocks": {f"n_rings={n}": stats for n, stats in blocks},
    }
    with open(os.path.join(W3, "rings_stats.json"), "w") as handle:
        json.dump(payload, handle, indent=1)
    print("wrote", os.path.join(W3, "rings_stats.json"))
    with open(os.path.join(W3, "rings_stats.md"), "w") as handle:
        handle.write(markdown(blocks))
    print("wrote", os.path.join(W3, "rings_stats.md"))


if __name__ == "__main__":
    main()
