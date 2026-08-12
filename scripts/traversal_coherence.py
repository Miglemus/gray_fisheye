#!/usr/bin/env python
"""Does a non-central camera cost BVH coherence?  hits/ray, `off` against `noncentral`.

Why this exists
---------------
The first question a graphics reviewer asks about a non-central camera is whether losing the
common ray origin wrecks traversal: neighbouring pixels no longer share an origin, so the
ray bundle is no longer a pencil and the BVH cannot amortise a node fetch across it. Nobody
in this project has measured it. Both answers are publishable -- "no measurable cost" is the
one the paper wants, "N % more hits per ray" is an honest cost line next to the +0.315 dB.

gray already logs the number, `traversal_stats.csv` (`iteration, num_hit_per_ray,
num_accum_per_ray`, written by `train.py` every `log_stats_interval`). Two modes here:

`--from-runs` (CPU, no GPU, no render)
    Reads what training already wrote for every `off` / rung pair on disk, and prints the
    paired statistic. Cheap, and it is what you can look at today -- but it is CONFOUNDED:
    the two runs prune to slightly different gaussian counts, and hits/ray depends on how
    many gaussians there are and how big they got, not only on the ray field. Read the
    `gaussian_ratio` column, and the rank correlation printed under the table, before reading
    the delta. This mode answers "is there a large effect", not "how large is it".

    Measured 2026-08-12 over the 13 pairs on disk: hits/ray ratio 0.9981 +- 0.0269 sd,
    CI95 [0.9835, 1.0127], Wilcoxon p = 0.735, and the confound does not explain the scatter
    (rho with the gaussian ratio -0.374, p = 0.209). So: no effect down to about +-1.3 %.

`--measure <model_path>` (GPU, one render pass per arm, no training)
    The controlled version, and the one to quote. ONE checkpoint, ONE view set, and the only
    thing that changes between the two arms is whether `z(theta)` is active:

        arm `noncentral`  the trained profile        -> one ray origin per pixel
        arm `z0`          the same model, z forced 0 -> one ray origin per view

    `z_weights` enters `CameraModel.camera_frame()` only through the origin offset; the ray
    DIRECTIONS are computed from `theta_weights`, `phi_weights` and `omega` and are bit-for-bit
    identical between the two arms (checked at run time and reported as `arms_are_a_control`;
    if that is False, throw the run away). So the two arms differ in exactly one thing:
    whether the primary rays share an origin. Same gaussians, same BVH, same directions, same
    views. That isolates the question completely.

    The headline `hits_ratio` is pooled over the pixels BOTH arms traced -- see
    `shared_mask_stats()` for why a per-arm mask biases it. The per-arm version is kept as
    `hits_ratio_per_arm_mask`; if the two disagree, the difference is pixels entering or
    leaving the traced set, not rays getting cheaper.

GOTCHA -- stats are gated on `grads_enabled`
    `params.stats.num_gaussians_hit[...]++` in `cuda/shaders.cu:65` and
    `num_gaussians_accumulated` in `cuda/forward_pass.cu:127` are both inside
    `if (grads_enabled)`, and `MetaDataHolder::update()` takes that flag from
    `torch::autograd::GradMode::is_enabled()` on every `forward_pass()`. A `no_grad` render
    therefore collects NOTHING -- silently, as zeros. `--measure` renders under
    `enable_grad()` for that reason alone; it never calls `backward()` and never steps. The
    price is that the backward PPLL is filled, so it needs the memory of a training
    iteration, not of an eval one.

Usage
-----
    # CPU, right now, on the runs already on disk
    python scripts/traversal_coherence.py --from-runs --out tmp/traversal_coherence.json

    # GPU, the controlled measurement (queue it, never run it bare)
    pueue add --group gpu1 --print-task-id -- "cd $PWD && \\
      PATH=/workspace/gray/.venv/bin:\\$PATH CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 \\
      python scripts/traversal_coherence.py --measure tmp/final/workshop_noncentral \\
        --out tmp/traversal_coherence_workshop.json"
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# --------------------------------------------------------------------------- mode A: CSV


def read_traversal_csv(run: Path):
    """Last logged (iteration, hits/ray, accum/ray) of a run, or None.

    The header is comma separated and the rows are SPACE separated (`train.py:176` against
    `train.py:473`), which is worth knowing before writing `split(",")` and getting nothing.
    """
    path = run / "traversal_stats.csv"
    if not path.is_file():
        return None
    rows = [line.split() for line in path.read_text().splitlines()[1:] if line.strip()]
    if not rows:
        return None
    last = rows[-1]
    return {"iteration": int(last[0]), "hits_per_ray": float(last[1]), "accum_per_ray": float(last[2])}


def read_last_gaussians(run: Path):
    path = run / "num_gaussians.csv"
    if not path.is_file():
        return None
    rows = [line.split() for line in path.read_text().splitlines()[1:] if line.strip()]
    return int(rows[-1][1]) if rows else None


# * (label, off run, rung run). Only pairs that were trained with the same recipe belong
# * here -- a pair from different resolutions or iteration counts is not a control.
DEFAULT_PAIRS = [
    ("tunnel_r4_7500", "tmp/ladder_final/off", "tmp/ladder_final/noncentral"),
    ("tunnel_r8_3000", "tmp/ladder_r8/tunnel_off", "tmp/ladder_r8/tunnel_noncentral"),
    ("tunnel_r8_long", "tmp/long_r8/off", "tmp/long_r8/noncentral"),
    ("workshop_immervision", "out/workshop_immervision_off", "out/workshop_immervision_noncentral"),
]
FULLCIRCLE_SCENES = ["room1", "room2", "room3", "flat1", "flat2", "lab", "lounge", "dark", "persons"]
DEFAULT_PAIRS += [
    (
        f"fullcircle_{scene}",
        f"out/fullcircle_rttpf/{scene}_refit_rttpf_off",
        f"out/fullcircle_rttpf/{scene}_refit_rttpf",
    )
    for scene in FULLCIRCLE_SCENES
]


def from_runs(root: Path, pairs):
    results = []
    for label, off_dir, rung_dir in pairs:
        off = read_traversal_csv(root / off_dir)
        rung = read_traversal_csv(root / rung_dir)
        if off is None or rung is None:
            results.append({"label": label, "present": False, "off_dir": off_dir, "rung_dir": rung_dir})
            continue
        n_off = read_last_gaussians(root / off_dir)
        n_rung = read_last_gaussians(root / rung_dir)
        results.append(
            {
                "label": label,
                "present": True,
                "off_dir": off_dir,
                "rung_dir": rung_dir,
                "iteration": min(off["iteration"], rung["iteration"]),
                "iterations_match": off["iteration"] == rung["iteration"],
                "off_hits_per_ray": off["hits_per_ray"],
                "rung_hits_per_ray": rung["hits_per_ray"],
                "hits_per_ray_ratio": rung["hits_per_ray"] / off["hits_per_ray"],
                "off_accum_per_ray": off["accum_per_ray"],
                "rung_accum_per_ray": rung["accum_per_ray"],
                "accum_per_ray_ratio": rung["accum_per_ray"] / off["accum_per_ray"],
                "off_gaussians": n_off,
                "rung_gaussians": n_rung,
                "gaussian_ratio": (n_rung / n_off) if (n_off and n_rung) else None,
            }
        )
    return results


def summarize(results) -> dict:
    """Paired statistics over the ratios, so the headline is one number and not an eyeball.

    Reported the way the rest of the project reports a null: a mean with a CI, the paired
    Wilcoxon against the no-effect value 1.0, AND the rank correlation of the ratio with the
    gaussian-count ratio -- which is the confound of this mode. If that correlation were
    strong the table would be measuring pruning, not traversal.

    `scipy` is optional here on purpose: the script must still run in an environment that
    only has numpy, and the mean / CI carry most of the message.
    """
    ratios = [row["hits_per_ray_ratio"] for row in results if row["present"]]
    if not ratios:
        return {"n": 0}
    import numpy as np

    hits = np.array(ratios, dtype=float)
    accum = np.array([row["accum_per_ray_ratio"] for row in results if row["present"]])
    gauss = np.array(
        [row["gaussian_ratio"] or float("nan") for row in results if row["present"]]
    )
    stderr = float(hits.std(ddof=1) / max(len(hits) ** 0.5, 1.0)) if len(hits) > 1 else 0.0
    summary = {
        "n": int(len(hits)),
        "hits_ratio_mean": float(hits.mean()),
        "hits_ratio_sd": float(hits.std(ddof=1)) if len(hits) > 1 else 0.0,
        "hits_ratio_median": float(np.median(hits)),
        "hits_ratio_min": float(hits.min()),
        "hits_ratio_max": float(hits.max()),
        "hits_ratio_ci95": [float(hits.mean() - 1.96 * stderr), float(hits.mean() + 1.96 * stderr)],
        "accum_ratio_mean": float(accum.mean()),
        "gaussian_ratio_mean": float(np.nanmean(gauss)),
    }
    try:
        from scipy import stats as scipy_stats

        summary["wilcoxon_p_vs_1"] = float(scipy_stats.wilcoxon(hits - 1.0).pvalue)
        finite = np.isfinite(gauss)
        if finite.sum() >= 3:
            rho, pvalue = scipy_stats.spearmanr(hits[finite], gauss[finite])
            summary["spearman_hits_vs_gaussian_ratio"] = [float(rho), float(pvalue)]
    except Exception as error:  # * numpy-only environment, or a degenerate all-ties sample
        summary["scipy_error"] = repr(error)
    return summary


def print_from_runs(results, summary=None):
    header = f"{'pair':<24}{'off':>8}{'nc':>8}{'ratio':>8}{'accum off':>11}{'accum nc':>10}{'gauss':>9}"
    print(header)
    print("-" * len(header))
    for row in results:
        if not row["present"]:
            print(f"{row['label']:<24}{'-- missing --':>45}")
            continue
        gaussian = f"{row['gaussian_ratio']:.3f}" if row["gaussian_ratio"] else "n/a"
        flag = "" if row["iterations_match"] else "  (!) iteration mismatch"
        print(
            f"{row['label']:<24}{row['off_hits_per_ray']:>8.2f}{row['rung_hits_per_ray']:>8.2f}"
            f"{row['hits_per_ray_ratio']:>8.3f}{row['off_accum_per_ray']:>11.2f}"
            f"{row['rung_accum_per_ray']:>10.2f}{gaussian:>9}{flag}"
        )
    if summary is None:
        summary = summarize(results)
    if not summary.get("n"):
        return
    low, high = summary["hits_ratio_ci95"]
    print(
        f"\nn = {summary['n']} pairs, hits/ray ratio (noncentral / off): "
        f"mean {summary['hits_ratio_mean']:.4f} +- {summary['hits_ratio_sd']:.4f} sd, "
        f"CI95 [{low:.4f}, {high:.4f}], median {summary['hits_ratio_median']:.4f}, "
        f"range [{summary['hits_ratio_min']:.3f}, {summary['hits_ratio_max']:.3f}]"
    )
    if "wilcoxon_p_vs_1" in summary:
        print(f"paired Wilcoxon against 1.0: p = {summary['wilcoxon_p_vs_1']:.3f}")
    if "spearman_hits_vs_gaussian_ratio" in summary:
        rho, pvalue = summary["spearman_hits_vs_gaussian_ratio"]
        print(f"confound check, rho(hits ratio, gaussian ratio) = {rho:+.3f} (p = {pvalue:.3f})")
    print(
        "CONFOUNDED: the two runs prune to different gaussian counts. Use --measure for "
        "the controlled number."
    )


# ------------------------------------------------------------------- mode B: measurement


def shared_mask_stats(hits_nc, hits_z0, accum_nc, accum_z0, num_views: int) -> dict:
    """Per-ray costs of the two arms on the pixels BOTH of them traced.

    Each arm's own `hits > 0` mask is a function of its own result, so masking per arm puts
    a different denominator on each side of the ratio: displacing the origin by `z` can push
    a grazing pixel over or under the "hit at least one gaussian" line, and that alone shows
    up as a ratio even when every surviving ray costs exactly the same. A shared mask removes
    that bias. Both per-arm numbers are kept alongside so the size of the effect stays visible.

    Pure tensor arithmetic, deliberately: it is the headline number of the GPU mode and it is
    the one thing in that mode a CPU test can pin down (`tests/test_traversal_coherence.py`).
    """
    shared = (hits_nc > 0) & (hits_z0 > 0)
    if int(shared.sum()) == 0:
        raise ValueError("no pixel was traced by both arms; the measurement is degenerate")
    return {
        "shared_traced_pixels": int(shared.sum()),
        "shared_traced_pixel_fraction": float(shared.float().mean()),
        "noncentral_hits_per_shared_ray": float(hits_nc[shared].float().mean()) / num_views,
        "z0_hits_per_shared_ray": float(hits_z0[shared].float().mean()) / num_views,
        "noncentral_accum_per_shared_ray": float(accum_nc[shared].float().mean()) / num_views,
        "z0_accum_per_shared_ray": float(accum_z0[shared].float().mean()) / num_views,
    }


def rung_is_noncentral(rung: str) -> bool:
    """Does this ablation rung move the ray ORIGIN? (i.e. is the z-on / z-off arm a control?)

    Read from `RUNGS` rather than pattern-matched on the name. The substring test this
    replaces -- `"z" not in rung and rung != "noncentral"` -- rejected `noncentral_no_ana`,
    which the error message it printed then recommended, and would have accepted any future
    rung with a `z` anywhere in its name.
    """
    from gray.camera_model import RUNGS

    if rung not in RUNGS:
        raise KeyError(f"unknown rung {rung!r}; known rungs: {sorted(RUNGS)}")
    return "z" in RUNGS[rung]


def measure(model_path: str, split: str, max_views: int, unknown_args):
    """Render one view set twice off one checkpoint: z(theta) active, then forced to zero."""
    import torch
    import tyro

    from gray.prelude import Config, Raytracer, SceneInfo, search_for_max_iteration

    model_dir = Path(model_path)
    cfg = tyro.cli(
        Config,
        args=unknown_args,
        default=Config(**json.loads((model_dir / "config.json").read_text())),
    )
    cfg.model_path = str(model_dir)
    if not rung_is_noncentral(cfg.camera_opt):
        raise SystemExit(
            f"--measure needs a rung whose origin is non-central; this run is "
            f"'{cfg.camera_opt}'. Use noncentral / noncentral_no_ana / z_only."
        )

    iteration = search_for_max_iteration(str(model_dir))
    checkpoint = str(model_dir / f"gaussians_{iteration:05d}.safetensors")

    scene = SceneInfo.from_colmap(cfg)
    cameras = scene.test_cameras if split == "test" else scene.train_cameras
    if not cameras:
        # * A held-out split can be empty (`--llffhold 0`), and `cameras[0]` would then raise
        # * an IndexError six lines from the config that caused it.
        raise SystemExit(f"split '{split}' is empty for {model_dir}; try --split train")
    if max_views:
        cameras = cameras[:max_views]
    reference = cameras[0]
    raytracer = Raytracer.from_safetensors(
        cfg, checkpoint, reference.image_width, reference.image_height
    )
    stats = raytracer.cuda_module.get_stats()
    lens_z = {key: lens.z_weights.detach().clone() for key, lens in raytracer.camera_model.lenses.items()}

    def run_arm(name: str):
        stats.reset()
        directions = None
        # * enable_grad is MANDATORY: the stat counters live behind `if (grads_enabled)`.
        # * We never call backward(); `output_channels` is cleared by hand because
        # * `Raytracer.__call__` asserts the previous forward was consumed.
        with torch.enable_grad():
            for camera in cameras:
                raytracer(camera)
                raytracer.output_channels = None
                raytracer._ray_origin = None
                raytracer._ray_direction = None
                if directions is None:
                    framebuffer = raytracer.cuda_module.get_framebuffer()
                    directions = framebuffer.ray_direction[
                        : raytracer.render_height, : raytracer.render_width
                    ].detach().clone()
        hits = stats.num_gaussians_hit.float()
        accum = stats.num_gaussians_accumulated.float()
        traced = hits > 0
        return {
            "arm": name,
            "views": len(cameras),
            # * Pooled over every pixel of the launch, including the ones outside the lens
            # * disk that trace nothing -- that is what train.py's column means too.
            "hits_per_ray_all_pixels": float(hits.mean()) / len(cameras),
            "accum_per_ray_all_pixels": float(accum.mean()) / len(cameras),
            # * ... and restricted to pixels that hit at least one gaussian, which is the
            # * coherence question proper: the invalid annulus is identical in both arms and
            # * only dilutes the contrast.
            "hits_per_traced_ray": float(hits[traced].mean()) / len(cameras),
            "accum_per_traced_ray": float(accum[traced].mean()) / len(cameras),
            "traced_pixel_fraction": float(traced.float().mean()),
        }, directions, hits.clone(), accum.clone()

    non_central, directions_nc, hits_nc, accum_nc = run_arm("noncentral")

    with torch.no_grad():
        for key, lens in raytracer.camera_model.lenses.items():
            lens.z_weights.zero_()
    raytracer.camera_model.invalidate_ray_cache()
    central, directions_z0, hits_z0, accum_z0 = run_arm("z0")

    # * The control is only a control if the directions really did not move.
    direction_delta = float((directions_nc - directions_z0).abs().max())

    shared_stats = shared_mask_stats(hits_nc, hits_z0, accum_nc, accum_z0, len(cameras))

    with torch.no_grad():
        for key, lens in raytracer.camera_model.lenses.items():
            lens.z_weights.copy_(lens_z[key])
    raytracer.camera_model.invalidate_ray_cache()

    return {
        "model_path": str(model_dir),
        "iteration": iteration,
        "rung": cfg.camera_opt,
        "split": split,
        "num_gaussians": int(raytracer.cuda_module.get_gaussians().mean.shape[0]),
        "direction_max_abs_delta_between_arms": direction_delta,
        # * The whole design rests on the two arms differing ONLY in the ray origin. If this
        # * is False the run measured something else and the ratio below is meaningless.
        "arms_are_a_control": direction_delta == 0.0,
        "arms": {"noncentral": non_central, "z0": central},
        "shared_mask": shared_stats,
        "hits_ratio": (
            shared_stats["noncentral_hits_per_shared_ray"] / shared_stats["z0_hits_per_shared_ray"]
        ),
        "accum_ratio": (
            shared_stats["noncentral_accum_per_shared_ray"]
            / shared_stats["z0_accum_per_shared_ray"]
        ),
        "hits_ratio_per_arm_mask": (
            non_central["hits_per_traced_ray"] / central["hits_per_traced_ray"]
        ),
        "accum_ratio_per_arm_mask": (
            non_central["accum_per_traced_ray"] / central["accum_per_traced_ray"]
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--from-runs", action="store_true", help="CPU: read traversal_stats.csv pairs")
    parser.add_argument("--measure", metavar="MODEL_PATH", help="GPU: the controlled z-on / z-off pass")
    parser.add_argument("--split", default="test", choices=["test", "train"])
    parser.add_argument("--max-views", type=int, default=0, help="0 = every view of the split")
    parser.add_argument("--root", default=str(REPO_ROOT))
    parser.add_argument("--out", help="write the result as JSON here")
    args, unknown = parser.parse_known_args()

    if not args.from_runs and not args.measure:
        parser.error("pass --from-runs or --measure")

    payload = {}
    if args.from_runs:
        results = from_runs(Path(args.root), DEFAULT_PAIRS)
        summary = summarize(results)
        print_from_runs(results, summary)
        payload["from_runs"] = results
        payload["from_runs_summary"] = summary
    if args.measure:
        os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
        result = measure(args.measure, args.split, args.max_views, unknown)
        print(json.dumps(result, indent=2))
        payload["measured"] = result

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(payload, indent=2))
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
