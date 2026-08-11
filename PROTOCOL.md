# PROTOCOL — running the learnable camera model

Worktree `/workspace/gray/worktrees/rttpf-intrinsics`, branch `rttpf-intrinsics`, forked
from `noncentral-camera` (whose worktree still exists and is where the parent branch's runs
live). See IMPLEMENTATION.md for what was changed and its limitations.

This branch adds one rung, `--camera_opt rttpf`: the **re-calibration control**, which
optimizes COLMAP's own 16 rttpf parameters photometrically instead of adding a residual
model on top of them. Jump to "The re-calibration control" below for how to run it.

## Setup

```bash
cd /workspace/gray/worktrees/rttpf-intrinsics
OPTIX_HOME=/workspace/gray/third_party/optix bash make.sh    # own build/, NOT shared
```

The CUDA in this branch is byte-identical to `noncentral-camera`'s, but keep the build
directories separate anyway: sharing one silently swaps the other worktree's kernels, and
that worktree has jobs queued from other sessions.

The venv is shared (`/workspace/gray/.venv`); `data/` is a symlink to `/workspace/gray/data`.
**Do not symlink `build/`** — this branch changes CUDA, and sharing the build directory would
silently swap `main`'s kernels (and therefore anyone else's runs) for these.

All GPU work goes through `pueue`. myscenes at `-r 4` peaks at **11.8-16.5 GB**, so it needs
the 24 GB card (`CUDA_VISIBLE_DEVICES=1`, group `gpu1`); `-r 8` fits the 12 GB card.

```bash
ENV1="PATH=/workspace/gray/.venv/bin:\$PATH CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1"
pueue add --group gpu1 --print-task-id -- "cd $PWD && $ENV1 bash scripts/train_myscenes.sh tunnel out/my_run --camera_opt noncentral"
pueue wait <id>
```

## Running a scene

`scripts/train_myscenes.sh <scene> <model_path> [extra train.py args]` handles the two
myscenes layouts (`atrium|library|reception|tunnel` are `<scene>_undistortion`,
`classroom|forest|workshop` are `<scene>`) and pins the baseline settings:
`-r 4 --camera_model rad_tan_thin_prism_fisheye --batch_size 2 --eval --vignetting_comp
--vignetting_terms 3`. It then calls `scripts/eval_rttpf.sh`.

**Never use bare `run.sh` for these scenes.** `render.py`'s `eval_models` defaults to
`["pinhole"]`, so `run.sh` produces no fisheye renders and `scripts/masked_eval.py` finds
nothing. `scripts/eval_rttpf.sh <model_path> <source_path>` renders both models with the
right `--intrinsics` and runs `metrics.py` + `measure_fps.py`. It can be run on its own
against an already-trained run.

Note `-c` on `train.py` is the preset path, not the camera model: write `-c configs/lq.json`
(literal `open()`, so `-c lq` raises), and select the camera with `--camera_model`.

### Scenes that are NOT myscenes (`data/others/*`)

`train_myscenes.sh` pins `-r 4` and the myscenes directory layout, so it does not apply.
Mirror the recipe that produced the scene's published `gray` row — for
`workshop_immervision`, `dataset/others_ocv/queue_immervision.sh` step `gray`:

```bash
$PY train.py -s /workspace/gray/data/others/workshop_immervision -r 1 \
  -m out/workshop_immervision_noncentral -y \
  --camera_model rad_tan_thin_prism_fisheye --batch_size 2 --eval \
  --vignetting_comp --vignetting_terms 3 --iterations 15000 \
  --camera_opt noncentral --camera_opt_from_iter 3000
$PY render.py -m out/workshop_immervision_noncentral \
  --eval-models pinhole rad_tan_thin_prism_fisheye \
  --intrinsics /workspace/gray/data/others/workshop_immervision/distorted/sparse/0/cameras.bin
$PY metrics.py -m ... && $PY result_to_csv.py -t .../results.json && $PY measure_fps.py -m ...
```

Traps, each of which silently corrupts the comparison:

* **`-r 1`, not `-r 4`.** 1440x1080 IS the working resolution here; `-r 4` would train at
  360x270 and the row would not be comparable to anything.
* **Always run an `--camera_opt off` control in THIS worktree**, not just a diff against the
  published `gray` number. The branch changes CUDA, and the control is what proves the change
  is neutral (measured: 25.282 against a published 25.220, i.e. inside the ±0.06 noise).
* **Cross-check `psnr.csv` (live) against `results.json` (re-rendered PNGs) on every new
  scene.** They agree to ~0.01 dB when nothing is wrong. This is what caught the stale-bearing
  collision that cost this scene 12 dB — see IMPLEMENTATION.md GOTCHAS.
* **The card must be nearly free.** This scene peaks ~16 GB (3.35 M init points at full res),
  and other agents run GPU-1 jobs in the `default` pueue group, i.e. outside the `gpu1` lane,
  so that group's 1-parallel setting does not by itself keep the card to one job. Guard the
  job with a free-memory wait (`gray/worktrees/masked-efficiency/scripts/gpuwait.sh 1 17000
  <cmd>`); without one it OOMs at `Raytracer.__init__`.

Cross-method numbers for this scene come from `gray/scripts/masked_eval_immervision.py
--track rttpf` then `gray/scripts/collect_immervision.py --track rttpf`
(-> `dataset/fisheye_baselines/immervision_results.csv`), which both carry a
`gray-non-central` entry. Do **not** read quality numbers off `results.json`: its LPIPS is a
different definition (0.5603 against the shared pass's 0.3514 for the same renders).

## The ablation ladder

One flag, `--camera_opt`, cumulative left to right:

| rung | adds | params |
|---|---|---|
| `off` | native ray generation | 0 |
| `passthrough` | rays from Python, residual pinned at zero | 0 |
| `tilt` | 3-DoF bearing rotation (~ principal point + roll) | 3 |

(counts are *trained* parameters; all rungs allocate 111 because the full `[5,K]` tensors
are always created, so unused channels sit at zero with zero gradient)
| `radial` | + `d(theta)(theta)` **and** a radially-varying `d(phi)` roll | 103 |
| `ana` | + anamorphic `cos/sin(k phi)`, k=1,2, on `d(theta)` and `d(phi)` | 103 |
| `noncentral` | + on-axis pupil profile `z(theta)`, gauged `z(0)=0` | **111** |
| `central_matched` | central control with a **larger** budget (not matched: 183 vs 111) | 183 |
| `raxel` | dense ray field — **central only, see IMPLEMENTATION.md limitation 8** | ~1e4 |

`--pose_opt` adds a per-view SE(3) residual on top of any rung. Report it separately: test
poses stay at COLMAP, so it can cost test PSNR (IMPLEMENTATION.md, limitation 2).

### Recipe (measured, on `tunnel` at -r 8 / 7500 it; `off` = 27.41)

| variant | test PSNR |
|---|---|
| unfreeze at 67% (`--camera_opt_from_iter 5000`) | 27.36 (worse than `off`) |
| unfreeze at 40% (3000) | 27.51 |
| **unfreeze at 20% (1500)** | **27.60** |
| 20 knots, unfreeze at 40% | 27.55 |
| 20 knots, unfreeze at 20% | 27.43 |
| lr 3e-4 | 27.47 |
| no regularization | 27.50 |

So: **unfreeze early** (~20% of the run — the opposite of what a long phase A would
suggest; the camera model is a low-dimensional global correction and the geometry is better
built on top of a corrected camera than made to fight it), keep the default 10 knots, and
use lr 1e-4. Combining early unfreezing *and* extra knots is worse than either alone.
Run-to-run noise is about ±0.05 dB (CUDA atomics make the backward non-deterministic), so
treat differences below that as ties.

Knobs: `--camera_opt_from_iter`, `--camera_opt_lr_{tilt,angular,z,raxel}`,
`--camera_opt_reg_l2`, `--camera_opt_reg_curvature`, `--camera_opt_knots{,_z}`.

`passthrough` is the control: it must reproduce `off`. If it does not, stop and debug —
everything downstream compares rungs against each other.

## The re-calibration control (`--camera_opt rttpf`)

The question it exists to answer: how much of the `noncentral` gain is the *model* and how
much is simply that the camera was allowed to move? It re-fits COLMAP's own 16
RAD_TAN_THIN_PRISM_FISHEYE parameters photometrically, with the same schedule and the same
freeze point as every residual rung.

```bash
# one scene, sweep resolution / iterations exposed
bash scripts/train_myscenes_at.sh tunnel tmp/probe/tunnel_rttpf 8 7500 \
    --camera_opt rttpf --camera_opt_from_iter 1500 [--camera_opt_lr_intrinsics 1e-3]

# the whole scene x rung matrix, with the VRAM guard and the 20 % freeze point
bash scripts/queue_rttpf_control.sh tmp/r4_control 4 15000 1 rttpf,rttpf_z

# the r4 rungs live in three different roots; the table script takes one, so alias them
mkdir -p tmp/r4_paired
for s in atrium classroom forest library reception tunnel workshop; do
  ln -sfn /workspace/gray/tmp/final/${s}_noncentral      tmp/r4_paired/${s}_noncentral
  ln -sfn "$PWD/tmp/r4_control/${s}_rttpf"               tmp/r4_paired/${s}_rttpf
  ln -sfn /workspace/gray/out/${s}_fisheye_baseline      tmp/r4_paired/${s}_off
done
ln -sfn /workspace/gray/tmp/noncentral/fix15k_workshop   tmp/r4_paired/workshop_off  # see below

# paired table, re-scored from the PNGs, with the live-vs-rendered cross-check
python scripts/rttpf_control_table.py --root tmp/r4_paired --rungs off rttpf noncentral
# do the two rungs move the image the same way? (2D fields, not just the radial slice)
python scripts/analysis/rttpf_fields.py --root tmp/r4_paired
```

**`workshop_off` must be `tmp/noncentral/fix15k_workshop`, not the published baseline.**
The published `out/workshop_fisheye_baseline` was trained without `vignetting_comp` and at
`batch_size 1`; pairing against it credits the control with +0.73 dB of configuration fix
that has nothing to do with the camera. This single substitution moves the headline
conclusion by more than the effect being measured.

**-r 4 does not fit on gpu0.** The 11 GB card OOMs during `Raytracer.from_point_cloud` on
every myscenes scene except tunnel (3.8 M init points). Queue the -r 4 matrix on gpu1 only;
`queue_rttpf_control.sh <root> 8 7500 0 ...` is the gpu0-safe variant, but see the noise
caveat below before trusting single-scene -r 8 deltas.

Things worth knowing before running it:

* **Learning rate is in normalized-plane units (~ `fx` pixels), not pixels or COLMAP
  units**, so it transfers across render resolutions. Swept on tunnel (-r 8, 7500 it,
  unfreeze at 1500), test PSNR against 28.06 for `off` and 28.26 for `noncentral`: 1e-6 ->
  28.17, 1e-5 -> 28.22, 1e-4 -> 28.19, 1e-3 -> **28.24**, 3e-3 -> 28.21. That is a plateau
  inside the +-0.06 noise floor, which is the useful part of the result: **the control is
  not learning-rate starved**, so a shortfall against `noncentral` cannot be blamed on
  tuning. `config.py` defaults to 1e-5; pass `--camera_opt_lr_intrinsics 1e-3` for the
  measured optimum.
* **Single-scene -r 8 deltas are near-worthless.** Repeats of the *same* rung land 0.07 dB
  apart, i.e. as far as the effect being measured. The tunnel -r 8 pair happens to show
  `rttpf` recovering ~85 % of the `noncentral` gain; the 7-scene -r 4 paired mean says 25 %.
  Trust the paired mean over seven scenes, never one scene.
* **It costs ~2.6x the `off` training time**, more than `noncentral` (2.0x). At -r 4 / 15k
  that is ~22 min per scene on the TITAN.
* **It only applies to RAD_TAN_THIN_PRISM_FISHEYE.** `train.py` refuses any other
  `--camera_model`; `render.py`'s pinhole eval pass silently renders the *uncorrected*
  camera (and prints one line saying so). Score the fisheye pass, as always.
* **`camera_intrinsics_<iter>.csv`** in the run directory holds the COLMAP value, the
  learned delta and the normalized coefficient for all 16 parameters. The `coefficient`
  column times `fx` is roughly the peak pixel displacement that channel contributes -- but
  the channels of the radial polynomial largely cancel each other, so read the *net* curve
  from `scripts/analysis/rttpf_vs_noncentral.py`, never the individual coefficients.

## Evaluation

**Cross-method numbers only from `scripts/masked_eval.py`** (shared mask, radius 0.95,
identical masked PSNR/SSIM/LPIPS for every method). Never a repo's self-reported metric.

**Where a rung acts** — `scripts/radial_eval.py`, which gray did not have:

```bash
python scripts/radial_eval.py \
  --runs out/tunnel_off out/tunnel_noncentral \
  --labels off noncentral --rings 6
```

It prints per-ring pooled masked PSNR plus deltas vs the first run, and reproduces the
canonical full-disk number exactly (validated: `disk(view)` = 28.537 on
`out/tunnel_fisheye_baseline`, matching the published gray tunnel score). The mask it uses
is the `valid_mask.png` that `render.py` saved, which is built from the COLMAP intrinsics
and is therefore identical across rungs — the learned residual never touches
`cam_info.intrinsics`.

On a **multi-camera rig** it now reads the run's `masks.json` and gives each view the mask
of the lens that took it, reporting `distinct_masks` so you can see it engaged. It used to
apply `valid_mask.png` — which is only the FIRST camera's — to every view; on FullCircle
the two lenses' disks differ by 1864 px (0.36 % of the frame) at the rim, i.e. exactly
where the residual is supposed to act. The bias was identical in every run and so cancelled
in a rung-vs-rung delta, but it corrupted the absolute number.

A radial/anamorphic residual vanishes at `theta = 0` and the non-central term scales as
`sin(theta) z(theta) / depth`, so **both are peripheral by construction**. Full-disk PSNR
averages that away; the ring table is what shows whether a rung did what it claims.

After any baseline run: `cd /workspace/fisheye-baseline-viewer && ./ctl.sh rebuild`.

## FullCircle, `refit_rttpf` track (9 scenes, dual back-to-back fisheye rig)

A second dataset for the camera model, and the one where it does **nothing** — see
IMPLEMENTATION.md "RESULTS — FullCircle". Run it with:

```bash
bash scripts/queue_fullcircle_rttpf.sh <scene> <rung> [gpu] [extra train.py args]
#   scene = room1 room2 room3 flat1 flat2 lab lounge dark persons
#   rung  = off | ... | noncentral        `off` is the paired control
NAME_SUFFIX=_lr1e3 bash scripts/queue_fullcircle_rttpf.sh room2 noncentral 1 \
    --camera_opt_lr_angular 1e-3       # a probe, lands in its own dir
```

The recipe is byte-for-byte `dataset/fullcircle_code/queue_fullcircle_gray_rttpf.sh` — the
script that produced the published `gray` row — with **only `--camera_opt` added**, so the
delta is the camera model. It writes `out/fullcircle_rttpf/<scene>_refit_rttpf[_<rung>]`,
which is where `masked_eval_rttpf.py` looks for methods `gray-nc` / `gray-nc-off`.

Three things this track taught, all of them traps:

* **Queue a VRAM guard, not just a pueue group.** pueue serialises its *own* group; a
  process started outside it can still hold the card. Four runs died on OOM 13 s after
  start because a foreign job held 12 GB of the TITAN RTX. The script now waits for
  >15 GB free before allocating (same guard as `queue_fullcircle_gray.sh`).
* **These scenes peak at ~13 GB at `-r 4`**, so they need the 24 GB card; the 12 GB one
  cannot run them at all.
* **Never quote a timing measured while the card was shared.** The first batch reports
  63-245 FPS and 10-16 min of training on comparable scenes, all of it contention.
  `scripts/measure_fullcircle_fps.sh` re-measures every run of the track, back to back,
  pinned to gpu1 — including the published `gray` runs, which had no `fps.csv` at all.

Evaluation is the shared pass, never a self-reported number:

```bash
python dataset/fullcircle_code/masked_eval_rttpf.py --variants refit_rttpf \
    --methods gray gray-nc gray-nc-off --out .../fullcircle_tracks/rttpf_masked_nc
python dataset/fullcircle_code/merge_masked_metrics.py \
    --from .../rttpf_masked_nc --into .../rttpf_masked
```

**Always eval to a side stem and merge.** Pointing `--out` at the canonical stem rewrites
it with only the methods you ran, silently dropping every other column.

## Analysis of the learned model (`scripts/analysis/`)

Written for the physics report; run in this order, each writes a JSON next to itself.

```bash
python scripts/analysis/collect.py            # per-view + per-ring masked PSNR, all methods
python scripts/analysis/rungs.py tunnel workshop reception   # same, per ablation rung
python scripts/analysis/optics.py             # z(theta), caustic, COLMAP (theta, depth) stats
python scripts/analysis/fields.py             # all 5 azimuthal channels from the checkpoints
python scripts/analysis/crops.py workshop reception   # mechanically-chosen comparison crops
python scripts/analysis/pack.py               # merge into report_data.json
```

Three things worth knowing before reusing them:

* `collect.py` recomputes masked PSNR on CPU in numpy and **reproduces the viewer's GPU
  numbers exactly** (27.878 / 28.036 / ... on all 21 scene-method pairs), so it is a valid
  independent check of the eval path, not just a convenience.
* `rungs.py` resolves runs from BOTH `/workspace/gray/tmp/final` and
  `<worktree>/tmp/final` — `tmp/` is relative to the launching shell's cwd, so the ablation
  runs ended up split between the two roots depending on when they were queued.
* `optics.py` reads `distorted/sparse/0`, the **fisheye** COLMAP model, not `sparse/0`.
  `sparse/0` is the 120-degree pinhole undistortion and its points do not reach the
  periphery, which is exactly the field angle range the whole analysis is about.

`optics.py` is where the physical claim lives: it computes the angular shift a non-central
camera induces, `z(theta) sin(theta) / t`, and the part of it no central model can remove,
`z(theta) sin(theta) * sigma(1/t | theta)` — 0.12 to 0.41 px depending on the scene.

Both conversions go through `plate_scale()`, the LOCAL `dr/dtheta`, never the paraxial
`fx`. On these calibrations `dr/dtheta` drops to 0.56 fx at theta = 90 deg, so using `fx`
inflates every peripheral pixel figure by ~1.8x. If you add an analysis that converts
radians to pixels, use `plate_scale()`.

## Tests

```bash
python -m pytest tests/test_ray_gradients.py tests/test_camera_model.py -q   # 16 tests
```

`pyproject.toml` sets `addopts = "--forked"` (mandatory: two `Raytracer` instances in one
process abort at teardown). Six failures in the full suite are pre-existing / stale fixture
data — see IMPLEMENTATION.md.

`tests/test_ray_gradients.py` is the one that matters: if the CUDA ray gradients are wrong
everything downstream is quietly wrong and nothing else would notice.

## Iteration-matched baselines

SPaGS runs 30k, gray's paper config is 15k. A plain `-t 30000` is **not** an
iteration-matched gray run, because `scale_decay` is per-iteration: the default shrinks
gaussians 0.153x over 15k but 0.0235x over 30k. Measured on `tunnel`, plain 30k scored
**28.42 against 28.54 at 15k**. Pass `--scale_decay 0.9999375` for a 30k run with the same
total shrink.
