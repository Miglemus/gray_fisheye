# PROTOCOL — running the learnable camera model

Worktree `/workspace/gray/worktrees/noncentral-camera`, branch `noncentral-camera`.
See IMPLEMENTATION.md for what was changed and its limitations.

## Setup

```bash
cd /workspace/gray/worktrees/noncentral-camera
OPTIX_HOME=/workspace/gray/third_party/optix bash make.sh    # own build/, NOT shared with main
```

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
  job with a free-memory wait (`/workspace/gray/worktrees/masked-efficiency/scripts/gpuwait.sh
  1 17000 <cmd>`); without one it OOMs at `Raytracer.__init__`.

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
| `radial` | + `d(theta)(theta)` **and** a radially-varying `d(phi)` roll | 23 |
| `ana` | + anamorphic `cos/sin(k phi)`, k=1,2, on `d(theta)` and `d(phi)` | 103 |
| `noncentral` | + on-axis pupil profile `z(theta)`, gauged `z(0)=0` | **111** |
| `central_matched` | central control with a **larger** budget (not matched: 183 vs 111) | 183 |
| `raxel` | dense ray field — **central only, see IMPLEMENTATION.md limitation 8** | ~1e4 |

Counts are *trained* parameters (all rungs allocate 111 either way: the `[5,K]` tensors are
always created, and unused channels sit at zero with zero gradient). Where they live, with
the defaults `--camera_opt_knots 10` and `--camera_opt_knots_z 8`:

| tensor | shape | component | params | what it is |
|---|---|---|---|---|
| `omega` | `[3]` | `tilt` | 3 | axis-angle rotation of the whole bearing field |
| `theta_weights[0]` | `[10]` | `radial` | 10 | `d(theta)` vs theta — the r(theta) mapping residual |
| `phi_weights[0]` | `[10]` | `radial` | 10 | `d(phi)` vs theta — a radially-varying roll (swirl) |
| `theta_weights[1:5]` | `[4,10]` | `ana` | 40 | `d(theta)` x {cos phi, sin phi, cos 2phi, sin 2phi} |
| `phi_weights[1:5]` | `[4,10]` | `ana` | 40 | same four harmonics on `d(phi)` |
| `z_weights` | `[1,8]` | `z` | 8 | `z(theta)`, the axial entrance-pupil profile |

Only `z_weights` is non-central: it moves the ray **origin** along the optical axis. Every
other term rotates the **direction** and leaves a single centre of projection intact.
`central_matched` is `ana` with `knots = 10 + 8 = 18`, i.e. `3 + 2 x 5 x 18 = 183`.

`--pose_opt` adds a per-view SE(3) residual on top of any rung. Report it separately: test
poses stay at COLMAP, so it can cost test PSNR (IMPLEMENTATION.md, limitation 2).

Two **subtractive** rungs sit outside the ladder — they remove capacity `noncentral` has,
to ask whether the non-central term can carry the model alone:

| rung | components | params | 7-scene mean |
|---|---|---|---|
| `noncentral_no_ana` | `noncentral` minus the anamorphic harmonics | 31 | 27.492 |
| `z_only` | the pupil profile alone, nothing central | 8 | 27.441 |

(for reference: `off` 27.228, SPaGS 27.333, `noncentral` 27.561 — both amputations still
beat SPaGS.) Do NOT read these as the ladder reversed: `z` and `radial` are not orthogonal,
so removing the central terms changes what `z` learns. In particular the *amplitude* of the
recovered `z(theta)` is only meaningful on rungs that keep `radial` — see IMPLEMENTATION.md,
"Subtractive rungs". Run them like any other rung:

```
bash scripts/train_myscenes.sh <scene> tmp/final/<scene>_z_only \
     --camera_opt z_only --camera_opt_from_iter 3000
```

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

## Phase 1 — one command for the whole measurement pipeline

Everything Phase 1 measures is joined into **one** file, `scripts/analysis/report_data.json`,
and **every figure is drawn from that file and nothing else**. One command does the lot:

```bash
python scripts/analysis/run_phase1.py          # run what is missing, pack, draw all figures
```

| flag | what it does |
|---|---|
| *(none)* | runs every stage whose **output file is missing**, then always re-runs `pack` and `figures` |
| `--list` | prints every stage, its cost, whether it needs a GPU, and whether its output is on disk |
| `--verify` | deletes `figures/*.{svg,pdf,png}` first, so a full regeneration is *proved*, not assumed |
| `--force <stage> [...]` | recompute those stages even though their output exists |
| `--force all` | recompute every CPU stage (~35 min; the two GPU stages are still refused) |
| `--only <stage> [...]` | run just those stages, still followed by `pack` + `figures` |

It is idempotent and cheap in the normal case (`pack` + `figures` ≈ 20 s). Use `--force` when
the **runs on disk changed**; nothing else invalidates a stage.

**It never launches a GPU task.** `rings_all` and `mask_radius_sweep` need one (LPIPS is a
VGG forward pass). If their output is missing the script prints the exact `pueue add` line and
refuses to run it — queue it yourself, `pueue wait <id>`, then re-run `run_phase1.py`. Every
CPU stage is launched with `CUDA_VISIBLE_DEVICES=""` so it cannot take a card by accident.

### The stages, in dependency order

| stage | writes | cost | GPU | what it is |
|---|---|---|---|---|
| `sfm_residual` | `sfm_residual.json` | ~4 min | no | **A1** — SfM reprojection residual of 62 scene×camera tracks, in µrad (`--workers 5`; never above 6, a worker OOMs at 10) |
| `plate_scale` | `plate_scale.json` | ~45 s | no | **A2** — `S(θ) = dr/dθ` per camera, the evaluated-disk edge, µrad/px |
| `calib_consistency[_urad]` | same name `.json` | ~30 s | no | `E_shared` / calibration disagreement (**use the `_urad` one**) |
| `residual_expressible` | `residual_expressible.json` | ~20 s | no | how much of the learned residual COLMAP's own `k0…k5` could already express (99.2 % on myscenes — but see IMPLEMENTATION.md: expressible ≠ profitable) |
| `dose_response` | `dose_response.json` | ~7 min | no | **W1** — paired gains, three estimators, the pre-registered fits |
| `ranking_flip` | `ranking_flip.json` | ~4 min | no | **W2** — six method-pair regressions + leave-one-dataset-out |
| `rings_all` | `tmp/w3_radial/rings_all.json` | ~25 min | **yes** | **W3** — PSNR/SSIM/LPIPS per equal-area ring, `off` vs `noncentral` |
| `rings_stats` | `tmp/w3_radial/rings_stats.json` | ~10 s | no | W3 — the paired rim-minus-centre statistics |
| `mask_radius_sweep` | `tmp/w3_radial/sweep_*.json` | ~3 h | **yes** | W3 — every method re-scored at r = 0.85 / 0.95 / 1.00 |
| `annulus_audit` | `tmp/w3_radial/annulus_audit.json` | ~10 min | no | who renders black in the 0.95→1.00 annulus, and against which GT |
| `optics` | `optics.json` | ~6 min | no | learned profiles, caustics, depth histograms per scene |
| `fields` | `fields.json` | ~10 s | no | all five azimuthal channels of the learned residual, from the checkpoints |
| `rings` | `rings.json` | ~4 min | no | per-view + 8-ring masked PSNR, gray / noncentral / SPaGS (`collect.py`) |
| `rungs` | `rungs.json` | ~6 min | no | the **six-rung** fisheye ladder per scene, with rings |
| `subtractive` | `subtractive.json` | ~6 min | no | the same ladder through a second entry point (cross-checked at pack time) |
| `crops` | `crops.json` | ~3 min | no | mechanically-chosen qualitative crops, base64 WEBP |
| `ladder` | `ladder.json` | ~2 min | no | **pinhole** ladder + the 2×2 interaction + the regularisation audit |
| `pack` | **`report_data.json`** | ~15 s | no | the join; re-run every time |
| `figures` | `figures/*.{svg,pdf}` | ~40 s | no | all eight figures, from `report_data.json` alone |

Any stage can still be run on its own — `python scripts/analysis/<stage>.py` — and several
take arguments (`rungs.py [scene ...]`, `crops.py [scene ...]`, `mask_radius_sweep.py --track
<name|all>`, `rings_all.py --rings 3`, `dose_response.py --from-json`, `pack.py --strict`,
`make_figures.py --only rings ladder`). `run_phase1.py` is only the orchestration.

### The eight figures

`figures/{dose_response, estimator_agreement, estimator_lens_level, ranking_flip, rings,
ladder, field, crops}.{svg,pdf}` — W1 ×3, W2 ×1, W3 ×1, and three that were previously
un-drawn: the ablation ladder on both camera families, the learned angular field, and the
qualitative crops. `python scripts/analysis/make_figures.py --only <name>` redraws one.

### What `pack.py` checks while joining

It re-verifies six cross-file agreements and prints PASS/FAIL: `rungs.json` vs
`subtractive.json` (0.0 dB over 34 rungs), W1's gains vs W3's ring pass (0.0 dB), the
`radial_eval` non-regression anchor (28.53749608 vs the canonical 28.53749677, −7e-07 dB),
`E_sfm` as used by W1 and by W2 against A1's own column (0.0 µrad, 198 joins), and the
regularisation audit's own falsification. `--strict` turns a missing **required** input into a
failure instead of a hole; `provenance` records the sha256, size and mtime of every input.

### The older per-script entry points (still valid)

```bash
python scripts/analysis/collect.py            # per-view + per-ring masked PSNR, all methods
python scripts/analysis/rungs.py tunnel workshop reception   # same, per ablation rung
python scripts/analysis/optics.py             # z(theta), caustic, COLMAP (theta, depth) stats
python scripts/analysis/fields.py             # all 5 azimuthal channels from the checkpoints
python scripts/analysis/crops.py workshop reception   # mechanically-chosen comparison crops
python scripts/analysis/pack.py               # merge into report_data.json
python scripts/analysis/subtractive.py        # the two subtractive rungs, all 7 scenes
python scripts/analysis/z_shape.py            # is z(theta) still the same curve alone?
```

`subtractive.py` re-scores the published rows first (gray 27.124, config-matched 27.228,
`noncentral` 27.561, SPaGS 27.333) and only then the new rungs, so a protocol drift shows up
as a mismatch on a known number instead of silently biasing the new one. Its per-scene
0.95-mask means agree with the viewer's independent GPU eval to **0.000 dB on all 7 scenes**.

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
`fx`. On these calibrations `dr/dtheta` = **0.554 fx at θ = 90°** and **0.633 fx at 85.5°**,
the angle actually evaluated (`plate_scale.json`, differentiated from the true COLMAP
projection, not from the radial polynomial alone — that shortcut is ~14 % off at the edge).
Converting with `fx` therefore inflates a peripheral figure by ~1.6×, and quoting the 90°
number for a disk cut at 85.5° inflates it by a further 14 %.

**Better still: do not convert at all.** Phase 1's rule is that every camera-error statistic
is reported in **microradians**, because the corpus mixes `-r 4`, `-r 2` and native
resolutions and a fixed angle is a different number of pixels in each. The learned residual is
already in radians in `camera_model_*.csv` (`delta_theta_rad`, `delta_phi_rad`); never round
trip it through pixels.

## Render cost: the pose-independent ray cache, and how to re-measure it

Every `--camera_opt` rung except `off` synthesises its rays in torch and hands them to the
raygen through the framebuffer. **That is the FPS tax, not the non-centrality.** Two readings
off the `fps.csv` already on disk say so:

| | FPS | vs `off` | note |
|---|---:|---:|---|
| `tunnel` -r 4 15k, `ana` / `central_matched` / `noncentral` | 117.44 / 115.98 / 112.08 | — | three rungs **within 5 %**; no `off` twin was timed |
| `tunnel` -r 8 7500 (`tmp/ladder_final`) `off` | 725.22 | 1.00x | |
| ... `passthrough` (rays from Python, **zero** residual) | 625.55 | **0.86x** | the copies alone |
| ... `noncentral` | 375.99 | **0.52x** | copies + synthesis |
| FullCircle `room1` `off` / `noncentral` (clean card) | 471.42 / 249.83 | 0.53x | gaussian counts within 0.5 % |

`passthrough` runs the same Python path with an empty residual, so **0.86x is the floor this
cache cannot go below** — two `[H,W,3]` framebuffer copies and one 3x3 GEMM per image — and
everything between 0.52x and 0.86x is synthesis the cache can hoist. That is the
pre-registered prediction for the benchmark below: cache-on should move `noncentral` from
~0.52x toward ~0.86x of native, and **not** past it.

So `CameraModel.forward()` caches the half of the synthesis that does not depend on the pose,
per `(camera, resolution)`; the rotation into the world stays per image, because it must.

⚠️ The `-r 4` and `-r 8` rows above come from runs whose card occupancy at measurement time is
not recorded (only the FullCircle row was swept on an idle card). Use them for **ratios within
one row**, never as absolute numbers, and re-measure before publishing any of it.

**It is on by default and it is bit-exact.** Two ways to turn it off:

```bash
GRAY_NO_RAY_CACHE=1 python render.py -m ...     # whole process
# or, in code / tests:  raytracer.camera_model.ray_cache_enabled = False
```

Anything that writes a lens parameter **by hand** must call
`raytracer.camera_model.invalidate_ray_cache()`. `step()`, `set_frozen()`, a `state_dict`
load, `set_render_resolution()` and `apply_camera_model_transfer()` already do.

Re-measuring the speedup (GPU — **queue it**, and never with `measure_fps.py`, which
overwrites the run's canonical `fps.csv`):

```bash
ENV1="PATH=/workspace/gray/.venv/bin:\$PATH CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1"
pueue add --group gpu1 --print-task-id -- "cd $PWD && $ENV1 \
  /workspace/gray/worktrees/masked-efficiency/scripts/gpuwait.sh 1 14000 python scripts/bench_ray_cache.py \
    -m /workspace/gray/tmp/final/tunnel_noncentral --repeats 3 \
    --out tmp/p2_ray_cache/tunnel.json"
pueue wait <id>
```

One process, one checkpoint, arms interleaved ABBA so a drift in card contention cancels, and
it **refuses to print a timing** unless the cached and un-cached renders are `torch.equal`.
Good targets:

* `/workspace/gray/tmp/final/{tunnel,workshop}_noncentral` — myscenes, single camera, the
  runs the 112.08 / 102.18 numbers come from. **No `off` twin was ever timed at -r 4**, so
  take the native reference from `tmp/ladder_final` (-r 8) or re-time an `off` run.
* `tmp/ladder_final/{off,passthrough,noncentral}` — the only place all three exist, and
  therefore the only place the 0.86x floor above can be checked directly.
* `out/fullcircle_rttpf/room1_refit_rttpf` — two-camera rig, with a paired `off` (471.42).
  Expect **less** here: the `masked-efficiency` worktree measured its analogous native-path
  bearing cache at 0.995x on a rig, because a rig alternates tables and rebinding costs about
  what the cache saves.

## Does the non-central origin cost BVH coherence?

`scripts/traversal_coherence.py`. Answer so far: **no, to ±1.3 %** — `hits/ray` ratio
0.9981 ± 0.0269 over the 13 `off` / `noncentral` pairs on disk (Wilcoxon p = 0.735).

```bash
python scripts/traversal_coherence.py --from-runs --out tmp/p2_bvh_coherence/from_runs.json
```

CPU, no render: it reads the `traversal_stats.csv` that `train.py` already writes. It is
**confounded** — the paired runs prune to slightly different gaussian counts — so it also
prints the rank correlation with that ratio (−0.374, p = 0.209, i.e. the confound does not
explain the scatter). The controlled version needs a card:

```bash
pueue add --group gpu1 --print-task-id -- "cd $PWD && $ENV1 \
  /workspace/gray/worktrees/masked-efficiency/scripts/gpuwait.sh 1 17000 python scripts/traversal_coherence.py \
    --measure /workspace/gray/tmp/final/workshop_noncentral \
    --out tmp/p2_bvh_coherence/workshop_measured.json"
```

Two arms off **one** checkpoint — the trained `z(theta)`, then `z` forced to zero — so the ray
*directions* are bit-identical and the only difference is whether the primary rays share an
origin. Check `arms_are_a_control` in the output before reading the ratio. It renders under
`enable_grad()` on purpose (the hit counters live behind `if (grads_enabled)`, so a `no_grad`
render silently counts nothing), which means it needs a **training** iteration's VRAM.

## Tests

```bash
# 1. The default suite. Forked, GPU-gated tests skipped. 6 failed / 143 passed / 4 skipped;
#    those 6 are pre-existing and fail identically on `main` (stale `mock_camera` fixtures).
python -m pytest tests/ -q

# 2. The GPU-gated tests. ONE MODULE AT A TIME, and `-o addopts=""` is not optional.
GRAY_RUN_GPU_TESTS=1 python -m pytest tests/test_camera_model_cache.py    -q -o addopts=""  # 36 pass
GRAY_RUN_GPU_TESTS=1 python -m pytest tests/test_camera_model_transfer.py -q -o addopts=""  # 24 pass
```

Three interlocking traps here, all measured on 2026-08-12, all of which produce a red suite
that looks like a code regression and is not:

* **`pyproject.toml` sets `addopts = "--forked"`**, and a forked child cannot initialise CUDA
  if the parent already holds a context. So **no module may call `torch.cuda.is_available()`
  at collection time** — a decorator argument counts. Gate on the env var FIRST and let `and`
  short-circuit: `skipif(not (RUN_GPU and torch.cuda.is_available()))`. One module doing this
  the naive way took the suite from 6 failures to **35**, poisoning every module collected
  after it, pure-CPU tests included.
* **Therefore `GRAY_RUN_GPU_TESTS=1` and `--forked` are mutually exclusive**: setting the env
  var is what lets the parent reach `torch.cuda.is_available()`. Clear `addopts` when you opt
  in. That is why the two commands above look inconsistent — they are not.
* **Unforked, one process can only build so many OptiX scenes**: run two GPU modules back to
  back and the *last* one dies in `rebuild_bvh` with `unspecified launch failure`, then aborts
  at teardown. Order-dependent, not a defect in either module. One module per process.

The cache tests are the ones to run after touching `camera_model.py`: they compare against a
**verbatim copy of the pre-split `forward()`**, so a refactor is checked against the code it
replaced and not only against itself, and they pin the pose down by reconstructing one view's
ray field from another's by the relative rotation.

`tests/test_ray_gradients.py` is the one that matters: if the CUDA ray gradients are wrong
everything downstream is quietly wrong and nothing else would notice.

## Iteration-matched baselines

SPaGS runs 30k, gray's paper config is 15k. A plain `-t 30000` is **not** an
iteration-matched gray run, because `scale_decay` is per-iteration: the default shrinks
gaussians 0.153x over 15k but 0.0235x over 30k. Measured on `tunnel`, plain 30k scored
**28.42 against 28.54 at 15k**. Pass `--scale_decay 0.9999375` for a 30k run with the same
total shrink.
