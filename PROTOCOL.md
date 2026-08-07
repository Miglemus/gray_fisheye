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

A radial/anamorphic residual vanishes at `theta = 0` and the non-central term scales as
`sin(theta) z(theta) / depth`, so **both are peripheral by construction**. Full-disk PSNR
averages that away; the ring table is what shows whether a rung did what it claims.

After any baseline run: `cd /workspace/fisheye-baseline-viewer && ./ctl.sh rebuild`.

## Tests

```bash
python -m pytest tests/test_ray_gradients.py tests/test_camera_model.py -q   # 15 tests
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
