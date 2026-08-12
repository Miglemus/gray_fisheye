# IMPLEMENTATION — learnable residual / non-central camera model in gray

Branch `rttpf-intrinsics`, a worktree of `noncentral-camera` (everything below the
"re-calibration control" section describes that parent branch and is unchanged). Adds the
ability to **learn the camera model jointly with the gaussians**: a residual on top of the
COLMAP calibration, including a genuinely **non-central** (axial entrance-pupil) term, plus
an optional per-view SE(3) pose residual.

**What this branch adds on top: `--camera_opt rttpf`, the re-calibration control.** It
optimizes COLMAP's *own* 16 RAD_TAN_THIN_PRISM_FISHEYE parameters photometrically during
training, so that "the non-central model wins" can be separated from "the camera was
allowed to move during training, and the baseline's was not". See
[the section on it](#the-re-calibration-control----camera_opt-rttpf).

The scientific point is that a ray tracer needs only the *backward* map `pixel -> (origin,
direction)`, which stays closed-form even when the camera has no single centre of
projection. gray's BVH is in world space and `__intersection__is` reads
`optixGetObjectRayOrigin()` per ray, so **the traversal needed no change at all**. The one
thing gray was missing was `dL/d(ray)`.

---

## What was added

| file | change |
|---|---|
| `cuda/core/framebuffer.h` | `grad_ray_origin`, `grad_ray_direction` (`[H,W,3]`) in `Framebuffer`, `FramebufferDataHolder`, `reify()` and the pybind `bind()` |
| `cuda/backward_pass.cu` | `rotate_grad_to_world()` helper; per-pixel accumulation of `dL/d(ray_origin)` and `dL/d(ray_direction)`, gated on `config.rays_from_python` |
| `gray/camera_model.py` | **new** — `CameraModel`, `LensResidual`, `PoseResidual`, B-spline basis helpers |
| `gray/raytracer.py` | `upload_camera_intrinsics()`, `base_bearings()`, `_rotation_c2w_cuda()`, ray injection in `__call__`, third backward stage, `camera_model.step()`, `materialize_from_state_dict()` on reload |
| `gray/config.py` | `camera_opt` rung enum + LRs/regularizers/knot counts, `pose_opt` |
| `train.py` | camera-model LR schedule, phase A/B freeze, regularization term, `scene_scale` |
| `tests/test_ray_gradients.py` | **new** — 3 tests validating the CUDA ray gradients |
| `tests/test_camera_model.py` | **new** — 11 tests (passthrough equality, gauge, subdivision, per-rung training step, monotonicity) |
| `scripts/radial_eval.py` | **new** — radially-binned masked metrics |
| `scripts/train_myscenes.sh`, `scripts/eval_rttpf.sh` | **new** — run wrappers (see PROTOCOL.md) |
| `tests/test_eval_modes.py` | bug fix, see "pre-existing issues" |

Added by `rttpf-intrinsics` on top of that:

| file | change |
|---|---|
| `gray/camera_model.py` | `rttpf` / `rttpf_z` rungs: `rttpf_project`, `rttpf_plate_scales`, `rttpf_normalization`, `rttpf_tables`, `rttpf_delta_params`, `rttpf_solve`, `LensResidual.intrinsics`, `CameraModel.rttpf_table` / `intrinsic_report` |
| `gray/raytracer.py` | `scaled_intrinsics()` extracted from `upload_camera_intrinsics()`; `base` dict now carries `model` and the render-resolution `intrinsics` |
| `gray/config.py` | two rungs, `camera_opt_lr_intrinsics`, `camera_opt_reg_intrinsics` |
| `train.py` | guard against a non-rttpf camera model; `camera_intrinsics_<iter>.csv`; the Newton residual as a scalar |
| `tests/test_camera_model.py` | 5 more tests (zero-exactness, agreement with the raygen under a perturbed calibration, resolution invariance, a training step per new rung) |
| `scripts/train_myscenes_at.sh` | **new** — `train_myscenes.sh` with resolution and iterations exposed, for sweeps |
| `scripts/queue_rttpf_control.sh` | **new** — queues the scene x rung matrix with the VRAM guard |
| `scripts/rttpf_control_table.py` | **new** — paired table, re-scored from the PNGs, with the live-vs-rendered cross-check |
| `scripts/analysis/rttpf_vs_noncentral.py` | **new** — do the two rungs learn the same radial correction? |
| `scripts/analysis/rttpf_solve_debug.py` | **new** — splits an inverse-solver discrepancy into base error / convergence / model mismatch |

## How it works

```
Python (torch autograd)                          CUDA / OptiX
base bearings (cached, detached) --------------> [probe: raygen with identity pose]
        v
residual (~40 shared params)
        v
o, d  --copy_()-->  framebuffer.ray_origin/ray_direction --> raygen (rays_from_python=True)
                                                                  v
                                                            forward / backward
                                                                  v
grads <----------- framebuffer.grad_ray_origin/grad_ray_direction
        v
torch.autograd.backward([o, d], [go, gd])  ->  the ~40 parameters
```

### The CUDA gradient (the load-bearing derivation)

Per hit, with `o, d` in world space and `W2L` the gaussian's world-to-local transform:

```
lo = W2L (o,1)   ld_raw = W2L_rot d   n = |ld_raw|   ld = ld_raw/n
t  = -(lo.ld)    h_un = lo + t ld  =  P_perp(ld) lo
```

`backward_pass.cu` already had `grad_x_local = dL/dh` and `s = scaling_factor`, and its own
`grad_x_world` line establishes `dL/dh_un = s * grad_x_local`. From there, with `g = dL/dh_un`:

```
dL/dlo     = g - (ld.g) ld
dL/dld     = t g - (ld.g) lo
dL/dld_raw = ( dL/dld - (dL/dld . ld) ld ) / n
dL/do      = W2L_rot^T dL/dlo          dL/dd = W2L_rot^T dL/dld_raw
```

The backward launch is one OptiX thread per pixel looping over that pixel's hits, so the
accumulation is a register and a single store — **no atomics**.

### Why the base bearings are probed, not recomputed

`Raytracer.base_bearings()` renders one throw-away frame with an identity pose, `zfar=1e-6`
and `needs_ray_output=True`, then reads `framebuffer.ray_direction`. The bearings are
therefore bit-for-bit the ones the native raygen traces, for *any* camera model, with no
duplicated math — and no exposure to the `converged` latch bug in `gray/fisheye_geometry.py`
(`:39, :89, :160`), whose numpy `valid` is systematically wider than what CUDA actually
traces. Cached per `(camera uid, render width, render height)`; a bearing never depends on
the pose.

### Why the residual is applied as rotations

`b = b0 cos(d) + (n x b0) sin(d)` with `n` the meridional axis is **exactly** `b0` when
`d = 0`, whereas a `b -> (theta, phi) -> b` round trip loses ~1e-7 in float32. `theta_base`
is then only ever a (detached) spline coordinate, so its precision is irrelevant. Every
operation is a rotation, so bearings stay unit-norm and the **invalid sentinel outside the
lens disk (a zero bearing) survives automatically** — rotating zero gives zero.

Measured: `passthrough` reproduces the native path to **1.8e-7** max chord.

---

## The re-calibration control — `--camera_opt rttpf`

### Why it exists

Every rung above learns *something the baseline camera cannot express*, and is compared
against a baseline whose camera is **frozen at the COLMAP fit**. Two explanations of the
myscenes gain survive that comparison:

1. the non-central model is richer, or
2. the non-central model was **allowed to move during training and the baseline was not**.

`residual_expressible.py` already argued for (2) offline — 98.6 % of the learned radial
correction lies in the span of rttpf's own `k0..k5` — but that is a statement about what a
re-fit *could* reach, not about what photometric descent *does* reach. The direct experiment
is to hand the optimizer COLMAP's own 16 parameters and change nothing else.

### How it works

The rung learns a delta on `fx, fy, cx, cy, k0..k5, p0, p1, s0..s3` and expresses it, like
every other rung, as a rotation of the base bearing. The chain, per pixel:

```
base bearing b0  --(theta/sin theta)-->  w0 = theta (cos phi, sin phi)     [fisheye plane]
target pixel p   =  project(w0, colmap_params)                            [cached, exact]
solve           project(w, colmap_params + delta) = p     by Newton on w
rotation        d_theta = |w| - |w0|,  d_phi = angle(w0 -> w)  --> the existing rotations
```

Four things about that are load-bearing:

* **Only the forward map is implemented.** gray's raygen inverts rttpf with a 100-step
  fixed-point iteration; re-implementing *that* in torch (and differentiating through it)
  would be a second source of truth for the camera model. Newton needs only
  `rttpf_project`, which is COLMAP's `Distortion` transcribed once.
* **The target pixel is `project(w0, colmap_params)`, not the pixel grid.** It makes a zero
  coefficient give a *bitwise* zero rotation, so the rung is exactly `off` at init in the
  same sense every other rung is — and it inherits the raygen's own solver error instead of
  competing with it. (Measured: the two agree to 1e-5 px anyway.)
* **The Jacobian must be recomputed at each Newton step.** Frozen at the base bearing it
  converges only *linearly*, at rate ~0.25, because `rho''/rho' ~ -1.9 per radian at the rim
  where the degree-13 polynomial turns over. That cost 1.4e-3 rad of error on a large
  perturbation — 300x the tolerance — and is the one non-obvious thing in the file.
* **The 16 coefficients are normalized**, each to "one unit of normalized-plane displacement
  at the worst-affected pixel", on an analytic theta grid rather than the pixel grid. They
  span nine orders of magnitude and a unit of `k0` moves the image ~1000x further than a
  unit of `cx`; Adam normalizes the gradient but not the step, so without this a single
  learning rate cannot be fair to all 16. Doing the normalization on the *pixel* grid
  instead makes a coefficient mean 5 % more at a different render resolution
  (`test_rttpf_coefficients_mean_the_same_lens_at_any_resolution` pins this).

`test_rttpf_reproduces_the_cuda_unprojection_of_a_perturbed_calibration` is the test that
makes the control legitimate: it writes a perturbed calibration straight into `cam_info`,
probes the *native* raygen with it, and requires the rung's rays to match — 4.6e-6 rad on a
perturbation 100x larger than training ever applies, which is the float32 noise floor of the
reference itself (a float64 solve of the same equation lands at 4.7e-6 from the raygen).

### Cost and caveats

* **~2.6x the training time of `off`** (tunnel, -r 8, 7500 it: 6:30 against 2:31; the
  `noncentral` rung is 5:00). Four Newton steps of a ~30-kernel projection per rendered
  frame, launch-bound rather than flop-bound. Two steps would converge for perturbations
  this small; four is kept because a diverging run must not also silently stop solving.
* **The rung is inert on any camera that is not RAD_TAN_THIN_PRISM_FISHEYE**, and says so
  once per camera. `train.py` refuses to start in that case, but `render.py --eval-models
  pinhole rttpf` legitimately hits it: the *pinhole* re-render of an rttpf run is the
  uncorrected camera. Only the fisheye pass is ever scored, so this affects nothing that is
  reported — but do not read a pinhole number off one of these runs.
* `monotonicity_margin()` returns 1.0 for this rung (it inspects the splines, which stay at
  zero). The Newton residual in the tensorboard log is the equivalent health signal:
  anything above ~1e-3 px means the solve stopped converging.

## RESULTS — myscenes, rttpf track, masked 0.95, per-view mean

Scored by `scripts/myscenes_table.py`, which reads the rendered PNGs through
`radial_eval.evaluate` (validated to reproduce gray's published numbers to +0.000 on all
seven scenes) and reads the rival from the canonical shared-mask aggregate. `ours` is
`--camera_opt noncentral`, 15k iterations, unfreeze at 20%, lr 1e-4.

| scene | gray (published) | ours | delta | SPaGS | ours - SPaGS |
|---|---|---|---|---|---|
| atrium | 27.878 | 28.036 | +0.158 | 27.724 | **+0.312** |
| classroom | 28.562 | 28.973 | +0.411 | 28.723 | **+0.250** |
| forest | 19.565 | 19.645 | +0.080 | 19.595 | **+0.050** |
| library | 31.580 | 31.853 | +0.273 | 31.642 | **+0.211** |
| reception | 27.277 | 27.751 | +0.474 | 27.352 | **+0.399** |
| tunnel | 28.537 | 28.729 | +0.191 | 28.983 | -0.255 |
| workshop | 26.470 | 27.942 | +1.471 | 27.312 | **+0.629** |
| **mean** | **27.124** | **27.561** | **+0.437** | **27.333** | **+0.228** |

Ahead on 6 of 7 scenes. Full metric set under the same shared-mask protocol
(`scripts/full_metrics.py`; LPIPS uses masked_eval's `frame_lpips`):

| method | PSNR | SSIM | LPIPS | trained camera params |
|---|---|---|---|---|
| gray (published) | 27.124 | 0.9458 | 0.1600 | 0 |
| SPaGS | 27.333 | **0.9514** | 0.1657 | 0 |
| DirectFisheye-GS | 26.818 | 0.9445 | 0.1872 | 0 |
| 3DGUT | 25.930 | 0.9306 | 0.2272 | 0 |
| **gray + camera model** (`noncentral`) | **27.561** | 0.9491 | 0.1547 | 111 |
| **gray + `rttpf_z`** | 27.554 | 0.9492 | **0.1546** | **17** |

`rttpf_z` ties the full residual model on **all three** metrics — PSNR within 0.007, SSIM
+0.0001, LPIPS -0.0001 — with 6x fewer trained camera parameters. Same shared-mask eval
pass, `scripts/full_metrics.py --runs-root tmp/r4_control --suffix _rttpf_z`.

We take PSNR and LPIPS; **SPaGS keeps SSIM** (0.9514 vs 0.9491). The camera model narrows
the SSIM gap from 0.0056 to 0.0023 but does not close it -- say so rather than reporting
PSNR alone. (FPS and train-time were measured while another job shared the GPU and are not
trustworthy; re-measure on an idle device before publishing them.)

Decomposition of the +0.437:

* **+0.104** is the `workshop` configuration fix (that scene alone had been run without
  `vignetting_comp` and at `batch_size 1`), which is *not* attributable to this work;
* **+0.333** is the camera model. On its own it puts gray at **27.457**, i.e. it clears
  SPaGS (27.333) without the configuration fix.

Run-to-run noise is about ±0.06 dB (`passthrough`, which is mathematically identical to
`off`, lands 0.06 apart on a repeat), so per-scene deltas below that are ties; the 7-scene
mean averages it down to ~±0.02.

Levers measured and **rejected**: plain 30k iterations (-0.12 dB, the `scale_decay` trap),
and per-view `pose_opt` (-0.13 dB on test while train rises).

## RESULTS — the re-calibration control: does the gain survive it?

The objection this branch exists to answer: `noncentral` is optimized by gradient descent
during training and every rival is frozen at its COLMAP fit, so the win could be *"one
method got to move its camera"* rather than *"one method has a better camera model"*.
`--camera_opt rttpf` removes that asymmetry — it hands the **baseline's own 16-parameter
rttpf calibration** to the same optimizer, on the same schedule, in the same renderer.

Seven myscenes, -r 4, 15k iterations, unfreeze at 20 %, everything but `--camera_opt`
identical, all three rungs starting from the same cached `point_cloud.safetensors`. Scored
from the PNGs by `scripts/rttpf_control_table.py` (shared 0.95 mask, `disk_per_view_mean`):

| scene | off (frozen COLMAP) | rttpf (re-calibrated) | noncentral | d rttpf | d noncentral | nc - rttpf |
|---|---|---|---|---|---|---|
| atrium | 27.878 | 27.933 | 28.036 | +0.055 | +0.158 | +0.102 |
| classroom | 28.562 | 28.653 | 28.973 | +0.091 | +0.411 | +0.319 |
| forest | 19.565 | 19.614 | 19.645 | +0.049 | +0.080 | +0.031 |
| library | 31.580 | 31.647 | 31.853 | +0.067 | +0.273 | +0.206 |
| reception | 27.277 | 27.519 | 27.751 | +0.241 | +0.474 | +0.233 |
| tunnel | 28.537 | 28.533 | 28.729 | -0.004 | +0.191 | +0.195 |
| workshop | 27.198 | 27.270 | 27.942 | +0.071 | +0.743 | +0.672 |
| **mean** | **27.228** | **27.310** | **27.561** | **+0.082** | **+0.333** | **+0.251** |

**Re-calibration is real but small: +0.082 dB (stderr 0.029, positive on 6/7).** It recovers
**25 %** of the camera model's +0.333. `noncentral` still beats the re-calibrated control by
**+0.251 dB, stderr 0.078, on 7/7 scenes** — so the answer to "is it just optimization?" is
no: about a quarter of the headline gain is the freedom to move the camera at all, and three
quarters needs the *shape* of the model.

Note the `off` column here uses the fixed `workshop` configuration (27.198, not the
published 26.470), so this table's +0.333 is the camera model alone with the configuration
fix already removed — consistent with the decomposition above.

### The control is not under-tuned

Every knob that could hobble it was swept (tunnel, -r 8, 7500 it):

* **intrinsic learning rate**, 7 points: 1e-6 → 28.17, 1e-5 → 28.22, 1e-4 → 28.19,
  **1e-3 → 28.24**, 3e-3 → 28.21, 1e-2 → 28.19, 3e-2 → 28.07 (diverging). A plateau 0.07 dB
  wide — the same size as run-to-run noise. 1e-3 is what the table above uses.
* **the same learning rate re-swept at -r 4 / 15k**, on the two scenes where `noncentral`
  wins by the most, because an optimum found at -r 8 / 7500 need not transfer:

  | scene | lr 1e-4 | lr 1e-3 (the table) | lr 1e-2 | `noncentral` |
  |---|---|---|---|---|
  | workshop | 27.28 | 27.26 | 27.21 | **27.94** |
  | classroom | 28.64 | 28.63 | 28.36 | **28.97** |

  Two orders of magnitude of learning rate move the control by ≤ 0.07 dB while `noncentral`
  stays 0.66 and 0.34 dB ahead. Even a per-scene oracle that picked the best of the three
  would gain the control 0.02 dB.
* **unfreeze point**: 0 % → 28.21, 20 % → 28.24, 40 % → 28.21.
* **noise floor**: repeats of the same rung land 0.07 dB apart (`nc` 28.26/28.19,
  `rttpf` 28.24/28.17), so single-scene deltas at -r 8 are worth little; the 7-scene paired
  mean is the number to read.

No setting of the control's own hyper-parameters comes within 0.2 dB of closing the gap.

**The one objection this does not answer: the control has not converged at 15k, and the
headline rung has.** Weighted rms displacement of the learned field between the 7500 and
15000 checkpoints, as a fraction of the field's own magnitude:

| | `noncentral` | `rttpf` |
|---|---|---|
| field drift, 7.5k → 15k | 5-47 % (median ~24 %) | **35-146 % (median ~103 %)** |
| `z` drift | 1-24 % | — |

The intrinsic rung is still moving by about its own size when training stops; the residual
rung has largely settled. The learning-rate sweep argues against this mattering (10x the
learning rate does not help and 100x hurts, which is not what a step-starved optimizer looks
like) but it is not a substitute for more iterations. Credit to the `paper-brainstorm` session
for raising it.

**The *a fortiori* test settles it: 30k, `--scale_decay 0.9999375` (the documented fix for
the per-iteration trap), same four rungs, on `workshop` (widest gap) and `tunnel` (where the
control scored exactly -0.004 at 15k).**

| scene / rung | 15k | 30k | budget effect |
|---|---|---|---|
| tunnel / off | 28.537 | 28.616 | +0.079 |
| tunnel / rttpf | 28.533 | 28.735 | +0.201 |
| tunnel / rttpf_z | 28.732 | 28.732 | -0.000 |
| tunnel / noncentral | 28.729 | 28.814 | +0.086 |
| workshop / off | 27.198 | 27.303 | +0.105 |
| workshop / rttpf | 27.270 | 27.423 | +0.153 |
| workshop / rttpf_z | 27.937 | 28.139 | +0.201 |
| workshop / noncentral | 27.942 | 28.104 | +0.162 |

Three things, in decreasing order of confidence:

1. **The under-training objection is dead.** At doubled budget `noncentral - rttpf` is still
   +0.080 (tunnel) and +0.681 (workshop). The control does not catch up.
2. **The 25 % figure is budget-dependent and must be quoted as "at 15k".** `rttpf - off` goes
   -0.004 -> +0.119 and +0.071 -> +0.120. Note the two 30k values agree to 0.001 dB across
   two very different scenes, which looks like a systematic lens re-calibration worth a
   scene-independent ~0.12 dB rather than a scene-dependent effect. If that holds on the
   other five, the converged re-calibration share is ~0.12/0.36 ~ **33 %**, not 25 % and not
   the 40-45 % a naive extrapolation of the two deltas suggests. **Not measured — do not
   quote either extrapolation.**
3. **`rttpf_z == noncentral` is established at 15k and simply not yet tested at 30k.** Mean
   PSNR difference over the two 30k scenes is -0.024 (workshop +0.035, tunnel -0.082). Two
   scenes at ~1.3x the 0.06 dB noise floor cannot establish or refute a tie either way. **The
   7-scene tie at 15k (p = 0.76) stands; a 30k tie needs the other five scenes.**

   Across all three metrics at 30k, nothing separates the two models consistently — PSNR
   splits by scene, SSIM favours `noncentral` by 0.0003 on both, LPIPS is a wash:

   | 30k | PSNR | SSIM | LPIPS |
   |---|---|---|---|
   | tunnel: rttpf_z / noncentral | 28.7320 / **28.8143** | 0.95442 / **0.95477** | **0.15474** / 0.15483 |
   | workshop: rttpf_z / noncentral | **28.1387** / 28.1036 | 0.95685 / **0.95712** | 0.17165 / **0.17159** |

   **Do not repeat the reading this section carried first**, that `tunnel/rttpf_z` "gained
   nothing from the doubled budget" because its PSNR read 28.7321 at 15k and 28.7320 at 30k.
   Only the *PSNR* coincided. Its SSIM moved +0.00147 and its LPIPS -0.00268, both inside the
   range of the other seven runs (+0.00147..+0.00229 and -0.00189..-0.00841). It is an
   ordinary run and a ~1 % coincidence over eight pairs, not an anomaly. Caught by the
   `paper-brainstorm` session; confirmed here independently. The lesson is the general one:
   **a single metric agreeing to 4 decimal places is a coincidence to check against the other
   metrics, not evidence about the run** — and the check costs one script.

### The two rungs are not finding the same correction

`scripts/analysis/rttpf_fields.py` compares the actual per-pixel displacement each rung
applies (both fields produced by the renderer's own code, on an analytic θ/φ grid). If the
re-calibration were merely a clumsier route to the same correction, the fields would be
parallel. They are not:

| scene | \|rttpf\| px | \|nc\| px | cos | explained | z at scene depth px |
|---|---|---|---|---|---|
| atrium | 0.119 | 0.103 | +0.21 | -0.83 | 0.122 |
| classroom | 0.178 | 0.127 | -0.15 | -2.37 | 0.209 |
| forest | 0.068 | 0.171 | +0.08 | -0.09 | 0.176 |
| library | 0.129 | 0.169 | -0.64 | -1.56 | 0.160 |
| reception | 0.124 | 0.159 | +0.13 | -0.41 | 0.207 |
| tunnel | 0.112 | 0.194 | -0.19 | -0.55 | 0.167 |
| workshop | 0.159 | 0.326 | -0.87 | -1.08 | 0.683 |

Cosines scatter around zero (three are strongly *negative*) and the explained fraction is
negative on all seven: the re-calibration does not approximate the residual model, it applies
an unrelated small correction that happens to also help a little.

### What the extra 0.251 dB actually is

The last column above is the non-central `z(θ)` displacement at the rim, divided by each
scene's own median point-to-camera distance — i.e. the image-space shift the term buys, in
pixels, at the depth the scene actually sits at. It predicts the residual gap:

* `z at scene depth` vs (nc - rttpf): **Pearson r = 0.93, p = 0.003**.
* Without the depth normalization it is only r = 0.72, p = 0.066 — `forest` is the outlier,
  and it is an outlier for a mechanical reason: 2.17 px per unit depth but a median depth of
  12.3, the largest of the seven. Dividing by depth puts it back on the line.

The direct decomposition agrees, and it is the cleanest result on this branch. `rttpf_z` is
the re-calibration **plus one scalar profile `z(θ)`** and nothing else — no splines, no
tilt, no anamorphic term. Same seven scenes, same -r 4 / 15k recipe:

| scene | off | rttpf | **rttpf_z** | noncentral | d rttpf | **d rttpf_z** | d noncentral |
|---|---|---|---|---|---|---|---|
| atrium | 27.878 | 27.933 | 28.004 | 28.036 | +0.055 | +0.126 | +0.158 |
| classroom | 28.562 | 28.653 | 28.899 | 28.973 | +0.091 | +0.337 | +0.411 |
| forest | 19.565 | 19.614 | 19.759 | 19.645 | +0.049 | **+0.194** | +0.080 |
| library | 31.580 | 31.647 | 31.809 | 31.853 | +0.067 | +0.229 | +0.273 |
| reception | 27.277 | 27.519 | 27.740 | 27.751 | +0.241 | +0.462 | +0.474 |
| tunnel | 28.537 | 28.533 | 28.732 | 28.729 | -0.004 | +0.195 | +0.191 |
| workshop | 27.198 | 27.270 | 27.937 | 27.942 | +0.071 | +0.739 | +0.743 |
| **mean** | **27.228** | **27.310** | **27.554** | **27.561** | **+0.082** | **+0.326** | **+0.333** |

Paired contrasts over the seven scenes:

| contrast | mean | stderr | t | p | Wilcoxon | positive |
|---|---|---|---|---|---|---|
| rttpf - off | +0.082 | 0.029 | 2.82 | 0.030 | 0.031 | 6/7 |
| rttpf_z - rttpf | **+0.244** | 0.074 | 3.31 | 0.016 | 0.016 | **7/7** |
| noncentral - rttpf | +0.251 | 0.078 | 3.21 | 0.018 | 0.016 | 7/7 |
| **noncentral - rttpf_z** | **+0.007** | 0.022 | 0.31 | **0.764** | 0.375 | 5/7 |

**Re-calibration plus one scalar `z(θ)` reproduces 98 % of the full camera model**, and the
difference between them is a statistical zero (+0.007 ± 0.022, p = 0.76; per-scene agreement
within 0.03 dB on four of seven, and on `forest` the two-term model is 0.114 dB *ahead*).
Everything the splines add on top — tilt, radial and anamorphic residuals, 100+ parameters —
is worth nothing once the camera is re-calibrated and allowed one non-central degree of
freedom.

So the gain is the non-central degree of freedom: the thing a re-calibration of a *central*
model cannot express at any learning rate. Not the extra optimizer freedom, and not the
extra parameter count — `rttpf_z` has 17 trained camera parameters against `noncentral`'s
111 and matches it.

(The earlier single-scene version of this table, tunnel at -r 8: off 28.091, rttpf 28.148,
rttpf_z 28.258, noncentral 28.254. Same conclusion, but both intrinsic rungs there ran at
lr 1e-5, the default at the time, so those rows are comparable to each other and not to the
-r 4 table.)

## RESULTS — `workshop_immervision` (rttpf), the hardest lens available

An **ImmerVision panomorph**: elliptical footprint, anamorphic `f_x/f_y = 1.294`, principal
point 39 px right / 19 px above centre, local magnification peaking at **1.65x near 60 deg**.
The prior was that a richer camera should pay off *most* here. It does not, and the honest
summary is that the gain is the same small one seen on myscenes.

Recipe identical to the published `gray` row (`-r 1` — 1440x1080 is already the working
resolution — rttpf, batch 2, `--vignetting_comp --vignetting_terms 3`, 15k, same shared EDGS
init), only `--camera_opt` differs. Shared masked pass, r=0.95 (47.9 % of the frame), n=26:

| method | PSNR | SSIM | LPIPS | FPS | train | #gauss |
|---|---|---|---|---|---|---|
| 3DGUT | **25.800** | **0.8534** | 0.3789 | 176 | 733 s | 64 k |
| **gray + camera model** | 25.363 | 0.8341 | **0.3514** | ~52 | 1249 s | 962 k |
| gray (published) | 25.220 | 0.8320 | 0.3539 | 123 | 813 s | 973 k |
| SPaGS | 23.826 | 0.8456 | 0.3773 | 221 | 1373 s | 251 k |
| DirectFisheye-GS | 20.552 | 0.7891 | 0.4692 | 182 | 2190 s | 131 k |

* **+0.143 dB over published gray, +0.081 over an `off` control run in this worktree**
  (25.282, which itself reproduces the published 25.220 to within the ±0.06 noise — so the
  branch's CUDA changes are neutral, as intended). **+0.081 is at the noise floor**: it is
  one run on one scene and must not be quoted as a real gain on its own.
* **It does not overturn the ranking.** 3DGUT keeps PSNR (+0.44) and SSIM (+0.019) with
  **15x fewer gaussians**. The camera model takes **LPIPS outright** (0.3514, best of all
  five), which is the one honest headline.
* **The richer lens did not buy a richer gain**, which is the interesting negative result:
  +0.14 here against +0.44 on the myscenes 7-scene mean. Consistent with the COLMAP finding
  that rttpf's 8 extra parameters only buy 3 % reprojection on this lens (1.103 -> 1.067 px)
  — the calibration is already close to saturated, so there is little for a residual to
  recover. Contrast myscenes, where COLMAP pinned `cx, cy` at the exact sensor centre and
  never refined them.
* **The camera model costs ~55 % more training time** (1249 s vs 813 s) and roughly halves
  render FPS at this resolution, because ray synthesis runs in Python per frame. Both numbers
  were measured on a shared GPU and are indicative only.

## RESULTS — FullCircle `refit_rttpf`, 9 scenes: the camera model buys nothing

The cleanest negative result of the branch, and the best-controlled: **+0.016 dB** on a
9-scene paired mean. Worth having precisely because the control is tight enough to say
"nothing" rather than "we could not tell".

Recipe: byte-for-byte the script that produced the published `gray` row
(`dataset/fullcircle_code/queue_fullcircle_gray_rttpf.sh` — 15k, `-r 4`, rttpf, batch 2,
vignetting 3 terms, `--llffhold 0`, person masks, `pruning_min_weight 1e-8`, the track's
shared `point_cloud.safetensors`), **one flag added**. A `--camera_opt off` control was run
in this worktree on all 9 scenes, so the comparison is paired and same-code; `gray` is the
published row, scored by the same shared masked pass.

| | room1 | room2 | room3 | flat1 | flat2 | lab | lounge | dark | persons | mean |
|---|---|---|---|---|---|---|---|---|---|---|
| gray (published) | 30.268 | 30.340 | 28.559 | 27.626 | 28.365 | 28.834 | 22.709 | 28.451 | 29.803 | 28.328 |
| `off` (this worktree) | 30.224 | 30.368 | 28.618 | 27.606 | 28.387 | 28.830 | 22.734 | 28.425 | 29.791 | **28.332** |
| `noncentral` | 30.282 | 30.422 | 28.607 | 27.600 | 28.382 | 28.816 | 22.754 | 28.491 | 29.770 | **28.347** |
| *nc − off* | +0.057 | +0.054 | −0.011 | −0.006 | −0.005 | −0.014 | +0.020 | +0.066 | −0.021 | **+0.016** |

* **The `off` control reproduces the published gray row to +0.003 dB on the mean** (and its
  gaussian counts land within 0.1-1 % — 156 206 vs 156 516 on room1). The branch's CUDA
  changes are neutral, and the published row is a legitimate baseline for this track.
* **+0.016 dB, std 0.033 across scenes** (standard error 0.011). Positive on 4 of 9,
  negative on 5. This is zero.
* **It does not change the ranking.** SPaGS' fisheye port keeps this track at **28.678**,
  0.33 dB ahead — the camera model closes none of it. Contrast myscenes, where the same
  rung took gray from 27.124 past SPaGS' 27.333.

### The residual is doing the right thing, there is just nothing to correct

Per equal-area ring, `noncentral` − `off`, averaged over the 9 scenes
(`scripts/radial_eval.py --rings 6`):

| ring (r/R) | 0-.41 | .41-.58 | .58-.71 | .71-.82 | .82-.91 | .91-1.0 | disk |
|---|---|---|---|---|---|---|---|
| mean | −0.014 | −0.002 | +0.011 | +0.018 | +0.019 | **+0.037** | +0.016 |
| std | 0.050 | 0.037 | 0.064 | 0.064 | 0.065 | 0.068 | 0.033 |

Monotonically increasing outward — **exactly the peripheral signature the model predicts**,
and the sign is right. It is simply 35x smaller than `workshop`'s +1.305 on the same ring.
So this is not a broken rung; it is a rung with nothing to eat.

Two independent measurements say the same thing:

* **The learned residual is ~2.5x smaller than on myscenes.** Checkpoint magnitudes:
  `theta_weights` 1.1-1.6e-3 here against 4.0e-3 on the myscenes ladder, `z_weights`
  2.3-3.9e-3 against 6.5e-3. At this plate scale 1.4e-3 rad is about **0.3 px**.
* **The calibration was already re-bundled with the full rttpf model.** This track is a
  bundle adjustment that took median reprojection error from 1.102 to 0.866 px
  (`dataset/fullcircle_baselines/refit_gains.md`), 0.062 px of which came from the rttpf
  parameters themselves. myscenes, by contrast, ran on a raw COLMAP fit with `cx, cy`
  pinned at the exact sensor centre and never refined.

The same pattern, weaker, was already visible on `workshop_immervision` (+0.081 over its
own `off` control, on a lens whose rttpf terms only bought 3 % reprojection). **The rule
that fits all three datasets: the camera model recovers what the calibration left on the
table, and nothing more.** It is worth its cost on a raw COLMAP fisheye fit and worth
nothing on a re-bundled one — which is a useful thing to know before spending it.

Note the principal point is pinned at *exactly* `(1440.0, 1440.0)` for both lenses even
after the re-fit, so the `tilt` term did still have a free 3 DoF here; it was not enough.

### Cost

| | train | #gauss |
|---|---|---|
| `off` | 4:30-5:11 | 62.9k-278k |
| `noncentral` | 9:12, 9:47 (clean card) | within 1 % of `off` |

**~2.1x training time** for nothing, on this track. The 14-16 min figures in the first
batch's `time.csv`, and every FPS in it (63 to 245 on comparable scenes), are contention
artefacts — a foreign process held 12 GB of the same card. Only `dark` and `persons` were
trained on a card this worktree had to itself. `scripts/measure_fullcircle_fps.sh` re-times
the whole track, gray included.

### Two traps this track exposed

1. **`radial_eval.py` ignored multi-camera rigs.** It applied `valid_mask.png` — only the
   FIRST camera's — to every view, though `render.py` also writes `valid_mask_cam<k>.png`
   and a `masks.json` naming each view's mask. On FullCircle the two lenses' disks differ by
   1864 px (0.36 % of the frame) **at the rim**, i.e. exactly where the residual acts. The
   bias cancelled in a rung-vs-rung delta but corrupted absolute numbers. Fixed: per-view
   masks from `masks.json`, ring geometry from their union so ring k is the same pixels in
   every run, and a `distinct_masks` field so you can see it engaged. Regression-checked
   against the single-camera reference: `disk(view)` on `out/tunnel_fisheye_baseline` is
   still exactly 28.537.
2. **A pueue group does not reserve a GPU.** Four runs died on OOM 13 s in because a process
   started outside `gpu1` held 12 GB of it. Queue a free-VRAM wait as well as a group.

## What each term corresponds to physically

Maximum pixel displacement contributed by each azimuthal order, measured from the trained
checkpoints (`scripts/analysis/fields.py`; local plate scale `dr/dtheta`):

| order | physical cause | atrium | classroom | forest | library | reception | tunnel | workshop |
|---|---|---|---|---|---|---|---|---|
| k=0 `s0` | mapping function r(theta) mis-fit | 0.118 | 0.280 | 0.301 | 0.297 | 0.175 | **0.556** | **0.717** |
| k=1 `s1,s2` | decentred / tilted element | 0.153 | 0.156 | 0.268 | 0.191 | 0.179 | 0.115 | 0.192 |
| k=2 `s3,s4` | anamorphism (cylindrical stress) | 0.109 | 0.075 | 0.047 | 0.197 | **0.252** | 0.056 | 0.060 |
| `omega` | global boresight tilt | 0.043 | 0.052 | 0.069 | 0.029 | 0.087 | 0.035 | 0.034 |

k=0 and k=1 correct the same physics as rttpf's `k` and `p` coefficients; what changes is
the **basis** (a degree-13 global polynomial versus a locally-supported spline).

**k=2 is the only order COLMAP cannot reach.** The obvious objection is that `fx != fy`
already produces an ellipse — but a *fixed, axis-aligned* one. What is measured is neither:
on `library` the anamorphic amplitude grows **10.5x** between 30 and 85 degrees, and on
`atrium` the ellipse axis **rotates by 85 degrees** across the field. A theta-dependent
orientation cannot be an fx/fy artefact.

`omega` measures 0.03-0.09 px, i.e. nothing, and that is expected: a global tilt is nearly
degenerate with a rotation of the scene, which the gaussians absorb for free. The `tilt`
rung scores 27.39 against 27.40 for `off`.

**Do not repeat this mistake:** an earlier version of the report called `forest` the
anamorphic scene. It is not — `forest`'s non-radial part is dominated by k=1, which is a
ONE-lobe pattern. The error came from summing |channels 1..4| together, which conflates
decentring with anamorphism. Separate the orders before attributing a visual structure.

## LIMITATIONS

1. **SH view directions are not non-central.** `cuda/utils/sh.cu:231,255` bakes per-gaussian
   view-dependent colour from `*camera.origin` once per camera. With a millimetre-scale
   pupil offset against a ~14-unit scene the angular error is ~1e-4 rad, so this is a
   deliberate approximation — but it also means **no gradient flows to the camera through
   SH**. For `pose_opt` the nominal pose is still pushed to `set_pose()`, so SH stays
   consistent, but the pose gradient through SH is likewise ignored (a partial gradient).
2. **`pose_opt` does not transfer to held-out views — measured, not hypothetical.** Test
   poses stay at COLMAP, so refining the training poses drifts the scene frame relative to
   them. On `tunnel` (-r 8, 7500 it), train / test PSNR:

   | rung | train | test |
   |---|---|---|
   | `off` | 30.71 | 27.41 |
   | `noncentral` | 30.82 | **27.51** |
   | `noncentral --pose_opt` | 30.83 | 27.38 |
   | `passthrough --pose_opt` | 30.78 | 27.33 |

   Pose refinement raises train PSNR and *lowers* test PSNR — the textbook signature. The
   shared lens parameters do not have this problem, which is exactly why they are the
   headline rungs and `pose_opt` is reported separately. Publishing a `pose_opt` number
   would require refining the test poses too, and then doing the same for every baseline.
3. **No anti-aliasing anywhere in gray.** One infinitesimal ray per pixel;
   `jitter_primary_rays` is false in every production run. The camera model does not change
   that. Enabling jitter would also invalidate the cached bearing/basis tables.
4. **`z(theta)` is in COLMAP units.** Scene scale enters only through the learning rate
   (exactly as `lr_mean` is scaled by the point-cloud radius in `train.py`), never through
   `forward()`, so the checkpoint is self-contained. Converting `z` to millimetres would
   still need a metric anchor these scenes do not carry.
5. **Peripheral gains are capped by the initialization.** `third_party/edgs.py:984-987`
   forces `camera_model="pinhole"` on `sparse/0`, i.e. the RoMa dense init only covers the
   central ~120 deg pinhole crop, and **gray never densifies** (`PreMLP.densify` has no
   caller). Beyond ~60 deg incidence only the sparse COLMAP cloud exists and no rung can add
   gaussians there. Keep `point_cloud.safetensors` identical across rungs or the comparison
   is void.
6. **The raxel rung is a per-(uid, resolution) grid.** Changing render resolution mid-run
   would create a second, independently-initialized field.
7. **`central_matched` does not match the parameter count** (183 vs 111 trained) — see the
   attribution section. Conservative, but mislabelled; a true matched control is future work.
8. **The `raxel` rung is central and cannot express non-centrality.** Its last 3 channels are
   added to the **world-space direction** after the c2w rotation
   (`camera_model.py:427-428`), not to `ray_origin` — and `RUNGS["raxel"]` has no `"z"`, so
   the origin stays a plain broadcast. Worse, being added post-rotation, the same world
   vector is applied to a pixel regardless of where the camera points: it is a per-pixel
   world-frame bias, not a camera model. It is therefore **not** the "generic ray field upper
   bound" it is documented as. Its 27.59 in the r8 ladder measures central ray-field capacity
   only. Fix: rotate the offset into world space and add it to `ray_origin`.
9. **The `radial` rung also learns a sagittal residual.** `forward()` derives `delta_phi`
   from the same active channel, so `radial` is 20 spline DoF (a radially-varying roll on top
   of the radial theta residual), not the 10 documented. `radial` and `ana` also both report
   103 trainable parameters because the full `[5,K]` tensors are always allocated; channels
   1-4 simply receive exactly zero gradient under `radial`. Any per-rung parameter column
   must be computed from *effective* DoF.
10. **The rung is not recorded in the checkpoint.** `passthrough/tilt/radial/ana/noncentral`
    all produce byte-identical state_dict schemas, so loading a `noncentral` checkpoint under
    `--camera_opt radial` silently renders a different model. Only `--camera_opt off` fails
    loudly. Register `camera_opt` as a buffer to close this.
11. ~~**`base_bearings` caches on `(uid, height, width)` only**~~ — **FIXED 2026-08-07**, after
    it silently destroyed a whole baseline. The key now covers everything
    `upload_camera_intrinsics()` pushes: model, `fov_y`, image size and the intrinsics
    themselves. See "the stale-bearing collision" under GOTCHAS — this was not hypothetical,
    it cost `workshop_immervision` 12 dB and it would have been reported as a result.

## Is the gain "the lens is underfit by rttpf"? Measured, and the answer is two-part

`scripts/analysis/calib_consistency.py`. The experiment exists because both datasets
calibrate **one physical lens independently on every scene** — myscenes' 7 fits all give
fx = 1240.2 +- 0.9 on the same sensor, FullCircle's 9 re-fits give 781.13 +- 0.18 — so the
learned residual can be split into a part shared by all scenes and a per-scene part, and
each can be checked against the calibration it is supposed to be correcting.

Everything in pixels **at the resolution the run was trained at** (`-r 4`, so COLMAP's
fx 1240 becomes 310), through the LOCAL plate scale `dr/dtheta`, on a common pixel-radius
grid. Two unit traps, both of which this script fell into before being fixed: quoting
COLMAP's sensor pixels overstates every figure by **4x** and is the wrong unit anyway,
since the renderer samples the downsampled grid; and normalising the radius by each
scene's own `r(theta_max)` pins every scene to the same value at the rim and manufactures
agreement exactly where the residual acts (it reported 0.000 px of disagreement there).

| | myscenes (1 lens, 7 fits) | FC lens 1 (9 re-fits) | FC lens 2 |
|---|---|---|---|
| images per scene | 332 | 1060 | 1060 |
| cross-scene calibration disagreement, rim | 2.970 px | 1.273 px | 2.393 px |
| learned residual, **shared** part, rim | **0.393 px** | 0.037 px | 0.035 px |
| learned residual, per-scene part, rim | 0.233 px | 0.033 px | 0.016 px |
| shared / per-scene | 1.58 | 1.71 | 1.54 |
| corr(scene's calib deviation, scene's residual) | **-0.427** | +0.125 | -0.051 |
| effect on cross-scene spread | **-5.6 %** | +0.9 % | -0.2 % |
| PSNR gain | +0.33 to +0.44 dB | +0.016 dB | |

Two distinct things are true on myscenes and false on FullCircle:

1. **A systematic ~0.4 px error at the rim, common to all 7 independent fits of the same
   lens.** That is the honest meaning of "the delivered calibration does not describe this
   lens". It is **10x smaller** on FullCircle.
2. **A per-scene error the residual actively cancels.** The correlation between a scene's
   own calibration deviation and its own learned residual is **-0.427** — the residual
   points *against* the error of the scene it was trained on. That is also the only reason
   the cross-scene spread can move at all: a shared component cannot change a standard
   deviation, so the -5.6 % is entirely this term. On FullCircle the correlation is zero
   and the spread does not move.

**Cross-scene disagreement alone predicts nothing.** FullCircle's lens 2 disagrees by
2.39 px, close to myscenes' 2.97, and gains nothing. The predictor is the SHARED bias
(0.393 vs 0.035 px), not the variance of the fits.

The practical threshold worth remembering: **0.04 px is far below what gray can express** —
one ray per pixel, `jitter_primary_rays` false, no anti-aliasing anywhere (limitation 3).
No camera model, however rich, can cash a correction that small.

### MISFIT, not an inexpressive model — and that reframes the whole result

`scripts/analysis/residual_expressible.py`. "The lens is underfit by rttpf" conflates two
claims, and the data picks one. The learned residual moves each pixel radius `r` from angle
`theta` to `theta + dtheta`, so the corrected forward map passes through the samples
`(theta_i + dtheta_i, r_i)`. `r = fx * t * (1 + k1 t^2 + ... + k4 t^8)` is LINEAR in
`(fx, fx*k1 ... fx*k4)` over the basis `[t, t^3, t^5, t^7, t^9]`, so "could rttpf have
expressed this?" is an exact least-squares question. Control: the same fit on the
UNcorrected samples must return ~0, since they were generated by that very form (0.0002 px).

| | correction learned | absorbable by re-fitting rttpf | outside its span |
|---|---|---|---|
| myscenes | 0.396 px | **98.6 %** | 0.0054 px rms |
| FC lens 1 | 0.046 px | 91.0 % | 0.0042 px |
| FC lens 2 | 0.062 px | 98.8 % | 0.0007 px |

**98.6 % of the radial correction was reachable by moving the k1..k4 COLMAP already had.**
The model class was adequate; the delivered coefficients were wrong. So the bulk of the
myscenes gain is **online re-calibration**, not a richer camera: the right rttpf, found by
photometric descent over whole images instead of reprojection of sparse features. FullCircle
closes the loop — its explicit bundle re-fit did that job offline, leaving 0.046 px, which
is why the model finds nothing there.

Combining this with the attribution ladder (tunnel, r4, 15k: `off` 28.49 -> `ana` 28.64 ->
`noncentral` 28.73), this section originally concluded a **~60 % re-calibration / ~40 %
non-centrality** split.

> **SUPERSEDED — the split is ~25 / 75 at 15k, measured directly.** The inference above goes
> from *"the correction lies inside rttpf's span"* to *"most of the gain is re-calibration"*,
> and that step does not hold. `--camera_opt rttpf` performs the re-calibration instead of
> arguing about it and recovers **+0.082 of the +0.333 dB** over seven scenes (see "RESULTS
> — the re-calibration control"). Expressibility was never in question — corrected for its
> index bugs the span figure is *99.2 %*, higher than the 98.6 % originally claimed, which
> makes the point sharper: **essentially the whole radial correction is inside rttpf's span,
> and actually re-fitting it is worth 0.08 dB.** Photometric value is a different quantity
> from expressibility, and only the former decomposes a PSNR gain.
>
> Quote the 25 % **with "at 15k" attached**. At 30k the control roughly doubles
> (`rttpf - off`: tunnel -0.004 -> +0.119, workshop +0.071 -> +0.120) while `noncentral - off`
> is flat to rising (+0.191 -> +0.198, +0.743 -> +0.801). The re-calibration share is
> budget-dependent; the ordering is not.

`scripts/analysis/rttpf_span.py` closes the loop by projecting the learned *central* field
(all azimuthal orders, `z` excluded) onto the span of all 16 rttpf parameters, using the
renderer's own solver linearized one coefficient at a time:

| scene | \|residual\| px | explained, 2D | best-fit px | what descent learned, px | found/best |
|---|---|---|---|---|---|
| atrium | 0.108 | 0.77 | 0.095 | 0.211 | 2.23 |
| classroom | 0.234 | 0.96 | 0.228 | 0.281 | 1.23 |
| forest | 0.320 | 0.99 | 0.319 | 0.082 | 0.26 |
| library | 0.306 | 0.80 | 0.275 | 0.264 | 0.96 |
| reception | 0.208 | 0.76 | 0.182 | 0.216 | 1.19 |
| tunnel | 0.366 | 0.99 | 0.364 | 0.174 | 0.48 |
| workshop | 0.616 | 1.00 | 0.615 | 0.314 | 0.51 |
| **mean** | **0.308** | **0.90** | **0.297** | **0.220** | 0.98 |

Two things to read off it, and they point the same way:

* **The span claim survives, and got stronger once its script was fixed.** The 98.6 % was
  computed with two index bugs (see "Pre-existing issues"); corrected it reads **99.2 %**.
  Radially, rttpf can express essentially all of it. This branch's independent 2D number is
  **90 %**, and the two are consistent rather than contradictory: 99.2 % is the *radial slice*,
  90 % is the *full 2D field*, and the difference is the azimuthal k=1/k=2 content, where
  rttpf answers with `p0,p1,s0..s3` against the splines' per-order profiles. Nothing in this
  branch ever inherited the bug — `rttpf_fields.plate_scale` loops `params[4+index]` over
  `range(6)` and `rttpf_span.py` imports from there, so the 90 % and the field tables above
  were always on the correct degree-13 polynomial.
* **And it does not matter.** Descent lands nowhere near the projection: the `found/best`
  column scatters from 0.26 to 2.23 (the mean of 0.98 is an averaging artefact), and the
  field comparison shows cosines around zero. The re-calibration is not a worse version of
  the same correction, and being able to express the central part buys +0.08 dB.

So the corrected reading is not "expressible therefore worth 60 %". It is: **the central
part of the correction is largely expressible by rttpf and is worth about a quarter of the
gain; the non-central part is expressible by nothing central and is worth the other three
quarters** — which is what `rttpf_z` reproducing `noncentral` on tunnel, and the r = 0.93
correlation with `z` at scene depth, both say independently.

One caveat carried over unchanged: "absorbable by re-fitting" is **not** "COLMAP would have
found it": the two optimise different objectives (photometric over all pixels vs
reprojection of sparse keypoints) on different data.

## Where the gain lands: it depends on the scene

The theory says a radial / anamorphic / non-central residual is **peripheral**: `d(theta)`
vanishes at `theta = 0` and the non-central term scales as `sin(theta) z(theta) / depth`.
Measured per equal-area ring (`scripts/radial_eval.py`, -r 4, 15k, `noncentral` vs `off`):

| ring (r/R) | 0-.41 | .41-.58 | .58-.71 | .71-.82 | .82-.91 | .91-1.0 | disk |
|---|---|---|---|---|---|---|---|
| `tunnel` | **+0.399** | +0.130 | +0.191 | +0.161 | +0.231 | +0.206 | +0.246 |
| `workshop` | +0.648 | +0.938 | +0.933 | +0.561 | +0.663 | **+1.305** | +0.743 |
| `workshop_immervision` (-r 1) | **-0.270** | +0.047 | +0.021 | +0.016 | +0.182 | +0.024 | **-0.002** |

`workshop` is the textbook peripheral profile the model predicts — the outer ring gains
twice the centre. `tunnel` is the opposite, centre-heavy. So the two scenes are dominated by
different error modes: on `workshop` by something that grows with field angle (what the lens
residual is designed for), on `tunnel` by a roughly uniform mis-registration, which the
3-DoF `tilt` term can absorb and which is plausible given COLMAP pinned `cx, cy` at the
exact sensor centre on every myscenes scene and never refined them.

**`workshop_immervision` is the null result, and it is worth reading carefully.** On the one
lens whose geometry most deserves a richer camera, the rung buys *nothing* per pixel: the
pooled disk delta is **-0.002 dB**, and the centre ring actually loses 0.27. The headline
+0.081 dB comes entirely from the **per-view mean** convention — the gain sits in views that
already score well, which pooling over all pixels of all views washes out. `radial_eval.py`
prints both (`disk(pool)` and `disk(view)`); when they disagree in *sign*, as here, the honest
report is "no effect", not the per-view number. Two candidate explanations, neither tested:
the calibration is already near-saturated (rttpf's extra parameters buy only 3 % reprojection
on this lens), and/or the r=0.95 mask keeps just 47.9 % of the frame, so the outer field where
the residual acts is largely masked away before it is ever scored.

One caveat applies to both: the periphery may not be free to show its full gain, because it
is reconstruction-starved. The EDGS/RoMa init only covers the central ~120 deg pinhole crop
and gray never densifies (limitation 5), so beyond ~60 deg incidence there is only the
sparse COLMAP cloud. A camera correction cannot sharpen geometry that is not there. Widening
the init's field -- or triangulating the RoMa matches with gray's own rays instead of the
pinhole DLT -- is the untested experiment that would settle it.

## Attribution: is the gain non-centrality, or just parameters?

Three scenes, -r 4, 15k, identical init / mask / schedule / frozen poses, one run each.
Scored with `scratchpad/analysis/rungs.py` (masked PSNR recomputed from the PNGs under the
shared protocol, NOT the run's own `psnr.csv`; `workshop`'s `off` is the config-fixed
re-run `tmp/noncentral/fix15k_workshop`, not the misconfigured published baseline):

| rung | trained params | tunnel | workshop | reception |
|---|---|---|---|---|
| `off` | 0 | 28.537 | 27.198 | 27.277 |
| `ana` (full central residual) | 103 | 28.638 | 27.297 | 27.618 |
| `central_matched` (more *central* capacity) | 183 | 28.593 | 27.329 | 27.486 |
| `noncentral` | 111 | **28.729** | **27.942** | **27.751** |

Differences, and the mean over the three scenes:

| difference | isolates | Δ params | tunnel | workshop | reception | mean |
|---|---|---|---|---|---|---|
| `ana − off` | tilt + radial + anamorphic | +103 | +0.100 | +0.099 | +0.340 | +0.180 |
| `central_matched − ana` | more central capacity | +80 | −0.045 | +0.032 | −0.132 | **−0.048** |
| `noncentral − ana` | **the z(theta) profile alone** | +8 | +0.091 | +0.645 | +0.134 | **+0.290** |
| `noncentral − central_matched` | vs the capacity control | −72 | +0.136 | +0.612 | +0.265 | +0.338 |
| `noncentral − off` | the whole model | +111 | +0.191 | +0.743 | +0.474 | +0.469 |

**The cleanest statement available: +80 central parameters buy −0.048 dB (nothing);
+8 non-central parameters buy +0.290 dB.** `noncentral` beats `central_matched` on all
three scenes despite training 72 fewer parameters.

`workshop` was the scene predicted in advance to show the largest non-central effect (its
|z| is 5x `tunnel`'s and its computed irreducible residual, 0.67 px, is the largest of the
seven) and it does, by 10x the noise floor. The prediction does NOT order the other two:
`tunnel` has a larger predicted residual than `reception` (0.45 vs 0.29 px) and measures
less (+0.091 vs +0.134) — but their 0.043 dB difference is below the ±0.06 noise floor, so
they are indistinguishable. The theory separates `workshop` from the rest; it does not rank
the rest.

Ring-resolved, `noncentral − ana` (8 equal-area annuli, centre → rim):

```
workshop   +0.44 +0.70 +0.89 +0.68 +0.40 +0.50 +0.75 +1.37
reception  +0.03 -0.04 +0.01 +0.27 +0.15 +0.26 +0.57 +0.69
tunnel     +0.24 +0.12 -0.03 +0.30 +0.26 -0.06 +0.02 +0.00
```

The two scenes where the term carries weight peak at the outermost ring — the sin(theta)
growth the model predicts. `tunnel`'s profile is flat: at 1.5x noise there is nothing to
read into it.

**`central_matched` is NOT parameter-matched — corrected 2026-08-06 after an independent
audit.** Measured by replaying `LensResidual`: `noncentral` trains **111** parameters,
`central_matched` trains **183** (+65%). The extra knots are added to both `theta_weights`
and `phi_weights` across all 5 harmonic channels (2 x 5 x 8 = 80), not the 8 that `z` costs.
`camera_model.py`, `config.py`, `PROTOCOL.md` and this file all claimed equality; they were
wrong.

The bias is **conservative**, so the conclusion survives and in fact strengthens: the
central control has 65% MORE free parameters than the non-central rung and still scores
lower on every scene. Read the row as *"a central model with a larger budget"*, not
*"the same budget spent centrally"*. A genuinely parameter-matched control still needs
building (add 8 knots to `theta_weights` channel 0 only, or drop 4 knots from each of
theta/phi) — though with `central_matched − ana` measuring −0.048 dB, there is little
reason to expect a smaller central control to do better.

**Still do not over-claim.** Three scenes, one run each, no seed repeats. `workshop` alone
carries the demonstration (+0.645 dB, 10x noise); on `tunnel` the isolated term is 1.5x
noise and on `reception` 2.2x. What is safe to say today: the central residual buys
+0.18 dB on average, more central capacity buys nothing further, and the non-central term
buys another +0.29 dB — but that mean is dominated by one scene, and the share of the
7-scene +0.333 dB attributable to non-centrality has not been measured. Per-scene shares of
`noncentral − off` carried by the z term: `workshop` 87%, `tunnel` 47%, `reception` 28%.

**Why the two terms behave differently across scenes** (this is the mechanism, not a
coincidence): the angular residual is a *calibration* correction, so it lands at roughly
+0.10 dB on both `tunnel` and `workshop`. The non-central term corrects an error that scales
as `z(theta) sin(theta) / depth`, so it depends on the *scene's* depth distribution and
varies by 7x between them. A central model can cancel that at one depth and one only; what
survives is `z(theta) sin(theta) * sigma(1/t | theta)`, measured at 0.12–0.41 px across the
seven scenes (`scripts/analysis/optics.py`). That is the reason a central control with
more parameters cannot close the gap.

**Pixel figures use the LOCAL plate scale `dr/dtheta`, not the paraxial `fx`** (corrected
2026-08-07). The rttpf radial polynomial compresses the rim: `dr/dtheta` falls to ~0.56 fx
at theta = 90 deg on all seven calibrations, so `fx * angle` overstates every peripheral
pixel number by ~1.8x. The image-space displacement caused by an angular ray error `d` is
`dr/dtheta * d`. Affected quantities, old (paraxial) -> new (local): max angular residual
0.13-1.20 -> **0.12-0.72 px**; irreducible non-central residual at 85 deg 0.20-0.67 ->
**0.12-0.41 px**; rim disparity spread p05->p95 0.65-3.05 -> **0.36-1.72 px**. No PSNR,
ring or ablation number is affected -- those are measured, not derived.

## GOTCHAS (things you would not guess)

* **`bspline_eval` by advanced indexing is a performance trap.** `weights[:, idx]` with an
  `[H,W]` index makes the backward a scatter-add of ~5M values into ~50 addresses — total
  atomic contention. Measured at 512x512: **130 ms/iteration versus 3.4 ms for the native
  path**. `theta_base` is constant per (camera, resolution), so the basis is precomputed
  once as a dense `[N, K]` matrix (`build_bspline_basis`) and the evaluation is a GEMM.
  After the fix `noncentral` costs ~10 ms/iteration. **Never reintroduce the gather.**
* **A branch on `|omega|` in the Rodrigues rotation costs 4 ms/iteration** — `float(angle)`
  is a GPU->CPU sync every call, more than gray's entire native ray generation. `skew_rotate`
  uses Taylor series in `|w|^2` instead: no branch, no sync, exactly the identity at zero.
* **`autograd.backward` raises on any input without a `grad_fn`.** The ray *origin* is a
  plain broadcast of the camera centre for every rung except `noncentral` and `pose_opt`, so
  the third backward stage must filter its inputs by `requires_grad`. This bit once and is
  now covered by `test_every_rung_completes_a_training_step`.
* **`base_bearings` must upload the intrinsics itself.** An unconfigured CUDA camera is
  PINHOLE with `fov_y = 0`, which probes as `theta == 0` for every pixel and *silently*
  returns a degenerate bearing field instead of failing.
* **`run.sh` never produces fisheye renders.** `render.py`'s `eval_models` defaults to
  `["pinhole"]` (`render.py:26-29`), so `scripts/masked_eval.py` finds nothing. Use
  `scripts/eval_rttpf.sh`.
* **Lazily-created parameter blocks break strict `load_state_dict`.** Lens/raxel/pose blocks
  are created on first render, so `from_safetensors` calls
  `CameraModel.materialize_from_state_dict()` before loading.
* **A training-time constant inside `forward()` is a render-time bug waiting to happen.**
  `scene_scale` used to multiply `z(theta)` inside the model. `train.py` set it from the
  scene radius; `render.py` had no reason to and left it at 1.0, so the *training* metrics
  were right (+0.17 dB on tunnel r4) while every *re-rendered* image was wrong by ~14x
  (-0.19 dB, and the damage concentrated in the outer ring exactly where `z` is largest).
  Nothing in the training logs could reveal this -- it only surfaced because
  `scripts/radial_eval.py`, which scores the PNGs, disagreed with `psnr.csv`, which scores
  the live render. **Always cross-check the two.** The fix removes the dependency rather
  than restoring it, and `test_ray_synthesis_does_not_depend_on_scene_scale` pins it.
* **The stale-bearing collision: two eval models, one cache key (fixed 2026-08-07).** Same
  family as the `scene_scale` bug above, same detection method, worse blast radius.
  `base_bearings()` was cached on `(uid, height, width)`. `render.py --eval-models pinhole
  rad_tan_thin_prism_fisheye` renders BOTH models for the same camera uid in one process, so
  whichever ran first won the cache and the second silently traced the first one's bearings.
  On myscenes this never fired, purely by luck: the pinhole copy is 400x266 while the fisheye
  is 1368x912, so the resolution separated them. On `workshop_immervision` both are
  **1440x1080** (`undistort_consistent.py --width 1440 --height 1080`), and the fisheye pass
  re-rendered at **13.31 dB** against the **25.37** the same checkpoint reached live. The
  training was perfect; only the PNGs everything downstream reads were garbage.
  Three things make this worth remembering:
  - **It is invisible to `--camera_opt off`**, which never calls `base_bearings` (native
    raygen). The control run re-rendered at 25.28 against its own live 25.29, so a
    control-vs-rung comparison looked like the *rung* had collapsed, not the renderer.
  - **`psnr.csv` vs `results.json` caught it again.** The live metric and the re-rendered
    metric disagreeing by 12 dB is the whole signal. Cross-check them on EVERY new scene.
  - **It is order-dependent**, so it silently vanishes if you render the fisheye first —
    which is exactly how a "fix" that isn't one gets committed. The real fix is in
    `raytracer.py:base_bearings`: the key now includes the model, `fov_y`, the image size
    and the intrinsics.

  **Retro-audit (2026-08-07): nothing else on disk was affected.** The test is free — compare
  each run's `psnr.csv` test column against its `results.json` fisheye entry. All 7 myscenes
  `tmp/final/*_noncentral` runs agree to <=0.04 dB, and the 7 `out/fullcircle_rttpf/*` runs to
  <=0.07 dB (those render a single eval model, so they cannot collide at all). Only
  `workshop_immervision` ever hit it. Re-run that check on any new scene before trusting a
  number.

* **Measuring "angular error" with `arccos(dot)` in float32 is misleading.** Near 1 it turns
  a 1e-7 dot-product error into an apparent 5e-4 rad. Use the chord `|a-b|`.
* **`scale_decay` is applied per iteration, not per schedule.** The default `0.999875`
  shrinks gaussians by 0.153x over 15k iterations but by **0.0235x over 30k** — 6.5x more.
  A 30k run is therefore *not* "the same training, longer". Measured on `tunnel`:
  15k = 28.54, plain 30k = **28.42**, 30k with `--scale_decay 0.9999375` (same total shrink)
  = **28.61**. So the extra iterations are worth about +0.07 dB once the decay is corrected,
  and are worth *negative* dB if it is not. Always pass the corrected decay for an
  iteration-matched comparison.
* **Two `Raytracer` instances cannot coexist in one process** (teardown aborts in
  `PipelineWrapper::~PipelineWrapper`). That is why `pyproject.toml` sets
  `addopts = "--forked"`, and why the passthrough test detaches/reattaches the camera model
  on a single instance instead of building two.
* **Finite-differencing gray needs a *large* epsilon.** The error scales as `1/eps` (float32
  round-off in the render), not as `eps^2`. Measured on a 24x24 probe: 1e-2 -> 2.4e-3,
  1e-4 -> 2.0e-1. Also keep per-pixel hit counts under `BUFFER_SIZE = 32`
  (`forward_pass.cu:4`) or the truncation approximation and its discontinuity dominate.

## Pre-existing issues found (not caused by this branch)

* ~~**`calib_consistency.py` used the 12-parameter radial layout on 16-parameter rttpf
  data**~~ — **FIXED 2026-08-12** by the `paper-brainstorm` session, patch ported into this
  worktree. `RADIAL = [4, 5, 8, 9]` (the `THIN_PRISM_FISHEYE` layout, as its own comment
  described) became `RADIAL_BY_NPARAM`, keyed on parameter count and raising on any unknown
  layout. rttpf's six radials are consecutive at **4..9** (`cuda/core/rtpf.cuh` evaluates
  `1 + k0 t^2 + ... + k5 t^12`).

  **`residual_expressible.py` carried a second, independent instance of the same mistake**,
  and the way it hid is the part worth remembering: `BASIS = [1,3,5,7,9]` was hardcoded to
  five terms for a span that needs seven (`t .. t^13`). The script *has* a control — fit the
  UNcorrected samples, which must return ~0 — and the control passed at 0.00018 px, because
  the sampler (`radial_poly`) and the fitting basis were truncated **the same way**. A
  self-consistency check cannot catch an error shared by both sides of the comparison; it
  needs an external reference, which here is `rtpf.cuh`. With both fixed the control returns
  0.0000 and expressibility reads **99.2 %**, up from 98.6 %.

  Consequences for numbers quoted elsewhere in this file: the 98.6 % becomes 99.2 % (and
  makes the "expressible is not profitable" point *stronger*, not weaker — see the SUPERSEDED
  note); the shared/per-scene ratio 1.58 is unaffected at 1.56, since the residual comes from
  the checkpoint spline and never passed through the bad indices; but **"cross-scene
  disagreement predicts nothing" was itself an artefact and is now false** — myscenes 2.970 px
  vs FullCircle 2.393 px become 0.397 vs 0.049, i.e. the disagreement now orders the two
  datasets the same way the gains do. Do not oppose shared bias and variance in the writeup;
  on this evidence they are not separable.
* `tests/test_single_gaussian.py`, `tests/test_multiple_gaussians.py`,
  `tests/test_fisheye_mask.py` — **5 tests fail on `main` too** (`mock_camera` has no
  `origin_cuda`, etc.). Untouched here.
* `tests/test_eval_modes.py` computes its fixture path as `Path(__file__).parents[1].parents[1]`.
  From `/workspace/gray` that resolves to `/data/...` and the tests **skip**; from a worktree
  (two levels deeper) it resolves to the real `data/` and the tests run **for the first
  time**. One then failed because it passed a `str` where `load_eval_views` expects a
  `GrayCameraModelClass` — fixed here. The other still fails on a genuine data mismatch:
  `data/myscenes/transmission_fe/distorted/sparse/0` is `thin_prism_fisheye` while the test
  asserts `opencv_fisheye`. Left failing on purpose; it is stale fixture data, not code.
* The published gray `workshop` myscenes baseline ran with `vignetting_comp=False` and
  `batch_size=1` while the other six scenes used `True`/`2`. Re-running it matched
  (15k, same everything else) scores **27.18 against 26.47**, i.e. **+0.71 dB** was lost to a
  configuration slip on gray's *worst* scene relative to SPaGS.
