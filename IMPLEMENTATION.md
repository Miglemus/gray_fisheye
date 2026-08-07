# IMPLEMENTATION — learnable residual / non-central camera model in gray

Branch `noncentral-camera`. Adds the ability to **learn the camera model jointly with the
gaussians**: a residual on top of the COLMAP calibration, including a genuinely
**non-central** (axial entrance-pupil) term, plus an optional per-view SE(3) pose residual.

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

| method | PSNR | SSIM | LPIPS |
|---|---|---|---|
| gray (published) | 27.124 | 0.9458 | 0.1600 |
| SPaGS | 27.333 | **0.9514** | 0.1657 |
| DirectFisheye-GS | 26.818 | 0.9445 | 0.1872 |
| 3DGUT | 25.930 | 0.9306 | 0.2272 |
| **gray + camera model** | **27.561** | 0.9491 | **0.1547** |

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
11. **`base_bearings` caches on `(uid, height, width)` only** — not on the camera model,
    intrinsics or `fov_y`. Within one training run that is safe (one model, one resolution),
    but multi-eval-mode rendering and the interactive viewer's FoV slider can silently reuse
    stale bearings.

## Where the gain lands: it depends on the scene

The theory says a radial / anamorphic / non-central residual is **peripheral**: `d(theta)`
vanishes at `theta = 0` and the non-central term scales as `sin(theta) z(theta) / depth`.
Measured per equal-area ring (`scripts/radial_eval.py`, -r 4, 15k, `noncentral` vs `off`):

| ring (r/R) | 0-.41 | .41-.58 | .58-.71 | .71-.82 | .82-.91 | .91-1.0 | disk |
|---|---|---|---|---|---|---|---|
| `tunnel` | **+0.399** | +0.130 | +0.191 | +0.161 | +0.231 | +0.206 | +0.246 |
| `workshop` | +0.648 | +0.938 | +0.933 | +0.561 | +0.663 | **+1.305** | +0.743 |

`workshop` is the textbook peripheral profile the model predicts — the outer ring gains
twice the centre. `tunnel` is the opposite, centre-heavy. So the two scenes are dominated by
different error modes: on `workshop` by something that grows with field angle (what the lens
residual is designed for), on `tunnel` by a roughly uniform mis-registration, which the
3-DoF `tilt` term can absorb and which is plausible given COLMAP pinned `cx, cy` at the
exact sensor centre on every myscenes scene and never refined them.

One caveat applies to both: the periphery may not be free to show its full gain, because it
is reconstruction-starved. The EDGS/RoMa init only covers the central ~120 deg pinhole crop
and gray never densifies (limitation 5), so beyond ~60 deg incidence there is only the
sparse COLMAP cloud. A camera correction cannot sharpen geometry that is not there. Widening
the init's field -- or triangulating the RoMa matches with gray's own rays instead of the
pinhole DLT -- is the untested experiment that would settle it.

## Attribution: is the gain non-centrality, or just parameters?

`tunnel`, -r 4, 15k, identical init and mask, one run each:

| rung | test PSNR | vs `off` |
|---|---|---|
| `off` | 28.49 | — |
| `ana` (full central residual) | 28.64 | +0.15 |
| `central_matched` (same budget as `noncentral`, spent centrally) | 28.60 | +0.11 |
| `noncentral` | **28.73** | **+0.24** |

**`central_matched` is NOT parameter-matched — corrected 2026-08-06 after an independent
audit.** Measured by replaying `LensResidual`: `noncentral` trains **111** parameters,
`central_matched` trains **183** (+65%). The extra knots are added to both `theta_weights`
and `phi_weights` across all 5 harmonic channels (2 x 5 x 8 = 80), not the 8 that `z` costs.
`camera_model.py`, `config.py`, `PROTOCOL.md` and this file all claimed equality; they were
wrong.

The bias is **conservative**, so the conclusion survives and in fact strengthens: the
central control has 65% MORE free parameters than the non-central rung and still scores
lower (28.60 vs 28.73). Read the row as *"a central model with a larger budget"*, not
*"the same budget spent centrally"*. Two things follow. First, `central_matched` has
strictly more parameters than `ana` and does not beat it (28.60 vs 28.64, inside the ±0.06
noise) — **extra central capacity buys nothing**. Second, the non-central degree of freedom
adds **+0.13** over a control that was handed more parameters, not fewer.

A genuinely parameter-matched control still needs building (add 8 knots to `theta_weights`
channel 0 only, or drop 4 knots from each of theta/phi).

**Do not over-claim this.** +0.13 is about 2x the run-to-run noise, from a single run on a
single scene. It is the right sign and the right control, but a paper claim needs repeats
across scenes and seeds. What is safe to say today: the central residual buys ~+0.15, more
central capacity buys nothing further, and the non-central term buys another ~+0.09-0.13.

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
