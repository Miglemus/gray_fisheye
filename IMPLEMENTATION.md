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
| `gray/camera_model.py` | `forward()` split into `camera_frame()` (pose-independent, **cached**) + `apply_pose()` (per image); `invalidate_ray_cache()` — see "The pose-independent ray cache" |
| `tests/test_camera_model_cache.py` | **new** — 33 CPU tests: bit-exactness over rungs x resolutions x poses, and the pose-freezing line the cache must not cross |
| `scripts/bench_ray_cache.py` | **new** — the FPS A/B for that cache, bit-exactness checked before any timing |
| `scripts/traversal_coherence.py`, `tests/test_traversal_coherence.py` | **new** — does the non-central origin cost BVH coherence? (`hits/ray`, `off` vs `noncentral`) |

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
| `noncentral`, seed 1 | 30.282 | 30.422 | 28.607 | 27.600 | 28.382 | 28.816 | 22.754 | 28.491 | 29.770 | 28.347 |
| `noncentral`, seed 2 | 30.265 | 30.420 | 28.492 | 27.648 | 28.360 | 28.824 | 22.742 | — | — | **28.335** |
| *nc (2 seeds) − off* | +0.050 | +0.053 | −0.069 | +0.018 | −0.016 | −0.010 | +0.014 | +0.066 | −0.021 | **+0.009** |

Seed 2 is the second, clean-card training of the seven scenes whose first run shared the
GPU; `dark` and `persons` were already clean and were not repeated. The canonical table and
the viewer carry seed 2, because those are the renders now on disk.

* **The `off` control reproduces the published gray row to +0.003 dB on the mean**, its
  gaussian counts land within 0.1-1 % (156 206 vs 156 516 on room1) and its render speed
  matches to 0.1 % (419.4 vs 419.9 FPS). The branch's CUDA changes are neutral, and the
  published row is a legitimate baseline for this track.
* **+0.009 dB over two seeds.** Seed 1 alone gave +0.016, seed 2 alone +0.003; the
  seed-to-seed spread on the seven repeated scenes is -0.115 to +0.048 (std 0.046),
  i.e. **larger than the effect being measured**. Positive on 5 of 9. This is zero, and the
  repeat is what makes it safe to say so.
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
* **It is not an optimisation artefact.** `room2` re-run with all three learning rates at
  10x (`NAME_SUFFIX=_lr1e3 ... --camera_opt_lr_{tilt,angular,z} 1e-3`) scores **30.398**
  against 30.422 at the default and 30.368 for `off` — no better, and the learned magnitude
  does not grow: `theta_weights` 1.32e-3 at lr 1e-3 versus 1.57e-3 at lr 1e-4, i.e.
  slightly *smaller*. The residual is small because the photometric gradient wants it
  small, not because the schedule cannot reach further.

The same pattern, weaker, was already visible on `workshop_immervision` (+0.081 over its
own `off` control, on a lens whose rttpf terms only bought 3 % reprojection). **The rule
that fits all three datasets: the camera model recovers what the calibration left on the
table, and nothing more.** It is worth its cost on a raw COLMAP fisheye fit and worth
nothing on a re-bundled one — which is a useful thing to know before spending it.

Note the principal point is pinned at *exactly* `(1440.0, 1440.0)` for both lenses even
after the re-fit, so the `tilt` term did still have a free 3 DoF here; it was not enough.

### Cost

Every run below trained and was timed on a card it had to itself, and every FPS comes from
one back-to-back sweep on gpu1 (`scripts/measure_fullcircle_fps.sh`), gray included — the
published `gray` runs had no `fps.csv` at all before this.

| | train | FPS | #gauss |
|---|---|---|---|
| gray (published) | 4.4 min | 419.9 | 130 953 |
| `off` (this worktree) | 4.6 min | 419.4 | 131 059 |
| `noncentral` | **10.2 min** | **245.6** | 130 246 |

**2.3x training time and 0.58x render speed, for nothing on this track.** Ray synthesis runs
in Python once per frame, which is where both go. Gaussian counts are unchanged, as expected
— the camera model touches rays, not geometry.

The whole first batch's `time.csv` (14-16 min) and FPS (63 to 245 on comparable scenes) were
contention artefacts: a foreign process held 12 GB of the same card. They are gone from the
runs now, but the lesson is in the traps below.

Full cross-method table, all six metrics, generated by `scripts/fullcircle_rttpf_table.py`
into `dataset/fullcircle_baselines/fullcircle_rttpf_results.md`:

| method | PSNR | SSIM | LPIPS | #gauss | FPS | train |
|---|---|---|---|---|---|---|
| SPaGS (fisheye port) | **28.678** | **0.9391** | **0.1869** | 203 377 | 341.0 | 16.4 min |
| **gray + camera model** | 28.335 | 0.9329 | 0.2064 | 130 246 | 245.6 | 10.2 min |
| gray | 28.328 | 0.9328 | 0.2065 | **130 953** | **419.9** | **4.4 min** |
| DirectFisheye-GS | 28.132 | 0.9331 | 0.2018 | 257 435 | 388.6 | 18.6 min |
| 3DGUT | 28.042 | 0.9321 | 0.2078 | 175 795 | 277.4 | 10.8 min |

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

**All numbers below were regenerated on 2026-08-12 after fixing the radial-index bug** (see
"Two silent bugs in the calibration analysis" further down). The pre-fix table claimed a
2.970 px cross-scene disagreement on myscenes and 2.393 px on FullCircle lens 2; those were
artefacts. Do not resurrect them from git history.

| | myscenes (1 lens, 7 fits) | FC lens 1 (9 re-fits) | FC lens 2 |
|---|---|---|---|
| images per scene | 332 | 1060 | 1060 |
| cross-scene calibration disagreement, rim | 0.397 px | 0.095 px | 0.049 px |
| learned residual, **shared** part, rim | **0.324 px** | 0.065 px | 0.058 px |
| learned residual, per-scene part, rim | 0.197 px | 0.058 px | 0.022 px |
| shared / per-scene | 1.56 | 1.70 | 1.54 |
| corr(scene's calib deviation, scene's residual) | **-0.856** | -0.726 | +0.169 |
| effect on cross-scene spread | **-36.0 %** | -22.1 % | +48.7 % |
| PSNR gain | +0.33 to +0.44 dB | +0.016 dB | |

What survives the fix, and is now much stronger than reported:

1. **A systematic ~0.32 px error at the rim, common to all 7 independent fits of the same
   lens** — the honest meaning of "the delivered calibration does not describe this lens".
   It is 5x smaller on FullCircle.
2. **A per-scene error the residual actively cancels**, at `corr = -0.856` (not -0.427): the
   residual points *against* the error of the scene it was trained on. A shared component
   cannot change a standard deviation, so the entire -36.0 % contraction of the cross-scene
   spread is this term.
3. The **shared / per-scene ratio (1.54-1.70)** is untouched by the fix, because the residual
   is read from the checkpoint spline and never passes through the radial indices.

What **died** with the fix, and must not be written up:

* ~~"Cross-scene disagreement alone predicts nothing"~~. Corrected, the disagreement orders
  myscenes **4.7-6.7x above** FullCircle (0.397 vs 0.095 / 0.049 px; 868.7 vs 184.4 /
  129.9 urad recomputed independently through pycolmap's real projection) — the same order as
  the gains. And it sits within 10-40 % of the shared bias (791/869, 208/184, 197/130 urad).
  **Shared bias and fit-to-fit variance are not separable on this evidence.** `plate_scale.py`
  reached the same conclusion independently (0.360 / 0.069 / 0.045 px) and left the warning
  that this argument had to be re-checked before write-up; it has been, and it fails.
* ~~"On FullCircle the correlation is zero and the spread does not move"~~. Lens 1 is at
  `corr = -0.726` and contracts by 22.1 % while gaining +0.016 dB. So the cancellation
  mechanism is real and general — it just does not, by itself, buy PSNR. Lens 2 goes the
  other way (+0.169, spread +48.7 %) on a residual of 0.058 px, i.e. inside the noise.

The practical threshold worth remembering: **0.05 px is far below what gray can express** —
one ray per pixel, `jitter_primary_rays` false, no anti-aliasing anywhere (limitation 3).
No camera model, however rich, can cash a correction that small.

### MISFIT, not an inexpressive model — and that reframes the whole result

`scripts/analysis/residual_expressible.py`. "The lens is underfit by rttpf" conflates two
claims, and the data picks one. The learned residual moves each pixel radius `r` from angle
`theta` to `theta + dtheta`, so the corrected forward map passes through the samples
`(theta_i + dtheta_i, r_i)`. `r = fx * t * (1 + k0 t^2 + ... + k5 t^12)` is LINEAR in
`(fx, fx*k0 ... fx*k5)` over the basis `[t, t^3, t^5, t^7, t^9, t^11, t^13]` — **seven**
terms, because rttpf carries six radials, not four — so "could rttpf have expressed this?"
is an exact least-squares question. Control: the same fit on the UNcorrected samples must
return ~0, since they were generated by that very form. It now returns 0.0000 px.

| | correction learned | absorbable by re-fitting rttpf | outside its span |
|---|---|---|---|
| myscenes | 0.335 px | **99.2 %** | 0.0027 px rms |
| FC lens 1 | 0.067 px | 94.4 % | 0.0038 px |
| FC lens 2 | 0.063 px | 97.8 % | 0.0014 px |

**99.2 % of the radial correction was reachable by moving the k0..k5 COLMAP already had.**
The model class was adequate; the delivered coefficients were wrong.

**But do NOT conclude from this that the gain is re-calibration.** That inference was made
here and has since been falsified by direct measurement: the `--camera_opt rttpf` rung hands
the same photometric optimiser exactly those 16 rttpf parameters, and it recovers only
**+0.082 of the +0.333 dB** at 15k (7 scenes, 6/7 positive), while `rttpf_z` — the same
16 parameters plus a single scalar non-central profile `z(theta)` — adds **+0.244 more on
7/7** and ties the full 111-parameter model. Expressibility and photometric value are
different quantities: the correction lies inside rttpf's span, and gradient descent still
does not land on it. See `worktrees/rttpf-intrinsics` for the rung table, and note the 15k
budget qualifier: at 30k the `rttpf` control climbs to +0.119 / +0.120 dB on the two scenes
measured, so the re-calibration share is budget-dependent and the 25 % figure must always be
quoted with "at 15k" attached.

### Two silent bugs in the calibration analysis, fixed 2026-08-12

Both were the same mistake — assuming a fisheye camera has four radial coefficients — in two
places, and the second one hid the first.

1. `calib_consistency.py` set `RADIAL = [4, 5, 8, 9]`, the **12-parameter
   `THIN_PRISM_FISHEYE`** layout (`fx fy cx cy k1 k2 p1 p2 k3 k4 sx1 sy1`). Every camera it
   reads is **16-parameter `RAD_TAN_THIN_PRISM_FISHEYE`** (`fx fy cx cy k0..k5 p0 p1 s0..s3`),
   whose six radials are **consecutive at 4..9** and carry `theta^2 .. theta^12`. The subset
   silently dropped `k2, k3` and re-labelled `k4, k5` as the `theta^6 / theta^8` terms. Now
   `RADIAL_BY_NPARAM = {16: [4..9], 12: [4,5,8,9], 8: [4..7]}`, and an unrecognised length
   raises instead of guessing. `residual_expressible.py` and `calib_consistency_urad.py`
   inherit the fix through `radial_poly()`; `plate_scale.py` already recomputed independently
   and keeps doing so — leave that redundancy in place, it is what caught this.
2. `residual_expressible.py` hardcoded `BASIS = [1, 3, 5, 7, 9]`, the same four-radial
   assumption on the fitting side. rttpf's span is **seven** terms, `t .. t^13`. Now built
   from the layout by `basis_for()`.

**Why the script's own control did not fire.** It fits the *uncorrected* curve and asserts
the residual is ~0, "because those samples were generated by exactly this form". They were —
by `calib_consistency.radial_poly`, which was truncated *the same way as the basis*. Sampler
and fit shared the error, agreed to 0.0002 px, and the control printed OK. A control only
tests what it does not share with the thing under test; this one was generating its own
ground truth from the same broken function. With both fixed it returns 0.0000.

Truncating the polynomial anywhere fabricates rim noise: plate-scale dispersion across the
7 myscenes fits was 5.0x with `[4:8]`, 1.12x with `[4,5,8,9]`, and is **1.01x** with the
correct 4..9 (`S(theta)/fx` = 0.631-0.638). The affected outputs and their new values are in
the two tables above.

Combining this with the attribution ladder (tunnel, r4, 15k: `off` 28.49 -> `ana` 28.64 ->
`noncentral` 28.73), this section originally concluded a **~60 % re-calibration / ~40 %
non-centrality** split.

> **SUPERSEDED — the split is ~25 / 75, measured directly, in the child worktree
> `gray/worktrees/rttpf-intrinsics`.** The step from *"the correction lies inside rttpf's
> span"* to *"most of the gain is re-calibration"* does not hold. `--camera_opt rttpf` hands
> the baseline's own 16 rttpf parameters to the same optimizer on the same schedule and
> **recovers only +0.082 of the +0.333 dB** (7 myscenes, stderr 0.029, 6/7 positive);
> `noncentral` still beats that re-calibrated control by **+0.251 dB on 7/7**. The two rungs
> are not even finding the same correction — field cosines scatter around zero, three
> strongly negative. And `rttpf_z` (re-calibration + `z` only) reproduces `noncentral`
> exactly (28.258 vs 28.254 on tunnel). **Expressibility was never in question; photometric
> value was, and they are not the same quantity.** See that worktree's IMPLEMENTATION.md,
> "RESULTS — the re-calibration control".

One caveat carries over unchanged: "absorbable by re-fitting" is **not** "COLMAP would have
found it": the two optimise different objectives (photometric over all pixels vs
reprojection of sparse keypoints) on different data.

**What this does to the FullCircle null (this document's main result): nothing — but it
changes the reason.** "The re-bundle did the re-calibration offline" can only ever have been
worth ~0.08 dB, so it explains at most a quarter of the missing gain. The other three
quarters are that **the non-central term itself finds less to eat on this lens**: `z_weights`
2.3-3.9e-3 here against 6.5e-3 on the myscenes ladder, and the outer-ring delta 35x smaller
than `workshop`'s. Two small terms, not one absent one.

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

## Subtractive rungs: can the non-central term stand on its own? (7 scenes, 2026-08-10)

The cumulative ladder answers *"what does z(theta) add on top of everything else"*. It does
NOT answer *"can z(theta) carry the model alone"*, and the two are not the same question,
because the mean-over-depth part of the non-central shift, `z(theta) sin(theta) E[1/t|theta]`,
has exactly the form of a *central* radial correction. `z` and `radial` are not orthogonal,
so a rung read backwards is not a rung removed. Two subtractive rungs were added:

    noncentral_no_ana  ("tilt","radial","z")   31 params   = noncentral minus the 80 anamorphic harmonics
    z_only             ("z",)                   8 params   = the non-central profile alone

7 scenes, -r 4, 15k, identical init / mask / schedule / frozen poses, one run each. Scored by
`scripts/analysis/subtractive.py` (shared masked protocol; the script reproduces the published
27.124 / 27.228 / 27.561 / 27.333 before it is used). `off` is the config-matched baseline.

| scene | `off` | `z_only` (8) | `no_ana` (31) | `noncentral` (111) | SPaGS |
|---|---|---|---|---|---|
| atrium | 27.878 | 27.969 | 27.974 | 28.036 | 27.724 |
| classroom | 28.562 | 28.877 | 28.921 | 28.973 | 28.723 |
| forest | 19.565 | **19.503** | **19.534** | 19.645 | 19.595 |
| library | 31.580 | 31.782 | 31.769 | 31.853 | 31.642 |
| reception | 27.277 | 27.698 | 27.637 | 27.751 | 27.352 |
| tunnel | 28.537 | 28.649 | 28.670 | 28.729 | **28.983** |
| workshop | 27.198 | 27.611 | 27.940 | 27.942 | 27.312 |
| **mean** | 27.228 | **27.441** | **27.492** | 27.561 | 27.333 |
| vs SPaGS | −0.105 | **+0.108** | **+0.159** | +0.228 | — |

**Both amputated models still beat SPaGS.** Dropping the 80 anamorphic harmonics costs
0.069 dB (79% of the gain kept for 28% of the parameters); dropping the central terms
entirely costs 0.120 dB and still leaves +0.108 dB over SPaGS with **8 parameters**.

Two things this measures that the additive ladder could not:

1. **8 non-central parameters beat 103–183 central ones.** On the three scenes where the
   central rungs exist, `z_only` 27.986 > `ana` 27.851 (103 params) > `central_matched`
   27.803 (183 params). This is a stronger control than `noncentral − ana`: it assumes no
   additivity, and it hands the losing family a 13–23x parameter advantage.
2. **The two families are largely redundant, not complementary.** On `reception`, `ana`
   alone was +0.341 over `off` and `noncentral_no_ana` reaches +0.360 *without* anamorphic
   harmonics at all. `z_only` (8 params) matches `no_ana` (31) to within noise on 6 of 7
   scenes; the entire 0.051 dB mean gap between them is `workshop` (−0.329). So
   `noncentral − ana` **understates** what z can do — it measures z only after the central
   terms have already taken what they could.
3. **Redundant is not the whole story: `workshop` is *super*-additive.** There the central
   terms alone give +0.099 over `off` and `z` alone +0.413, but together **+0.744** — well
   above the sum, where `reception` is well below it (+0.341, +0.421, together +0.474). Same
   two families, opposite interaction, so "redundant" is a per-scene property, not a property
   of the model. The reading that fits both: on `workshop` the geometry is wrong enough that
   the central terms cannot do their own job until `z` has fixed it. (Caveat: the
   "central alone" reference is `ana`, 103 params, since `radial` alone was never run on
   these scenes, so these are not exact complements of `noncentral_no_ana`.)

**GOTCHA — the amplitude of `z(theta)` is not identifiable without the central terms.**
Measured by `scripts/analysis/z_shape.py`. The learned profile keeps its *shape*
(corr 0.967–0.998 with the full model's curve) but its amplitude collapses: `z(85 deg)` in
`z_only` is 0.31–0.83x the full model's value, median 0.63x (on `max |z|` rather than
`z(85 deg)`: 0.28–0.86x, median 0.65x — same conclusion, and the script prints both). So the physical readout ("the entrance pupil moves X mm") depends on which
rung produced it, and only the rungs that contain `radial` give the transferable number.
Note the direction: the prediction from the non-orthogonality argument was that z would be
*inflated* by absorbing the radial job; it *shrinks* instead. Also note that a correlation
near 1 between two smooth monotone curves is nearly free — the amplitude carries the
information, not the shape.

LIMITATIONS of this section: one run per scene, no seed repeats, noise floor ±0.05–0.06 dB.
`forest` sits *below* its own baseline in both amputated rungs (−0.062 and −0.031) where the
full model gained +0.080 — the only scene where removing capacity actively hurts, but it is
within noise and should not be read as a mechanism. Neither amputation changes any per-scene
verdict: like the full model, both lose `tunnel` (−0.33 vs SPaGS) and `forest`, and win the
other five. `ana` and `central_matched` were never run on the other four scenes, so the
"8 params beat 183" comparison rests on three scenes.

## PHASE 1 (W4) — the measurement pipeline, and what it exposed (2026-08-11)

The paper's central contribution **is the measurement**, so a measurement pipeline that only
one person can re-run is a contradiction. Phase 1 W4 turns the pile of analysis scripts into
one command and one file.

### What was added / changed

| file | status | what |
|---|---|---|
| `scripts/analysis/run_phase1.py` | **new** | the single command. Runs every stage whose output is missing, then always `pack` + `figures`. `--list`, `--verify`, `--force <stage>|all`, `--only`. |
| `scripts/analysis/pack.py` | **rewritten** | was a 100-line per-scene page builder that crashed on four missing inputs. Now the join point: embeds A1, A2, W1, W2, W3, the ladder and the calibration files verbatim, plus provenance (sha256 + mtime + producing command), a pre-joined figure payload, a headline block and six cross-file checks. |
| `scripts/analysis/make_figures.py` | **new** | draws **all eight** figures from `report_data.json` and nothing else. Imports the drawing code of `dose_response.py` / `ranking_flip.py` / `make_rings_figure.py` so a figure cannot drift from the statistics it illustrates. |
| `scripts/analysis/ladder.py` | **new** | the **pinhole** ablation ladder (there was none), the 2×2 interaction table, and the regularisation audit recomputed exactly from the saved weights. |
| `scripts/analysis/rungs.py` | extended | `RUNGS` went from 4 to the full 6 (`noncentral_no_ana`, `z_only` added), default scene list from `tunnel` to all 7. `rungs.json` therefore changed shape. |
| `scripts/analysis/make_rings_figure.py` | refactored | `main()` split into `draw(data, outdir)`; reads the pack, falls back to `rings_all.json`. |
| `scripts/analysis/dose_response.py` | 3-char patch | `fig.savefig(path, format="svg")` → format from the extension, so the same function writes the pdf the plan asks for. |
| `scripts/analysis/report_data.json` | **new output** | 2.23 MB, everything Phase 1 measured. |

The four inputs `pack.py` had always wanted — `fields.json`, `rings.json`, `rungs.json`,
`crops.json` — turned out to have **producers that had simply never been run**
(`fields.py`, `collect.py`, `rungs.py`, `crops.py`). Nothing was rewritten; they were run,
and `rungs.py` was widened to the six rungs the ladder argument actually needs.

**Verified reproducibility:** `rm figures/*; python scripts/analysis/run_phase1.py --verify`
brings back all 16 files, and the five figures that pre-dated this work come back
**byte-identical in size** (`dose_response.svg` 192 787 B, `estimator_agreement.svg` 216 301 B,
`estimator_lens_level.svg` 81 797 B, `ranking_flip.svg` 250 709 B, `rings.svg` 167 494 B) —
so routing their data through the pack changed nothing.

### The three Phase-1 results, in one page (all of it in `report_data.json → headline`)

**W1 — the dose-response law.** Paired `noncentral − off`, one shared masked eval pass, never
averaged across tracks: myscenes rttpf **+0.333 dB** (n=7, CI95 [+0.189, +0.498], Wilcoxon
p = 0.016, 7/7 positive), mip-NeRF 360 pinhole **+0.196** (n=7, [+0.073, +0.328], p = 0.031,
6/7), workshop_immervision **+0.081** (n=1), FullCircle refit_rttpf **+0.003** (n=9,
[−0.035, +0.035], p = 0.57 — a null, and a same-configuration repeat moves scenes by more than
that). The **pre-registered x axis is falsified**: against `E_sfm` the rank correlation is
**negative** (ρ = −0.446, p = 0.029, n = 24), and the one-parameter fit has r² = −0.52, worse
than the mean. The only training-free statistic that orders the effect is the cross-fit
calibration disagreement **E_calib** (myscenes 958 µrad → +0.333 dB; FullCircle lenses 198 and
139 µrad → +0.003 dB), giving a **practical threshold of 0.43–0.94 mrad** at the 0.068 dB noise
floor — **descriptive only, n = 3 lenses, no CI is quotable**.

**W2 — the ranking-flip predictor.** Of six method pairs, exactly one survives:
**3DGUT − gray**, ρ = +0.506, permutation p = 0.0027, Holm 0.0165, cluster-bootstrap CI95
[+0.15, +0.77] over 34 tracks / **18 independent captures**. At the honest unit of independence
(one median per capture) ρ = +0.387, **p = 0.113** — the two levels disagree, so the headline
must quote the conservative one and the result is **exploratory, not a law**. The other five
pairs are NO EVIDENCE (|ρ| ≤ 0.07 for three of them). Leave-one-dataset-out beats the trivial
"3DGUT always loses" baseline by **one track out of 34** (sign accuracy 0.912 vs 0.882); what
it really buys is amplitude (MAE 0.252 vs 0.340 dB) and 2 of the 4 real inversions.

**W3 — the gain is peripheral, and the ranking survives the radius.** On myscenes the paired
rim-minus-centre delta is **+0.359 dB** for PSNR (CI95 [+0.141, +0.575], p = 0.023, 6/7 scenes)
and **+0.0093** for SSIM (p = 0.015, **7/7**), while LPIPS is flat (−0.0008, p = 0.15) — two
metrics agree, the third says nothing, and that is reported as such. On FullCircle the same
contrast is +0.051 dB, p = 0.10 — consistent with its null. Under the mask-radius sweep,
**23 of 1333** method pairs that are separated at both radii change order between r = 0.95 and
r = 0.85 (**98.3 % survival**; PSNR 96.3 %, SSIM 99.3 %, LPIPS 99.3 %). **r = 1.00 is an
annotated row, not a ranking** on 4 of 7 tracks, and on `immervision_rttpf` the three radii are
literally the same mask (the polynomial-inversion criterion binds before the θ cut, so its
"stability" is arithmetic).

### The regularisation confound is FALSIFIED — recomputed here, not quoted

`ladder.py` recomputes `LensResidual.regularization(l2=1e-2, curvature=1e-2)` exactly, from
`camera_model.lenses.<uid>.{theta,phi,z}_weights` in `gaussians_15000.safetensors`, and
divides by the final `l1` in `losses.csv`. Over **37 runs** (7 myscenes scenes × their rungs,
plus bicycle and stump) the penalty is between **0.0000 % and 0.1395 %** of the data loss.

| scene / rung | #regularised par | penalty | final L1 | penalty / L1 |
|---|---:|---:|---:|---:|
| workshop / `central_matched` | 180 | 2.915e-07 | 1.581e-2 | 0.0018 % |
| workshop / `z_only` | **8** | **3.998e-06** | 1.554e-2 | 0.0257 % |
| workshop / `noncentral_no_ana` | 28 | 2.079e-05 | 1.525e-2 | **0.1364 %** (worst case) |
| reception / `central_matched` | 180 | 3.842e-07 | 1.976e-2 | 0.0019 % |
| reception / `z_only` | 8 | 9.476e-07 | 1.958e-2 | 0.0048 % |

Two independent reasons the confound cannot hold: the term is **inert** (0.14 % at worst
cannot move a 2.4× difference), and **its sign is inverted** — `z_only` (8 params) pays
**13.7× more** than `central_matched` (180 params) on workshop and 2.5× more on reception. It
pays 0.5× on tunnel, i.e. the sign is not even consistent, which is what "inert" looks like.

> **But the L2 is NOT commensurable between channels, and that is the reusable trap.**
> `z_weights` is in units of the **scene radius** (`config.py:208` — `camera_opt_lr_z` is
> documented as scene-scale free), `theta_weights` / `phi_weights` are in **radians** (~1e-4).
> One coefficient, `camera_opt_reg_l2 = 1e-2`, is applied to both, so equal numbers are
> completely different physical pressures. It is harmless *here only because the whole term is
> negligible*; any increase of that coefficient makes the ladder uninterpretable, and no test
> would catch it.

### The additive ladder, and the control that decides it

Mean Δ masked PSNR vs each scene's own `off`, all from `ladder.json` (`interaction_2x2`):

| rung | #reg. par | fisheye, 3 full-ladder scenes | fisheye, all scenes present | bicycle (pinhole) | stump (pinhole) |
|---|---:|---:|---:|---:|---:|
| `tilt` | 0 | — | — | −0.037 | — |
| `radial` | 20 | — | — | +0.026 | +0.028 |
| `ana` | 100 | **+0.180** | +0.180 (3) | **+0.400** | — |
| `central_matched` | 180 | **+0.132** | +0.132 (3) | **+0.362** | +0.088 |
| `z_only` | **8** | **+0.315** | +0.213 (7) | **+0.039** | +0.023 |
| `noncentral_no_ana` | 28 | +0.411 | +0.264 (7) | — | — |
| `noncentral` | 108 | **+0.469** | +0.333 (7) | +0.437 | +0.061 |

**Both tables in this section count *regularised* parameters** — `{theta,phi,z}_weights`
only — because that is what the confound is about. The trained count is 3 higher wherever
`tilt` is on (`omega` is trained but not regularised), which is why `noncentral` reads 108
here and **111** everywhere else, and `noncentral_no_ana` 28 here and **31**. Verified
against the checkpoints by counting non-zero entries: 31 / 8 / 111 for
`noncentral_no_ana` / `z_only` / `noncentral`.

Four readings, and the last two are new:

1. **`z` alone beats every central model put together** on fisheye: +0.315 against +0.180
   (`ana`) and +0.132 (`central_matched`, which has 22× more parameters).
2. **The premise of `plan_camera_non_centrale_3dgrt.md` §4 is dead.** It predicted the
   anamorphic term would carry the PSNR and the non-central term would carry only the
   argument. It is the exact opposite: anamorphic is the *smallest* marginal slice
   (+0.058 = 0.469 − 0.411) and `z` carries both.
3. **THE NEGATIVE CONTROL HAS LANDED, AND IT PASSES.** On a pinhole camera a non-central
   pupil cannot exist, so `z_only` there must be zero. It is: **+0.039 dB on bicycle and
   +0.023 on stump, both under the 0.068 dB noise floor**, while `central_matched` takes
   +0.362 of bicycle's +0.437. The order between the two rungs **inverts completely between
   camera families** (pinhole: central ×9 over z; fisheye: z ×2.4 over central). That 2×2 is
   much stronger than either cell alone — a capacity absorber would have absorbed on pinhole
   too. Until this run the ladder table could not be written down at all.
4. **The `ana` rung on pinhole (new here) separates two things that were confounded.**
   `ana` (100 params, 10 knots, azimuthal freedom) gets **+0.400** where `central_matched`
   (180 params, 18 knots, same azimuthal freedom) gets +0.362. So on bicycle the gain is
   bought by **azimuthal freedom, not by knot count** — extra knots buy nothing, possibly
   less than nothing. This also re-confirms that "bicycle's learned residual is purely
   radial" is false: `radial` alone gets +0.026, one fifteenth of `ana`.

**Caveats that must travel with that table.** `n = 1` per cell, no seeds. The fisheye
3-scene column is tunnel/workshop/reception; the 7-scene column is lower (+0.213 vs +0.315 for
`z_only`) because the four extra scenes are the low-signal ones — quote which column you mean.
`tunnel/z_only` was still drifting 58 % at 15k, so its +0.111 is not a result in either
direction. And the two families are scored differently on purpose (pinhole = full frame, no
mask; fisheye = masked r=0.95): **only deltas against each scene's own `off` are ever
compared, never absolute dB**.

### LIMITATIONS and gotchas of the pipeline itself

* **`report_data.json` is a JOIN, not a recomputation.** `pack.py` computes no metric of its
  own. If an input is stale the pack is stale — which is why every input carries its sha256,
  size and mtime in `provenance`, and why `--strict` exists.
* **Its inputs were written at different times against checkpoints that MOVED.** pueue tasks
  1539-1545 retrained 7 of the 9 FullCircle `noncentral` runs on 2026-08-10 between 19:30 and
  20:40 local, i.e. **after** `calib_consistency.json` (13:16) and `plate_scale.json` (16:58)
  were written and **before** `dose_response.json` and `ladder.json` (2026-08-11). Their
  FullCircle learned-residual numbers therefore describe two different checkpoint instances.
  The differences are below the noise floor but they are not zero. `provenance` carries every
  mtime in UTC — read it before joining those three files by hand.
* **Two of the six pack-time checks are plumbing, not evidence.** `rungs_vs_subtractive`
  (0.0 dB over 34 rungs) and `w1_gain_vs_w3_rings` (0.0 dB) compare two invocations of the
  *same* scoring code; they prove both saw the same runs, not that the metric is right. The
  informative ones are the `radial_eval` non-regression anchor (−7e-07 dB against the
  canonical 28.53749677) and the `E_sfm` joins (0.0 µrad on 198 rows).
* **`crops.json` is base64 WEBP inside the JSON.** It is 177 KB of the 2.23 MB. If you ever
  need the pack small, that is the knob; the crops figure is the only consumer.
* **The figures import the analysis modules for their drawing code**, but take their data from
  the pack. Importing `dose_response` pulls in scipy; it does no work at import time. If you
  add a figure, take the payload as an argument — never re-open a source JSON.
* **`run_phase1.py` will not launch a GPU stage, ever.** `rings_all` (~25 min) and
  `mask_radius_sweep` (~3 h) need LPIPS on a card. It prints the `pueue add` line and stops.
  If you "fix" that, you have broken the house rule, not the script.
* **The ladder is a snapshot of the runs on disk at pack time.** Another session was still
  adding pinhole rungs while this was written (`bicycle_ana` landed 2026-08-10 21:26,
  `stump_z_only` at 2026-08-11 01:03). `python scripts/analysis/run_phase1.py --force ladder`
  re-reads them. One benign config difference: `bicycle_off` has
  `camera_opt_from_iter = 8000` against 3000 for the rungs — irrelevant, `off` has no camera
  model to unfreeze, and every other parity key matches.
* **`fields.py`'s console line prints pixels via `fx`.** `radial=1.197px` on workshop is
  `max|Δθ| · fx`, not `max|Δθ| · S(θ)`. It is illustrative only and it is *not* the number in
  the pack: `report_data.json` carries the angles, and A2's `plate_scale.json` carries the
  true local plate scale (0.633 fx at the evaluated 85.5° edge, 0.554 fx at 90°).
* **The `*_px` columns of `calib_consistency.json` were in QUARANTINE, and the cause is now
  FIXED (2026-08-12) — see "Two silent bugs in the calibration analysis" below.** They are
  still embedded under `calib_consistency_px_LEGACY` in old packs; re-run the pack to refresh
  them. Nothing in `report_data.json`'s x axes ever depended on the bug — `E_sfm` and
  `E_calib` project through `pycolmap`'s own camera object, `E_learned` through the checkpoint
  spline — which is why `dose_response.json`'s `e_shared_bugfixed` (myscenes 634 µrad) is
  *still* the number to prefer over the freshly regenerated 791 µrad: it integrates to the
  85.5° mask edge through the full projection, while `calib_consistency` integrates to the
  monotonicity fold through the radial polynomial alone. Same ordering, different estimator —
  never mix them in one table.
* **`pack.py` will happily pack a hole.** Missing optional inputs print a warning and land as
  `null` with `provenance[...]["present"] = false`. Read that block before trusting a section.

## The pose-independent ray cache — where the FPS tax actually is (2026-08-11/12)

**The paper would today publish a ~2x FPS penalty on its headline contribution, and the
penalty is not caused by the contribution.** Two independent readings of the `fps.csv` on
disk say so:

* **It does not depend on the rung.** `tunnel` at -r 4 / 15k: `ana` 117.44, `central_matched`
  115.98, `noncentral` 112.08 — within 5 %, though `central_matched` has 183 parameters and
  no non-central term at all.
* **It is already there with an EMPTY residual.** The -r 8 ladder (`tmp/ladder_final`, 7500
  it, gaussian counts within 2 %): `off` 725.22, **`passthrough` 625.55 (0.86x)**,
  `noncentral` 375.99 (0.52x). `passthrough` runs the same Python path with a residual pinned
  at zero, so 0.86x is the cost of the **two `[H,W,3]` framebuffer copies plus the per-image
  GEMM alone**, and only the 0.52x → 0.86x span is synthesis. Independently, FullCircle
  `room1` on an idle card: `off` 471.42 against `noncentral` 249.83 (0.53x).

What the rungs share is not non-centrality — it is that `raytracer.py` synthesises the rays in
**torch, per image**, and sets `rays_from_python=True`.

> **Pre-registered prediction for the benchmark**, so the result cannot be read after the
> fact: the cache should move `noncentral` from ~0.52x of native toward the `passthrough`
> floor of ~0.86x, and **must not** exceed it. Anything past 0.86x would mean it also skipped
> work that is genuinely per-image, i.e. a correctness bug that the timing found first.

### What was split

`CameraModel.forward()` is now two halves (`gray/camera_model.py`):

| half | depends on | contents | cached |
|---|---|---|---|
| `camera_frame()` | lens parameters + base bearings | residual-rotated `bearings [H,W,3]`, `z_profile [H,W]`, `raxel_offset` | **yes** |
| `apply_pose()` | + the view | c2w rotation, world origin, per-view SE(3), the axis `z` displaces along | never |

Key: `(base["cache_key"], uid, height, width)`, and `base["cache_key"]` already carries the
camera model, `fov_y`, the image size and the intrinsics themselves — the key that was
widened after the stale-bearing collision. So the two eval models of
`render.py --eval-models pinhole rad_tan_thin_prism_fisheye` cannot collide here either.

**The one line a naive cache crosses.** `z(theta)` is a scalar *in the camera frame*, but the
direction it displaces the ray origin along is `rotation[:, 2]`, the world-space optical
axis — it rotates with the view. Caching one operation further would freeze the first view's
pose into every later render, and the result would look like a plausible image, not like a
crash. `tests/test_camera_model_cache.py::test_cpu_cache_does_not_freeze_the_pose` therefore
does not check "the two fields differ": it reconstructs pose B's field from pose A's by the
relative rotation `R_B R_A^T` and requires a chord below 1e-6, and it rebuilds the non-central
origin from pose B's own axis with `torch.equal`, then asserts that pose A's axis would have
been visibly wrong.

### The rules that keep it correct

* **Only under `no_grad`** — render, eval, FPS, and training previews. Under autograd the
  tensors carry a graph a previous backward has freed and the parameters move every step, so
  training always recomputes. `test_cpu_training_path_is_never_cached` pins that.
* **Invalidated by** `step()`, `set_frozen()`, `_load_from_state_dict()`,
  `materialize_from_state_dict()`, `Raytracer.set_render_resolution()` and
  `apply_camera_model_transfer()` (`gray/config.py`, which pokes the parameters by hand).
  Anything else that writes a parameter tensor directly **must** call
  `invalidate_ray_cache()`; there is no way to detect it without a device sync.
* **Escape hatch**: `GRAY_NO_RAY_CACHE=1`, or `model.ray_cache_enabled = False`. Deliberately
  an environment variable and not a `config.py` flag — several sessions edit that file in
  parallel and a new field there would collide.
* **Bit-exact, by `torch.equal` and not by a tolerance**: 7 rungs x 2 resolutions x 3 poses,
  plus a verbatim copy of the pre-split `forward()` as the reference (33 CPU tests, no GPU).

### What it does NOT remove — read this before quoting a speedup

Per image the rung still pays **one `[H,W,3]x[3,3]` GEMM, a renormalisation, and two `[H,W,3]`
`copy_()` into the framebuffer**. Those are per-image by construction (they depend on the
pose) or by the framebuffer contract. So this recovers the *synthesis*, not the whole tax, and
native parity is not reachable this way: the rest needs the two cached tables to live on the
CUDA side as a `Camera::bearing_table` / `origin_offset_table`, which is exactly the shape
`camera_frame()` returns them in.

> ⚠️ **No post-cache FPS number is quoted here, on purpose.** This work ran under an
> instruction to queue no GPU task (four other sessions held the queue), so the speedup is
> **unmeasured**; the numbers above are all pre-cache runs already on disk, and their card
> occupancy at measurement time is unrecorded except for the FullCircle sweep.
> `scripts/bench_ray_cache.py` is the A/B — one process, one checkpoint, ABBA-interleaved
> arms, and it refuses to print a timing unless the two arms' pixels are `torch.equal`. The
> `masked-efficiency` worktree measured 1.03–1.05x for the analogous *native*-path bearing
> cache on single-camera fisheye scenes and **0.995x on a two-camera rig** (rebinding costs
> what it saves) — so a rig may show nothing here either.

**Memory.** One entry per `(camera uid, render resolution)`: ~4 floats/pixel on top of the
~9 that `_base_bearing_cache` already holds. Nothing evicts either of them. On a COLMAP model
with one camera per image this grows with the number of images — pre-existing, since the base
bearings already do it, but the ray cache adds ~45 % to that footprint.

## Does the non-central origin cost BVH coherence? First measurement — no

The first question a graphics reviewer asks: neighbouring pixels no longer share an origin,
so the primary bundle is no longer a pencil and the BVH cannot amortise a node fetch across
it. `scripts/traversal_coherence.py --from-runs` reads the `hits/ray` that `train.py` already
logs into `traversal_stats.csv` for every `off` / `noncentral` pair on disk (CPU, no render):

| | n | mean | sd | CI95 | median | range |
|---|---:|---:|---:|---:|---:|---:|
| hits/ray ratio (`noncentral` / `off`) | 13 | **0.9981** | 0.0269 | [0.9835, 1.0127] | 0.9989 | 0.957–1.050 |
| accum/ray ratio | 13 | 0.9993 | 0.0240 | — | — | — |

Paired Wilcoxon against 1.0: **p = 0.735**. So: **no cost, down to about ±1.3 %.**

**Read the confound before quoting it.** The two runs of a pair prune to slightly different
gaussian counts (ratio 0.990 ± 0.007), and hits/ray depends on how many gaussians there are.
It does not explain the scatter — Spearman(hits ratio, gaussian ratio) = **−0.374, p = 0.209**,
and the sign is *opposite* to what the confound would produce — but this mode answers "is
there a large effect", not "how large is it".

The controlled version is `--measure` and it needs a GPU (not run here). One checkpoint, one
view set, two arms that differ **only** in whether `z_weights` is zeroed: the ray directions
are bit-for-bit identical between arms and the script reports `arms_are_a_control` to prove
it. Its headline ratio pools over the pixels **both** arms traced, because a per-arm
`hits > 0` mask puts a different denominator on each side — a grazing pixel pushed over or
under that line by the origin shift would show up as a "coherence cost" on its own.

## GOTCHAS (things you would not guess)

* **A `no_grad` render collects NO traversal statistics, silently.**
  `params.stats.num_gaussians_hit[...]++` (`cuda/shaders.cu:65`) and
  `num_gaussians_accumulated` (`cuda/forward_pass.cu:127`) are both inside `if (grads_enabled)`,
  and that flag comes from `torch::autograd::GradMode::is_enabled()` on every `forward_pass()`.
  Measuring hits/ray under `no_grad` returns zeros, not an error — which is why
  `traversal_coherence.py --measure` renders under `enable_grad()` (never calling `backward()`)
  and therefore needs a training iteration's memory, not an eval one's.
* **`traversal_stats.csv` has a comma-separated header and space-separated rows**
  (`train.py:217` against `train.py:511`). `split(",")` silently returns one field per row.
  Same shape in `num_gaussians.csv`. `tests/test_traversal_coherence.py` pins both.
* **`measure_fps.py` writes `fps.csv` into the run directory.** Running it twice to A/B
  anything leaves the *last* arm in the canonical file that the FullCircle and myscenes FPS
  columns are read from. Use `scripts/bench_ray_cache.py`, which writes only where `--out`
  points.
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

<!-- ===== merged from branch `rttpf-intrinsics` on 2026-08-12 ===== -->
*The two sections below arrived with the `rttpf` / `rttpf_z` rungs. They were
written in the `rttpf-intrinsics` worktree; paths and worktree names in them refer
to that worktree and are kept verbatim rather than rewritten, so the provenance of
each measurement stays checkable.*

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
