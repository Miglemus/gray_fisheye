# `scripts/perf/` — measuring speed so the number can be defended

Two things live here:

* **K1.a — a harness that makes an unciteable speed measurement impossible.**
* **K1.b — the protocol for the FoV sweep**, `FPS(field of view)` for a ray tracer and a
  rasteriser on the *same* gaussians. Ready to run, **not run**.

Nothing here has been executed on a GPU. At the time of writing `pueue status` holds ~180
queued tasks from four other sessions, and adding to that would manufacture exactly the
contention this harness exists to detect.

---

## 0. Why: not one FPS number in this project is citable

Run it yourself:

```bash
python scripts/perf/collect_fps.py --legacy-audit \
    /workspace/gray/tmp/final \
    /workspace/gray/worktrees/noncentral-camera/out \
    /workspace/gray/worktrees/fullcircle-erp/out
```

**80 of 80** `fps.csv` / `fps.json` value files under those three roots have no provenance
(**230 of 230** if `tmp/` is included). None of them records which of the machine's two
cards produced it — and the two cards differ by roughly 1.6×. On top of that:

| defect | where |
|---|---|
| four `gray`-rttpf head values written inside a 31 s window (impossible sequentially: each rebuilds a BVH over ~5·10⁵ instances and warms up 57 views) | `dataset/` FPS columns |
| the same scene, resolution and model size reading 118.87 on one track and 149.74 on another | myscenes vs FullCircle tracks |
| five FullCircle values that `IMPLEMENTATION.md` itself calls contention artefacts | `out/fullcircle_rttpf/*` |
| SPaGS reports a **best of 10**, gray / DFGS / 3dgrut report a **single pass**, 3DGEER reports an **unsynchronised Python wall clock** | `protocol.py --diff` |
| `dataset/fullcircle_code/queue_perf.sh` pointed gray at `worktrees/person-masks`, deleted 2026-08-04 → gray reads `FPS=None` on all 18 FullCircle rows while its three rivals carry numbers | **fixed, see §4** |

`plan_phase1_loi_calibration.md` §8 already forbids publishing any of it. This directory
is what replaces it.

---

## 1. The harness

```bash
# one provenanced measurement (GPU — always queue it)
python scripts/perf/bench_fps.py -m <run dir> --gpu 1 --repeats 3 \
    --out tmp/perf/<name>.perf.json

# the table; it refuses anything unprovenanced, loudly, with exit status 1
python scripts/perf/collect_fps.py --roots tmp/perf

# what each method's number actually measures
python scripts/perf/protocol.py            # the table
python scripts/perf/protocol.py --diff     # only the columns that break comparability
```

| file | what |
|---|---|
| `provenance.py` | the schema (`gray.perf.provenance/1`), the nvidia-smi probes, and `validate()`. Pure stdlib so it runs in DFGS'/3dgrut's/SPaGS'/3DGEER's venvs too. |
| `bench_fps.py` | the harness: exclusive card, sequential, explicit warmup, explicit sync, median of 3, sidecar on every measurement. Exposes `GrayBench` (one load, many measurements), `bench_one()` and `timed_passes()`, so the sweep and the single measurement share one clock. |
| `collect_fps.py` | the collector. Per-row validation **and** cross-row checks. `--legacy-audit` explains why every existing value file is rejected. |
| `protocol.py` | the printable protocol table, per column: repetitions, aggregation, warmup, synchronisation, timed region, readback, provenance. |
| `stamp_provenance.py`, `stamp_gray_run.py` | attach conditions to a value some *other* program produced. Writes `tier="attested"`, which the collector rejects by default. |
| `fovmath.py` | the FoV → intrinsics mapping and the 180° wall. No torch. |
| `fov_sweep.py` | the sweep driver. `--plan` prints the camera plan with no GPU at all. |
| `queue_fov_sweep.sh` | prints (never submits, unless `--queue`) the exact `pueue add` lines. |
| `test_perf.py` | 54 CPU tests. |

### What makes a row citable

`collect_fps.py` rejects a row when **any** of these holds. Each rule blocks a defect that
has already produced a wrong number here:

| rule | blocks |
|---|---|
| no sidecar | 100 % of what is on disk today |
| `tier="attested"` (post-hoc stamp) | exclusivity claimed but not observed around the clock |
| a foreign CUDA context on the card, before **or** after the timed region | the five contention artefacts; four OOM deaths |
| `CUDA_DEVICE_ORDER != PCI_BUS_ID` | torch's index ≠ nvidia-smi's → the sidecar names the wrong card |
| repeat spread > 5 % | the card was not in a steady state |
| any required field missing | see `provenance.REQUIRED` — and `test_perf.py` proves every entry there is actually enforced |

and rejects the **whole table** when rows disagree on: GPU uuid, aggregation, repeat
count, timed region, resolution, view count, driver version — or when the timeline is
physically impossible (the 31-second-window check).

> **A pueue group does not reserve a GPU.** It serialises its own tasks; a process started
> outside it holds the same card. Every queued line therefore carries `gpuwait.sh` as well
> as `--group gpu1`, and `bench_fps.py` aborts on top of that.

### The one thing it deliberately does not do

It never writes `fps.csv`. The repo-root `measure_fps.py` does, and running it twice to
A/B anything leaves the *last* arm in the canonical file that the FullCircle and myscenes
tables read (`IMPLEMENTATION.md`, GOTCHAS). `measure_fps.py` is left untouched on purpose:
four other sessions are using it right now, and the safe move was to add a harness rather
than to change one under them.

---

## 2. K1.b — the FoV sweep

> **The claim to be tested.** Every published speed comparison in this line of work — GRay's
> own 248 / 253 / 68 FPS on an RTX 4090, and 3DGUT, 3DGEER, Radiant Foam, GRTX — is
> measured on **pinhole** data (MipNeRF360, Tanks & Temples). A rasteriser's cost grows
> with distortion (more tiles touched per primitive, sigma-point re-estimation, degenerate
> footprints at the rim); a ray tracer's does not, because it never projects a primitive.
> **Nobody has published the RT/raster ratio as a function of field of view.**

Preview it with no GPU:

```bash
python scripts/perf/fov_sweep.py --plan -m /workspace/gray/tmp/r4/tunnel_off
python scripts/perf/fovmath.py --width 1024 --height 1024
```

### 2.1 The camera axis: one scalar, exactly

`OPENCV_FISHEYE` with `k1 = k2 = k3 = k4 = 0` **is** the equidistant mapping `r = f·θ`.
At a fixed output size, choosing the half-field `θ_max` fixes `f = R / θ_max` with
`R = min(W,H)/2`, and the whole sweep is a sweep of one number.

Three reasons this beats every alternative:

1. **Nothing has to be inverted.** gray inverts `r(θ)` by bisection
   (`cuda/core/opencv_fisheye.cuh`) and 3dgrut folds the radial polynomial at its first
   stationary point. With non-zero `k`s the two engines disagree about where the lens stops
   being invertible, and the disagreement lands *at the rim* — the part of the field the
   sweep is about. With `k = 0` there is nothing to disagree on.
2. **The lens family never changes.** Re-fitting a real rttpf lens per field angle would
   change mapping shape and distortion at once.
3. **The rectilinear control lives in the same construction**: `f_p = R / tan(θ_max)`, same
   `R`, same resolution, same gaussians. It exists for 60–120° and is *undefined* past
   180°, which is itself a result worth stating.

The field is taken across the **inscribed circle** (the shorter axis), so the image circle
sits inside the frame at every point and the background fraction is constant — otherwise a
growing corner-background area would read as a speed-up.

### 2.2 ⛔ The 180° wall — measured in the source, not assumed

```
cuda/core/opencv_fisheye.cuh:31   constexpr float FISHEYE_MAX_THETA = 1.5707963f;
cuda/core/rtpf.cuh:120            constexpr float kMaxTheta        = 1.5707963f;
cuda/core/tpf.cuh:96              constexpr float kMaxTheta        = 1.5707963f;
```

All three fisheye raygens return the invalid bearing `(0,0,0)` past θ = 90°. **gray cannot
render more than 180° of field with any camera model it has today**, so the task's
"~60 to ~200 degrees" is currently **blocked above 180°**, not a design choice. The sweep
stops at 175° and `fovmath.check_gray_can_trace()` raises rather than letting a 200° point
render a black annulus — which would read as *fewer hits, more background, higher FPS*,
i.e. a silent 2× speed-up in the wrong direction. `test_perf.py` pins that guard.

Two consequences to state in the paper rather than hide:

* Reaching 200° needs an **uncapped equidistant camera model** (~15 lines in `cuda/core/`,
  the same shape as the equirectangular model already on branch `erp-camera`,
  `cuda/core/camera.h:73`). That is a CUDA change plus a rebuild, outside this task's write
  perimeter.
* **3DGUT is not capped**: its `FThetaCamera` carries an explicit `max_angle`
  (`threedgut_tracer/include/3dgut/kernels/cuda/sensors/cameraProjections.cuh:131`). So on
  today's code the *rasteriser* reaches fields the *ray tracer* cannot. Report the
  asymmetry; do not quietly truncate the raster arm to match.

### 2.3 The transfer is lossless — of **what**, exactly

The protocol rests on "the same gaussians can be rendered by several engines". Verified,
not assumed, on 2026-08-12 (`test_perf.py::test_gray_to_ply_roundtrip_is_bit_exact_...`),
on a 156 206-gaussian FullCircle checkpoint:

| | result |
|---|---|
| `mean`, `opacity`, `rotation`, `scale`, `sh_coeffs_dc`, `sh_coeffs_rest`, `current_sh_degree` | **bit-exact** through safetensors → PLY → safetensors (`torch.equal`, not a tolerance) |
| `vignetting.coefficients`, `vignetting.principal_point` | **silently dropped** |
| `camera_model.lenses.*` (`omega`, `theta_weights`, `phi_weights`, `z_weights`) | **silently dropped** |

And read the README's own sentence, which is stronger than the summary usually quoted:

> "The gaussians produced by this method are **incompatible with 3DGS**; … different
> kernel, different sorting, and perspective accuracy … The parameter conversion is
> lossless (a GRay → 3DGS → GRay round-trip reproduces identical metrics) but **3DGS-based
> viewers will produce blurrier images with some differences**. 3DGRT renders should look
> identical."

So the honest statement of the asset is:

* ✅ **the parameter set transfers exactly**, which is what a *cost* comparison needs;
* ❌ **it is not an image-identical comparison** against a 3DGS-family rasteriser. The
  rasteriser renders a different (blurrier) image from the same parameters.
* ⚠️ `convert/to_3dgrt.py --match-kernel` (**default true**) *adds a log-scale offset to
  every scale* so 3dgrut's generalized gaussian reproduces gray's `exp_power` kernel. It
  makes the images comparable and makes the parameters *not* identical. Run both arms and
  say which one a number came from.
* ⚠️ `to_3dgrt.py` moves tensors to `.cuda()` and its `--template-checkpoint` default
  (`~/Desktop/3dgrut/...`) does not exist here. Queue it and pass a real template.
* ⚠️ Use an **`--camera_opt off`** run. A `noncentral` checkpoint's camera model does not
  survive the transfer, and it would also put ray synthesis on the Python path — a 0.52–0.86×
  tax that has nothing to do with the field of view. `fov_sweep.py` refuses a non-`off` run.

### 2.4 What is controlled, and what is not

**Held fixed by construction** — one checkpoint, one scene, one output resolution, one set
of 30 poses, one physical card, all points back to back in **one process and one load**,
timed by the same `timed_passes()`.

> The single load is not an optimisation. A second `Raytracer` in one process aborts at
> teardown (`PipelineWrapper::~PipelineWrapper` — the reason `pyproject.toml` sets
> `addopts = "--forked"`), so an 11-point sweep that reloaded per point would die halfway
> through; and reloading would rebuild the BVH, adding a second variable to a
> single-variable experiment. `GrayBench` enforces one instance per process and
> `test_perf.py` pins the guard.

**Cannot be held fixed — report as columns:**

| variable | why it moves | mitigation |
|---|---|---|
| **content in frustum** | a 175° frame sees far more gaussians than a 60° one | both engines see the *same* content at each point, so the **RT/raster ratio is controlled**; the per-engine absolute curve is not. Report each engine normalised to its own 60° point, and `visible_fraction` (fraction of gaussian centres inside the cone, computed outside the clock). |
| **angular sampling density** | at fixed pixels a 175° frame samples ~7× fewer px/sr than a 60° one | `pixels_per_sr` is a printed column of the plan. "FPS at fixed resolution" and "FPS at fixed angular resolution" are different curves; this sweep is the first. |
| **image agreement between engines** | gray traces one exact ray per pixel; a rasteriser approximates the projection, and that approximation is what degrades with field angle | a speed ratio with no fidelity column is meaningless — the rasteriser can always be fast by being wrong. Score the two engines against each other per sweep point before reading any ratio. |
| **kernel / sorting / hybrid transparency** | different by construction between gray, 3DGS and 3DGRT | 3DGRT is the near-identical one (README: "renders should look identical"); 3DGS is not. Prefer 3DGRT/3DGUT as the rival and say why. |
| **BVH build cost** | excluded from the clock (steady-state throughput) — but BVH *quality* under a non-central origin is a separate question | already measured: `hits/ray` ratio 0.9981 ± 0.0269, Wilcoxon p = 0.735 (`scripts/traversal_coherence.py`). |
| **RT-core generation** | see §3 | the confirmation pass |
| **no anti-aliasing anywhere in gray** | one infinitesimal ray per pixel, `jitter_primary_rays` false (`IMPLEMENTATION.md` limitation 3) | the rasteriser's footprint filtering is not free; this is a fidelity difference the ratio does not see |

### 2.5 The sweep points

| arm | FoV (°) | note |
|---|---|---|
| `fisheye_equidistant` | 60, 80, 100, 120, 140, 160, 175 | the main curve |
| `pinhole_control` | 60, 80, 100, 120 | same field, same resolution, same gaussians, rectilinear projection — isolates *cost of a non-rectilinear projection* from *cost of a wider field* |
| *(blocked)* | 180, 200 | needs an uncapped equidistant model, §2.2 |

Three resolutions (512², 1024², 1920×1080) because resolution is not a free variable: a
ratio that only holds at one output size is not a law.

---

## 3. ⚠️ The most attackable variable: both cards here are Turing

```
GPU 0  NVIDIA GeForce RTX 2080 Ti  11 GB  cc 7.5   driver 570.207
GPU 1  NVIDIA TITAN RTX            24 GB  cc 7.5   driver 570.207
```

`sm_75` = **first-generation RT cores (2018)**. Every later generation changed the
traversal hardware substantially (Ampere: 2× triangle throughput and hardware motion blur;
Ada: opacity micromaps, displaced micro-meshes, 2–3× again). A reviewer's first question
about an RT-vs-raster ratio measured on Turing is whether it survives on modern hardware,
and the honest answer today is *unknown*. The confirmation pass is not optional.

**Where to get an Ada/Ampere card with RT cores** (see the `ccc` skill,
`references/clusters.md` §"Which clusters have RT cores"):

| cluster | GPU | RT cores | agent-reachable |
|---|---|---|---|
| **Killarney** | L40S-48GB (Ada AD102) | **yes, 3rd gen** | ❌ human-only, no automation node |
| **Vulcan** | L40S-48GB (Ada AD102) | **yes, 3rd gen** | ❌ human-only, no automation node |
| Fir, Nibi, Rorqual, Trillium, tamIA | H100 (Hopper GH100) | **none** | ✅ |
| Narval | A100-40GB (Ampere GA100) | **none** | ✅ |
| Nibi (MI300A) | AMD CDNA3 | n/a — no CUDA, no OptiX | ✅ |

> **Datacenter Hopper and Ampere ship zero RT cores**, contrary to many secondhand spec
> tables. OptiX still runs there, on the SMs. So an H100 run is **not** a confirmation
> pass — it is a *software-traversal* data point, which is interesting in its own right
> (it isolates how much of gray's speed is the RT hardware) but it does not answer the
> reviewer's question.

Practical consequence: the only hardware-RT clusters in the allocation are **human-only**,
and the automation-node request is still open (memory: *ccc cluster setup state*). So the
confirmation pass is currently **blocked on a human**, on a ticket, or on a local Ada card.
Do not silently substitute an H100 for it.

A third option, cheapest of all: run the identical sweep on the 2080 Ti as well as the
TITAN RTX. Both are Turing, so it does not answer the generation question, but two cards
of the same generation and different memory bandwidth do tell you whether the curve is
bandwidth-shaped or traversal-shaped — which is the mechanism a reviewer will ask about
next. `bench_fps.py --gpu 0` and one extra sweep directory.

---

## 4. What was fixed in `dataset/fullcircle_code/queue_perf.sh`

* the gray branch pointed at `/workspace/gray/worktrees/person-masks`, **deleted
  2026-08-04**; its `[ -d ]` guard then failed on every gray invocation, which is why gray
  reads `FPS=None` on all 18 FullCircle rows while DFGS / 3dgrut / SPaGS carry numbers.
  It now resolves gray's root from a candidate list (`fullcircle-erp`, then `/workspace/gray`,
  then the old path), the same resolution `build_manifest.py` already does. The DFGS and
  3dgrut `person-masks` worktrees **do** still exist, so only the gray branch was broken.
* added a `gray-nc` method for the `gray-non-central` column the viewer already has.
* every branch now goes through `gpuwait.sh`: the header's claim that *"pueue's gpu1 group
  is 1-parallel, so the timed pass also gets the card to itself"* is false, and believing
  it is how five values became contention artefacts.
* `CUDA_DEVICE_ORDER=PCI_BUS_ID` was `export`ed in the gray branch only; `gpuwait.sh` now
  sets it for all three.
* `DRY_RUN=1` prints instead of queueing. `STAMP=1` appends a `stamp_gray_run.py` call so
  the value at least carries its conditions (`tier="attested"` — still not publishable).

---

## 5. Tests

```bash
python -m pytest scripts/perf/test_perf.py -q -p no:cacheprovider     # 54 tests, CPU only
```

They are the non-regression proof for four things: every field in
`provenance.REQUIRED` is genuinely enforced (parametrised over the whole list, so adding a
field without reading it fails); the cross-row checks catch mixed cards, mixed estimators
and the 31-second window; the FoV→intrinsics mapping is exact and the 180° wall raises;
and the `convert/` round-trip really is bit-exact on the gaussians **and** really does drop
`vignetting.*` / `camera_model.*`. That last one is a *characterisation* test — if a future
change makes vignetting survive, it fails and the protocol note above must be updated.

The protocol table cannot drift from the harness: `protocol.py` imports `TIMED_REGION` and
`TIMED_REGION_EXCLUDES` from `bench_fps.py` rather than restating them, and a test asserts
the row matches.

---

## 6. What is still missing

| | why it matters | where it is blocked |
|---|---|---|
| `fov_sweep_3dgut.py` — the raster arm | without it the sweep has one arm and no ratio | needs a synthetic-camera injection point in 3dgrut's dataset layer (`threedgrut/datasets/datasetNcore.py`, which already has `camera_max_fov_deg` and a per-camera `max_angle`). That is a different repo, outside this task's write perimeter. |
| an uncapped equidistant camera model in gray | the 180–200° end of the sweep | ~15 lines in `cuda/core/` + a rebuild; mirror `erp-camera`'s `equirectangular_unproject` |
| a per-point fidelity score between the two engines | a speed ratio without it is not a result | needs the raster arm first |
| the Ada/Ampere confirmation pass | the Turing objection | §3 — human-only clusters, open automation ticket |
| re-measuring every existing FPS number | 80/80 are unciteable | queued lines are printed by `queue_fov_sweep.sh`; the queue must drain first |
