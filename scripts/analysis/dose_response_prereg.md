# Pre-registration — W1, the dose-response curve

**Written 2026-08-10T21:46:30Z, worktree commit `8e0a15bf6765fd75c72a6c9b9015b2044b04ac67`.**

This file was written **before** any curve was fitted and before any x-estimator was joined to
any gain. It fixes the design, the point list, the functional form and the decision rules, so
that whatever comes out is not a post-hoc story. Anything added after the fit is marked
EXPLORATORY in `dose_response.json` and must be read as such.

Author: W1 agent (Phase 1). Zero GPU is used anywhere in this analysis.

---

## 1. The dependent variable `y`

`y` = masked-PSNR gain in dB of `--camera_opt noncentral` over `--camera_opt off`,
**paired scene by scene**, both sides scored by ONE shared evaluation pass over the raw
rendered PNGs (`scripts/radial_eval.evaluate`, `disk_per_view_mean`, mask radius 0.95 for the
fisheye tracks; full frame for the pinhole track, which has no mask).

Rules fixed in advance:

* No self-reported metric from any repo is used, on either side.
* The `off` control must have **identical training configuration** to its `noncentral`
  partner apart from the `--camera_opt` flag. Verified per scene from `config.json`
  (`vignetting_comp`, `vignetting_terms`, `batch_size`, `downsampling`, `iterations`).
  Consequence, decided in advance: for myscenes `workshop`, the published baseline
  `/workspace/gray/out/workshop_fisheye_baseline` was trained with `vignetting_comp=False`
  and `batch_size=1` and is therefore **disqualified as the paired control**; the matched
  control is `/workspace/gray/tmp/noncentral/fix15k_workshop`. A sensitivity line using the
  published run is reported but is not the primary number.
* Tracks are never averaged with each other (different lens, different frame regime,
  different resolution convention). Every fit that pools tracks is reported alongside the
  per-track breakdown.

## 2. The independent variable `x` — three estimators, all in microradians

Pixels are forbidden as an axis (heterogeneous `-r`, and a fisheye plate scale that is not
`fx`). All three estimators are angular.

| estimator | definition | source | role |
|---|---|---|---|
| **E_sfm** | median SfM reprojection residual of the COLMAP model that gray trained on, expressed as an angle by pushing each per-observation pixel residual through the inverse Jacobian of the true COLMAP projection, restricted to the evaluated disk (theta <= 85.5 deg) | `scripts/analysis/sfm_residual.json`, key `residual_urad_theta_le_85_5.median` | **PRIMARY axis.** The only estimator that exists for every point and needs no training. |
| **E_shared** | area-weighted RMS over the valid disk of the cross-scene MEAN of the learned angular residual of the same physical lens, calibrated independently on >= 3 scenes | recomputed here from `calib_consistency.py` helpers; area weight ~ rho on the uniform pixel-radius grid | secondary axis. Isolates the SYSTEMATIC part; exists only for myscenes and FullCircle. |
| **E_learned** | area-weighted RMS over the valid disk of the per-scene learned angular residual (`camera_model_15000.csv`, `delta_theta_rad`, already in radians) | run checkpoints + `plate_scale.json` for the area element | **NEVER an x axis.** Consistency check only: did the optimiser find what E_sfm / E_shared predicted? |

Aggregation rule fixed in advance, per plan section 2.2 ter: the statistic is an
**area-weighted RMS over the valid disk**, never a rim peak, wherever a profile is
available. E_sfm is an exception in kind — it is an empirical distribution over
observations, not a profile — so its primary statistic is the **median** as specified by the
task, with the observation-weighted RMS reported as a secondary column.

E_shared and E_sfm do **not** measure the same thing, and this is stated as a result, not as
a caveat: E_sfm mixes camera-model misfit with feature-detection noise and scene difficulty
(texture, motion blur, baseline); E_shared isolates the part that is systematic across
independent calibrations of the same glass and therefore cannot be feature noise.

## 3. The point list (fixed before fitting)

22 paired scenes, four camera families:

| track | family | n scenes | camera model | eval resolution |
|---|---|---|---|---|
| myscenes rttpf | fisheye_circular | 7 | RAD_TAN_THIN_PRISM_FISHEYE | 1368x912 (`-r 4`) |
| FullCircle refit_rttpf | fisheye_circular (2-lens rig) | 9 | RAD_TAN_THIN_PRISM_FISHEYE | 720x720 (`-r 4`) |
| mip-NeRF 360 | pinhole | 5 (bicycle, stump, garden, bonsai, counter) | PINHOLE | `-r 4` outdoor / `-r 2` indoor |
| workshop_immervision | panomorph | 1 | RAD_TAN_THIN_PRISM_FISHEYE | 1440x1080 (`-r 1`) |

`workshop_fujinon` (fisheye_fullframe) has **no `noncentral` run** and therefore contributes
no y; it is carried in the JSON with `y = null` as a prediction target only.
mip-NeRF 360 `kitchen` and `room` are excluded: `kitchen_off` was still training when this
was written and `room` has no run at all. If they land later they are additions, not
substitutions.

## 4. The transfer function (pre-registered form)

```
gain_dB(x) = 10 * log10( 1 + min(x^2 * G, C) / MSE0 )
```

with `x` in radians (so `G` is dimensionless per rad^2), and starting values taken from the
plan: `G = 0.0070`, `MSE0 = 2e-3`, `C = 0.12`.

* Fitted by least squares on the 22 paired points, in dB, unweighted.
* `G`, `MSE0`, `C` are re-fitted. `MSE0` and `C` are only jointly identifiable with `G`
  through the saturation elbow; if the data do not reach the elbow, `C` is reported as
  unidentified and the fit is reduced to the unsaturated two-parameter form
  `gain_dB = 10*log10(1 + x^2 * G / MSE0)`, i.e. one effective parameter `G/MSE0`.
* Confidence intervals by **paired bootstrap over scenes**, 10 000 resamples, stratified by
  track so a resample cannot delete a whole camera family. Percentile intervals at 95 %.
* **Practical threshold** `x_star`: the x at which the fitted curve crosses the measured
  run-to-run noise floor of **0.068 dB**. Reported with its bootstrap interval.

## 5. Statistics (fixed before fitting)

* Per-track gain significance: **Wilcoxon signed-rank on the paired per-scene deltas**, not
  a t-test, two-sided, exact where n allows. `n`, `p` and a bootstrap CI of the mean delta
  are reported for every track without exception.
* Association between `y` and each x estimator: **Spearman rank correlation**, with `n`, `p`
  and a bootstrap CI. Pearson is reported only as a secondary number.
* A track with `n = 1` (workshop_immervision) yields no test and is labelled as such.
* Declared in advance: with 4 camera families and 3 distinct E_shared values, **no fit on
  E_shared can be more than descriptive**. If E_sfm does not order the gains, that is
  reported as a falsification of the primary hypothesis, not repaired by switching axes.

## 6. Falsification conditions, declared in advance

The primary hypothesis "gain is a monotone increasing function of pre-training angular
calibration error" is **falsified** if any of the following holds:

1. Spearman(y, E_sfm) over the 22 points is not significantly positive at the 5 % level.
2. A track with strictly larger E_sfm than another shows a strictly smaller mean gain, by
   more than the noise floor, on paired data.
3. The fitted `G/MSE0` has a bootstrap CI containing 0.

A prior warning is already on record from the A1 hand-off and is repeated here so it cannot
be presented as a discovery: myscenes sits at 519-743 urad and gains ~+0.33 dB, while
FullCircle refit_rttpf sits at 768-1176 urad — i.e. **worse** — and gains +0.016 dB, and
bicycle at 221 urad — the cleanest model in the whole dataset — gains +0.437 dB.
Condition 2 is therefore expected to trigger. The pre-registered response is to **report the
falsification**, quantify it, and then, clearly labelled EXPLORATORY and post-hoc, test the
single mechanistic explanation already on record (from `calib_consistency.json`): that only
the SYSTEMATIC, cross-fit-shared part of the calibration error is recoverable by a
photometric residual, while the feature-noise part that dominates E_sfm is not. No other
estimator will be introduced.

## 7. Outputs

* `scripts/analysis/dose_response.json`
* `scripts/analysis/dose_response.py` (re-runnable, CPU only)
* `figures/dose_response.svg`
* `figures/estimator_agreement.svg`
