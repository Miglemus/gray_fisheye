# PRE-REGISTRATION — W2, ranking-flip predictor

**Frozen at 2026-08-10T21:44:29Z (UTC).** Written and saved BEFORE any gap-versus-residual
statistic was computed. Nothing below may be revised after the first fit; deviations, if any,
must be listed in a separate "deviations" section of `ranking_flip.json` and labelled as
post-hoc.

Author: W2 agent, Phase 1 (`/workspace/plan_phase1_loi_calibration.md` §3).
Analysis script: `scripts/analysis/ranking_flip.py` → `scripts/analysis/ranking_flip.json`.

## 0. What has been inspected so far (audit trail)

Before freezing this document I looked at, and only at, *structural* information:
which tracks exist in `scripts/analysis/sfm_residual.json`, which method keys exist in the four
metric stores, the shape of the per-method records, and the SfM residual values themselves
(the x variable). **No method-gap (y) value has been read, printed or plotted.** The x values
are legitimately visible because they are the predictor and were produced by task A1 for this
purpose; freezing after seeing x but before seeing y is what makes the inclusion rules below
non-circular.

## 1. Hypothesis under test

The claim to be generalised (plan §3.1) is currently supported by **one** dataset
(`workshop_immervision`), where `corr(median SfM reprojection residual at eval resolution,
3DGUT − gray PSNR gap) = +0.86`. H1 states that this is a property of the capture, not of that
scene:

> **H1 (confirmatory, directional).** Across independent scene×camera tracks spanning several
> datasets and several lens types, the per-track PSNR gap `Δ(3dgrut − gray)` increases
> monotonically with the SfM reprojection residual of that track.
> Predicted sign: **ρ_Spearman > 0**.

All other method pairs are **exploratory and two-sided**.

The mechanistic reading being tested — declared in advance so that a null is interpretable —
is that a large SfM residual marks a capture whose camera model is a poor fit, and that
methods differ in how much of that misfit they can absorb; when the residual is small every
method sees essentially the same geometry and the ranking is decided by the rendering
primitive instead.

## 2. Data

### 2.1 x — the predictor (fixed, computed by task A1, not re-derived here)

* **Primary x**: `residual_urad_theta_le_85_5.median` from
  `scripts/analysis/sfm_residual.json` — the median per-observation SfM reprojection residual,
  restricted to observations with field angle θ ≤ 85.5° (exactly the cut of the shared r=0.95
  evaluation mask), expressed as an **angle in µrad**. Angular units are mandated by plan §2.2:
  the tracks span `-r 4` and native resolution, and a pixel is not the same quantity across
  them.
* **Secondary x**: `residual_eval_px_theta_le_85_5.median` — the same statistic in pixels at
  the evaluation resolution. This is the form in which the original ImmerVision observation was
  made, and is reported so that the two forms can be compared, not because it is preferred.
* Spearman is invariant under any monotone transform of x, so `log10` is applied only for the
  OLS/LODO model of §5 and for the figure's x axis.

### 2.2 y — the outcome

Per-track masked PSNR from the four shared-evaluation stores (plan §1.2), never a
self-reported number:

| store | key path |
|---|---|
| `dataset/fisheye_baselines/masked_metrics.json` | `[scene][method]` |
| `dataset/fisheye_baselines_ocv/masked_metrics.json` | `[track][method]` |
| `dataset/fullcircle_baselines/fullcircle_masked_metrics.json` | `[scene][method]` |
| `dataset/fullcircle_tracks/rttpf_masked_metrics.json` | `[variant][scene][method]` |

`Δ_AB = PSNR_A − PSNR_B` on the same track, both methods from the same shared pass.

* **Primary metric**: PSNR gap, in dB.
* **Secondary metrics** (exploratory, reported with the same statistics): SSIM gap `SSIM_A −
  SSIM_B`, and LPIPS gap `LPIPS_B − LPIPS_A` (sign flipped so that positive always means "A is
  better").

### 2.3 Canonical methods and the alias map (frozen)

Four methods enter the analysis: **gray, 3dgrut, DFGS, SPaGS**. Store-specific keys:

| canonical | fisheye_baselines | fisheye_baselines_ocv | fullcircle_baselines | fullcircle_tracks |
|---|---|---|---|---|
| gray | `gray` | `gray` | `gray_masked` | `gray` |
| 3dgrut | `3dgrut` | `3dgrut` | `3dgrut_masked` | `3dgrut` |
| DFGS | `DirectFisheye-GS` | `DirectFisheye-GS` | `DFGS_masked` | `DFGS` |
| SPaGS | `SPaGS` | `SPaGS` | `SPaGS-fe_masked` | `SPaGS-fe` |

**Excluded method keys, and why — decided now, not after seeing results:**

* `3dgrut-oldbase`, `3dgrut-oldrays`, `3dgrut-nofix(_masked)` — superseded/known-buggy columns.
* `3dgrut-mcmc` — a different densification regime; the project rule is paper-default configs
  only.
* `gray-non-central`, `gray-nc` — the camera-model treatment arm, i.e. W1's y, not a baseline.
* `SPaGS(_masked)` and `SPaGS-panofe(_masked)` **on FullCircle only** — those columns are the
  panorama-trained checkpoint evaluated in the fisheye domain; the fisheye-domain like-for-like
  column is `SPaGS-fe`. On myscenes/others, `SPaGS` *is* the fisheye-trained port and is used.
  This is the single place where the alias map is store-dependent, and it is deliberate.
* `3dgeer` — present on 6 tracks of a single dataset; it cannot contribute to a cross-dataset
  test. Reported descriptively in the ranking tables only.

### 2.4 Track inclusion (frozen)

* Include every track of `sfm_residual.json` whose `stores` field is non-empty (**43 tracks**).
* **Exclude the 9 `fullcircle_relabel` tracks**, for two reasons both established before this
  document: (i) they are *geometrically identical* to `fullcircle_ocv` (residuals agree to
  1e-13 px), so they are duplicate x values, not independent points; (ii) their store carries a
  single method (`SPaGS-panofe`), so they generate no pair at all.
* Remaining: **34 tracks** in 6 dataset tracks — `myscenes_rttpf` (4), `myscenes_ocv` (10),
  `others_rttpf` (2), `others_ocv` (1), `fullcircle_ocv` (9), `fullcircle_refit_rttpf` (9).
* A track missing one method of a pair is dropped **for that pair only**. No track is ever
  dropped after inspecting its y.
* The 19 residual-only tracks (7 mip-NeRF 360, 9 `fullcircle_refit_ocv`, 3 myscenes rttpf
  scenes absent from the store) have no cross-method metrics and are out of scope for W2.

### 2.5 Method pairs (frozen list, frozen directions)

Six ordered pairs, all 4-choose-2 combinations of the canonical methods:

1. `3dgrut − gray`   ← **the confirmatory pair (H1)**
2. `DFGS − gray`
3. `SPaGS − gray`
4. `3dgrut − DFGS`
5. `3dgrut − SPaGS`
6. `DFGS − SPaGS`

## 3. Functional form (frozen)

* **Primary statistic**: Spearman rank correlation ρ between x and Δ_AB across tracks. Rank,
  not Pearson, per plan §3.3, because the gaps are bounded and heavy-tailed at the ImmerVision
  end. Kendall τ_b is reported alongside as a robustness statistic.
* **Model used for prediction (§5 only)**: `Δ_AB = a_AB + b_AB · log10(x_µrad)`, ordinary least
  squares, one intercept and one slope per pair. No interaction terms, no per-dataset
  intercepts, no weighting. Chosen because it is the smallest form that can extrapolate to a
  held-out dataset; a model with per-dataset intercepts cannot, by construction.
* No other functional form will be fitted. If the scatter suggests one, it is reported in words
  as a post-hoc observation and never as a result.

## 4. Inference (frozen)

For every pair and every statistic:

* **n** = number of contributing tracks, and **n_clusters** (see below), both always reported.
* **p**: two-sided permutation test, 20 000 random permutations of x within the analysis set,
  seed 20260810. For H1 the one-sided p in the predicted direction is additionally reported.
  Nominal p from `scipy.stats.spearmanr` is reported for reference only.
* **95 % CI**: percentile bootstrap, 10 000 resamples, seed 20260810, **resampling clusters,
  not tracks**.
* **Cluster** = `(capture family, scene)` with capture family ∈ {myscenes, fullcircle, others}.
  This is what makes the CI honest: `myscenes_rttpf/tunnel`, `myscenes_ocv/tunnel_warmstart`
  and `myscenes_ocv/tunnel_remap` are the same physical capture recalibrated, and
  `fullcircle_ocv/room1` and `fullcircle_refit_rttpf/room1` likewise. Expected ≈ **18 clusters**
  for 34 tracks. The effective sample size of this study is the number of clusters, not the
  number of tracks, and the report must say so.
* **Multiplicity**: Holm–Bonferroni across the 6 pairs, family-wise α = 0.05, applied to the
  primary (PSNR, primary x) tests. Secondary metrics and secondary x are not corrected and are
  labelled exploratory.

### 4.1 Three levels of variation, reported separately (frozen)

The failure mode this whole work package exists to avoid is an intra-dataset pattern sold as a
law, so pooled and within/between variation are never reported as one number:

* **S1 pooled** — ρ over all tracks. Mixes within- and between-dataset variation.
* **S2 within-dataset** — ρ computed separately inside each dataset track with n ≥ 4, plus a
  stratified permutation test in which x is shuffled **only within** a dataset. This is the
  variation the ImmerVision observation actually exhibited.
* **S3 between-dataset** — ρ over per-dataset medians of (x, Δ). Declared **a priori
  underpowered**: at most 6 points, so p can never fall below ≈0.01 and any value will be
  reported as descriptive.

## 5. Leave-one-dataset-out cross-validation (frozen)

* **Folds (primary)**: the 6 dataset tracks of §2.4.
* **Folds (secondary, coarser)**: the 3 capture families {myscenes, fullcircle, others}, which
  is the genuinely out-of-distribution test since the two myscenes tracks share a sensor.
* Fit `Δ = a + b·log10(x)` on the training folds, predict every track of the held-out fold.
* **Scores**, all declared now:
  1. **Sign accuracy** of the predicted gap on held-out tracks, i.e. "does it predict which
     method wins".
  2. **Baseline to beat**: the constant-sign predictor — the majority sign of Δ in the training
     folds. A slope model that does not beat this has predicted nothing.
  3. **MAE (dB)** of the held-out prediction, against the intercept-only model fitted on the
     same training folds. Count of folds where the slope model wins.
  4. **Within-fold Kendall τ** between predicted and actual Δ, on folds with ≥ 4 tracks.
* Aggregated over folds by pooling all held-out tracks.

## 6. Decision rule (frozen, applied verbatim)

For the confirmatory pair, and for each exploratory pair:

* **SUPPORTED** — Holm-adjusted p < 0.05 **and** cluster-bootstrap 95 % CI excludes 0 **and**
  LODO sign accuracy strictly beats the constant-sign baseline **and** n_clusters ≥ 12.
* **SUGGESTIVE (exploratory)** — nominal p < 0.05 but at least one of the other three fails.
* **NO EVIDENCE** — otherwise.

Additional hard rule: **if n_clusters < 12 for a pair, its result is labelled exploratory no
matter what p says.** The project has already nearly published a `corr = +0.43, n = 7,
p ≈ 0.34` as a law; this rule exists specifically so that cannot happen twice. A negative or
underpowered result is an acceptable, publishable outcome of W2 and will be written as such.

## 7. Anti-averaging rules (frozen, plan §3.4)

* No statistic is ever computed on a mean taken across lens types. The r=0.95 disk keeps 95.5 %
  of the frame on `workshop_fujinon`, ~44 % on myscenes, 47.9 % / 65.6 % on ImmerVision
  (rttpf / ocv); an absolute SSIM or PSNR averaged over those is meaningless.
* Method **gaps** are paired within a track, which is what makes them comparable at all across
  regimes; even so, every plot and table is coloured/split by dataset and the pooled statistic
  is always shown next to the within-dataset ones.
* The ranking tables are reported **per lens/track**, never merged.
* `-r 4` tracks and native-resolution tracks are never merged in a pixel-unit statistic; this
  is why the primary x is angular.

## 8. Deliverables (frozen)

1. `scripts/analysis/ranking_flip.py`, `scripts/analysis/ranking_flip.json`.
2. `figures/ranking_flip.svg` — one panel per method pair, Δ vs x (log µrad), colour by
   dataset, marker by lens family, per-panel ρ/n/p annotation.
3. A per-lens ranking table exhibiting the inversions.
4. This file, plus the LODO result, reported whatever it says.
