# Addendum A1: Full FPI Primary Grid (authorized 2026-10-10)

Applies to `PROTOCOL_v2_FROZEN.md` (SHA256 d98ee18951e3d0fe3e3f62f09d459211890570b31d757e9283035826b6f5a623). The frozen protocol, `frozen_protocol.json`
and `pilot_v1/` are preserved unchanged. Authorized here: the primary grid only. Sensitivity inner tuning and the GaitRec stage remain NOT authorized.
Pilot numbers were not used to change n, k, R, C, margin or hypothesis direction.

## Design executed
- Recorded outer splits: 3 repeats x 5 folds (56 train / 14 test people). Grid n in {21,28,35,42,49,56} x k in {1,3,5}, R = 20 draws per fold.
- Nested draws: per draw and class, one random permutation of that class's training people; the n-level set takes the first allocation
  (9/6/6, 12/8/8, 15/10/10, 18/12/12, 21/14/14, 24/16/16), so 21 ⊂ 28 ⊂ 35 ⊂ 42 ⊂ 49 ⊂ 56. Per person one random permutation of trials 1..5; k takes the first k, so k1 ⊂ k3 ⊂ k5.
- Fixed C = 0.1, balanced multinomial logistic regression, `max_iter=3000`, `tol=1e-7`; scaler fitted on acquired person vectors only; test = 14 people, five-trial median.

## 1. Uncertainty (exact computation)
- Unit of inference: the 70 unique participant IDs. Each person is a test person exactly once per repeat and once per (repeat, draw, cell).
- Point estimate per cell: for each (repeat, draw), pool the 70 held-out predictions across the five folds and compute person-level balanced accuracy
  BA = mean over classes of (correct persons in class / persons in class). Cell BA = mean over 3 repeats x 20 draws = 60 values. Log loss analogous (mean over persons).
- Primary contrast: per (repeat, draw), D = BA(35x3) - BA(21x5) using the same draw's nested selections and the same test people; point estimate = mean of 60 D values.
- Bootstrap: 5,000 class-stratified resamples of participant IDs (30 Neutral, 20 Supinated, 20 Pronated, with replacement). Each resample yields one multiplicity
  weight per person; **the same weights are applied to every fold, repeat, draw, cell and both designs.** Weighted BA uses weighted class-wise correctness;
  weighted log loss uses weighted person means. Repeated prediction rows for a person are never independent observations.
- Interval: percentile 2.5/97.5 of the bootstrap distribution of the cell mean and of the mean paired contrast. Conditional on the fitted models, this cohort,
  and the recorded splits; does not include refitting, draw-selection or prior-exploration uncertainty.
- Between-draw variability: SD over 20 draws of the repeat-pooled BA, reported per cell and per repeat; between-repeat SD of draw-averaged BA reported separately.

## 2. Model-run control
- Every model file (`coef`, `intercept`, `scale_mean`, `scale_scale`, `classes`, `n_iter`) is saved and hashed in `output_manifest.json`.
- Class order asserted as [0,1,2] for every model; probability columns p0,p1,p2 follow `model.classes_`.
- Script SHA256 and this addendum's SHA256 recorded in `frozen_protocol_A1.json` before execution.
- Convergence: `ConvergenceWarning` captured per fit; `n_iter_` recorded; any fit with `n_iter_ >= max_iter` flagged in `convergence_audit.csv`.

## 3. Full verifier (no refits)
Checks: protocol/addendum/script/input hashes; manifest hashes of all outputs and models; nested people and trials over the whole grid;
class counts and three-class presence in every acquired set; test people identical to recorded folds and disjoint from training;
exclusion of unacquired trials (scaler mean must equal the mean of retained-trial medians computed from memberships alone; recorded exclusion-proof flags all true);
probability replay from saved models; recomputation of BA, log loss, primary contrast and bootstrap CI from predictions; class order; fit counts.
PASS is claimed only for these listed checks.

## 4. Fit accounting
| Run | Primary fits | Exclusion-control duplicate fits |
|---|---:|---:|
| Pilot (done) | 50 | 50 |
| Full primary grid (this addendum) | 15 x 18 x 20 = 5,400 | 5,400 |
Sensitivity and GaitRec: 0 (not authorized).

## 5. Evidence preservation
Outputs go to `grid_v1/`. From now on a failed run is not deleted: its directory is renamed `grid_v1_FAILED_<timestamp>/` with `FAILURE.json` giving status and cause.
