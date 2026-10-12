# Full FPI Primary Grid Report: Restricted-Acquisition Learning Surface (grid_v1)

Date: 2026-10-10. Exploratory single-cohort study on 70 previously examined healthy men. Authorized scope executed: primary grid only.
Sensitivity inner tuning and GaitRec stage: NOT run. Pilot numbers were not used to change any design element.

Frozen inputs: `PROTOCOL_v2_FROZEN.md` (d98ee189...5a623), `ADDENDUM_A1_FULL_GRID.md` (1e43e2c0...eebae), `restricted_acquisition_grid.py` (6f25db27...82475),
`frozen_protocol_A1.json`. Verifier `verify_restricted_acquisition_grid.py` (223ec2df...9d6c6). Interpreter: existing `economy_ml_p02_llm`; no environment change; pilot and earlier files preserved.

## Execution and control
| Item | Value |
|---|---|
| Primary fits | 5,400 (15 folds x 6 n x 3 k x 20 draws) |
| Exclusion-control duplicate fits | 5,400 (counted separately; unacquired entries replaced by noise; coefficients and probabilities bit-identical in 5,400/5,400) |
| Wall time | 75.8 s fitting (6.1 ms per fit) + about 5 min bootstrap |
| Convergence | 0 ConvergenceWarnings; n_iter min 19, median 25, max 30 of max_iter 3000; 0 fits reached max_iter |
| Class order | [0,1,2] in all 5,400 models; p0,p1,p2 follow `classes_` |
| Manifest | all outputs and all 5,400 model files hashed in `output_manifest.json` |

## Independent verifier: PASS_LISTED_CHECKS_ONLY (0 refits)
Hashes of protocol, addendum, script and inputs; manifest; nesting of people (21⊂28⊂...⊂56) and trials (k1⊂k3⊂k5, identical trials for a person across n within a draw)
over the whole grid; class counts and three-class presence in all 5,400 acquired sets; test people equal to recorded folds and disjoint from training;
exclusion of unacquired trials verified from memberships alone (scaler mean and scale rebuilt from retained-trial medians: max error 0.0);
probability replay of 5,400 models (max error 1.1e-16); recomputation of surface, contrasts and bootstrap CI (match to 1e-12); plain recomputation of the primary point estimate; fit counts.

One verifier correction during this stage, recorded: the first verifier run reported FAIL on the scaler check with max error 1.0. Cause: scikit-learn stores
`scale_=1.0` for zero-variance columns (one-hot toe-shape columns constant in 6 small acquired sets), while the verifier compared against SD 0. The verifier was
corrected to the documented convention and rerun; no model, prediction or metric file was changed. The initial FAIL and its cause are kept here.

## Learning surface (person-level balanced accuracy; mean over 3 repeats x 20 draws; 95% class-stratified participant bootstrap, shared weights across all cells)
| n | k=1 | k=3 | k=5 |
|---:|---|---|---|
| 21 | 0.610 [0.547, 0.670] | 0.621 [0.555, 0.684] | 0.622 [0.556, 0.684] |
| 28 | 0.636 [0.564, 0.703] | 0.648 [0.574, 0.717] | 0.648 [0.574, 0.718] |
| 35 | 0.654 [0.577, 0.727] | 0.659 [0.578, 0.733] | 0.664 [0.581, 0.739] |
| 42 | 0.671 [0.587, 0.749] | 0.679 [0.592, 0.758] | 0.678 [0.591, 0.758] |
| 49 | 0.682 [0.593, 0.764] | 0.692 [0.601, 0.776] | 0.696 [0.605, 0.780] |
| 56 | 0.692 [0.596, 0.780] | 0.699 [0.602, 0.791] | 0.722 [0.624, 0.815] |

Sanity anchor: 56x5 (0.7222) reproduces the earlier median-five result of `internal_trial_budget_study_v1` on the same splits, as expected since the acquired set is then the full training set.
Log loss decreases monotonically with n and with k (table in `learning_surface.csv`).

## Prespecified contrasts (paired per repeat and draw; 60 pairs)
| Contrast | Recordings | Mean BA difference | 95% CI | Between-draw SD of difference | Pairs positive |
|---|---|---:|---|---:|---:|
| Primary: 35x3 minus 21x5 | 105 vs 105 | **+0.037** | [+0.013, +0.061] | 0.052 | 45/60 |
| Secondary: 56x3 minus 35x5 | 168 vs 175 | +0.035 | [+0.004, +0.067] | 0.043 | 47/60 |

The primary hypothesis direction (more people, fewer trials, equal recordings) is supported in this cohort under this pipeline. The point estimate exceeds the
3-point practical margin; the lower CI bound (1.3 points) does not, so the wording is "a positive difference is supported; a difference larger than three points
is not resolved".

## Finite differences (neighbouring cells; `finite_differences.csv`)
- Along n at fixed k: all 15 steps positive in point estimate; 9 of 15 have CI excluding zero (mostly the steps from 21 to 42 people).
- Along k at fixed n: steps 1→3 are small and positive (0.4 to 1.2 points; CI excludes zero only at n=28); steps 3→5 are near zero at n≤49
  (−0.1 to +0.5 points, CIs include zero) and +2.3 points [0.2, 4.9] at n=56.
- No derivative, ratio, extrapolation or saturation claim is made.

## Variability
- Between-draw SD of the repeat-pooled BA: 0.016 (56x5, single composition) to 0.048 (21x3); it decreases with n.
- Between-repeat SD of draw means: 0.008 to 0.019 across cells.
- Both are acquisition-selection variability on the same 70 people, not population variability.

## Interpretation boundaries (apply to every number above)
Intervals are conditional on the fitted models, the recorded splits and this repeatedly examined cohort; they exclude refitting, draw-selection design, prior
exploratory choices and independent-cohort uncertainty. Recording count is not time or money. Healthy men, extreme posture classes, single session, five accepted
trials of unverified chronology. Results describe this fixed-C logistic pipeline; no participant-to-trial exchange rate, D-study prediction, clinical benefit, external
validation, or priority claim. The 56x5 cell has only one composition, so its draw SD reflects nothing about acquisition choice.

## Stopped here
Awaiting the decision on (a) sensitivity inner tuning on the two primary cells and (b) the GaitRec stage. Neither was run.
