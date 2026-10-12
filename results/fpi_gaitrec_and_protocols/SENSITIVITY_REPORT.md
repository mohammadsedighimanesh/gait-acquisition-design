# Sensitivity Report: Inner-Tuned C on the Primary Cells (sensitivity_v1, Addendum A2)

Date: 2026-10-10. Exploratory single-cohort study. `grid_v1/` and all earlier files preserved. GaitRec NOT run.
Frozen: `ADDENDUM_A2_SENSITIVITY_TUNING.md` (c9a788f0...60119), `restricted_acquisition_sensitivity.py` (0b71b1ca...a1891), `frozen_protocol_A2.json`.
Verifier `verify_restricted_acquisition_sensitivity.py` (91e708e4...c494aa): PASS_LISTED_CHECKS_ONLY, 0 refits.

## Locked design executed
Same memberships, retained trials, outer folds and 20 draws as grid_v1 for cells 35x3 and 21x5. Three class-stratified inner person folds inside each acquired set;
scaler fitted on the inner-training people of each inner fold only; C in {0.1, 1, 10}; selection = argmin mean inner-validation log loss, ties to smaller C (locked before execution).
Refit on all acquired people with the selected C; test on the recorded 14 people with five-trial median.

## Execution and control
| Item | Value |
|---|---|
| Inner fits | 5,400 (15 folds x 2 cells x 20 draws x 3 C x 3 inner folds) |
| Refits | 600 |
| Exclusion-control duplicate refits | 600 (bit-identical coefficients and probabilities in 600/600; counted separately) |
| Wall time | 50 s |
| Convergence | 0 warnings; 0 fits reached max_iter; max n_iter 57 of 3000 |
| Verifier | hashes; grid inputs unchanged; manifest of all 600 models; inner folds exactly partition the acquired people and are disjoint from test; selection rule replayed from saved inner losses for all 600 refits; probability replay error 1.1e-16; scaler rebuilt from retained-trial medians, error 0.0; endpoints and CIs recomputed to 1e-12 |

One verifier defect, recorded: the first verifier run aborted because it accessed `row.repeat` on a pandas Series (a method, not the column). Corrected to `row['repeat']`; no data or model file changed.

## Selected C distribution (100 selections per cell per repeat)
| Cell | Repeat | C=0.1 | C=1 | C=10 |
|---|---|---:|---:|---:|
| 35x3 | 0 / 1 / 2 | 80 / 85 / 84 | 19 / 15 / 15 | 1 / 0 / 1 |
| 35x3 | all | 249 (83%) | 49 (16%) | 2 (1%) |
| 21x5 | 0 / 1 / 2 | 81 / 78 / 78 | 16 / 19 / 17 | 3 / 3 / 5 |
| 21x5 | all | 237 (79%) | 52 (17%) | 11 (4%) |

The fixed value C=0.1 used in grid_v1 was also the inner-selected value in about four fifths of the acquired sets.

## Primary contrast, tuned versus fixed (paired per repeat and draw; 60 pairs; 95% shared-weight participant bootstrap)
| Regime | Metric | 35x3 minus 21x5 | 95% CI | Within-repeat SD (mean of 3) | SD of repeat means | Pairs favouring 35x3 |
|---|---|---:|---|---:|---:|---:|
| Tuned C | balanced accuracy | **+0.040** | [+0.016, +0.062] | 0.055 | 0.007 | 46/60 |
| Fixed C=0.1 (grid_v1) | balanced accuracy | +0.037 | [+0.013, +0.060] | 0.052 | 0.006 | 45/60 |
| Tuned C | log loss | −0.094 | [−0.127, −0.060] | 0.098 | 0.017 | 52/60 |
| Fixed C=0.1 (grid_v1) | log loss | −0.087 | [−0.112, −0.061] | 0.046 | 0.024 | 58/60 |

(Negative log-loss difference favours 35x3.) The direction and size of the primary contrast are unchanged by inner tuning.

## Tuned minus fixed, per cell (paired on the same draw)
| Cell | Metric | Tuned mean [95% CI] | Fixed mean | Tuned − fixed | 95% CI |
|---|---|---|---:|---:|---|
| 35x3 | BA | 0.659 [0.581, 0.734] | 0.659 | −0.000 | [−0.003, +0.003] |
| 21x5 | BA | 0.619 [0.556, 0.682] | 0.622 | −0.003 | [−0.006, +0.000] |
| 35x3 | log loss | 0.775 [0.655, 0.899] | 0.739 | +0.036 | [+0.015, +0.060] |
| 21x5 | log loss | 0.869 [0.759, 0.981] | 0.826 | +0.043 | [+0.019, +0.070] |

Inner tuning did not change balanced accuracy and slightly worsened held-out log loss in both cells, i.e. the quality of the predicted probabilities under a proper
scoring rule was worse; a larger C was selected in about one fifth of the acquired sets. A worse log loss alone does not establish worse calibration, which was not
measured here. This is a within-pipeline sensitivity statement, not evidence that fixed C is generally preferable.

## Boundaries
Conditional on fitted models, recorded splits and this repeatedly examined cohort of 70 healthy men; no refitting, draw-design or independent-cohort uncertainty;
no exchange rate, D-study, clinical, external-validation or priority claim. Stopped here; GaitRec stage awaits a decision.
