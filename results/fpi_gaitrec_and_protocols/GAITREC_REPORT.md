# GaitRec Stage Report: Restricted-Acquisition Learning Surface (gaitrec_grid_v1, Addendum A3)

Date: 2026-10-10. Second task, separate cohort and label; exploratory development-cohort analysis. **Not external FPI validation.** Official TEST never used.
FPI results (`grid_v1/`, `sensitivity_v1/`) preserved. The FPI result was not used to select this design; a contrary result would have been reported identically.

Frozen before any fit: `ADDENDUM_A3_GAITREC.md`, `restricted_acquisition_gaitrec.py`, `frozen_protocol_A3.json` (hashes inside; the grid stage asserted the script and
addendum were unchanged since the pilot freeze). Verifier `verify_restricted_acquisition_gaitrec.py`: **PASS_LISTED_CHECKS_ONLY**, 0 refits: addendum/script/input hashes; manifest of all 4,500 models; nesting of people and
trials over the whole grid; class counts equal to the A3 table and five classes in every acquired set; test people equal to recorded folds and disjoint; no official TEST
person anywhere; scaler-1 statistics rebuilt from retained-trial medians (error 0.0); probability replay through saved scaler-PCA-scaler-logistic (max error 1.7e-15);
surface, contrasts and CIs recomputed to 1e-12; plain recomputation of the primary point estimate; convergence; fit counts.

## Task and cohort (reused unchanged from `gaitrec_trial_stability_pilot_v1`)
Five-class GaitRec diagnosis (HC/H/K/A/C) from bilateral processed vertical GRF (202 values per trial). Official TRAIN only; self-selected speed, normal shoes, no insole;
first eligible session, five lowest trial IDs. 1,301 people (146/234/368/350/203), 6,505 real trials; these people were explored in earlier development pilots.
Recorded 3 x 5 person folds (1,040-1,041 train / 260-261 test). No demographic input.

## Design executed
Restricted acquisition identical to FPI: nested people (300 ⊂ 500 ⊂ 600 ⊂ 800 ⊂ 1000, class allocation by largest remainder, table in A3) and nested trials (k1 ⊂ k3 ⊂ k5);
median of retained trials -> scaler -> PCA12 -> scaler -> balanced logistic, fixed C=0.1 (selected 15/15 in the prior GaitRec nested tuning), all fitted on acquired people only.
Test: recorded fold people with all five trials. R = 20 draws per fold.

## Execution and control
| Item | Value |
|---|---|
| Mechanical pilot | repeat 0, cells 1000x3 and 600x5, 5 draws: 50 fits + 50 exclusion controls; verifier PASS (replay error 7.8e-16, scaler error 0.0, no TEST person) |
| Grid primary fits | 4,500 (15 folds x 5 n x 3 k x 20 draws) |
| Exclusion-control duplicates | 4,500; coefficients and probabilities bit-identical in 4,500/4,500 |
| Wall time | 493 s fitting (40 ms per fit) + bootstrap |
| Convergence | 0 warnings; 0 reached max_iter; max n_iter 19 of 3000 |
| Class order | [0,1,2,3,4] in all models |

Two verifier corrections during this stage, recorded: (1) the pilot verifier crashed on a cell absent from the pilot grid (KeyError) and was patched to skip absent cells; (2) no data or model file changed.

## Learning surface (five-class person-level balanced accuracy; mean over 3 repeats x 20 draws; 95% class-stratified participant bootstrap over 1,301 IDs, shared weights)
| n | k=1 | k=3 | k=5 |
|---:|---|---|---|
| 300 | 0.404 [0.385, 0.423] | 0.408 [0.389, 0.427] | 0.408 [0.389, 0.428] |
| 500 | 0.414 [0.393, 0.435] | 0.416 [0.395, 0.438] | 0.419 [0.397, 0.440] |
| 600 | 0.418 [0.396, 0.439] | 0.421 [0.399, 0.444] | 0.422 [0.399, 0.444] |
| 800 | 0.422 [0.400, 0.445] | 0.427 [0.404, 0.451] | 0.427 [0.404, 0.451] |
| 1000 | 0.427 [0.404, 0.451] | 0.430 [0.406, 0.455] | 0.431 [0.407, 0.457] |

Chance level for five balanced classes is 0.20; the 1000x5 cell (0.431) is consistent with the earlier five-trial development result of this pipeline (0.43). Log loss decreases
with n (1.430 at 300x1 to 1.385 at 1000x3) and is nearly flat in k.

## Prespecified exact-budget contrasts (paired per repeat and draw; 60 pairs)
| Contrast | Recordings | Metric | Mean difference | 95% CI | Within-repeat SD (mean of 3) | SD of repeat means | Pairs favouring more people |
|---|---|---|---:|---|---:|---:|---:|
| Primary 1000x3 − 600x5 | 3,000 vs 3,000 | BA | **+0.0083** | [+0.0026, +0.0142] | 0.0080 | 0.0007 | 51/60 |
| Primary 1000x3 − 600x5 | 3,000 vs 3,000 | log loss | −0.0156 | [−0.0181, −0.0126] | 0.0059 | 0.0010 | 59/60 |
| Secondary 500x3 − 300x5 | 1,500 vs 1,500 | BA | +0.0078 | [+0.0038, +0.0119] | 0.0096 | 0.0026 | 49/60 |
| Secondary 500x3 − 300x5 | 1,500 vs 1,500 | log loss | −0.0247 | [−0.0278, −0.0218] | 0.0079 | 0.0017 | 60/60 |

(Negative log-loss difference favours the more-people design.) Direction agrees with FPI in both contrasts and both metrics. The BA effect is about 0.8 points, far below the
3-point practical margin used for FPI wording; it is a small, precisely estimated difference in a task where the whole surface spans only 2.7 points.

## Finite differences (neighbouring cells)
- Along n at fixed k: all 12 steps positive; 11 of 12 CIs exclude zero (800x3→1000x3: [−0.0003, +0.0064]).
- Along k at fixed n: 1→3 small positive (0.2 to 0.5 points; CI excludes zero at n=300, 600, 800); 3→5 within ±0.25 points, CI includes zero at every n except 500x3→500x5 (+0.25 [0.06, 0.44]).
- No derivative, ratio, extrapolation or saturation claim.

## Variability
Within-repeat (acquisition-draw) SD of BA falls from about 0.010 at n=300 to 0.0035 at 1000x5; SD of repeat means is 0.003 to 0.007. For the primary difference, within-repeat SD 0.008 and SD of repeat means 0.0007: the paired difference is stable across the three split repeats.

## Cross-task reading (descriptive, not a pooled estimate)
| | FPI (70 men, 3 classes, 47 inputs) | GaitRec (1,301 patients, 5 classes, PCA12 of GRF) |
|---|---|---|
| Exact-budget contrast, more people vs more trials | +3.7 points [1.3, 6.0] (35x3 vs 21x5) | +0.8 points [0.3, 1.4] (1000x3 vs 600x5) |
| Second exact-budget contrast | — | +0.8 points [0.4, 1.2] (500x3 vs 300x5) |
| k 3→5 at the largest n | +2.3 points [0.2, 4.9] (n=56, single composition) | +0.15 points [−0.09, +0.41] (n=1000) |
| Dominant axis | n | n |
The two tasks differ in modality, label, classifier input, cohort size and baseline accuracy; the agreement is in direction only. Effect sizes are not comparable across tasks.

## Boundaries
Conditional on fitted models, recorded splits and a previously explored development cohort; no refitting or draw-design uncertainty; GaitRec classes, sessions and speed categories
are as released; no clinical, deployment, external-validation, exchange-rate, D-study or priority claim. Both cohorts have been examined repeatedly in this project.
