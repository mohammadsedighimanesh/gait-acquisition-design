# Third Task Report: StepUP-P150 footsteps (Addendum A5 with amendment A5.1), 2026-10-11

Exploratory; public dataset; 42 of 150 people were used earlier in this project for a different question, 108 were untouched. All stages pre-registered in
`ADDENDUM_A5_STEPUP_THIRD_TASK.md` (hash 1669d991...9985) before any feature was computed; amendment A5.1 (per-fold class allocation) applied before any result was read.
Data: 150 participants, BF/W1, pipeline 2, first 20 eligible right steps per person (min eligible 31, median 43); 300 source files verified against the official FRDR checksum file
(`stepup_third_task_v1/download_log*.json`, `build/build_info.json`). Features: 200 per step (10 time bins x 5x4 blocks), computed from that step only; replayed by the verifier (error 0.0).
Labels: L1 = sex (76 F / 74 M), L2 = BMI >= 25 (74/150). Pipeline: median of retained steps -> scaler -> PCA12 -> scaler -> balanced logistic, C = 0.1 (inner tuning selected 0.1 in 597/600 cells; results unchanged).
All verifiers PASS_LISTED_CHECKS_ONLY with 0 refits (build; pilot; surface L1, L2; policy L1, L2; sensitivity L1). Fits: surfaces 6,000 + 6,000 controls per label; policies 166,050 internal + 2,700 final + 2,700 controls per label; sensitivity 5,400 inner + 600 refits. No convergence warnings.

## Learning surfaces (person-level BA; 95% participant bootstrap; 3 repeats x 20 draws)
L1 sex (weak signal after size/amplitude normalisation):
| n \ k | 1 | 3 | 5 | 10 | 20 |
|---:|---|---|---|---|---|
| 30 | 0.530 | 0.543 | 0.553 | 0.556 | 0.557 |
| 60 | 0.526 | 0.549 | 0.550 | 0.555 | 0.561 |
| 90 | 0.532 | 0.553 | 0.562 | 0.567 | 0.572 |
| 120 | 0.535 | 0.553 | 0.562 | 0.580 | 0.606 |

L2 BMI >= 25:
| n \ k | 1 | 3 | 5 | 10 | 20 |
|---:|---|---|---|---|---|
| 30 | 0.598 | 0.619 | 0.618 | 0.616 | 0.618 |
| 60 | 0.633 | 0.653 | 0.660 | 0.653 | 0.654 |
| 90 | 0.654 | 0.665 | 0.671 | 0.669 | 0.664 |
| 120 | 0.667 | 0.674 | 0.679 | 0.671 | 0.658 |

(120x20 is the whole training set, one composition.) CIs are wide (about ±0.05) because test sets have 30 people.

## Pre-registered equal-budget contrasts
| Label | Contrast | Recordings | BA difference | 95% CI | Pairs > 0 | Log-loss difference |
|---|---|---|---:|---|---:|---|
| L1 | 120x5 − 60x10 (primary) | 600 | +0.007 | [−0.013, +0.027] | 35/60 | −0.036 [−0.069, −0.012] |
| L1 | 60x5 − 30x10 | 300 | −0.006 | [−0.024, +0.011] | 24/60 | −0.057 [−0.106, −0.024] |
| L1 | 60x10 − 30x20 | 600 | −0.002 | [−0.019, +0.014] | 31/60 | −0.046 [−0.082, −0.014] |
| L2 | 120x5 − 60x10 (primary) | 600 | **+0.026** | [+0.009, +0.043] | 46/60 | −0.057 [−0.089, −0.030] |
| L2 | 60x5 − 30x10 | 300 | +0.044 | [+0.026, +0.061] | 50/60 | −0.085 [−0.122, −0.052] |
| L2 | 60x10 − 30x20 | 600 | +0.035 | [+0.014, +0.056] | 47/60 | −0.082 [−0.125, −0.042] |

Finite differences along k: in L1 at n=120 the steps 5→10 (+1.8 [0.3, 3.3]) and 10→20 (+2.6 [0.0, 5.2]) are positive; in L2 the steps beyond k=3 are null or negative (120: 10→20 −1.3 [−3.4, +0.7]).

**Reading.** The third dataset shows two regimes inside one cohort. For BMI (L2) people dominate exactly as in FPI and GaitRec. For sex (L1) extra footsteps keep helping up to 20
and the equal-budget contrasts are indeterminate in BA (people and trials roughly exchangeable at these budgets), while log loss still favours more people. Seibel et al. noted that
tasks whose informative pattern is present in few segments may favour more segments; L1 is such a low-signal task. This is the regime the first two tasks could not reach.

## Sequential policy (A4 rules, K_max 20, 10 seeds, budgets 600 and 300)
**B = 600 is pool-saturated and uninformative:** with 120 recruitable people, CONV5, FIX1–FIX4, RANDOM and MARGIN all end as the whole pool with 5 steps each, so they coincide
by construction (differences within ±0.001). Only B = 300 discriminates policies.

B = 300, BA (mean over 3 repeats x 10 seeds; 95% participant bootstrap):
| Policy | L1 sex | L2 BMI |
|---|---:|---:|
| CONV5 (60 people x 5) | 0.545 | 0.654 |
| FIX1/FIX2 (120 people x 2.5) | 0.551 | 0.680 |
| FIX3 | 0.547 | 0.676 |
| FIX4 | 0.547 | 0.671 |
| FIX10 (30 x 10) | 0.554 | 0.616 |
| FIX20 (15 x 20) | 0.553 | 0.558 |
| RANDOM | 0.546 | 0.681 |
| MARGIN | 0.552 | 0.681 |

| Label | MARGIN − CONV5 | MARGIN − RANDOM | MARGIN − best fixed (post hoc) |
|---|---|---|---|
| L1 | +0.008 [−0.018, +0.033], 17/30 | +0.007 [−0.003, +0.016], 18/30 | −0.001 [−0.033, +0.029] vs FIX10 |
| L2 | **+0.027** [+0.004, +0.050], 24/30 | +0.001 [−0.005, +0.007] | +0.002 [−0.001, +0.005] vs FIX1 |

MARGIN's budget share spent on extra trials: L1 44% of batches, L2 39%; final composition about 119 people x 2.5 steps in both labels (the 120-person pool is nearly exhausted even at B=300).
Undefined internal estimates (A4.1/A5 PCA guard, default "people"): 7.1% of decisions at B=300, 3.4% at B=600.

**Reading.** In the people-dominated regime (L2) the policy again matches the post-hoc best fixed design and beats the convention. In the trial-relevant regime (L1) every design is
within about one point and the policy neither wins nor loses; it did shift a larger share of its budget to trials (44% vs 39%) but still recruited almost the whole pool. The
algorithmic claim is therefore unchanged: **never worse than the convention, equal to the best fixed rule where one axis dominates, and not demonstrably better than fixed rules where
the axes are exchangeable.** The small pool (120 training people) and 30-person test sets limit what this task can show; a policy advantage in a trial-favouring regime would need a
budget small relative to the pool (e.g. B ≈ 100–150 with K_max 20), which was not pre-registered here and is proposed as A5.2, not run.

## Boundaries
Exploratory; conditional on fitted models, recorded splits and this pipeline; 30-person test folds; sex and BMI labels are used only as between-person labels with many trials and carry no
clinical meaning here; pipeline-2 normalisation removes most foot-size information; no claim of priority, cost, or generality. The 42 previously examined people are flagged in `eligibility_audit.csv` by source.
