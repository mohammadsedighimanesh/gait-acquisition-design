# Sequential Acquisition Policy Report: "next person or next trial?" (policy_grid_v1, Addendum A4 with amendments A4.1 and A4.2)

Date: 2026-10-10. Exploratory simulation on real recordings of two previously examined cohorts. No synthetic data. Official GaitRec TEST never used.
Frozen: `ADDENDUM_A4_SEQUENTIAL_POLICY.md` (f29b8d14...776a), amendments in `ADDENDUM_A4_1_AMENDMENT.md` (a6b8d66f...7a78), script `restricted_acquisition_policy.py`
(8784077...5ffc), `frozen_protocol_A4.json`. Verifier `verify_restricted_acquisition_policy.py` (508b0b49...c489): **PASS_LISTED_CHECKS_ONLY**, 0 refits.
Preserved: three pilot runs (`policy_pilot_v1_before_A4.1/`, `_before_A4.1b/`, `_before_A4.2/`, `policy_pilot_v1/`), the failed first grid attempt
(`policy_grid_v1_FAILED_20261010_173839/` with FAILURE.json) and its log. No hyperparameter was tuned anywhere.

## What was run
Policies CONV5 (lab convention, 5 trials per recruited person), FIX4/3/2/1, RANDOM (coin per batch) and MARGIN (proposed) on paired recruitment streams
(same person order and trial order per fold and seed). MARGIN estimates, from its acquired data only, the marginal gain per recording of "more people" versus "more trials"
by backward differences in an internal 3-fold person CV (3 replicates, one drawn at random to decide), then spends the next batch accordingly.
Final model per policy = task pipeline (A1 for FPI, A3 for GaitRec; fixed C=0.1) on the acquired set; scored on the recorded fold's test people (five trials, median).

| Task | Repeats x folds | Seeds | Budgets | Warm start / batch |
|---|---|---:|---|---|
| FPI (56 recruitable per fold) | 3 x 5 | 20 | 105 (primary), 70 | 7 people x 2 / 7 |
| GaitRec (1,040-1,041 per fold) | 3 x 5 | 5 | 3,000 (primary), 1,500 | 100 x 2 / 100 |

## Execution and control
| Item | Value |
|---|---|
| Internal (policy) fits | 243,161 |
| Final fits | 5,250; exclusion-control duplicates 5,250 (bit-identical 5,250/5,250) |
| Wall time | 4,129 s |
| Convergence | 0 warnings; max n_iter 35 of 3000 |
| Acquired sets lacking a class (A4.2 rule) | 5 final models, all FPI B=70: CONV5 4, FIX4 1 (each with its duplicate; 10 fits counted) |
| MARGIN decisions with undefined internal estimate (A4.1 rule, default "people") | FPI 3.9% (B=70), 2.3% (B=105); GaitRec 0% |
| Verifier | hashes; every recording a distinct real (person, trial) from the fold pool, totals equal B, <=5 per person, no test person; MARGIN decisions replayed from logged estimates (294 near-ties at CSV precision accepted, 3 of which were not resolvable from the CSV); 5,250 models replayed (error 1.1e-15); scaler rebuilt from acquired trials only (error 0.0); summaries and CIs recomputed to 1e-12 |

Two verifier corrections recorded: budget-order mismatch in the recomputation (ordering only) and CSV-precision ties in decision replay. No data or model file changed.

## Primary result: MARGIN minus the convention (paired per repeat, fold, seed; 95% class-stratified participant bootstrap, shared weights)
| Task | Budget | MARGIN − CONV5, BA | 95% CI | Pairs favouring MARGIN | Log loss difference (negative favours MARGIN) |
|---|---:|---:|---|---:|---|
| FPI | 105 | **+0.094** | [+0.046, +0.140] | 59/60 | −0.165 [−0.213, −0.118] |
| FPI | 70 | +0.116 | [+0.066, +0.164] | 56/60 | −0.349 [−0.425, −0.274] |
| GaitRec | 3,000 | **+0.010** | [+0.003, +0.018] | 14/15 | −0.015 [−0.019, −0.011] |
| GaitRec | 1,500 | +0.023 | [+0.013, +0.032] | 15/15 | −0.044 [−0.051, −0.036] |

H1 (MARGIN > CONV5) is supported at both budgets in both tasks. FPI differences exceed the 3-point wording margin; GaitRec differences exceed the 1-point margin at 1,500 and in point estimate at 3,000 (lower CI 0.3 points).

## Secondary comparisons
| Task | Budget | MARGIN − RANDOM | MARGIN − best fixed design (post hoc) |
|---|---:|---|---|
| FPI | 105 | +0.016 [+0.008, +0.025], 36/60 | −0.004 [−0.011, +0.003] vs FIX1 |
| FPI | 70 | +0.017 [−0.003, +0.037], 35/60 | −0.023 [−0.041, −0.004] vs FIX1 |
| GaitRec | 3,000 | −0.001 [−0.002, +0.001], 4/15 | −0.001 [−0.002, +0.000] vs FIX2 (pool-saturated, see below) |
| GaitRec | 1,500 | +0.002 [−0.002, +0.006], 9/15 | −0.005 [−0.008, −0.001] vs FIX1 |

## All policies (BA, mean over repeats x seeds)
| Policy | FPI 105 | FPI 70 | GaitRec 3,000 | GaitRec 1,500 |
|---|---:|---:|---:|---:|
| CONV5 | 0.606 | 0.557 | 0.421 | 0.404 |
| FIX4 | 0.640 | 0.592 | 0.425 | 0.414 |
| FIX3 | 0.668 | 0.629 | 0.428 | 0.416 |
| FIX2 | 0.694 | 0.666 | 0.431 | 0.425 |
| FIX1 | 0.703 | 0.696 | 0.431 | 0.432 |
| RANDOM | 0.684 | 0.657 | 0.432 | 0.424 |
| MARGIN | 0.700 | 0.673 | 0.431 | 0.427 |

Acquired composition (mean people / trials per person): FPI 105: CONV5 21/5.0, MARGIN 54.6/1.94, RANDOM 50.3/2.16; FPI 70: MARGIN 44.3/1.67; GaitRec 1,500: CONV5 300/5.0, MARGIN 890/1.74, RANDOM 758/2.10.
MARGIN chose "people" in 66% (FPI) and 63% (GaitRec) of its batches.

## Honest reading
1. **Against the convention, the policy wins clearly in both tasks and both budgets.** This is the pre-registered primary result and it is not an artefact of the FPI finding: the
   policy learns the allocation online from its own acquired data, and the GaitRec result comes from a different cohort, label and pipeline.
2. **Against simple alternatives the advantage is small or absent.** MARGIN beats RANDOM by 1.6 points on FPI at 105 (CI excludes zero) but not on GaitRec; it is within
   half a point of the best fixed design chosen after the fact, and slightly worse than FIX1 at the lowest budgets. In these two tasks the best fixed rule is "one trial per person,
   recruit as many people as possible", and the policy approximately rediscovers it. The algorithm's value is that it does so without prior knowledge of the task; it does not
   outperform the oracle rule.
3. **GaitRec at B=3,000 is pool-saturated:** FIX1, FIX2 and MARGIN all end with every recruitable person (1,041) plus round-robin extra trials, so they coincide by construction.
   The informative GaitRec budget is 1,500. FPI pools are also small (56); FIX1 at B=105 is "everyone plus extra trials". Larger pools would be needed to see whether a trials-favouring
   regime ever appears; none appeared here.
4. Small-budget class-blind recruitment can produce training sets lacking a class (5 of 60 CONV5 runs at FPI B=70); the convention's poor score at that budget partly reflects this real risk.

## Boundaries
Conditional on fitted models, recorded splits, two previously explored cohorts and fixed-C pipelines; bootstrap over unique test IDs with shared weights; seed-level and repeat-level
SDs reported in the summary files. Recording count is not time or money; recruitment order is random and class-blind by design; no between-day data; no claim of clinical benefit,
external validation, optimality, regret bounds or priority. The policy's internal estimator is one simple backward-difference rule; no alternatives were tried, and none should be
tried on these data without a new pre-registered addendum.
