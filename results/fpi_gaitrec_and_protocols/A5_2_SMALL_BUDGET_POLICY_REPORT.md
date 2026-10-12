# A5.2 Report: Small-Budget Sequential Policy on StepUP (2026-10-11)

Exploratory follow-up chosen after the A5 results (declared as such in the addendum). Budgets 150 (primary) and 100; warm start 10 x 2; batch 10; 10 seeds; 3 x 5 folds; K_max 20.
Two mechanical amendments before any result was read (A5.2.1 adaptive PCA for tiny sets; A5.2.2 constant predictor for single-class sets), both triggered only by the tiny fixed
designs FIX10/FIX20. Failed/stopped attempts preserved. Verifiers for both labels: PASS_LISTED_CHECKS_ONLY (2,700 models replayed each, replay error < 1e-15, scaler rebuilt from
acquired trials error 0.0). Fits per label: 74,250 internal + 2,700 final + 2,700 exclusion controls; 0 convergence warnings; 900 reduced-PCA fits (FIX10/FIX20); 22–28 single-class fits (FIX20 at B=100).

## Results (BA, mean over 3 repeats x 10 seeds; 95% participant bootstrap; paired differences)
| Label | Budget | CONV5 | FIX1 | FIX2 | FIX3 | FIX10 | FIX20 | RANDOM | MARGIN | MARGIN − CONV5 | MARGIN − RANDOM | MARGIN − best fixed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|
| L1 sex | 150 | 0.549 | 0.543 | 0.553 | 0.549 | **0.556** | 0.528 | 0.543 | 0.535 | −0.014 [−0.039, +0.012] | −0.008 [−0.022, +0.006] | −0.021 [−0.053, +0.011] vs FIX10 |
| L1 sex | 100 | 0.535 | 0.530 | 0.546 | **0.546** | 0.531 | 0.515 | 0.542 | 0.538 | +0.003 [−0.020, +0.026] | −0.004 [−0.019, +0.012] | −0.008 [−0.028, +0.012] vs FIX3 |
| L2 BMI | 150 | 0.628 | **0.670** | 0.660 | 0.658 | 0.574 | 0.544 | 0.657 | 0.661 | **+0.034** [+0.005, +0.063] | +0.004 [−0.009, +0.017] | −0.009 [−0.020, +0.003] vs FIX1 |
| L2 BMI | 100 | 0.598 | **0.662** | 0.638 | 0.638 | 0.553 | 0.540 | 0.624 | 0.642 | **+0.044** [+0.019, +0.069] | +0.018 [+0.005, +0.031] | −0.020 [−0.034, −0.006] vs FIX1 |

Log loss: MARGIN better than CONV5 in all four cells; worse than the best fixed design in L1 (B=150: +0.20 vs FIX10) and marginally better in L2.
Composition: MARGIN at B=150 ends with about 85–89 people x 1.8 steps (RANDOM 76 x 2.1; CONV5 30 x 5; FIX10 15 x 10). MARGIN spent 42% (L1) / 39% (L2) of batches on trials.
Undefined internal estimates (default "people"): 25% of decisions at B=100 and 15% at B=150 (the internal 3-fold CV with PCA needs > 12 people per inner-train fold).

## Honest reading
1. **Trial-relevant regime (L1):** the small budgets finally make a trial-heavy fixed design the best (FIX10 at B=150, FIX2/FIX3 at B=100), but **MARGIN does not find it**: it is
   indistinguishable from the convention and from RANDOM, and about 2 points below the best fixed design (CI includes zero). Its internal backward-difference estimator is too noisy
   at 20–150 recordings with a 0.55-accuracy signal, and a quarter of its early decisions are undefined.
2. **People-dominated regime (L2):** MARGIN beats the convention clearly (+3.4 / +4.4 points) and RANDOM at B=100, but is **below** the oracle fixed rule "one trial per person"
   by 0.9–2.0 points (CI excludes zero at B=100). At these budgets the exploration cost of the policy is visible.
3. Combined with A4 and A5: the policy is never worse than the five-trial convention except in L1 where everything ties; it matches the best fixed rule only when the budget is large
   relative to the pool; at small budgets it pays an exploration cost in the easy regime and fails to exploit the hard regime. **The algorithmic claim must be stated as this limitation,
   not as a demonstrated advantage over simple fixed rules.**

## Boundaries
Exploratory follow-up; 30-person test folds; pool 120; results conditional on this estimator (one backward-difference rule, no tuning), this pipeline and these labels. No claim of priority
or generality. A different internal estimator or a smaller warm start would be a new pre-registered addendum, not a revision of this one.
