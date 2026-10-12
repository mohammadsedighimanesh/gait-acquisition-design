# Addendum A5.2: Small-budget sequential policy on StepUP (authorized 2026-10-11)

Motivation (stated honestly): in A5 the policy budgets (600, 300) were large relative to the 120-person pool, so every policy recruited nearly the whole pool and the
trial-favouring regime seen in the L1 surface could not express itself. This addendum pre-registers budgets small relative to the pool. It is a follow-up chosen after seeing
the A5 surfaces and policy results; it is labelled exploratory follow-up, not a confirmatory test, and both labels are reported whatever the result.

## Locked design
- Same data, features, labels (L1 sex, L2 BMI>=25), folds, pipeline (median -> scaler -> PCA12 -> scaler -> balanced logistic, C=0.1), K_max 20, and A4/A4.1/A4.2/A5 policy rules.
- Budgets: **150 (primary)** and 100. Warm start 10 people x 2 (20 recordings); batch **10** recordings (smaller than A5's 20 so that decisions are finer at small budgets).
- Policies: CONV5, FIX1, FIX2, FIX3, FIX4, FIX10, FIX20, RANDOM, MARGIN; 10 seeds; 3 repeats x 5 folds; common random numbers per (fold, seed) as before.
- Primary endpoint: MARGIN − CONV5 BA at B=150, per label; secondary: MARGIN − RANDOM, MARGIN − best fixed (post hoc), share of budget on extra trials, final composition.
- Uncertainty: participant bootstrap with shared weights; within-repeat and between-repeat SD; same as A4.
- Fit accounting recorded in progress.json; exclusion-control duplicate for every final fit.
- Script: `restricted_acquisition_stepup_a52.py` (imports the frozen A5 module without modifying it; overrides only WARM, BATCH and the budget list, recorded in `frozen_protocol_A5_2.json`).
- Verifier: `verify_restricted_acquisition_stepup_a52.py` (A4 policy checks; no refits).

## Not allowed
Any change to policies, warm start, batch or budgets after seeing results; any "first" wording; any claim beyond these two labels and this pool size.

## Amendment A5.2.1 (mechanical, before any result was read; 2026-10-11)
The first run (`policy_a52_L1_v1_FAILED_20261011_001956/`, preserved) aborted because FIX20 at B=100 acquires only 5 people and PCA(12) cannot be fitted. Rule: the final-model
PCA uses min(12, n_people − 1) components; such reduced-PCA fits are counted in progress.json. This affects only the tiny fixed designs (FIX10/FIX20 at these budgets) and is a real
property of those designs, not of the policy. Nothing else changed; the protocol is re-frozen with the new script hash (`frozen_protocol_A5_2_attempt1.json` preserved).

## Amendment A5.2.2 (mechanical, before any result was read; 2026-10-11)
The second attempt (L2 `policy_a52_L2_v1_FAILED_20261011_004833/`, preserved; the parallel L1 run was stopped and its partial folder renamed) aborted because FIX20 at B=100
can acquire 5 people of a single class, for which no classifier can be trained. Rule: a single-class acquired set yields the constant predictor of that class (probability 1),
flagged and counted (`single_class_fits`); the A4.2 missing-class rule then applies. This only affects tiny fixed designs. Protocol re-frozen with the new script hash
(`frozen_protocol_A5_2_attempt2.json` preserved). The A5 verifier's replay was extended for the constant model (verifier only; no A5 data changed).
