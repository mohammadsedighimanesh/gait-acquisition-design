# Feasibility Pilot Report: Restricted-Acquisition Learning Surface

Date: 2026-10-10. Scope executed: mechanical pilot only, exactly as authorized. Full grid NOT run.
Protocol frozen: `PROTOCOL_v2_FROZEN.md` (SHA256 d98ee189...5a623), `frozen_protocol.json` (status FROZEN_FOR_PILOT).
Scripts: `restricted_acquisition_pilot.py` (SHA256 800915...efafc), `verify_restricted_acquisition_pilot.py`.
Interpreter: existing `economy_ml_p02_llm` (Python 3.11.15, scikit-learn 1.7.2, numpy 2.3.4). No package or environment change. No original file modified.

## What was run
- Repeat 0 of the recorded outer splits, 5 folds, 14 held-out people per fold (all five trials, median features).
- Two designs per fold and draw, 5 draws: A = 35 people x 3 trials, B = 21 people x 5 trials (105 recordings each).
- B people are a class-stratified subset of A people; A's 3 retained trials are a subset of B's 5.
- Fixed C = 0.1, balanced multinomial logistic regression, scaler fitted on acquired person vectors only.

## Mechanical checks (independent verifier, 0 refits): PASS
| Check | Result |
|---|---|
| Protocol and input hashes (cache, splits, raw CSV, DATA.mat) | match |
| Nested membership (people and trials) | pass, 25 draws |
| Class counts 15/10/10 and 9/6/6; all three classes present | pass in all 50 acquired sets |
| Test people identical to recorded fold, disjoint from training | pass |
| Exclusion proof (unacquired entries NaN vs random noise give identical coefficients and probabilities) | pass, 50/50 |
| Provenance replay: 20 random trials recomputed from DATA.mat vs cache | max abs diff 1.1e-13, pass |
| Replay of saved models on saved memberships | max error 1.1e-16, 50 models |
| Fit count | 50 primary fits (+50 duplicate fits used only for the exclusion proof) |
| Runtime | 0.26 s of model fitting; 5 ms per fit; ~27 s wall time including DATA.mat reloads for provenance |

## One defect found and fixed during the pilot (recorded honestly)
The first execution aborted on the exclusion-proof assertion. Cause: the feature function used `nanmedian`, which selects trials by NaN presence; when the proof replaced unacquired entries by noise, those entries were silently included. Fix: features are built from the explicit retained-trial index set, with NaN abort kept as a second guard. The failed partial output directory was removed and the pilot rerun. This confirms the review's point that NaN alone is not sufficient as an exclusion mechanism.

## Descriptive pilot numbers (NOT a basis for any design change)
Mechanical pilot, repeat 0 only, 5 draws, fixed C=0.1; conditional on this cohort and these splits. No interval, no hypothesis test.

| Design | Mean BA over 25 fold-draw pairs | SD | Mean log loss |
|---|---:|---:|---:|
| A 35x3 | 0.664 | 0.118 | 0.746 |
| B 21x5 | 0.594 | 0.114 | 0.846 |
| Paired A minus B | +0.070 | | |

Runtime projection for the full primary grid (5,400 fits at ~5 ms each): under one minute of fitting; the exclusion-proof duplicate would double it. Computation is not a constraint.

## Status
- Pilot pass criteria 1-7 of the frozen protocol: met.
- Full grid, sensitivity tuning and GaitRec stage: **not authorized, not run**.
- Design, grid, R, C, margin and hypothesis direction: unchanged.
