# Addendum A5: Third Task — StepUP-P150 plantar-pressure footsteps, up to 20 trials per person (authorized 2026-10-10)

Applies to `PROTOCOL_v2_FROZEN.md`, A1–A4. All previous outputs preserved. Written and hashed before any feature is computed or any fit is run.
Purpose: a third task in a regime the first two could not reach (many separately recorded trials per person), to test whether the learning surface and the sequential
policy behave differently when up to 20 trials per person are available. Chosen before any outcome is seen; both pre-registered labels are reported whatever the result.

## Source and people
- StepUP-P150 (Larracy et al. 2025, Sci Data 12:1415, DOI 10.1038/s41597-025-05792-1; FRDR DOI 10.20383/103.01285), official preprocessed footsteps, **pipeline 2**
  (per-footstep normalised orientation/size/amplitude as released), condition **BF/W1** (barefoot or sockfoot, preferred speed) only. Arrays: (steps, 101 frames, 75, 40).
- 150 participants; `participant_metadata.csv` (Sex 76 F / 74 M; Age 19–91; DominantLeg 140 R / 10 L; Weight/Height for BMI).
- 42 participants were already local and examined in earlier project pilots on temporal loading shares (different question); the remaining 108 are downloaded now with
  verification against the official checksum file (`stepup_third_task_v1/download_log.json`) and are untouched by this project. This is still a public dataset explored
  by others; not a confirmatory holdout in any stronger sense.

## Footstep eligibility and trial bag (label-blind)
Right-foot steps with `Outlier=0`, `Exclude=0`, `Standing=0`, `Incomplete=0`. Ordered by `FootstepID`; the first **20** eligible steps form the person's bag (K_max=20).
People with fewer than 20 eligible steps are excluded and counted (in the 42 local people the minimum was 33, so exclusions are expected to be rare).
Trial k-subsets are nested per person (first k of a random permutation), as in A1/A3.

## Per-step features (function of that step only)
The 101x75x40 normalised pressure tensor of one step is reduced to **10 time bins x (5x4) spatial blocks = 200 mean-pressure values** (bins over normalised stance; blocks over the
75x40 grid). No cross-step or cross-person operation. Person vector = median over the k retained steps (200 values). Pipeline (fitted on acquired people only):
StandardScaler -> PCA(12, full SVD) -> StandardScaler -> balanced logistic regression, **C = 0.1** (chosen for consistency with A1/A3; no prior StepUP tuning exists, so an
inner-tuning sensitivity on the primary-contrast cells is pre-registered below).

## Labels (both pre-registered, both reported)
- L1 (primary): **Sex** (binary, balanced). Expected to be a comparatively strong signal (foot geometry survives size normalisation only partly).
- L2 (secondary, low-signal probe): **BMI >= 25** (74/150). Expected weak, where trial noise may matter relatively more.
No label is used in step eligibility, feature construction or bag selection.

## Splits
5 label-stratified person folds x 3 repeats per label (about 120 train / 30 test), recorded before any fit (`outer_splits_<label>.csv`). Test people always use all 20 steps (median).

## Learning surface
n in {30, 60, 90, 120} (class allocation by largest remainder on the label proportions, nested) x k in {1, 3, 5, 10, 20}; R = 20 draws per fold.
**Primary equal-budget contrast:** 120x5 versus 60x10 (600 recordings each); H1: BA(120x5) − BA(60x10) > 0. Secondary equal-budget: 60x5 versus 30x10 (300), and 60x10 versus 30x20 (600).
Finite differences along n and k only; no derivatives/extrapolation. Uncertainty exactly as A1 (class-stratified participant bootstrap, shared weights; within-repeat and between-repeat SD).

## Sequential policy (A4 rules, K_max = 20)
Policies CONV5, FIX1, FIX2, FIX3, FIX4, FIX10, FIX20, RANDOM, MARGIN; budgets 600 (primary) and 300; warm start 10 people x 2; batch 20; 10 seeds; internal estimator and
A4.1/A4.2 rules unchanged; "trials" action adds one step to the recruited people with the fewest steps (< 20). Primary: MARGIN − CONV5 at 600; secondary: MARGIN − RANDOM,
MARGIN − best fixed (post hoc), budget share spent on extra trials. ORACLE-FIX is reported, not competed against.

## Sensitivity (pre-registered, run after the surface)
Inner-tuning C in {0.1, 1, 10} (A2 rule, scaler per inner fold on inner-train people) on the two primary-contrast cells only, for L1.

## Fit accounting (estimates; actual counts in progress.json)
Surface: 15 folds x 20 cells x 20 draws = 6,000 fits + 6,000 exclusion controls **per label** (12,000 + 12,000). Policy: internal ≈ decisions x 27 per MARGIN run; final 9 policies x 2 budgets x 15 folds x 10 seeds per label.
Pilot (mechanical): L1, repeat 0, cells 120x5 and 60x10, 5 draws (50 fits + 50 controls) — must pass before anything else.

## Verifier (no refits)
Hashes (addendum, scripts, download log, feature cache); eligibility replay from metadata; feature replay for sampled steps from the npz arrays; nesting; class counts; test disjointness;
exclusion via scaler statistics; probability replay; metric/contrast/CI recomputation; convergence; fit counts; policy logs as in A4. PASS only for listed checks.

## Claims not allowed
Any "first"; clinical meaning of sex or BMI classification from footsteps; cost in time/money; generality beyond these pipelines; external validation of FPI or GaitRec.
Sex classification from plantar pressure has prior literature (gait biometrics); it is used here only as a between-person label with many trials.
