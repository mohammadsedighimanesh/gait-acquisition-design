# Addendum A3: GaitRec Stage of the Restricted-Acquisition Learning Surface (authorized 2026-10-10)

Applies to `PROTOCOL_v2_FROZEN.md`, A1 and A2. All FPI results (`grid_v1/`, `sensitivity_v1/`) preserved. Written and hashed before any GaitRec fit.
This is a second, separate task used to see whether the FPI finding recurs under a different cohort, modality and label; it is **not** external FPI validation.
The positive FPI result is not a condition for selecting this design, and a contrary GaitRec result will be reported in full.

## Task, classes and cohort (as previously frozen in `gaitrec_trial_stability_pilot_v1`, reused unchanged)
- Source: pinned GaitRec processed vertical GRF, left/right, 101 stance samples each (202 predictors per trial), body-weight and stance normalised. No demographic or diagnostic inputs.
  Source hashes: GRF_metadata 0d89799a...191d, left 1b1c898a...4ead, right 2688367e...4469 (recorded in that pilot's `protocol_before_fit.json`).
- Classification target: five GaitRec classes HC / H (hip) / K (knee) / A (ankle) / C (calcaneus), label index 0..4 in that order.
- Eligibility: official TRAIN=1 participants only; self-selected speed (SPEED=2); normal shoes (SHOD_CONDITION=1); no orthopedic insole (ORTHOPEDIC_INSOLE=0);
  first chronological eligible session with >=5 valid paired trials; 54 people without five valid paired trials excluded (listed in `cohort_audit.json`).
  Result: 1,301 people (HC 146, H 234, K 368, A 350, C 203), 6,505 real trials. **All official TEST participants reserved; no TEST data touched.**
- Session and trials: first eligible session; the five lowest trial IDs of that session, label-blind. Array `development_bags.npz` X (1301, 5, 202), y, person_ids
  (SHA256 e487d759...8149). Session table `selected_development_sessions.csv` (ff33d400...1b58).
- Feature provenance: each trial vector is that trial's own processed waveform as released; no cross-trial operation. The only learned transforms (scaler, PCA, scaler, classifier)
  are fitted inside each acquired training set.
- Outer splits: recorded `outer_person_splits_before_fit.csv` (15fd4d4d...452e): 3 repeats x 5 person folds; 1,040-1,041 train / 260-261 test; test people never appear in training.
  These 1,301 people were explored in earlier development pilots; this is not fresh confirmatory data.

## Restricted acquisition and pipeline
Identical rule to the FPI protocol: select n training people (class-stratified, nested), then k retained trials per person (nested k1⊂k3⊂k5); unselected entries are NaN and
features are built from the explicit retained index set; an exclusion-control duplicate fit with noise in unacquired entries must reproduce coefficients and probabilities exactly.
Person vector = median over the k retained trials of the 202-vector. Pipeline (fitted on acquired people only): StandardScaler -> PCA(12, full SVD) -> StandardScaler ->
balanced multinomial logistic regression, max_iter 3000, tol 1e-7, **fixed C = 0.1** (basis recorded before execution: in the prior GaitRec pilot's nested inner tuning,
C=0.1 was selected in 15/15 outer folds for both Mean and Median pooling). Test people always use all five trials with the median rule.

## Grid, allocation and contrasts
n in {300, 500, 600, 800, 1000}; k in {1, 3, 5}; R = 20 draws per fold. Class allocation by largest remainder on the 1,301-person proportions, checked to be
monotone in n and to fit every fold's training class minimum (116/187/294/280/162):

| n | HC | H | K | A | C |
|---:|---:|---:|---:|---:|---:|
| 300 | 33 | 54 | 85 | 81 | 47 |
| 500 | 56 | 90 | 141 | 135 | 78 |
| 600 | 67 | 108 | 170 | 161 | 94 |
| 800 | 90 | 144 | 226 | 215 | 125 |
| 1000 | 112 | 180 | 283 | 269 | 156 |

**Primary contrast (exact budget, locked):** 1000 x 3 versus 600 x 5, both 3,000 recordings; H1: BA(1000x3) − BA(600x5) > 0; practical margin 3 points for wording only.
**Secondary exact-budget contrast:** 500 x 3 versus 300 x 5, both 1,500 recordings.
Pairing: the 600 people of design B are a stratified subset of the 1000 of design A in the same draw; A's 3 trials are a subset of B's 5; same test people. Same for 300 ⊂ 500.
Endpoints: person-level balanced accuracy (5 classes) and log loss; finite differences along n and k only; no derivatives, ratios, extrapolation or saturation wording.

## Uncertainty
Class-stratified participant bootstrap over the 1,301 unique person IDs (5,000 resamples), one multiplicity weight per person shared across all folds, repeats, draws and cells.
Point estimate per cell = mean over 3 repeats x 20 draws of the repeat-pooled 1,301-person BA. Within-repeat SD and SD of repeat means reported separately (as in GRID_REPORT_v2).
Intervals are conditional on fitted models, recorded splits and this previously explored development cohort.

## Fit accounting
| Run | Primary fits | Exclusion-control duplicates |
|---|---:|---:|
| Pilot (repeat 0, cells 1000x3 and 600x5, 5 draws) | 50 | 50 |
| Grid | 15 folds x 5 n x 3 k x 20 draws = 4,500 | 4,500 |
Inner tuning sensitivity on GaitRec: not authorized.

## Pilot pass criteria (mechanical; performance not used for design decisions)
Hash checks; exclusion proof 50/50; nesting; class counts per table; test people equal to recorded fold and disjoint; no TEST participant anywhere; replay by verifier; fit count 50.

## Verifier (no refits)
Hashes of addendum, script, inputs; manifest of all outputs and models; nesting of people and trials across the whole grid; class counts and five-class presence;
test identity/disjointness; absence of official TEST IDs; scaler-1 statistics rebuilt from retained-trial medians (exclusion); probability replay through saved scaler-PCA-scaler-logistic;
recomputation of surface, contrasts and CIs; convergence; fit counts. PASS only for listed checks.
