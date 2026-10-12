# Restricted-Acquisition Learning Surface: Protocol v2 (FINAL FOR FREEZE)

Date: 2026-10-10. Status: EXPLORATORY SINGLE-COHORT PROTOCOL. Authorized scope: mechanical feasibility pilot only (about 50 fits); the full grid is NOT authorized.
Study type: **exploratory empirical evaluation of a learning surface under restricted acquisition.** No priority or first-ever wording is used anywhere in this study until the literature gate in section 8 is complete.
Supersedes the Idea-1 design in `Paper/New_Idea_Proposal_2026_10_10.md` after the corrections in
`Paper/Claude_Proposal_Critical_Review_2026_10_10.md`. All seven corrections are accepted and incorporated below.

## 0. Research question (revised)

For a participant-level FPI classifier trained under **restricted acquisition** (only n training people and only
k recorded trials per training person exist for the entire training pipeline), how does held-out person-level
performance compare between designs with the same number of recordings?

Primary claim allowed if supported: an empirical, retrospective, within-cohort learning surface over (n, k)
with an exactly matched-recording contrast. Not allowed: a participant-to-trial exchange rate, a D-study
accuracy predictor, monetary cost-effectiveness, between-day reliability, clinical benefit, or external FPI validation.

## 1. What is already done versus what is new (correction 1)

| Done | Source | New here |
|---|---|---|
| k-specific training models using all within-person k-subsets of five available trials, equal person mass | `exhaustive_trial_budget_study.py` lines 158, 181-198; `internal_trial_budget_study_v1` | Only k trials exist for training people; unacquired trials are physically excluded from every step |
| Fixed-k5 model versus budget-specific models | `fixed_model_trial_budget_audit_v1` | Training participant count n varies jointly with k |
| Scaling fitted on all five trials of training people | `Prepared.fit` standardizes `X.reshape(-1,28)` over all five trials | Scaling, median features, and tuning use retained trials only |

Verified facts used by this protocol: `source_feature_cache.npz` (SHA256 1cdef1a7...37c924) has X (70,5,28), A (70,19), labels (30/20/20);
recorded outer splits in `learned_multitrial_aggregation_pilot_v1/outer_splits_before_fit.csv` (SHA256 42b7d70a...ffd025c, byte-identical to the
`original_paper_incremental_value_pilot_v1` copy) have 3 repeats x 5 folds, 56 train / 14 test people each. In `internal_trial_budget_study_v1/inner_tuning.csv`,
C=0.1 was selected in 14/15 outer folds for every k in {1..5} and for both Mean and Median aggregation (the remaining fold selected C=1.0).

### Feature-cache provenance (correction 5, checked 2026-10-10)

- Cache built by `learned_multitrial_aggregation_pilot.py::load()` from `idea1_2026/corrected_raw_pipeline_diagnostic_v2/raw_trial_level_features_v2.csv`
  (SHA256 60e72d4a...30ed, identical hash recorded in five earlier frozen configs).
- That CSV is built by `corrected_raw_pipeline_diagnostic.py::build_raw()` directly from `dataset/DATA.mat` (SHA256 6fa44121...9374): for each trial, the nine
  R_* angle signals are summarized by `summarize_signal` (mean/std/min/max/median of that trial's own 100 samples) and the ten pressure descriptors are computed from
  that trial's own 40x40x100 tensor. No operation uses another trial of the same person.
- The cache keeps only `*_mean`, `*_max - *_min` (range) and the ten pressure columns. The `*_subject_centered` columns that exist in the CSV are **not** loaded.
  Anthropometric `*_first` fields are person-level constants; toe shape is one-hot encoded. No M_* moments and no FPI columns (asserted in `load()`).
- Therefore every trial feature is a function of that single trial only, and excluding a trial from acquisition removes all of its information.
  The pilot additionally recomputes descriptors for sampled trials from DATA.mat and compares them with the cache (section 6, check 6).

## 2. Restricted acquisition rule (correction 2)

Order of operations inside each outer fold and each acquisition draw:
1. Select the n training participant IDs (class-stratified from the 56 recorded training people).
2. Select k trial indices per selected person (uniform without replacement from that person's five).
3. Build the training array `X_acq` of shape (n, k, 28); unselected trials and unselected people are replaced by NaN
   and an assertion fails if any NaN reaches a fitted object.
4. Person feature vector = **median over the k retained trials** of each of the 28 descriptors, concatenated with the
   19 anthropometric fields (47 inputs). This is median of features, not median of probabilities. For k=1 the median is the trial itself.
5. StandardScaler fitted on the n x 47 person vectors only. No subject-centering, no feature selection, no auxiliary targets.
6. Inner tuning, if used, splits only these n people.
7. Test people (14, fixed by the recorded outer fold) always use all five trials with the same median rule and the training-fitted scaler.

## 3. Design grid and primary contrast (correction 3)

Grid: n in {21, 28, 35, 42, 49, 56}; k in {1, 3, 5}. Eighteen cells.

Class allocation (exact counts; every training fold has 24 Neutral / 16 Highly Supinated / 16 Highly Pronated, verified from the recorded splits):

| n | Neutral (0) | Highly Supinated (1) | Highly Pronated (2) | all three classes present |
|---:|---:|---:|---:|---|
| 21 | 9 | 6 | 6 | yes |
| 28 | 12 | 8 | 8 | yes |
| 35 | 15 | 10 | 10 | yes |
| 42 | 18 | 12 | 12 | yes |
| 49 | 21 | 14 | 14 | yes |
| 56 | 24 | 16 | 16 | yes |

Nesting rule for every draw: the people selected at a smaller n are a class-stratified subset of the people selected at the next larger n in the same draw
(21 ⊂ 28 ⊂ 35 ⊂ 42 ⊂ 49 ⊂ 56), and the trials retained at k=1 ⊂ k=3 ⊂ k=5 for each person. The pilot asserts nesting and the presence of all three classes in every acquired set.

**Primary contrast (exact recording match, 105 recordings each):**
D_A = 35 people x 3 trials versus D_B = 21 people x 5 trials.
Hypothesis H1: BA(D_A) - BA(D_B) > 0 on held-out people. Reported with a prespecified practical margin of 3 percentage points
used only for wording ("difference smaller than margin could not be resolved"), not as a significance substitute.

**Paired, nested draws:** within an outer fold and draw j, the 21 people of D_B are a stratified subset of the 35 people
of D_A, and the 3 retained trials of D_A are a subset of the 5 trials of D_B for the shared people. The same 14 test people
score both designs. Differences are therefore paired at the draw level and at the test-person level.

Secondary near-budget contrast: 56 x 3 (168) versus 35 x 5 (175); labelled as unequal by seven recordings.

## 4. Endpoints and reporting (corrections 4 and 5)

- Primary: person-level balanced accuracy averaged over draws and outer folds; log loss secondary.
- Secondary: **observed finite differences only** along n at fixed k and along k at fixed n (for example BA(35,5)-BA(35,3), BA(49,3)-BA(35,3)).
  No derivatives, no ratio, no extrapolation beyond n=56, no "saturation" wording from a null interval.
- Auxiliary: within-session ICC(1,1) and ICC(1,k) of the 28 descriptors estimated on training people only, reported as measurement
  reliability context for k. It is explicitly **not** a predictor of the accuracy surface and has no n argument.
- Uncertainty: (a) class-stratified person bootstrap over the 14-person test sets, retaining all draws and repeats, conditional on fitted
  models; (b) between-draw SD of each cell reported separately as acquisition-selection variability. Overlapping repeats are not independent cohorts.

## 5. Regularization and fit accounting (corrections 6 and 7)

- Primary run: **fixed C = 0.1**, balanced class weights, multinomial logistic regression, `max_iter=3000`, `tol=1e-7`.
  Historical basis, recorded before any new result: in `internal_trial_budget_study_v1/inner_tuning.csv` (1,500 fits, three repeats, nested inner selection by clean
  log loss), C=0.1 was selected in 14 of 15 outer folds for every k in {1,2,3,4,5} and both aggregations; the other fold selected C=1.0. Fixing C removes 9 inner fits per
  cell and makes the study explicitly **exploratory**: the regularization strength is inherited from a prior nested study on the same cohort, not re-tuned per acquired set.
  Sample weight 1 per training person (no subset expansion), so penalty strength is confounded only with n, which is the design variable; this is stated, not hidden.
- Sensitivity run (after primary, same draws): nested 3-fold inner tuning over C in {0.1, 1, 10} restricted to the acquired n people,
  on the primary-contrast cells only.
- Draws R = 20 per cell per outer fold.
- Fit accounting (FPI), each run counted separately in `progress.json`:
  | Run | Formula | Fits |
  |---|---|---:|
  | Feasibility pilot (authorized) | 5 folds x 2 cells x 5 draws | 50 |
  | Primary grid (not yet authorized) | 15 folds x 6 n x 3 k x 20 draws | 5,400 |
  | Sensitivity inner tuning (not yet authorized) | 15 folds x 2 cells x 20 draws x 3 C x 3 inner folds | 5,400 |
  | Sensitivity refits (not yet authorized) | 15 x 2 x 20 | 600 |
  Any change to R, grid, or C must be written here before results are viewed.

## 6. Feasibility pilot (must run and pass before the frozen full run)

Scope: repeat 0 only (5 folds), cells {35x3, 21x5}, R = 5, fixed C = 0.1. About 50 fits.
Pass criteria, all mechanical:
1. Exclusion proof: features, scaler and model are computed twice per fit, once with unacquired entries set to NaN and once with them replaced by random noise; both runs must give bit-identical features, coefficients and test probabilities. A NaN reaching any fitted object aborts the run.
2. Nested membership: every D_B person is in D_A for the same draw; D_A retained trials subset of D_B trials; test people disjoint from training and identical across designs.
3. Class counts in each acquired set equal the table in section 3 and all three classes are present in every acquired set.
4. Fit count, wall time per fit and total runtime recorded.
5. Independent verifier replays every saved model on the saved memberships without refitting and reproduces the saved probabilities.
6. Provenance replay: descriptors of at least 20 randomly chosen trials are recomputed from `dataset/DATA.mat` and must match the cache within 1e-9.
7. No performance number from the pilot is used to change the grid, R, C, margin or hypothesis direction. Pilot metrics are reported but labelled "mechanical pilot, 5 draws, repeat 0 only".

## 7. GaitRec stage (after FPI protocol is frozen)

Same restricted-acquisition rule on the 1,301-person development cache used in E5; n in {50, 100, 200, 400, 800, 1000}, k in {1, 3, 5},
5 person-disjoint folds, same recorded representation as the previous GaitRec development task. Official TEST is not used.
Presented as a second task with its own estimand, not as external FPI validation and not as proof that larger cohorts guarantee accuracy.

## 8. Literature gate before any priority wording (correction 8)

- Seibel, Haueisen, Jochmann, J. Neural Eng. 2026, DOI 10.1088/1741-2552/ae453d: institution-verified; full text not yet read. **No "first" wording of any kind**
  is permitted until the full text and its citation chain are reviewed. Current permitted description: "an empirical evaluation of the learning surface under restricted acquisition".
- Chen et al., NeuroImage 247:118786, DOI 10.1016/j.neuroimage.2021.118786; Figueroa et al. 2012, DOI 10.1186/1472-6947-12-8;
  Slijepcevic et al. 2020, DOI 10.1016/j.gaitpost.2019.10.021; Carvalho et al. 2026, DOI 10.1016/j.gaitpost.2025.110020;
  Pe et al. 2026, DOI 10.1016/j.ijmedinf.2026.106464; Bell et al., arXiv:2206.05985 (NeurIPS 2022).
- Required before submission: citation chain of Seibel 2026 and a PubMed/Scopus query with (subjects OR participants) AND (trials OR repetitions) AND (gait OR plantar OR biomechanics) AND (classification OR "machine learning").

## 9. Boundaries that stay in every report

Healthy men, selected extreme classes, single session, five accepted trials of unverified chronology, cohort repeatedly explored.
Recording count is not time or money. Results describe this pipeline and these two tasks, not a universal acquisition rule.

## 10. Freeze checklist

- [x] Section 3 class table and primary contrast 35x3 vs 21x5 confirmed by the author (message of 2026-10-10).
- [x] Fixed C=0.1 accepted; historical basis documented in section 5; study named exploratory.
- [x] "First" wording removed (section 0 and 8).
- [x] Fit counts recomputed per run (section 5).
- [x] Feature-cache provenance checked (section 1).
- [ ] `frozen_protocol.json` written with SHA256 of this file (done by `restricted_acquisition_pilot.py` at freeze time).
- [ ] Feasibility pilot passed; verifier PASS; fit count recorded.
- [ ] Full grid: NOT authorized.
