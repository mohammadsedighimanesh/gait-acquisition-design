# Addendum A2: Prespecified Inner-Tuning Sensitivity on the Primary Cells (authorized 2026-10-10)

Applies to `PROTOCOL_v2_FROZEN.md` and `ADDENDUM_A1_FULL_GRID.md`. `grid_v1/` is preserved unchanged. GaitRec remains NOT authorized.
Purpose: test whether the primary contrast 35x3 minus 21x5 and the two cell estimates change when C is tuned inside each acquired training set instead of fixed at 0.1.

## Locked design
- Cells: (35,3) and (21,5) only. Memberships, retained trials, outer folds and draws are read from `grid_v1/acquisition_memberships.csv` and reused exactly.
- Inner partition: 3 class-stratified person folds within the acquired people (21 -> 7/7/7 people with 3/2/2 per fold; 35 -> 12/12/11 with 5/4/3-4 per fold),
  `StratifiedKFold(3, shuffle=True, random_state=SEED_A2 + repeat*10000 + fold*100 + draw)`. Inner folds never include test people.
- Per inner fold: person feature vectors from retained trials only; **StandardScaler fitted on the inner-training people of that inner fold only**; logistic regression
  (balanced, max_iter 3000, tol 1e-7) for each C in {0.1, 1, 10}; log loss on the inner-validation people.
- **Selection rule (locked):** choose the C with the smallest mean inner-validation log loss over the three inner folds; exact ties go to the smaller C.
  The rule, grid and tie-break are fixed here before any fit.
- Refit: scaler and model on all acquired people with the selected C; predict the recorded 14 test people (five-trial median). One exclusion-control duplicate per refit.
- Fit accounting: inner 15 x 2 x 20 x 3 C x 3 folds = 5,400; refits 600; exclusion-control duplicates 600 (counted separately).

## Endpoints
1. Tuned primary contrast: BA(35x3 tuned) − BA(21x5 tuned), paired per (repeat, draw); mean and 95% participant bootstrap CI with the same shared-weight construction and seed as A1.
2. Tuned-minus-fixed per cell: BA and log loss, paired per (repeat, draw) against `grid_v1` predictions of the same draw; mean and CI.
3. Distribution of selected C per cell, per repeat.
4. Within-repeat SD and SD of repeat means for the tuned contrast (same decomposition as GRID_REPORT_v2).
Convergence (warnings, n_iter) recorded for all 6,000 fits.

## Verifier (no refits)
Hashes; memberships identical to grid_v1 for the two cells; inner folds partition exactly the acquired people, disjoint from test; selected C equals the argmin of the
saved inner losses under the locked tie rule; replay of 600 refit models; exclusion from memberships via scaler statistics; recomputation of all endpoints and CIs; fit counts.
PASS is claimed only for the listed checks.
