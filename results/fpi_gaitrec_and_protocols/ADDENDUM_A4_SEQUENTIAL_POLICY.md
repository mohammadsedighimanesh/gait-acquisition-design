# Addendum A4: Sequential Acquisition Policy — "next person or next trial?" (authorized 2026-10-10)

Applies to `PROTOCOL_v2_FROZEN.md` and A1–A3; all previous outputs preserved. Written and hashed before any fit. Exploratory; no "first" wording; no Q1 or accuracy promise.

## Question
Under a fixed recording budget B, can an online policy that decides at each step whether to **recruit a new person (1 trial)** or **record another trial of an already recruited person**,
using only the data it has acquired so far, reach held-out person-level performance at least as good as the laboratory convention of five trials per person?

## Literature position (targeted search, 2026-10-10)
Closest conceptual precedent: Sheng, Provost & Ipeirotis, "Get another label? Improving data quality and data mining using multiple, noisy labelers", KDD 2008
(repeated labels vs new examples under a budget; DOI to be verified at ACM DL). Related: Chen et al. 2022 NeuroImage 247:118786 (trial/subject trade-off, group inference);
Seibel et al. 2026 JNE (subjects vs recording length, offline learning curves); Forrester 2015 Int Biomech 2:62-72 (trials vs subjects for power);
cost-per-risk greedy sequential design (Pauwels et al. 2014, BMC Syst Biol; PMC4181721); biomarker replicate-vs-subject allocation (PMC4268307).
No gait/EEG/EMG study found that allocates recordings between new participants and repeated trials online for a between-person classifier. Absence from a targeted search is not evidence of absence.
Structural difference from Sheng 2008: input-measurement repetition in a hierarchical person/trial design, not label repetition; stated, not hidden.

## Simulation environment (real recordings only)
- Tasks and pipelines exactly as A1 (FPI: 47-input balanced logistic, C=0.1) and A3 (GaitRec: median -> scaler -> PCA12 -> scaler -> balanced logistic, C=0.1).
- Recorded outer folds: FPI 3x5 (56 train / 14 test), GaitRec 3x5 (1,040-1,041 train / 260-261 test). Test people always all five trials, median.
- Recruitable pool = the fold's training people; each has exactly five real trials. A "recording" = one real trial of one real person. No synthetic data.
- Common random numbers per (fold, seed): one class-blind recruitment order of the pool and one trial order per person, shared by every policy, so policies are paired.
- Budgets: FPI B = 105 (primary; matches A1 contrast 35x3 vs 21x5) and B = 70 (secondary). GaitRec B = 3,000 (primary; matches A3) and B = 1,500 (secondary).

## Policies (all locked; no hyperparameter tuning anywhere)
| Policy | Rule |
|---|---|
| CONV5 (reference) | recruit next person in order, record 5 trials; repeat until B |
| FIX3, FIX1 | same with 3 or 1 trials per person; if the pool is exhausted before B, add trials round-robin to recruited people (<5) |
| RANDOM | warm start as MARGIN; then each batch: fair coin -> people or trials |
| MARGIN (proposed) | see below |
| ORACLE-FIX (post hoc) | best of FIX1..FIX5 at that budget, chosen after seeing results; reported as reference only |

### MARGIN policy
1. Warm start: recruit W people with 2 trials each (FPI W=7 -> 14 recordings; GaitRec W=100 -> 200).
2. Then in batches of b recordings (FPI b=7; GaitRec b=100) until B is spent, at each decision estimate from the **acquired data only**:
   - internal 3-fold person CV, repeated with 3 fold seeds; the inner-validation people use the median of all their acquired trials;
   - BA_full: train on inner-train people with all their acquired trials;
   - BA_people: train on a random 80% of inner-train people (same trials);
   - BA_trial: train on all inner-train people with one random acquired trial removed from every person having >= 2 trials;
   - marginal gain per recording: g_people = (BA_full − BA_people) / (recordings removed), g_trial = (BA_full − BA_trial) / (recordings removed);
   - exploration: one of the three fold-seed replicates is drawn at random (Thompson-style) and its sign of (g_people − g_trial) decides.
3. Action "people": recruit the next b people in order with 1 trial each (fewer if the pool is exhausted; remainder goes to trials).
   Action "trials": add one trial to the b recruited people with the fewest trials (<5), ties by recruitment order; if none can take a trial, the batch goes to people.
4. Every recording is a real, previously unrecorded trial; the acquired index set is explicit; an assertion checks total recordings == B and no trial outside the index set reaches any fit.

## Final evaluation
For each policy, fit the task pipeline on the acquired set (median of each person's acquired trials; all learned transforms on acquired people only) and score the fold's test people.
An exclusion-control duplicate fit (noise in unacquired entries) must reproduce the final model's coefficients and probabilities exactly.

## Endpoints (locked)
- Primary: BA(MARGIN) − BA(CONV5) at the primary budget, paired per (repeat, fold, seed); 95% class-stratified participant bootstrap over unique test IDs with shared weights; H1: > 0.
  Wording margin 3 points (FPI) / 1 point (GaitRec) for "practically larger than" statements only.
- Secondary: MARGIN − RANDOM; MARGIN − ORACLE-FIX (post hoc, descriptive); log loss versions; share of the budget MARGIN spent on extra trials; the same at the secondary budget.
- Variability: within-repeat SD over seeds and SD of repeat means, separately.
- Failure gate: if the primary CI includes zero in both tasks, report "no advantage over the convention established"; if MARGIN is worse than RANDOM, say so.

## Seeds and fit accounting (estimates; actual counts recorded in progress.json)
FPI: 20 seeds; GaitRec: 5 seeds. Internal fits per MARGIN run ≈ decisions x 3 replicates x 3 folds x 3 fits (FPI ≈ 13 x 27 = 351; GaitRec ≈ 28 x 27 = 756).
Final fits: 7 policies (CONV5, FIX1, FIX2, FIX3, FIX4, RANDOM, MARGIN; FIX5 = CONV5) + 7 exclusion controls per (fold, seed, budget).
Pilot (mechanical): FPI repeat 0, 2 seeds, B = 105 only. Grid: everything else. All counts separately recorded.

## Verifier (no refits)
Hashes; acquisition logs: every recording is a distinct (person, trial) from the fold's pool, counts equal B, trials per person <= 5, no test person; MARGIN decisions reproducible from the
logged internal estimates and logged replicate draws; final-model replay from saved parameters on the logged acquired sets (scaler statistics rebuilt from acquired trials only);
endpoint and CI recomputation; convergence; fit counts. PASS only for the listed checks.

## Claims not allowed
Clinical benefit, cost in time or money, external validation, exchange rate, D-study prediction, generality beyond these two tasks and pipelines, any "first".
