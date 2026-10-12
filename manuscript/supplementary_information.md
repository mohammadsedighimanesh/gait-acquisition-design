# Supplementary Information

for: Participants or repeated trials? Equal-budget learning surfaces and an evaluated sequential acquisition rule in three gait datasets

[Authors: to be completed by the authors]

### S1. Execution log, verification and amendments

| Stage | Model fits | Policy-internal fits | Inner-tuning fits | Refits | Exclusion-control fits | Independent verifier |
|---|---|---|---|---|---|---|
| FPI mechanical pilot | 50 | - | - | - | 50 | pass (listed checks) |
| FPI learning surface (A1) | 5,400 | - | - | - | 5,400 | pass (listed checks) |
| FPI inner-tuning sensitivity (A2) | - | - | 5,400 | 600 | 600 | pass (listed checks) |
| GaitRec mechanical pilot | 50 | - | - | - | 50 | pass (listed checks) |
| GaitRec learning surface (A3) | 4,500 | - | - | - | 4,500 | pass (listed checks) |
| FPI + GaitRec sequential rule (A4) | 5,250 | 243,161 | - | - | 5,250 | pass (listed checks) |
| StepUP pilot (A5) | 50 | - | - | - | 50 | pass (listed checks) |
| StepUP surface, sex (A5) | 6,000 | - | - | - | 6,000 | pass (listed checks) |
| StepUP surface, BMI (A5) | 6,000 | - | - | - | 6,000 | pass (listed checks) |
| StepUP sensitivity, sex (A5) | - | - | 5,400 | 600 | - | pass (listed checks) |
| StepUP sequential rule, sex (A5) | 2,700 | 166,050 | - | - | 2,700 | pass (listed checks) |
| StepUP sequential rule, BMI (A5) | 2,700 | 166,050 | - | - | 2,700 | pass (listed checks) |
| StepUP small-budget rule, sex (A5.2) | 2,700 | 74,250 | - | - | 2,700 | pass (listed checks) |
| StepUP small-budget rule, BMI (A5.2) | 2,700 | 74,250 | - | - | 2,700 | pass (listed checks) |


Counts are model fits whose results enter the tables (final models and, where stated, policy-internal, inner-tuning and refit fits) and the exclusion-control duplicates. Across all stages there were 38,100 final model fits, 723,761 policy-internal fits, 10,800 inner-tuning fits, 1,200 refits and 38,700 exclusion-control fits. The StepUP sensitivity stage had no exclusion-control duplicates. "Pass (listed checks)" means that the independent verifier passed the checks described in Section 2.9 and no more.

Amendments, each written before any performance result of its stage was read:

- A4.1: the internal balanced accuracy of the sequential rule is computed over the classes present in the inner-validation fold, and an undefined internal estimate defaults to recruiting (the first pilot had silently resolved undefined comparisons to "trials").
- A4.2: the first full policy grid stopped when a class-blind acquired set lacked a class; models then assign probability zero to the absent class and are counted.
- A5.1: with 150 people and five folds each training set has 120 people, so StepUP class allocation was computed per fold from that fold's class counts.
- A5.2.1 and A5.2.2: in the small-budget follow-up the largest fixed designs acquire only 5 to 15 people, which cannot support 12 principal components or, sometimes, two classes; the number of components is min(12, n − 1) and a single-class set yields a constant predictor.

Verifier corrections: the FPI learning-surface verifier first failed because scikit-learn stores a scale of 1 for zero-variance columns, and it was corrected; the policy verifier was corrected for the order of budgets and for ties at the precision stored in the decision log (294 ties accepted); two smaller defects (a pandas attribute lookup and a cell absent from the pilot grid) were corrected. No result file was changed by any verifier correction. Superseded or aborted runs were preserved with their records: three superseded policy pilots, the first policy grid (aborted on a missing class), one StepUP pilot (aborted on class allocation) and three small-budget attempts (aborted). The first learning-surface pilot attempt was aborted by the exclusion test, because a median function that skips missing values had selected trials by their missing status; its partial output was deleted before the rule of keeping failed runs was adopted, and the defect and its correction are recorded in the pilot report.

### S2. Protocol hashes

| Protocol file | Field | SHA-256 (recorded before execution) |
|---|---|---|
| frozen_protocol_A1.json | protocol_md_sha256 | d98ee18951e3d0fe3e3f62f09d459211890570b31d757e9283035826b6f5a623 |
| frozen_protocol_A1.json | addendum_sha256 | 1e43e2c0da3c6b2265a063b997dc947503dd27f52b3a611c486f7455b7ceebae |
| frozen_protocol_A1.json | grid_script_sha256 | 6f25db2781578141dcb57404d3cf7c459a2c587c114fb5a3a04a05bc4e682475 |
| frozen_protocol_A2.json | addendum_A2_sha256 | c9a788f058d517861be434608cac3845958151a9c8731c5126541221b4f60119 |
| frozen_protocol_A2.json | script_sha256 | 0b71b1ca619b03dcf36138471a8dc974d49f8cddacf9b47d64aad94c2b4a1891 |
| frozen_protocol_A3.json | addendum_A3_sha256 | b86692d409e0f3ac8a8849f38fd2e5e9acc5734d049a268f082dfce5f7df0185 |
| frozen_protocol_A3.json | script_sha256 | 8b6e8b474a990a1152025392b784d8894581ec6cea9493133e0c02241a52e37e |
| frozen_protocol_A4.json | addendum_A4_sha256 | f29b8d14bf9d00e0a91d70dd89c75b9c518b248e11b47a9b3b8cb768aa57776a |
| frozen_protocol_A4.json | script_sha256 | 8784077192577e679e1d3ec975e2a273772f79daddb74e6731da090903d15ffc |
| frozen_protocol_A5.json | addendum_A5_sha256 | 1669d9910f99c6576a0e731ebb7df46f7560776e57da753a212f94f2a9999985 |
| frozen_protocol_A5.json | script_sha256 | 0a00bd52172b41a00d100ba6f5b16a11af1bc54348517c283e7daabe3735451e |
| frozen_protocol_A5_2.json | addendum_A5_2_sha256 | 40b1240b5eba65d56939ea2a20b62ce8cc99cc10444ce7bc0e31e5f50c1c8c21 |
| frozen_protocol_A5_2.json | script_sha256 | 49c40e9cacf3e748850b228a231506912c191443c4aa1922fc7e29f68a869111 |
| frozen_protocol.json | protocol_md_sha256 | d98ee18951e3d0fe3e3f62f09d459211890570b31d757e9283035826b6f5a623 |


### S3. FPI learning surface (balanced accuracy [95% CI])

| People (n) | k = 1 | k = 3 | k = 5 |
|---|---|---|---|
| 21 | 0.610 [0.547, 0.670] | 0.621 [0.555, 0.684] | 0.622 [0.556, 0.684] |
| 28 | 0.636 [0.564, 0.703] | 0.648 [0.574, 0.717] | 0.648 [0.574, 0.718] |
| 35 | 0.654 [0.577, 0.727] | 0.659 [0.577, 0.733] | 0.664 [0.581, 0.739] |
| 42 | 0.671 [0.587, 0.749] | 0.679 [0.592, 0.758] | 0.678 [0.591, 0.758] |
| 49 | 0.682 [0.593, 0.764] | 0.691 [0.601, 0.776] | 0.696 [0.605, 0.780] |
| 56 | 0.691 [0.596, 0.780] | 0.699 [0.601, 0.791] | 0.722 [0.624, 0.815] |


### S4. GaitRec learning surface (balanced accuracy [95% CI])

| People (n) | k = 1 | k = 3 | k = 5 |
|---|---|---|---|
| 300 | 0.404 [0.385, 0.423] | 0.408 [0.389, 0.427] | 0.408 [0.389, 0.428] |
| 500 | 0.414 [0.393, 0.435] | 0.416 [0.395, 0.437] | 0.418 [0.397, 0.440] |
| 600 | 0.417 [0.396, 0.439] | 0.421 [0.399, 0.443] | 0.422 [0.399, 0.444] |
| 800 | 0.422 [0.400, 0.445] | 0.427 [0.404, 0.451] | 0.427 [0.404, 0.451] |
| 1000 | 0.427 [0.404, 0.451] | 0.430 [0.406, 0.455] | 0.431 [0.407, 0.457] |


### S5. StepUP sex learning surface (balanced accuracy [95% CI])

| People (n) | k = 1 | k = 3 | k = 5 | k = 10 | k = 20 |
|---|---|---|---|---|---|
| 30 | 0.530 [0.507, 0.553] | 0.543 [0.516, 0.568] | 0.553 [0.524, 0.580] | 0.556 [0.525, 0.585] | 0.557 [0.526, 0.587] |
| 60 | 0.526 [0.494, 0.556] | 0.549 [0.513, 0.584] | 0.550 [0.511, 0.587] | 0.555 [0.513, 0.595] | 0.561 [0.519, 0.602] |
| 90 | 0.532 [0.494, 0.568] | 0.553 [0.510, 0.594] | 0.562 [0.516, 0.606] | 0.567 [0.518, 0.614] | 0.572 [0.520, 0.621] |
| 120 | 0.535 [0.492, 0.575] | 0.553 [0.502, 0.601] | 0.562 [0.508, 0.613] | 0.580 [0.521, 0.636] | 0.606 [0.539, 0.668] |


### S6. StepUP BMI learning surface (balanced accuracy [95% CI])

| People (n) | k = 1 | k = 3 | k = 5 | k = 10 | k = 20 |
|---|---|---|---|---|---|
| 30 | 0.598 [0.562, 0.633] | 0.619 [0.580, 0.657] | 0.618 [0.578, 0.656] | 0.616 [0.577, 0.656] | 0.618 [0.578, 0.658] |
| 60 | 0.633 [0.585, 0.679] | 0.653 [0.602, 0.703] | 0.660 [0.610, 0.710] | 0.653 [0.600, 0.705] | 0.654 [0.600, 0.707] |
| 90 | 0.654 [0.600, 0.707] | 0.665 [0.607, 0.721] | 0.671 [0.614, 0.727] | 0.669 [0.611, 0.728] | 0.664 [0.602, 0.724] |
| 120 | 0.667 [0.610, 0.722] | 0.674 [0.612, 0.733] | 0.679 [0.618, 0.739] | 0.671 [0.606, 0.735] | 0.658 [0.587, 0.727] |


### S7. Gain from three to five trials per person

| Task | People (n) | Gain from 3 to 5 trials, points [95% CI] |
|---|---|---|
| FPI posture classes (70 men) | 21 | +0.1 [-0.8, +0.9] |
| FPI posture classes (70 men) | 28 | +0.0 [-0.7, +0.8] |
| FPI posture classes (70 men) | 35 | +0.5 [-0.4, +1.4] |
| FPI posture classes (70 men) | 42 | -0.1 [-1.0, +0.9] |
| FPI posture classes (70 men) | 49 | +0.4 [-0.6, +1.6] |
| FPI posture classes (70 men) | 56 | +2.3 [+0.2, +4.9] |
| GaitRec diagnosis classes (1,301 patients) | 300 | +0.0 [-0.2, +0.2] |
| GaitRec diagnosis classes (1,301 patients) | 500 | +0.3 [+0.1, +0.4] |
| GaitRec diagnosis classes (1,301 patients) | 600 | +0.1 [-0.1, +0.3] |
| GaitRec diagnosis classes (1,301 patients) | 800 | +0.0 [-0.2, +0.2] |
| GaitRec diagnosis classes (1,301 patients) | 1000 | +0.2 [-0.1, +0.4] |
| StepUP sex (150) | 30 | +1.0 [-0.1, +2.1] |
| StepUP sex (150) | 60 | +0.1 [-1.0, +1.2] |
| StepUP sex (150) | 90 | +0.9 [-0.3, +2.0] |
| StepUP sex (150) | 120 | +0.9 [-0.3, +2.0] |
| StepUP BMI ≥ 25 (150) | 30 | -0.1 [-1.1, +0.8] |
| StepUP BMI ≥ 25 (150) | 60 | +0.7 [-0.2, +1.6] |
| StepUP BMI ≥ 25 (150) | 90 | +0.6 [-0.3, +1.6] |
| StepUP BMI ≥ 25 (150) | 120 | +0.6 [-0.6, +1.7] |


### S8. FPI contrast variability by repeat

| Dataset | Contrast | Mean (points) | Within-repeat SD over 20 draws (points; repeats 1/2/3) | SD of the three repeat means (points) | Pairs favouring A per repeat |
|---|---|---|---|---|---|
| FPI | 35x3 minus 21x5 | +3.7 | 5.2 / 6.0 / 4.3 | 0.55 | 14 / 15 / 16 |
| FPI | 56x3 minus 35x5 | +3.5 | 3.8 / 3.8 / 5.3 | 0.35 | 16 / 17 / 14 |


### S9. Regularisation chosen in the inner-tuning sensitivity analysis

| Dataset | Cell | Inner fits selecting C = 0.1 | C = 1 | C = 10 | Acquired sets |
|---|---|---|---|---|---|
| FPI | 21x5 | 474 (79%) | 104 | 22 | 600 |
| FPI | 35x3 | 498 (83%) | 98 | 4 | 600 |
| StepUP sex | 60x10 | 299 (100%) | 1 | 0 | 300 |
| StepUP sex | 120x5 | 298 (99%) | 1 | 1 | 300 |


### S10. Counterpart on the assessment side (FPI, earlier analysis of the same cohort)

The main analyses vary the training data. An earlier analysis of the FPI cohort varied the number of trials available at assessment while holding the model fixed (a model trained with five trials per person, applied to every subset of k of the five recorded trials of held-out people, with the median of the retained trials as person input; probabilities were never pooled across trials). Balanced accuracy increased from one to five trials, and the class assigned to a person depended on which k trials had been recorded in a large share of assessments; at five trials this share is zero by construction because only one subset exists.

| Trials available at assessment (k) | Median-pooled BA [95% CI] | Person-repeat assessments whose class depends on which k trials were recorded |
|---|---|---|
| 1 | 0.680 [0.596, 0.760] | 39.5% |
| 2 | 0.692 [0.602, 0.779] | 28.6% |
| 3 | 0.706 [0.616, 0.792] | 22.9% |
| 4 | 0.710 [0.616, 0.800] | 12.4% |
| 5 | 0.722 [0.626, 0.813] | 0.0% |


With the fixed model, five trials exceeded one trial by 4.3 points (95% CI 0.6 to 8.1), and three trials were 1.6 points below five (−3.6 to 0.6). These intervals are conditional on the fitted models and do not include re-fitting. The comparison shows that the weak value of additional trials at training reported here does not imply that trials are unimportant at assessment.

### S11. Data provenance

FPI inputs were rebuilt from the released raw file (SHA-256 6fa44121…9374) so that every descriptor depends on one trial only. Earlier feature tables that contained subject-centred features and preprocessing shared across splits were not used. GaitRec and StepUP inputs were taken from the released files; for StepUP, all 300 source files were verified against the official checksum file (the published checksums of the arrays refer to the uncompressed array payload). Detailed provenance is archived with the study materials.

### S12. Literature search

Searches were run in October 2026 in OpenAlex (full-text search, for example "gait classification number of trials per participant number of participants" and "learning curve number of subjects per subject gait OR plantar pressure OR ground reaction force", restricted to 2018 or 2020 onwards), in Crossref and on the web, together with the reference list of Seibel and colleagues and the list of works citing that paper (one citing work in OpenAlex, on heart-rate variability and EEG in mental health). Scopus and Web of Science were not available to us. No retrieved gait, plantar-pressure or ground-reaction-force study varied training participants and trials per participant jointly under participant-disjoint testing. Because the searches were targeted and not systematic, the absence of such a study is stated as a search result, not as proof that none exists.
