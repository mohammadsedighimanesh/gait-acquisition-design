# Critical Review of the Empirical FPI Trial-Budget Contribution

## Verdict
The numerical evidence supports a narrow internal empirical contribution. It does not establish a new method, universal acquisition rule, or Q1-level novelty. The claim should be restricted to exhaustive sensitivity within the five observed accepted trials, using the corrected specified FPI pipeline. A closer predecessor was identified during this review, so any broad claim that multi-trial aggregation effects in gait classification are new must be removed.

## 1. Exact Scientific Question
For a specified FPI classifier and aggregation rule, how much do expected person-level predictive performance and prediction agreement change when one observes different numbers and compositions of recorded gait trials?

For budget k, subsets S range over all binomial(5,k) combinations. Expected balanced correctness equals the average over classes, people within each class, held-out repeat exposures, and uniform subset choices of the indicator that f(aggregate(x_i,S),a_i) equals the recorded FPI class. Averaging predicted probabilities over all subsets is deliberately NOT used, because it would reveal all five trials while claiming a smaller acquisition budget. Measurement dispersion, categorical agreement, and predictive correctness are separate concepts; an always-wrong prediction can be perfectly stable.

This endpoint is a finite, retrospective within-recorded-trial expectation. It cannot estimate the population of all future strides, other visits, walking speeds or devices without further data.

## 2. Closest Prior Work and Actual Distinction
Hafer et al. evaluated pressure-parameter reliability and stabilization with additional trials [1]. Slijepcevic et al. directly studied aggregation and preprocessing effects in gait classification [2]. The latter compared mean and median early fusion, representative-trial selection and voting on a 910-person gait-disorder dataset with eight trials per person. Thus aggregation-based classification gains are not new in general. Stride-count reliability from truncated recordings is also established [3].

Our limited distinction is the FPI-specific outcome, exhaustive composition and count of the existing five-trial sets, and conditional person-level analysis under both budget-specific and unchanged-model evaluation. No exact matching FPI all-subset study was located in this targeted search, but that is NOT proof of priority or uniqueness. The search is not a systematic PubMed/Embase/Scopus review. Sources [1] and [3] concern measurement reliability, while [2] is the closer classification comparator; none provides a directly comparable FPI SOTA accuracy target.

## 3. Design Confound Identified and Controlled
The original budget study fitted a different classifier for each k. Its five-versus-one contrast therefore changes both assessment information and the fitted training representation/selected C. It is a valid pipeline comparison but does not isolate an inference-budget effect.

The new audit reuses the same frozen five-trial outer classifier for all k=1..5, holding coefficients, C, preprocessing and training participants constant. All 31 subsets are evaluated on held-out people. Thirty existing models yielded 13,020 predictions; no new classifier was trained, no seed was selected, and no subject was excluded. k5 probabilities exactly reproduced the original outputs. This post hoc sensitivity control was independently replayed; it does not replace the original primary comparison or erase exploratory cohort reuse.

### Performance: Fixed Model versus Budget-Specific Models

```text
method  k  expected_balanced_accuracy  expected_accuracy  macro_f1_of_expected_confusion  expected_log_loss  expected_brier  budget_specific_training_ba  fixed_minus_budget_specific_ba
  Mean  1                    0.670741           0.660952                        0.665173           0.708625        0.436673                     0.671481                       -0.000741
  Mean  2                    0.683704           0.673810                        0.678048           0.689436        0.424474                     0.679074                        0.004630
  Mean  3                    0.680370           0.670000                        0.674614           0.683076        0.420269                     0.680741                       -0.000370
  Mean  4                    0.688889           0.680000                        0.684154           0.679900        0.418128                     0.687037                        0.001852
  Mean  5                    0.705556           0.695238                        0.699794           0.677996        0.416832                     0.705556                        0.000000
Median  1                    0.679630           0.672381                        0.676689           0.702267        0.432707                     0.671481                        0.008148
Median  2                    0.691852           0.683333                        0.687582           0.684364        0.421194                     0.679074                        0.012778
Median  3                    0.706481           0.697143                        0.701928           0.675386        0.413028                     0.700556                        0.005926
Median  4                    0.710000           0.700952                        0.706308           0.669390        0.409305                     0.710000                        0.000000
Median  5                    0.722222           0.714286                        0.719101           0.669862        0.408419                     0.722222                        0.000000
```

### Fixed-Model Paired Contrasts

```text
method  inference_k_candidate  inference_k_baseline  fixed_model_ba_delta  conditional_ci_low  conditional_ci_high
  Mean                      5                     1              0.034815            0.002213             0.070000
  Mean                      3                     5             -0.025185           -0.049074            -0.004074
Median                      5                     1              0.042593            0.006296             0.080741
Median                      3                     5             -0.015741           -0.035741             0.005556
```

For Median, holding the model fixed gives five-versus-one expected BA +4.2593 percentage points, conditional 95% CI [+0.6296, +8.0741]. The original refitted-pipeline increase was +5.0741 points. The new control supports a positive conditional internal acquisition-information effect beyond merely switching models, not a causal clinical benefit or a universal guarantee.

Fixed-model Median three-minus-five is -1.5741 points, CI [-3.5741, +0.5556]. The lower bound still exceeds the allowed loss in magnitude for the illustrative 3pp engineering margin, so three-trial compatibility is not established here either. Failure to establish compatibility is not proof of incompatibility. Mean five-versus-one is +3.4815 points, CI [+0.2213, +7.0000], but the Mean point-estimate curve is not strictly monotone.

Intervals resample class-stratified people and keep repeated predictions together. They are conditional on these trained models and do not account for full refitting uncertainty, shared training sets, exploratory model selection or multiple historical hypotheses. Effective participant count is 70, not 350 trials or 13,020 predictions. These paired intervals are sensitivity estimates, not additional confirmatory primary tests.

### Composition Sensitivity with a Fixed Model

```text
method  k  fraction_person_repeat_any_disagreement
  Mean  1                                 0.390476
  Mean  2                                 0.285714
  Mean  3                                 0.209524
  Mean  4                                 0.133333
  Mean  5                                 0.000000
Median  1                                 0.395238
Median  2                                 0.285714
Median  3                                 0.228571
Median  4                                 0.123810
Median  5                                 0.000000
```

Median k3 has prediction disagreement across possible subsets in 22.8571% of person-repeat assessments (48/210). This is not 48 independent people, nor the percentage of pairs that disagree. The previously reported 23.3333% used the budget-specific k3 models; the new value is a different, explicitly fixed-model regime. k5 zero disagreement is structural because only one subset exists. It should not be a headline claim of perfect reliability.

## 4. Strengths that Make the Empirical Analysis Defensible
- Source-corrected, raw-unit-derived descriptors rather than the legacy pressure normalization artifact.
- All available within-five compositions instead of selecting favorable trials.
- Unseen-person outer evaluation; learned transformations use training people only.
- Equal person mass despite different numbers of subset representations.
- Direct comparison of budget-specific fitting with unchanged-model assessment.
- Independent probability replay, subset-identity, grouping and conditional interval verification.
- Negative results retained: no proven median superiority, no established three-trial compatibility, no new classifier advantage.

These are strengths of execution and reporting, not individually invented methods.

## 5. Reviewer Risks and Required Claim Restrictions
- The 70 participants are healthy men with selected Neutral/Highly Supinated/Highly Pronated source classes; mild categories and symptomatic populations are not represented.
- The source supplies five ACCEPTED recordings, not the rejected acquisition attempts; chronology is not verified. Results cannot be sold as a sequential real-time stopping policy.
- All development people originally had five recordings. Smaller assessment k does not prove that a training cohort can be collected with only k recordings.
- ICCs describe pooled/class-specific feature repeatability, not individual reliability or between-day reproducibility. The variance-instability link remained weak under multiplicity correction.
- Collection time, financial cost, missed clinical diagnoses, treatment response and net clinical benefit were not measured. Do not quantify a cost saving or declare a clinical optimum.
- No external cohort, untouched confirmatory sample or corrected full 100x/LOPO result is supplied by this audit.
- Exact subset enumeration is standard combinatorial evaluation. Different outcome/cohort alone does not prove strong methodological novelty, particularly after previous novelty-based desk rejections.

## 6. Recommended Contribution Wording
"We quantify the sensitivity of person-level FPI classification to the number and composition of repeated gait recordings by evaluating all subsets of five accepted trials under participant-disjoint internal validation. A fixed-model control separates inference-budget sensitivity from budget-specific model refitting, while person-clustered uncertainty and explicit claim boundaries distinguish measurement repeatability, decision stability and predictive correctness."

Preferred title: "How Many Recorded Gait Trials Inform Foot Posture Prediction? An Exhaustive Within-Cohort Sensitivity Study". This is a question, not a claim that a clinical optimum was established.

## 7. Publication Decision
This can be defended as an empirical acquisition-sensitivity study if positioned against [2] and presented with transparent limitations. It cannot honestly be presented as a new aggregation algorithm or certified Q1 contribution. A journal prioritizing technically novel AI may still find the contribution incremental. A question-focused biomechanics audience is a more coherent scientific fit, but no journal ranking or acceptance assessment has been made. Do not respond to previous novelty objections merely by renaming the framework.

The manuscript should report two complementary regimes separately: (a) original budget-specific models, preserving the original primary three-versus-five result, and (b) fixed-model inference-budget sensitivity. No favorable result from (b) should retroactively become a prespecified primary endpoint. Expert ratings from older models must not be transferred to the present pipeline without a matched new review.

## References
[1] J. F. Hafer et al., "Reliability of plantar pressure platforms," Gait & Posture, vol. 38, no. 3, pp. 544-548, 2013, doi: [10.1016/j.gaitpost.2013.01.028](https://doi.org/10.1016/j.gaitpost.2013.01.028).

[2] D. Slijepcevic et al., "Input representations and classification strategies for automated human gait analysis," Gait & Posture, vol. 76, pp. 198-203, 2020, doi: [10.1016/j.gaitpost.2019.10.021](https://doi.org/10.1016/j.gaitpost.2019.10.021). Primary full text reviewed: https://phaidra.ustp.at/api/object/o:4462/download . This is an internally split cohort, not an automatically external/multicentre comparator.

[3] D. Commandeur, M. Klimstra, K. Yoshida, and S. Hundza, "The Minimum Number of Strides Required for Reliable Gait Measurements in Older Adult Fallers and Non-Fallers," Sensors, vol. 24, no. 23, Art. no. 7666, 2024, doi: [10.3390/s24237666](https://doi.org/10.3390/s24237666). Verified primary record: https://pubmed.ncbi.nlm.nih.gov/39686203/ .

## Provenance
frozen_protocol.json, fixed_model_subset_predictions.csv, independent_verification.json and the CSVs above accompany this report. Original manuscripts and results were not overwritten; Python environments were unchanged.
