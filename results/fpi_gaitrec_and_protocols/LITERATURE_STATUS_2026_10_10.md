# Literature Status for the Restricted-Acquisition Study (2026-10-10)

Targeted search, not a systematic review. No "first" or priority wording is permitted anywhere in the study until this gate is closed.

## Closest comparator: Seibel, Haueisen, Jochmann 2026
- Title: "How much EEG is needed for deep learning with convolutional neural networks? Predicting the benefit from additional data." J. Neural Eng. 23(1), published online 2026-02-25.
  DOI 10.1088/1741-2552/ae453d; PMID 41736475. Code: Zenodo 10.5281/zenodo.18403060 (CC BY 4.0, "generation of training data subsets, fitting of different learning curve models, prediction for larger datasets").
- Access this turn: publisher page behind a bot-check; PubMed page returned no abstract to the fetch tool; Europe PMC record confirms subscription-only full text (no PMC). Institutional press release (TU Ilmenau, 2026-03-04) read.
- What is established from the abstract-level sources: CNNs on EEG, three architectures, three tasks (incl. normal/abnormal and eyes open/closed); number of subjects and recording length per subject varied; learning-curve characteristics differ by task, consistent across architectures; reliable extrapolation needs several hundred subjects; gains from longer recordings plateau after a few seconds.
- What cannot be claimed without the full text: whether they used equal-budget paired contrasts, nested subsets, variance decomposition, or any gait/biomechanics data; whether they fitted scaling laws jointly in (subjects, duration).
- Required before submission: obtain the full text (library access or author request), read Methods, and chain its references and citing papers. The Zenodo archive (200 MB) could be inspected for the subset-generation design if the user authorizes the download.

## Closest biomechanics comparator (verified via Crossref)
- Forrester SE. "Selecting the number of trials in experimental biomechanics studies." International Biomechanics 2(1):62-72, 2015, DOI 10.1080/23335432.2015.1049296.
  Subjects-versus-trials trade-off for the statistical power of group comparisons (e.g. 15 subjects with >=19 trials versus >=20 subjects with 3 trials for the same power).
  Difference from our study: hypothesis-test power for group means, not participant-disjoint predictive performance. Must be cited as the biomechanics precedent for the question.

## Other verified comparators (already in the protocol)
Chen et al. 2021/2022 NeuroImage 247:118786 (trial-subject hyperbolic trade-off, fMRI group effects); Figueroa et al. 2012 BMC MIDM 12:8 (learning-curve extrapolation);
Slijepcevic et al. 2020 Gait & Posture 76:198-203 (trial aggregation in gait classification); Carvalho et al. 2026 Gait & Posture 123:110020 (strides for reliability);
Pe et al. 2026 IJMI 216:106464 (split scheme); Lin & Dobriban 2021 JMLR (ANOVA of test error); Brennan 2001 (G-theory).

## Searched and not found (targeted, this turn)
No gait, plantar-pressure or GRF study was found that varies training participants and trials per participant jointly under participant-disjoint evaluation with equal-recording contrasts.
A BCI federated-learning paper (arXiv 2308.11636) reports trials per subject as the dominant factor in its setting; it is a different task and model class. Absence from a targeted search is not evidence of absence.

## Permitted wording now
"An empirical evaluation of the learning surface under restricted acquisition in two gait tasks, positioned against the subjects-versus-trials literature in biomechanics power analysis (Forrester 2015), neuroimaging (Chen 2022) and EEG deep learning (Seibel 2026)."
