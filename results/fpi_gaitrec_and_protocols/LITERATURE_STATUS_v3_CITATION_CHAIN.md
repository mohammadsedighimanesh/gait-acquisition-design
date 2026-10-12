# Literature Status v3: citation chain around Seibel 2026 and the gait search (2026-10-11)

Scope: targeted verification of the references Seibel et al. cite for the subjects-versus-data-per-subject question, plus two further targeted searches for gait studies that
vary trials/steps per participant jointly with participant count. Not a systematic review; no priority certificate.

## Verified citations (Crossref / publisher-indexed)
| Ref | Citation | Relevance | Verified |
|---|---|---|---|
| Viering & Loog 2023 | "The Shape of Learning Curves: A Review", IEEE TPAMI 45(6):7799–7819, DOI 10.1109/TPAMI.2022.3220744 | taxonomy of learning-curve models (POW3 etc.); we fit none, cite for terminology | DOI via search result; arXiv 2103.10948 |
| Kiessner et al. 2024 | "Reaching the ceiling? Empirical scaling behaviour for deep EEG pathology classification", Comput Biol Med 178:108681, DOI 10.1016/j.compbiomed.2024.108681 | subject-count scaling in EEG pathology; subjects only | Crossref |
| Śliwowski et al. 2023 | "Impact of dataset size and long-term ECoG-based BCI usage on deep learning decoders performance", Front Hum Neurosci 17:1111645, DOI 10.3389/fnhum.2023.1111645 | stagnating returns from more recording time (single subject BCI) | Crossref (thesis reference list) |
| Kapoor & Narayanan 2023 | "Leakage and the reproducibility crisis in machine-learning-based science", Patterns 4(9):100804, DOI 10.1016/j.patter.2023.100804 | subject-level leakage; motivates our exclusion proof and nested subsets | PMC/search |
| Hogan et al. 2025 | "Scaling convolutional neural networks achieves expert-level seizure detection in neonatal EEG", arXiv 2405.09911 (journal version listed by UCC, Dec 2025; journal DOI not retrieved) | gains from more data in a rare-event task (Seibel's counter-example) | arXiv only; **journal DOI to be confirmed before citing** |
| Seibel et al. 2026 | J Neural Eng 23:016030, DOI 10.1088/1741-2552/ae453d (full text read) | closest comparator | full text |
| Forrester 2015 | Int Biomech 2(1):62–72, DOI 10.1080/23335432.2015.1049296 | biomechanics subjects-vs-trials for power | Crossref |
| Chen et al. 2022 | NeuroImage 247:118786, DOI 10.1016/j.neuroimage.2021.118786 | trial/subject trade-off, fMRI | PMC |
| Figueroa et al. 2012 | BMC Med Inform Decis Mak 12:8, DOI 10.1186/1472-6947-12-8 | learning-curve weighting (used by Seibel) | PMC |
| Sheng, Provost & Ipeirotis 2008 | KDD 2008, "Get another label?" | repeated labels vs new examples under budget | NYU archive; **ACM DOI to be confirmed** |
| Slijepcevic et al. 2020 | Gait & Posture 76:198–203, DOI 10.1016/j.gaitpost.2019.10.021 | trial aggregation in gait classification | full text |
| Carvalho et al. 2026 | Gait & Posture 123:110020, DOI 10.1016/j.gaitpost.2025.110020 | strides needed for reliability | full PDF |
| Pe et al. 2026 | Int J Med Inform 216:106464, DOI 10.1016/j.ijmedinf.2026.106464 | split scheme and generalisability | PubMed |

Citing papers of Seibel 2026: none found by web search (the article is seven months old); Google Scholar "cited by" should be checked by the author at submission time.

## Gait-specific search (two further targeted queries, 2026-10-11)
No gait, plantar-pressure or GRF study was found that varies the number of trials/steps per participant jointly with participant count under participant-disjoint evaluation, or that
plots a learning curve against steps per participant. Found and relevant only as context: StepUP-P150 descriptor (many steps per person; no design study), a CoP gait-recognition
CNN with few-stride fine-tuning (identification task), and a 2026 Parkinson preprint stating that steps were never treated as independent observations. Absence from targeted searches
is not evidence of absence; a Scopus/Web of Science query by the author remains advisable.

## Permitted novelty wording (final)
"We replicate, in three gait tasks with separately recorded trials, the subjects-versus-data-per-subject learning-surface analysis previously reported for EEG (Seibel et al. 2026),
add pre-registered equal-recording-budget contrasts with participant-level uncertainty, show that both a people-dominated and a trial-relevant regime occur within one gait dataset,
and operationalise and evaluate the incremental-acquisition policy that Seibel et al. recommended but did not test; the policy improves on the five-trial convention but not on the best
fixed design." No "first" wording.
