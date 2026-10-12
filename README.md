# Participants or repeated trials? — study materials

Code, pre-execution protocols, derived results, verification records and manuscript files for:

> *Participants or repeated trials? Equal-budget learning surfaces and an evaluated sequential acquisition rule in three gait datasets.*

The study asks how a fixed number of gait recordings should be split between recruiting more people and repeating trials, for classifiers that must work on unseen people. It uses three public datasets (foot-posture classes, GaitRec diagnosis classes, StepUP-P150 sex and BMI category), learning surfaces over people × trials, equal-budget contrasts, a regularisation sensitivity analysis and a sequential acquisition rule. All analyses are exploratory secondary analyses.

## Contents

| Folder | What it holds |
|---|---|
| `code/` | Analysis scripts for every stage, independent verifiers (no refitting), the scripts that build the manuscript tables, figures and Word files, and the number audit |
| `results/fpi_gaitrec_and_protocols/` | Protocol (`PROTOCOL_v2_FROZEN.md`), addenda A1–A4 and amendments, frozen hash files, stage reports, and results for FPI and GaitRec (pilots, learning surfaces, sensitivity, sequential rule), including superseded and failed runs |
| `results/stepup/` | StepUP addenda results: feature-build records, surfaces, sensitivity, sequential rule (A5) and small-budget follow-up (A5.2), including failed attempts and download verification logs |
| `results/fpi_assessment_side_earlier/` | Earlier FPI analysis used in Supplementary S10 |
| `results/*_cohort_definition/` | Recorded splits, cohort audits and earlier inner-tuning records used to fix C = 0.1 |
| `manuscript/` | `Manuscript_final.docx`, `Supplementary_Information_final.docx`, PDF previews, Markdown sources, figures, tables, `claim_to_evidence_table.md`, `number_audit.txt` |
| `RELEASE_MANIFEST.json` | SHA-256 of every released file and the list of excluded files |

CSV files larger than 20 MB are stored as `.csv.gz`; decompress them before running a verifier.

## What is not included

- **Raw data.** Obtain the datasets from their sources (all CC BY 4.0):
  - Foot posture: Sanchis-Sales et al., *Sci Data* 2024, doi:10.1038/s41597-024-04166-3; data doi:10.5281/zenodo.10838067 (`DATA.mat`).
  - GaitRec: Horsak et al., *Sci Data* 2020, doi:10.1038/s41597-020-0481-z (`GRF_F_V_PRO_left.csv`, `GRF_F_V_PRO_right.csv`, `GRF_metadata.csv`).
  - StepUP-P150: Larracy et al., *Sci Data* 2025, doi:10.1038/s41597-025-05792-1; data doi:10.20383/103.01285 (`py/<id>/BF/W1/pipeline_2.npz`, `metadata.csv`, `participant_metadata.csv`). `code/download_stepup_third_task.py` downloads and checks the files against the official checksums.
- **Feature caches and fitted-model files** (about 600 MB of `.npz`). They are rebuilt by the scripts; the verifiers that replay saved models need them, so rerun a stage before verifying it.

## Reproducing

Python 3.11 with numpy 2.3, pandas, scikit-learn 1.7.2, scipy and matplotlib. Scripts expect the original project layout (`idea1_2026/...`, `dataset/DATA.mat`); place the raw files accordingly and run, in order:

1. Inputs: `corrected_raw_pipeline_diagnostic.py` and `learned_multitrial_aggregation_pilot.py` (FPI per-trial features), `gaitrec_trial_stability_pilot.py` (GaitRec bags), `restricted_acquisition_stepup.py build` (StepUP features).
2. Stages: `restricted_acquisition_pilot.py`, `restricted_acquisition_grid.py`, `restricted_acquisition_sensitivity.py`, `restricted_acquisition_gaitrec.py pilot|grid`, `restricted_acquisition_policy.py pilot|grid`, `restricted_acquisition_stepup.py pilot|surface L1|surface L2|sensitivity L1|policy L1|policy L2`, `restricted_acquisition_stepup_a52.py L1|L2`.
3. Each stage has a `verify_*.py` script. A pass certifies only the listed checks.
4. Manuscript: `build_manuscript_assets.py`, `build_manuscript_docx.py`, `audit_manuscript_numbers.py`.

Each stage script refuses to overwrite existing outputs and checks the SHA-256 of its protocol and inputs against the frozen values.

## Licence

Code: MIT (see `LICENSE`). Protocols, reports, derived tables and manuscript files: CC BY 4.0. Derived results include participant identifiers and labels from the public datasets above, which are themselves CC BY 4.0; please cite them.
