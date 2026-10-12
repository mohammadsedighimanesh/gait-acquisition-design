# Literature Status v2: Seibel et al. 2026 read in full (2026-10-10)

Source: `PaperForWritting/how much EEG is needed.pdf` (J. Neural Eng. 23:016030, open access CC BY 4.0, published 2026-02-25, DOI 10.1088/1741-2552/ae453d), 20 pages, read in full.
Code archive `dataset/Zenodo.zip` (SHA256 fdb38637...be150; Zenodo 10.5281/zenodo.18403060): README and split/learning-curve scripts inspected as data in an isolated scratch folder; nothing executed.
The earlier LITERATURE_STATUS file is superseded for Seibel only.

## What Seibel et al. did (verified from the full text)
- Three binary EEG tasks (sex, TUEG 13,307 subjects; normal/abnormal, TUAB 2,294; eyes open/closed, TDBRAIN 1,254), three CNNs.
- Varied the number of training subjects (16 to 10,240) **and** the number of EEG segments per subject (segments = random few-second windows cut from one recording), reported as
  accuracy heat maps (their figure 3). Conclusion: more subjects always helped; no substantial gain beyond four segments per subject; segment benefit shrinks as subjects grow.
- Nested, leakage-aware subsampling ("imaginary institution" sees only a visible subset; smaller subsets drawn from it) (their figure 1).
- Eight parametric learning-curve models in the number of subjects only; POW3 best; extrapolation reliable only with several hundred subjects.
- Variance decomposition of accuracy into sample-selection vs seed (selection 1.46x larger at the largest N).
- A separate experiment: a second recording (different date) from the same subjects versus recordings from new subjects; new subjects gave larger gains (their figure 4).
- Discussion 4.4 **recommends** "incremental data collection strategies combined with periodic model evaluations" but does not implement or evaluate any such policy.

## Exact relation to our study (what may and may not be claimed)
| Our element | Seibel 2026 | Honest statement |
|---|---|---|
| Learning surface over (people, trials per person) with nested subsets | Yes, for (subjects, segments per subject) in EEG CNNs | We **replicate their design question in gait**; the surface itself is not new. Our "trials" are separately recorded gait trials (independent walks), their "segments" are windows of one recording; this is a different measurement structure and must be said explicitly |
| Restricted acquisition (unacquired trials excluded from every learned step, exclusion proof) | Subsets are drawn before training; preprocessing is per-segment, so exclusion is implicit | Not a differentiator; we only make the guarantee explicit and verified |
| Exact equal-recording paired contrasts (35x3 vs 21x5; 1000x3 vs 600x5; 500x3 vs 300x5) | No equal-budget contrast; heat map cells compared informally | Modest methodological addition: a pre-registered equal-budget paired test with person-level CI |
| Within-repeat vs between-repeat variance of the surface; person-bootstrap with shared weights | Selection vs seed variance decomposition | Comparable in spirit; ours is at the participant level with paired designs |
| Finite-k saturation (3 to 5 trials adds nothing) | Four segments saturate | Same qualitative finding in a different modality; this is confirmation, not discovery |
| **Sequential acquisition policy (MARGIN) evaluated against the convention, random, and fixed designs on real recordings in two tasks** | **Recommended in Discussion 4.4, not implemented or evaluated** | This is the one element with a defensible algorithmic contribution relative to Seibel: an operational online rule with pre-registered evaluation. It still must be positioned against Sheng et al. 2008 (repeated labels vs new examples) and cost-aware sequential design (Pauwels 2014) |
| Learning-curve extrapolation | Central to Seibel | We deliberately do not extrapolate; nothing to claim |
| Second-session vs new-subject | Studied by Seibel | We have no second-session FPI data; GaitRec longitudinal data exist but were not used for this question |

## Consequences for the manuscript
1. Title/abstract must frame the work as: (a) replication of the subjects-versus-repeats question in gait with separately recorded trials, (b) equal-budget pre-registered contrasts,
   (c) an online acquisition policy that operationalises the "incremental collection with periodic evaluation" recommendation and is evaluated in two tasks (third task pending).
2. Remove any wording implying that the people-versus-trials surface is new; cite Seibel 2026 (EEG), Chen 2022 (fMRI), Forrester 2015 (biomechanics power).
3. The policy's limitation stays: in the two tasks so far it approximately rediscovers "one trial per person" and does not beat the best fixed rule; the third task (StepUP, up to 20 trials per person) is where a different regime could appear. Seibel's own hint: tasks where the informative pattern is present in few segments (rare events) may favour more segments.
4. Citation chaining still to do: Viering & Loog 2022 (learning-curve review), Kiessner et al. 2024 (EEG pathology scaling), Hogan et al. 2025 (seizure detection, gains from longer recordings), Śliwowski et al. 2023 (ECoG BCI stagnation), Kapoor & Narayanan 2023 (leakage).
