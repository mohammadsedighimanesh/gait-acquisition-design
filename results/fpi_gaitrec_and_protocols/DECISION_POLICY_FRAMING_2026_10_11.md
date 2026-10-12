# Decision: how the sequential policy enters the manuscript (2026-10-11)

Options considered: (a) keep as a secondary section with an honest negative/limitation result; (b) drop it to sharpen the paper.

Decision: **(a) keep as a secondary section**, for three reasons.
1. It answers a recommendation made explicitly in the closest comparator (Seibel et al. 2026, Discussion 4.4: "incremental data collection strategies combined with periodic model
   evaluations") with a pre-registered implementation and a clear outcome; omitting it would hide the one element that goes beyond replication.
2. The negative part is informative for practitioners: with the simplest online estimator, exploration cost is visible at small budgets and the trial-relevant regime is not exploited,
   so the practical recommendation stays "recruit people; stop at about three trials" rather than "run an adaptive scheme".
3. All runs, amendments and failed attempts are preserved and verified; the section can be written from source-linked tables with no new analysis.

Manuscript placement and wording
- Results, final subsection: "An online allocation rule improves on the five-trial convention but not on the best fixed design" (FPI, GaitRec, StepUP; budgets large and small).
- Methods: the MARGIN rule in one paragraph; A4.1/A4.2/A5.2.x mechanical rules in supplement.
- Discussion: state plainly that the policy approximately rediscovers "one trial per person" where people dominate and fails to find the trial-heavy design in the low-signal regime;
  list the estimator noise (undefined estimates 15–25% at B<=150) as the identified cause; propose better estimators as future work without running them here.
- Title/abstract: the policy is mentioned as an evaluated recommendation, not as a contribution of a new algorithm.

Not allowed: any further estimator variant on these data without a new pre-registered addendum; any wording implying the policy is superior to fixed designs.
