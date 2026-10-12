# Amendment A4.1 (mechanical, before the grid; 2026-10-10)

The first pilot (`policy_pilot_v1_before_A4.1/`, preserved) passed all mechanical checks but emitted NumPy "Mean of empty slice" warnings inside the MARGIN policy's internal
estimates: with very small acquired sets (warm start of 7 FPI people) an inner-validation fold can lack a class, and the internal balanced accuracy became NaN; NaN comparisons
silently resolved to the "trials" action. This is a definitional defect in the internal estimator, not a performance-driven change.

Fix (two parts): (a) the internal balanced accuracy is defined over the classes present in the inner-validation fold (absent classes skipped);
(b) when an internal gain is still undefined (a class has too few acquired people for any inner-train fold to contain all classes, which happens only at the FPI warm start of
7 people), the decision defaults to "people", because recruiting is the only action that can make the estimate defined; such decisions are flagged `undefined_estimate=True`
in the decision log and counted in the report. The second pilot run (`policy_pilot_v1_before_A4.1b/`, preserved) showed that part (a) alone left this case silently resolving
to "trials" through NaN comparisons. NumPy/sklearn small-fold warnings are silenced inside the internal estimator only. Nothing else changed: policies, budgets, batches,
seeds, warm start, endpoints, bootstrap and final-model pipeline are identical to A4. The pilot is rerun under the amended script and the protocol is re-frozen with the new
script hash. Pilot performance numbers from either run are not used for any design decision.

# Amendment A4.2 (mechanical, after the first grid attempt; 2026-10-10)

The first full-grid attempt (`policy_grid_v1_FAILED_20261010_173839/`, preserved with FAILURE.json) stopped at FPI repeat 0, fold 2, seed 14 because a final acquired set
contained no person of one class, so the classifier assertion on the class list failed. With class-blind recruitment and small budgets (B=70 under CONV5 recruits only 14 people)
this is a genuine property of the convention, not a defect of the data. Rule added: a model fitted on an acquired set lacking a class predicts probability 0 for that class
(clipped to 1e-15 in log loss) and that class is counted as unrecognised in balanced accuracy; such fits are flagged (`model_missing_class`) and counted in progress.json.
No policy, budget, seed or endpoint changed. The pilot is rerun under the amended script and the protocol re-frozen with the new script hash.
