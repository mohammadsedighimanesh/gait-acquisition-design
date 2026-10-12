# Amendment A5.1 (mechanical, before any result was read; 2026-10-10)

The first pilot attempt (`stepup_third_task_v1/pilot_v1_FAILED_*`, preserved) aborted on an allocation assertion: with 150 people and 5 folds each training set has exactly 120
people, and a global largest-remainder allocation (61/59) is infeasible in folds whose training set has 60/60. Rule: the class allocation for every n is computed by largest
remainder on the current fold's training class counts (so n=120 is the whole training set); allocations are recorded per fold in progress.json and checked by the verifier.
Nothing else changed. The protocol is re-frozen with the new script hash; the aborted pilot's fits are not used.
