#!/usr/bin/env bash
# Phase 1c -- strict twins of C2 (te_smooth=5, OOF 0.945225), the best single probe so far.
# Phase 1b left two brackets pointing the same way and three encoder fixes that may be
# redundant with each other. This round pushes both brackets past their edge and tests
# each encoder fix ON TOP of the new baseline, so redundancy shows up as a shrunken delta.
set -u
C2='{"run_tag":"C2","te_cols":["Annual_Income_USD","Daily_Commute_km","Age"],"te_smooth":5.0}'
G="GATE: +0.0001 OOF (interim, ~2.6x the 0.000038 seed-noise floor)."

run () {
  echo "################ $1 ################"
  python scripts/run_local.py --cfg "$2" --diff-vs "$C2" --description "$3" --notes "$4 $G" \
    2>&1 | grep -E "^(strict-twin|OOF AUC = .*folds|Appended)"
}

run D1 '{"run_tag":"D1","te_cols":["Annual_Income_USD","Daily_Commute_km","Age"],"te_smooth":2.0}' \
  "D1: te_smooth 5 -> 2" \
  "HYPOTHESIS: the smoothing bracket is monotone and has not turned over yet - 100 gave -0.000667, 20 is the B5 baseline, 5 gave +0.000487. MECHANISM: at ~50 rows per income value the per-value rate is already well estimated, so shrinkage is mostly discarding signal rather than controlling noise. Push until it turns over; the turnover point is the answer, a single tested point is not."

run D2 '{"run_tag":"D2","te_cols":["Annual_Income_USD","Daily_Commute_km","Age"],"te_smooth":5.0,"te_backoff":"neighborhood"}' \
  "D2: C2 + hierarchical neighborhood backoff (C1 stacked on C2)" \
  "HYPOTHESIS: C1 gave +0.000428 on top of B5 (smooth=20). MECHANISM: backoff target and smoothing STRENGTH are two ways to fix the same problem - an unreliable rare-value estimate. C2 already fixed part of it by shrinking less. If D2's delta over C2 is much smaller than C1's over B5, they are redundant and only one should ship; if it holds up, they address different parts of the rare-value tail and both belong."

run D3 '{"run_tag":"D3","te_cols":["Annual_Income_USD","Daily_Commute_km","Age"],"te_smooth":5.0,"te_count_feature":true}' \
  "D3: C2 + value-count feature (C4 stacked on C2)" \
  "HYPOTHESIS: C4 gave +0.000456 on top of B5. MECHANISM: as D2 - the count is a third route to the same rare-value problem, and it is the most general of the three because it lets the tree decide how to discount rather than fixing the discount in advance. Same redundancy test."

run D4 '{"run_tag":"D4","te_cols":["Annual_Income_USD","Daily_Commute_km","Age"],"te_smooth":5.0,"params":{"n_estimators":12000,"learning_rate":0.05,"num_leaves":7,"colsample_bytree":0.8,"subsample":0.8,"subsample_freq":1,"min_child_samples":20,"reg_lambda":0.0}}' \
  "D4: C2 + 7 leaves (C6 stacked on C2)" \
  "HYPOTHESIS: capacity is monotone downward - 255 leaves -0.000894, 63 baseline, 15 +0.000212, 7 +0.000268. MECHANISM: the generator is additive (all 28 pairwise interactions worth +0.000033), so leaves beyond what the per-feature shapes need can only fit interactions that do not exist. Unlike the encoder fixes this axis is ORTHOGONAL to the TE, so it should stack cleanly rather than redundantly."

run D5 '{"run_tag":"D5","te_cols":["Annual_Income_USD","Daily_Commute_km","Age"],"te_smooth":5.0,"params":{"n_estimators":25000,"learning_rate":0.05,"num_leaves":3,"colsample_bytree":0.8,"subsample":0.8,"subsample_freq":1,"min_child_samples":20,"reg_lambda":0.0}}' \
  "D5: C2 + 3 leaves (one step from stumps)" \
  "HYPOTHESIS: D4 pushed one step further, to find where the capacity bracket turns over. MECHANISM: num_leaves=3 permits a depth-2 interaction at most; num_leaves=2 would be stumps, i.e. a literal GAM with no interactions at all. If D5 >= D4 the additive reading is confirmed to its limit and stumps are the next probe."
