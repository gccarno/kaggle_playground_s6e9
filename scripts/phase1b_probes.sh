#!/usr/bin/env bash
# Phase 1b -- squeeze the lookup. Every probe is a STRICT TWIN of B5 (the new champion:
# per-value TE of income/commute/age, OOF 0.944738). Gate: +0.0001 OOF.
set -u
B5='{"run_tag":"B5","te_cols":["Annual_Income_USD","Daily_Commute_km","Age"]}'
G="GATE: +0.0001 OOF (interim, ~2.6x the 0.000038 seed-noise floor)."

run () {
  echo "################ $1 ################"
  python scripts/run_local.py --cfg "$2" --diff-vs "$B5" --description "$3" --notes "$4 $G" \
    2>&1 | grep -E "^(strict-twin|OOF AUC = .*folds|Appended)"
}

run C1 '{"run_tag":"C1","te_cols":["Annual_Income_USD","Daily_Commute_km","Age"],"te_backoff":"neighborhood"}' \
  "C1: hierarchical TE - rare values back off to their income neighborhood, not the global prior" \
  "HYPOTHESIS: the biggest single win available on top of B5. MECHANISM: income carries a REAL monotone trend underneath the lookup (Spearman(value, per-value rate)=0.68). B5 shrinks a value seen 3 times toward the global prior 0.1746, discarding that trend entirely; a value at income 160k seen 3 times should shrink toward the ~0.34 rate of its neighbourhood instead. With 13,214 values over 668k rows the tail of rare values is where most of the encoding error lives, so fixing the backoff target should be worth more than fixing the smoothing strength."

run C2 '{"run_tag":"C2","te_cols":["Annual_Income_USD","Daily_Commute_km","Age"],"te_smooth":5.0}' \
  "C2: te_smooth 20 -> 5 (trust rare values more)" \
  "HYPOTHESIS: B5's smoothing was an untested default. MECHANISM: smoothing trades lookup fidelity against noise in rare values. If the lookup is strong and the per-value counts are large (mean ~50 rows/value), less smoothing should recover more of it. Paired with C3 this brackets the optimum rather than testing one arbitrary point."

run C3 '{"run_tag":"C3","te_cols":["Annual_Income_USD","Daily_Commute_km","Age"],"te_smooth":100.0}' \
  "C3: te_smooth 20 -> 100 (trust rare values less)" \
  "HYPOTHESIS: the other bracket of C2. MECHANISM: if C3 beats B5 the encoding is currently overfitting rare values; if C2 beats B5 it is over-shrinking them. Both losing means 20 was already near-optimal and the smoothing axis is closed."

run C4 '{"run_tag":"C4","te_cols":["Annual_Income_USD","Daily_Commute_km","Age"],"te_count_feature":true}' \
  "C4: add log1p(train-fold value count) alongside each TE column" \
  "HYPOTHESIS: the model cannot currently tell a TE built from 1,000 rows from one built from 3. MECHANISM: the count is the encoding's own reliability, so handing it over lets the tree discount unreliable TE values in exactly the region where the encoding is worst. This is the cheap alternative to fixing the backoff (C1) and the two may be redundant - if both win, test them together before shipping both."

run C5 '{"run_tag":"C5","te_cols":["Annual_Income_USD","Daily_Commute_km","Age"],"params":{"n_estimators":6000,"learning_rate":0.05,"num_leaves":15,"colsample_bytree":0.8,"subsample":0.8,"subsample_freq":1,"min_child_samples":20,"reg_lambda":0.0}}' \
  "C5: capacity 63 -> 15 leaves" \
  "HYPOTHESIS: LOWER capacity should win, which is the opposite of the usual reflex. MECHANISM: two measurements point the same way - the GLM probe found all 28 pairwise interactions worth +0.000033, and B7 found that RAISING capacity to 255 leaves cost -0.000894. With a genuinely additive generator, extra leaves can only fit interactions that do not exist, which is pure variance. Now that TE has moved the per-value signal into the feature itself, the tree needs even less depth to reach it."

run C6 '{"run_tag":"C6","te_cols":["Annual_Income_USD","Daily_Commute_km","Age"],"params":{"n_estimators":12000,"learning_rate":0.05,"num_leaves":7,"colsample_bytree":0.8,"subsample":0.8,"subsample_freq":1,"min_child_samples":20,"reg_lambda":0.0}}' \
  "C6: capacity 63 -> 7 leaves" \
  "HYPOTHESIS: C5 pushed further, to bracket the capacity optimum rather than testing one point. MECHANISM: as C5. If C6 > C5 > B5 the trend is monotone toward an additive model and the next probe is stumps (num_leaves=2), which is a GAM."

run C7 '{"run_tag":"C7","te_cols":["Annual_Income_USD","Daily_Commute_km","Age","Charging_Stations_Near_Home","Charging_Stations_Near_Work","Number_of_Cars_Owned","Environmental_Concern_Level"]}' \
  "C7: extend TE to all 7 numeric columns" \
  "HYPOTHESIS: expected null, run as a control. MECHANISM: the 4 added columns have 4-20 distinct values each, and LightGBM's 255-bin histogram already resolves them exactly - there is no lookup structure a TE can reveal that a split cannot already reach. If this WINS, the mechanism is not resolution but something else (e.g. TE giving the tree a pre-ordered axis), and that would change what C5/C6 mean."
