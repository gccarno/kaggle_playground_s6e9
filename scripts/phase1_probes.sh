#!/usr/bin/env bash
# Phase 1 -- response-shape probes. Every one is a STRICT TWIN of A0: exactly one
# config field differs, enforced by --diff-vs. Gate: +0.0001 OOF (README section 4).
set -u
G="GATE: +0.0001 OOF (interim, ~2.6x the 0.000038 seed-noise floor)."

run () {  # run <tag> <cfg-json> <description> <notes>
  echo "################ $1 ################"
  python scripts/run_local.py --cfg "$2" --diff-vs '{"run_tag":"A0"}' \
    --description "$3" --notes "$4 $G" 2>&1 | grep -E "^(strict-twin|fold |OOF AUC = .*folds|Appended)"
}

run B1 '{"run_tag":"B1","fe_clip_flags":true}' \
  "B1: censoring indicators for the income=30000 and commute=5.0 floors" \
  "HYPOTHESIS: both numerics are censored at their floor (9.2% and 21.6% of rows sit exactly there), so each floor value mixes two populations - the genuinely-low and the clipped. MECHANISM: a tree can only separate them with an exact-equality split it has no reason to find; an explicit indicator hands it the partition. This is what makes the income logit curve non-monotone at its bottom decile."

run B2 '{"run_tag":"B2","fe_log_income":true}' \
  "B2: log(Annual_Income_USD) alongside the raw column" \
  "HYPOTHESIS: none, expected null. MECHANISM: trees are invariant to monotone transforms of a split variable, so log-income can only help via the histogram binning (equal-width bins in log space resolve the dense low end more finely). Run as a deliberate negative control: if this moves, the binning is doing more work than assumed."

run B3 '{"run_tag":"B3","drop_noise":true}' \
  "B3: drop the 4 flat-in-logit columns (Gender, Number_of_Cars_Owned, both Charging_Stations_*)" \
  "HYPOTHESIS: these 4 are generator noise - per-value logits are flat to within +/-0.1 and single-feature AUCs are 0.505-0.524. MECHANISM: if noise, removing them is variance reduction (fewer spurious splits, less colsample dilution of the real drivers), not a wash. If OOF DROPS, they carry conditional signal and the flat marginal was hiding it."

run B4 '{"run_tag":"B4","te_cols":["Annual_Income_USD"]}' \
  "B4: per-value target encoding of Annual_Income_USD (13,214 values), fold-fit" \
  "HYPOTHESIS: TE was S6E8's single largest feature win. MECHANISM: it hands the tree a column already monotone in log-odds, so one split captures what otherwise needs many. BUT S6E8's data was a value->target lookup table and this one is smooth-monotone in income, so the prior here is weaker than it was there. SUPERVISED transform - fit on the train fold only, training rows encoded via an inner 5-fold."

run B5 '{"run_tag":"B5","te_cols":["Annual_Income_USD","Daily_Commute_km","Age"]}' \
  "B5: per-value target encoding of the 3 numerics (income, commute, age)" \
  "HYPOTHESIS: B4 extended. MECHANISM: Age is non-monotone (rate dips at 29-34, peaks at 51-56) and commute is non-monotone with a clipped floor, so both are shapes a TE recovers directly and a tree must spend depth on. If B5 > B4, the win is coming from the NON-monotone columns, which is the mechanism worth knowing."

run B6 '{"run_tag":"B6","monotone_income":true}' \
  "B6: monotone-increasing constraint on Annual_Income_USD" \
  "HYPOTHESIS: the income->target relation is monotone increasing everywhere except the clipped bottom decile, which is an artifact rather than signal. MECHANISM: the constraint is a regularizer - it forbids the tree from fitting the non-monotonic wiggle, which should be noise. If it LOSES, the bottom-decile non-monotonicity is real signal and B1's clip indicator is the better tool for it."

run B7 '{"run_tag":"B7","params":{"n_estimators":6000,"learning_rate":0.02,"num_leaves":255,"colsample_bytree":0.8,"subsample":0.8,"subsample_freq":1,"min_child_samples":100,"reg_lambda":1.0}}' \
  "B7: more capacity, slower learning rate (255 leaves, lr 0.02, min_child 100)" \
  "HYPOTHESIS: A0's parameters were never tuned - they were a first guess, and it early-stopped at 213-381 iterations, which is fast enough to suggest it saturates before exhausting the data. MECHANISM: 668k rows can support far more capacity than 63 leaves; a lower LR with more regularization should reach a finer per-feature shape. This sizes how much of the 0.0048 gap to the public top is plain tuning."
