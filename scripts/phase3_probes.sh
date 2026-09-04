#!/usr/bin/env bash
# Phase 3. I1/I2 are STRICT TWINS chained one field at a time: I1 against E1
# (OOF 0.945642), I2 against I1 -- NOT against E1, which would differ in two fields and
# be refused by the guard, correctly.
#
# THE HYPOTHESIS. Three independent measurements say the generator is additive in
# log-odds: the GLM (all 28 pairwise interactions among the 8 strong drivers = +0.000033),
# the joint-key residual test (8 of 9 keys at 0.85-0.95), and H2/H3 (handing the tree five
# pre-computed interactions made it WORSE, by displacement, best_iter 998 -> 333). Yet the
# champion is a 7-leaf tree that CAN build 6-way interactions and spends ~1000 rounds
# doing so. Forbidding interactions outright deletes variance the additivity finding
# predicts is pure noise.
#
# THE PAIR IS THE POINT. Capacity was cut 63 -> 7 (B7: 255 leaves cost 0.000894) BECAUSE
# interactions were harmful. Once interactions are forbidden, depth no longer buys
# interaction -- it buys per-feature SHAPE RESOLUTION, which is the one thing this data
# rewards. So the two knobs should stop being independent, and I2 should beat I1.
set -u
LOW='"n_estimators":12000,"learning_rate":0.05,"num_leaves":7,"colsample_bytree":0.8,"subsample":0.8,"subsample_freq":1,"min_child_samples":20,"reg_lambda":0.0'
HIGH='"n_estimators":12000,"learning_rate":0.05,"num_leaves":63,"colsample_bytree":0.8,"subsample":0.8,"subsample_freq":1,"min_child_samples":20,"reg_lambda":0.0'
TE='"te_cols":["Annual_Income_USD","Daily_Commute_km","Age"],"te_smooth":5.0'
E1="{\"run_tag\":\"E1\",$TE,\"te_backoff\":\"neighborhood\",\"params\":{$LOW}}"
I1="{\"run_tag\":\"I1\",$TE,\"te_backoff\":\"neighborhood\",\"additive_only\":true,\"params\":{$LOW}}"
G="GATE: +0.0001 OOF solo (interim, ~2.6x the 0.000038 seed-noise floor), then ADD and SWAP at constant pool size - only the second is a shipping decision."

twin () {
  echo "################ $1 ################"
  python scripts/run_local.py --cfg "$2" --diff-vs "$3" --description "$4" --notes "$5 $G" \
    2>&1 | grep -E "^(strict-twin|OOF AUC = .*folds|Appended)"
}

twin I1 "$I1" "$E1" \
  "I1: E1 with ALL feature interactions forbidden (a boosted GAM)" \
  "HYPOTHESIS: >= E1. MECHANISM: the generator is measured additive three independent ways, so every interaction the 7-leaf tree fits is noise it is paying variance for. Forbidding them is a pure variance cut with no bias cost -- IF the additivity finding is right. If I1 loses materially, real interactions survive at 7 leaves and the additive story has a crack, which is worth as much as the win."

twin I2 "{\"run_tag\":\"I2\",$TE,\"te_backoff\":\"neighborhood\",\"additive_only\":true,\"params\":{$HIGH}}" "$I1" \
  "I2: I1 with capacity restored to 63 leaves" \
  "HYPOTHESIS: I2 > I1, recovering something like B7's -0.000894. MECHANISM: capacity was cut 63 -> 7 because capacity BOUGHT INTERACTIONS and interactions are absent. With interactions forbidden by construction, num_leaves buys only per-feature shape resolution, which is exactly the lever this data rewards (B4's per-value TE was +0.003017). If the capacity and additivity axes do NOT interact this way, the mechanism attributed to B7 is wrong."
