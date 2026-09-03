#!/usr/bin/env bash
# Phase 1d. E1-E3 are STRICT TWINS of D4 (OOF 0.945437) and close out the single-model
# recipe. E4/E5 are POOL MEMBERS, not ablations: a different learner changes many things
# at once by construction, so --diff-vs is deliberately not used and their solo score is
# not a shipping decision (playbook s6 - judge legs by stack contribution).
set -u
LOW='"n_estimators":12000,"learning_rate":0.05,"num_leaves":7,"colsample_bytree":0.8,"subsample":0.8,"subsample_freq":1,"min_child_samples":20,"reg_lambda":0.0'
TE='"te_cols":["Annual_Income_USD","Daily_Commute_km","Age"],"te_smooth":5.0'
D4="{\"run_tag\":\"D4\",$TE,\"params\":{$LOW}}"
G="GATE: +0.0001 OOF (interim, ~2.6x the 0.000038 seed-noise floor)."

twin () {
  echo "################ $1 ################"
  python scripts/run_local.py --cfg "$2" --diff-vs "$D4" --description "$3" --notes "$4 $G" \
    2>&1 | grep -E "^(strict-twin|OOF AUC = .*folds|Appended)"
}
leg () {
  echo "################ $1 ################"
  python scripts/run_local.py --cfg "$2" --description "$3" --notes "$4 $G" \
    2>&1 | grep -E "^(OOF AUC = .*folds|Appended)"
}

twin E1 "{\"run_tag\":\"E1\",$TE,\"te_backoff\":\"neighborhood\",\"params\":{$LOW}}" \
  "E1: D4 + neighborhood backoff (the two winning axes combined)" \
  "HYPOTHESIS: this should stack close to additively, unlike C1/C2/C4 which did not. MECHANISM: the encoder axis (what a rare value shrinks toward) and the capacity axis (how many interactions the tree may fit) address different error sources - encoding bias vs interaction variance - so unlike the three encoder fixes they are not three routes to one problem. If the combined delta is materially less than +0.000177+0.000212, that reading is wrong."

twin E2 "{\"run_tag\":\"E2\",\"te_cols\":[\"Annual_Income_USD\",\"Daily_Commute_km\",\"Age\"],\"te_smooth\":2.0,\"params\":{$LOW}}" \
  "E2: D4 with te_smooth 5 -> 2" \
  "HYPOTHESIS: D1 found +0.000123 for this step at 63 leaves. MECHANISM: at lower capacity the tree can no longer partially correct a noisy encoding by splitting around it, so the optimal smoothing may be HIGHER, not lower, than it was at 63 leaves. This probe tests whether the smoothing optimum moved when capacity changed - if it did, the two axes are not separable and E1's premise is weakened."

twin E3 "{\"run_tag\":\"E3\",$TE,\"te_inner_repeats\":4,\"params\":{$LOW}}" \
  "E3: average the training-row TE over 4 independent inner splits" \
  "HYPOTHESIS: the headline variance-reduction probe. MECHANISM: a training row's TE is built from 4/5 of the fold, and WHICH 4/5 is an arbitrary draw that injects noise into the feature the model leans on hardest. Averaging over 4 draws cancels it without weakening the leakage guarantee - every draw still excludes the row's own label. If the competition is now a variance problem rather than a signal problem, this is where it shows."

leg E4 "{\"run_tag\":\"E4\",\"learner\":\"xgb\",$TE,\"params\":{$LOW}}" \
  "E4: XGBoost leg on the D4 representation" \
  "POOL MEMBER, not an ablation. MECHANISM: same features, different split-finding and regularization, so its errors should point somewhere slightly different from LightGBM's. Judged by ADD/SWAP contribution to the stack, never by solo score - S6E8 measured solo deltas spanning 0.033 AUC while the corresponding stack contributions spanned 0.00006."

leg E5 "{\"run_tag\":\"E5\",\"learner\":\"cat\",$TE,\"params\":{\"n_estimators\":4000,\"learning_rate\":0.05,\"reg_lambda\":3.0,\"num_leaves\":7,\"colsample_bytree\":0.8,\"subsample\":0.8,\"subsample_freq\":1,\"min_child_samples\":20}}" \
  "E5: CatBoost leg on the D4 representation (symmetric trees)" \
  "POOL MEMBER, not an ablation. MECHANISM: CatBoost's oblivious/symmetric trees apply the SAME split at every node of a level, which is a strong structural regularizer and a genuinely different inductive bias from leaf-wise LightGBM - exactly the kind of difference that can subtract correlated error in a fitted stack. Iterations reduced to 4000 for CPU wall-clock; it is a pool member so this is not a twin violation."
