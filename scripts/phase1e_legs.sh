#!/usr/bin/env bash
# E4/E5 relaunch -- pool members on the E1 representation (the current champion:
# TE(income,commute,age), te_smooth=5, neighborhood backoff, 7 leaves).
# NOT ablations: a different learner changes many things at once by construction, so
# --diff-vs is deliberately unused and solo score is not a shipping decision.
set -u
TE='"te_cols":["Annual_Income_USD","Daily_Commute_km","Age"],"te_smooth":5.0,"te_backoff":"neighborhood"'
N="POOL MEMBER, judged by ADD/SWAP contribution to the stack, never by solo score - S6E8 measured solo deltas spanning 0.033 AUC against stack contributions spanning 0.00006."

python scripts/run_local.py \
  --cfg "{\"run_tag\":\"E4\",\"learner\":\"xgb\",$TE,\"params\":{\"n_estimators\":12000,\"learning_rate\":0.05,\"num_leaves\":7,\"colsample_bytree\":0.8,\"subsample\":0.8,\"subsample_freq\":1,\"min_child_samples\":20,\"reg_lambda\":1.0}}" \
  --description "E4: XGBoost leg on the E1 representation" \
  --notes "$N MECHANISM: same features, different split-finding and regularization, so its errors should point somewhere slightly different from LightGBM's. Both are leaf-wise here (grow_policy=lossguide, max_leaves=7), so expect HIGH correlation with E1 - this leg's value, if any, is in the residual disagreement a fitted stack can subtract, not in diversity of inductive bias." \
  2>&1 | grep --line-buffered -E "^(fold |OOF AUC = .*folds|Appended)"

python scripts/run_local.py \
  --cfg "{\"run_tag\":\"E5\",\"learner\":\"cat\",$TE,\"params\":{\"n_estimators\":3000,\"learning_rate\":0.08,\"reg_lambda\":3.0}}" \
  --description "E5: CatBoost leg on the E1 representation (symmetric trees)" \
  --notes "$N MECHANISM: CatBoost's oblivious trees apply the SAME split at every node of a level - a strong structural regularizer and a genuinely different inductive bias from leaf-wise LightGBM, which is the kind of difference a fitted stack can use to subtract correlated error. This is the most promising pool member for that reason. 3000 iterations at lr 0.08 for CPU wall-clock; depth is CatBoost's default 6, so unlike E1/E4 it is NOT capacity-matched - a confound accepted because this is a pool member, not a twin." \
  2>&1 | grep --line-buffered -E "^(fold |OOF AUC = .*folds|Appended)"
