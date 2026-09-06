#!/usr/bin/env python3
"""SHAP diagnostic on the champion recipe (E1), fold 0 only.

Not a probe in the run_local.py sense -- this doesn't touch OOF/LB, it is a read-only
lens on a model already fully characterized by the aggregate structural tests (GLM
interactions, monotone-fit residuals, joint-key residual-variance ratios). The question
here is whether SHAP's per-row, per-feature view surfaces anything those aggregate tests
average away: a shape irregularity in a column that never got its own encoder, or an
interaction those tests didn't happen to key on.

Fits ONE fold (fold 0) of E1's exact recipe (see scripts/phase3_probes.sh for the
reconstructed cfg, itself confirmed exact via README Phase 4 -- G1r reproduced G1
bit-for-bit using the same reconstruction method for its own baseline). Reuses
src/pipeline.py's base_features/apply_target_encoding so the design matrix is IDENTICAL
to what run_local.py would build, not a re-implementation that could drift.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
import pipeline as pl  # noqa: E402

CFG = {
    **pl.DEFAULTS,
    "run_tag": "E1",
    "te_cols": ["Annual_Income_USD", "Daily_Commute_km", "Age"],
    "te_smooth": 5.0,
    "te_backoff": "neighborhood",
    "params": {**pl.DEFAULTS["params"], "n_estimators": 12000, "learning_rate": 0.05,
               "num_leaves": 7, "colsample_bytree": 0.8, "subsample": 0.8,
               "subsample_freq": 1, "min_child_samples": 20, "reg_lambda": 0.0},
}


def main():
    import lightgbm as lgb
    import shap

    train = pd.read_csv(pl.DATA_DIR / "train.csv")
    test = pd.read_csv(pl.DATA_DIR / "test.csv")
    y = (train[pl.TARGET] == "Yes").astype(int).values

    train, test, feats_num, feats_cat = pl.base_features(train, test, CFG)
    feats = feats_num + feats_cat

    skf = StratifiedKFold(CFG["n_folds"], shuffle=True, random_state=CFG["cv_seed"])
    i, j = next(iter(skf.split(train[feats], y)))
    Xtr, Xva = train[feats].iloc[i].copy(), train[feats].iloc[j].copy()
    Xte = test[feats].copy()
    ytr, yva = y[i], y[j]

    pl.apply_target_encoding(Xtr, ytr, Xva, Xte, CFG["te_cols"], CFG, CFG["cv_seed"])

    m = lgb.LGBMClassifier(random_state=CFG["model_seed"], n_jobs=-1, verbose=-1,
                            **CFG["params"])
    m.fit(Xtr, ytr, eval_X=Xva, eval_y=yva, eval_metric="auc",
          callbacks=[lgb.early_stopping(CFG["early_stopping_rounds"], verbose=False)])
    print(f"fold 0 fit: best_iter={m.best_iteration_}  n_features={Xtr.shape[1]}")
    print(f"columns: {list(Xtr.columns)}")

    explainer = shap.TreeExplainer(m)
    sv = explainer.shap_values(Xva)
    if isinstance(sv, list):   # older shap API returns [neg_class, pos_class]
        sv = sv[1]
    sv = np.asarray(sv)
    print(f"shap_values shape: {sv.shape}")

    mean_abs = pd.Series(np.abs(sv).mean(0), index=Xva.columns).sort_values(ascending=False)
    print("\n=== mean |SHAP| per feature (fold-0 validation set) ===")
    print(mean_abs.to_string())

    print("\n=== per-feature: Spearman(raw value, shap value) -- shape check ===")
    for c in Xva.columns:
        col = Xva[c]
        if col.dtype.name == "category":
            col = col.cat.codes
        rho = pd.Series(col.values).corr(pd.Series(sv[:, Xva.columns.get_loc(c)]), method="spearman")
        print(f"  {c:32s} spearman={rho: .4f}  mean|shap|={mean_abs[c]:.5f}")

    # Binned dependence for the "noise" columns and Age -- text dependence plot: does the
    # SHAP contribution move with the raw value in a way the "flat in logit" GLM screen
    # (which assumes near-linearity) could have missed?
    print("\n=== binned SHAP dependence: noise candidates + Age ===")
    for c in pl.NOISE_CANDIDATES + ["Age"]:
        if c not in Xva.columns:
            continue
        k = Xva.columns.get_loc(c)
        vals = Xva[c]
        if vals.dtype.name == "category":
            vals = vals.astype(str)
        df = pd.DataFrame({"v": vals.values, "shap": sv[:, k]})
        agg = df.groupby("v")["shap"].agg(["mean", "count"]).sort_index()
        print(f"\n  -- {c} --")
        print(agg.to_string())

    # SHAP interaction values on a sample: cheap sanity check against the additive-only
    # finding (Phase 3b) and the one borderline joint-key ratio (income x concern, 1.074).
    rng = np.random.default_rng(0)
    samp = rng.choice(len(Xva), size=min(3000, len(Xva)), replace=False)
    print(f"\n=== SHAP interaction values on a {len(samp)}-row sample ===")
    Xva_num = Xva.copy()
    for c in Xva_num.columns:
        if Xva_num[c].dtype.name == "category":
            Xva_num[c] = Xva_num[c].cat.codes.astype(np.float32)
    iv = explainer.shap_interaction_values(Xva_num.iloc[samp])
    if isinstance(iv, list):
        iv = iv[1]
    iv = np.asarray(iv)
    n_feat = iv.shape[1]
    mean_abs_inter = np.abs(iv).mean(0)
    np.fill_diagonal(mean_abs_inter, 0.0)   # zero out main effects on the diagonal
    cols = list(Xva.columns)
    pairs = []
    for a in range(n_feat):
        for b in range(a + 1, n_feat):
            pairs.append((cols[a], cols[b], mean_abs_inter[a, b] * 2))  # symmetric halves
    pairs.sort(key=lambda t: -t[2])
    print("\ntop 15 interaction pairs by mean |SHAP interaction|:")
    for a, b, val in pairs[:15]:
        print(f"  {a:28s} x {b:28s} {val:.6f}")


if __name__ == "__main__":
    main()
