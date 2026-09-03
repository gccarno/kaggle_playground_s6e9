#!/usr/bin/env python3
"""S6E9 model pipeline -- the single source of truth.

Every probe is a CONFIG OVERRIDE, never an edit to this file's body. Overrides arrive
as JSON in the S6E9_CFG env var; the output directory arrives in S6E9_OUT. Both have
sane defaults so `python src/pipeline.py` runs the champion recipe standalone.

Emits, into S6E9_OUT:
    oof_proba_<learner>.csv    id, fold, proba   (aligned to the frozen split)
    test_proba_<learner>.csv   id, proba
    submission.csv             id, Will_Buy_EV
and prints one final `RUN_METRICS_JSON:{...}` line, which is the parsing contract with
scripts/run_local.py and scripts/collect_run.py.

Leakage rule (README.md section 3): every supervised transform is fit on the training
fold ONLY. Target encoding additionally uses an inner K-fold within the training fold to
produce the training rows' own encodings, so the learner never sees a value's encoding
computed from that same row's label.
"""
import json, os, sys, time, warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------- frozen constants
# README.md section 2. These are the contract. Do not change them.
SEED = 42
CV_SEED = 42
N_FOLDS = 5
TARGET = "Will_Buy_EV"
ID = "id"

ON_KAGGLE = Path("/kaggle/input").exists()
REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = (Path("/kaggle/input/playground-series-s6e9") if ON_KAGGLE
            else REPO_ROOT / "data")
OUT_DIR = Path(os.environ.get("S6E9_OUT", "/kaggle/working" if ON_KAGGLE else
                              REPO_ROOT / ".kaggle_output" / "scratch"))

# Measured flat-in-logit in Phase 0 (README.md section 6); candidates for removal, but
# removal is a PROBE (`drop_noise`), not an assumption baked into the pipeline.
NOISE_CANDIDATES = ["Gender", "Number_of_Cars_Owned",
                    "Charging_Stations_Near_Home", "Charging_Stations_Near_Work"]

RAW_CAT = ["Gender", "City_Type", "Current_Car_Type",
           "Home_Charging_Possible", "Subsidy_Available", "Range_Anxiety_Level"]
RAW_NUM = ["Age", "Annual_Income_USD", "Daily_Commute_km", "Number_of_Cars_Owned",
           "Charging_Stations_Near_Home", "Charging_Stations_Near_Work",
           "Environmental_Concern_Level"]

# ---------------------------------------------------------------------- config
DEFAULTS = {
    "run_tag": "baseline",
    "learner": "lgb",                # lgb | xgb | cat | glm
    "model_seed": SEED,
    "n_folds": N_FOLDS,
    "cv_seed": CV_SEED,

    # -- feature engineering, each a single independently-togglable field so that
    #    run_local.py --diff-vs can enforce strict-twin ablations.
    "drop_noise": False,             # drop NOISE_CANDIDATES
    "fe_clip_flags": False,          # income==30000 / commute==5.0 censoring indicators
    "fe_log_income": False,          # log(Annual_Income_USD) alongside the raw column
    "te_cols": [],                   # SUPERVISED per-value target encoding (fold-fit)
    "te_smooth": 20.0,               # additive smoothing toward the backoff target
    "te_inner_folds": 5,             # inner K-fold used for the training rows' own TE
    "te_inner_repeats": 1,           # average the inner-fold TE over this many splits
    "te_backoff": "prior",           # "prior" | "neighborhood" -- what a rare value falls back to
    "te_backoff_bins": 200,          # quantile bins defining a neighborhood (fold-fit)
    "te_backoff_smooth": 50.0,       # smoothing of the neighborhood level toward the prior
    "te_count_feature": False,       # also emit log1p(train-fold count) per encoded column
    "monotone_income": False,        # monotone increasing constraint on income

    "params": {
        "n_estimators": 3000, "learning_rate": 0.05, "num_leaves": 63,
        "colsample_bytree": 0.8, "subsample": 0.8, "subsample_freq": 1,
        "min_child_samples": 20, "reg_lambda": 0.0,
    },
    "early_stopping_rounds": 100,
}


def load_cfg():
    cfg = {**DEFAULTS, **json.loads(os.environ.get("S6E9_CFG", "{}"))}
    cfg["params"] = {**DEFAULTS["params"], **cfg.get("params", {})}
    return cfg


# ------------------------------------------------------------------ feature build
def base_features(train, test, cfg):
    """UNSUPERVISED transforms only -- safe to compute once over train + test.

    A label mapping or a log() over train union test is not a leak; nothing here reads
    the target. Everything supervised lives in fit_target_encoders() below.
    """
    feats_num = list(RAW_NUM)
    feats_cat = list(RAW_CAT)
    tr, te = train.copy(), test.copy()

    if cfg["fe_clip_flags"]:
        # Both numerics are censored at their floor: 9.2% of rows sit exactly at
        # income 30000 and 21.6% at commute 5.0. The clip mixes two populations, which
        # is why the income logit curve is non-monotone at its bottom decile.
        for df in (tr, te):
            df["inc_at_floor"] = (df["Annual_Income_USD"] == 30000).astype(np.int8)
            df["com_at_floor"] = (df["Daily_Commute_km"] == 5.0).astype(np.int8)
        feats_num += ["inc_at_floor", "com_at_floor"]

    if cfg["fe_log_income"]:
        for df in (tr, te):
            df["log_income"] = np.log(df["Annual_Income_USD"])
        feats_num += ["log_income"]

    if cfg["drop_noise"]:
        feats_num = [c for c in feats_num if c not in NOISE_CANDIDATES]
        feats_cat = [c for c in feats_cat if c not in NOISE_CANDIDATES]

    # Category vocabulary over train union test -- unsupervised, so not a leak, and it
    # guarantees an unseen test level can never raise at inference.
    for c in feats_cat:
        vocab = pd.api.types.CategoricalDtype(sorted(set(tr[c]) | set(te[c])))
        tr[c] = tr[c].astype(vocab)
        te[c] = te[c].astype(vocab)

    return tr, te, feats_num, feats_cat


def _te_stats(values, y):
    """Per-value (sum, count). Supervised -- callers must pass a train-fold slice."""
    return pd.DataFrame({"v": values, "y": y}).groupby("v")["y"].agg(["sum", "count"])


def _fit_encoder(values, y, bins, cfg, prior):
    """Build a value -> encoded-rate mapping, plus the backoff needed for unseen values.

    Two backoff modes, and the difference is the whole point of the `te_backoff` knob:

    "prior"        a rare value falls back to the fold's global positive rate (0.1746).
    "neighborhood" a rare value falls back to the rate of its QUANTILE BIN first, and only
                   the bin falls back to the global prior. For a lookup-table column this
                   is strictly better: an income value seen 3 times should be shrunk toward
                   the rate of nearby incomes, not toward the population average, because
                   the column carries a real monotone trend UNDERNEATH the lookup
                   (Spearman(value, rate) = 0.68) that the global prior throws away.
    """
    st = _te_stats(values, y)
    if cfg["te_backoff"] == "neighborhood" and bins is not None:
        bst = _te_stats(bins, y)
        bin_rate = ((bst["sum"] + prior * cfg["te_backoff_smooth"]) /
                    (bst["count"] + cfg["te_backoff_smooth"]))
        # each value's backoff target is its own bin's smoothed rate
        vbin = pd.Series(bins.values, index=values).groupby(level=0).first()
        target = vbin.reindex(st.index).map(bin_rate).fillna(prior)
    else:
        bin_rate, target = None, pd.Series(prior, index=st.index)
    enc = (st["sum"] + target * cfg["te_smooth"]) / (st["count"] + cfg["te_smooth"])
    return enc, st["count"], bin_rate


def apply_target_encoding(Xtr, ytr, Xva, Xte, cols, cfg, rng_seed):
    """Fit per-value target encodings on the TRAINING FOLD ONLY.

    Training rows get their encoding from an inner K-fold so a row never contributes to
    its own encoded value -- without this, a 13,214-value column like Annual_Income_USD
    memorizes the fold outright. Val and test rows use the full training-fold statistics.

    Why this transform matters here (README.md section 6): after fitting the best possible
    MONOTONE function of income, per-value residual rates still have SD 0.0748 against a
    binomial expectation of 0.0284. The column is a value->target lookup table, and a
    lookup is invisible to any model that reads the value as a magnitude.
    """
    prior = float(ytr.mean())
    new_cols = []
    for c in cols:
        name = f"te_{c}"
        new_cols.append(name)

        # Quantile bin edges are fit on the TRAINING FOLD ONLY, per the leakage rule.
        btr = bva = bte = None
        if cfg["te_backoff"] == "neighborhood":
            edges = np.unique(np.quantile(Xtr[c].values,
                                          np.linspace(0, 1, cfg["te_backoff_bins"] + 1)))
            btr = pd.Series(np.searchsorted(edges, Xtr[c].values), index=Xtr.index)
            bva = pd.Series(np.searchsorted(edges, Xva[c].values), index=Xva.index)
            bte = pd.Series(np.searchsorted(edges, Xte[c].values), index=Xte.index)

        # The inner-fold TE a training row receives is itself noisy: it is built from
        # 4/5 of the fold, and WHICH 4/5 is an arbitrary draw. Averaging over several
        # independent inner splits cancels that draw without touching the leakage
        # property -- every split still excludes the row's own label from its encoding.
        # Pure variance reduction, which is what an additive generator rewards.
        enc_tr = np.zeros(len(Xtr), dtype=np.float64)
        cnt_tr = np.zeros(len(Xtr), dtype=np.float64)
        reps = max(1, int(cfg["te_inner_repeats"]))
        for rep in range(reps):
            inner = StratifiedKFold(cfg["te_inner_folds"], shuffle=True,
                                    random_state=rng_seed + 1000 * rep)
            for i, j in inner.split(Xtr, ytr):
                e, n, br = _fit_encoder(Xtr[c].iloc[i].values, ytr[i],
                                        None if btr is None else btr.iloc[i], cfg, prior)
                fb = (pd.Series(prior, index=Xtr.index[j]) if br is None
                      else btr.iloc[j].map(br).fillna(prior))
                enc_tr[j] += Xtr[c].iloc[j].map(e).fillna(fb).values / reps
                if rep == 0:
                    cnt_tr[j] = Xtr[c].iloc[j].map(n).fillna(0.0).values
        Xtr[name] = enc_tr

        e, n, br = _fit_encoder(Xtr[c].values, ytr, btr, cfg, prior)
        for X, b in ((Xva, bva), (Xte, bte)):
            fb = prior if br is None else b.map(br).fillna(prior)
            X[name] = X[c].map(e).fillna(pd.Series(fb, index=X.index) if br is not None
                                        else prior).values

        if cfg["te_count_feature"]:
            cname = f"cnt_{c}"
            new_cols.append(cname)
            Xtr[cname] = np.log1p(cnt_tr)
            Xva[cname] = np.log1p(Xva[c].map(n).fillna(0.0).values)
            Xte[cname] = np.log1p(Xte[c].map(n).fillna(0.0).values)
    return new_cols


# ----------------------------------------------------------------------- learners
def fit_predict(cfg, Xtr, ytr, Xva, yva, Xte, feats_cat):
    """Return (val_proba, test_proba, best_iteration). AUC is the eval metric everywhere."""
    name, p = cfg["learner"], cfg["params"]

    if name == "lgb":
        import lightgbm as lgb
        mono = None
        if cfg["monotone_income"]:
            mono = [1 if c == "Annual_Income_USD" else 0 for c in Xtr.columns]
        m = lgb.LGBMClassifier(random_state=cfg["model_seed"], n_jobs=-1, verbose=-1,
                               monotone_constraints=mono, **p)
        m.fit(Xtr, ytr, eval_X=Xva, eval_y=yva, eval_metric="auc",
              callbacks=[lgb.early_stopping(cfg["early_stopping_rounds"], verbose=False)])
        return (m.predict_proba(Xva)[:, 1], m.predict_proba(Xte)[:, 1],
                int(m.best_iteration_ or p["n_estimators"]))

    if name == "xgb":
        import xgboost as xgb
        m = xgb.XGBClassifier(
            random_state=cfg["model_seed"], n_jobs=-1, tree_method="hist",
            enable_categorical=True, eval_metric="auc",
            early_stopping_rounds=cfg["early_stopping_rounds"],
            n_estimators=p["n_estimators"], learning_rate=p["learning_rate"],
            max_leaves=p["num_leaves"], colsample_bytree=p["colsample_bytree"],
            subsample=p["subsample"], reg_lambda=p["reg_lambda"], max_depth=0,
            grow_policy="lossguide")
        m.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
        return (m.predict_proba(Xva)[:, 1], m.predict_proba(Xte)[:, 1],
                int(m.best_iteration or p["n_estimators"]))

    if name == "cat":
        from catboost import CatBoostClassifier, Pool
        cat_idx = [Xtr.columns.get_loc(c) for c in feats_cat if c in Xtr.columns]
        A, B, C = (Xtr.copy(), Xva.copy(), Xte.copy())
        for c in feats_cat:
            if c in A.columns:
                A[c], B[c], C[c] = A[c].astype(str), B[c].astype(str), C[c].astype(str)
        m = CatBoostClassifier(
            random_seed=cfg["model_seed"], eval_metric="AUC", verbose=0,
            iterations=p["n_estimators"], learning_rate=p["learning_rate"],
            l2_leaf_reg=max(p["reg_lambda"], 1.0),
            early_stopping_rounds=cfg["early_stopping_rounds"])
        m.fit(Pool(A, ytr, cat_features=cat_idx),
              eval_set=Pool(B, yva, cat_features=cat_idx))
        return (m.predict_proba(B)[:, 1], m.predict_proba(C)[:, 1],
                int(m.get_best_iteration() or p["n_estimators"]))

    if name == "glm":
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        A = pd.get_dummies(Xtr, columns=[c for c in feats_cat if c in Xtr.columns],
                           drop_first=True).astype(float)
        B = pd.get_dummies(Xva, columns=[c for c in feats_cat if c in Xva.columns],
                           drop_first=True).astype(float).reindex(columns=A.columns, fill_value=0)
        C = pd.get_dummies(Xte, columns=[c for c in feats_cat if c in Xte.columns],
                           drop_first=True).astype(float).reindex(columns=A.columns, fill_value=0)
        # Scaler fit on the training fold only, per the leakage rule.
        sc = StandardScaler().fit(A)
        m = LogisticRegression(max_iter=3000, C=p.get("C", 1.0)).fit(sc.transform(A), ytr)
        return (m.predict_proba(sc.transform(B))[:, 1],
                m.predict_proba(sc.transform(C))[:, 1], 0)

    raise ValueError(f"unknown learner {name!r}")


# --------------------------------------------------------------------------- main
def main():
    t0 = time.time()
    cfg = load_cfg()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"cfg = {json.dumps(cfg)}\nout = {OUT_DIR}\n")

    train = pd.read_csv(DATA_DIR / "train.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")
    sample = pd.read_csv(DATA_DIR / "sample_submission.csv")
    y = (train[TARGET] == "Yes").astype(int).values

    train, test, feats_num, feats_cat = base_features(train, test, cfg)
    feats = feats_num + feats_cat
    print(f"{len(feats)} base features: {feats}")

    skf = StratifiedKFold(cfg["n_folds"], shuffle=True, random_state=cfg["cv_seed"])
    oof = np.zeros(len(train))
    fold_id = np.full(len(train), -1, dtype=np.int8)
    test_proba = np.zeros(len(test))
    fold_aucs, best_iters = [], []

    for f, (i, j) in enumerate(skf.split(train[feats], y)):
        Xtr, Xva, Xte = train[feats].iloc[i].copy(), train[feats].iloc[j].copy(), test[feats].copy()
        ytr, yva = y[i], y[j]

        # Supervised transforms live INSIDE the fold loop. This is the load-bearing line.
        if cfg["te_cols"]:
            apply_target_encoding(Xtr, ytr, Xva, Xte, cfg["te_cols"], cfg,
                                  cfg["cv_seed"] + f)

        n_model_features = Xtr.shape[1]
        pv, pt, bi = fit_predict(cfg, Xtr, ytr, Xva, yva, Xte, feats_cat)
        oof[j], fold_id[j] = pv, f
        test_proba += pt / cfg["n_folds"]
        fold_aucs.append(roc_auc_score(yva, pv))
        best_iters.append(bi)
        print(f"fold {f}  auc={fold_aucs[-1]:.6f}  best_iter={bi}", flush=True)

    final_auc = roc_auc_score(y, oof)
    lr = cfg["learner"]
    print(f"\nOOF AUC = {final_auc:.6f}   folds {np.mean(fold_aucs):.6f} "
          f"+/- {np.std(fold_aucs):.6f}")

    pd.DataFrame({ID: train[ID], "fold": fold_id, "proba": oof}).to_csv(
        OUT_DIR / f"oof_proba_{lr}.csv", index=False)
    pd.DataFrame({ID: test[ID], "proba": test_proba}).to_csv(
        OUT_DIR / f"test_proba_{lr}.csv", index=False)
    sub = pd.DataFrame({ID: test[ID], TARGET: test_proba})
    # Validate against sample_submission before trusting the file (playbook section 1).
    assert list(sub.columns) == list(sample.columns), (sub.columns, sample.columns)
    assert len(sub) == len(sample) and (sub[ID].values == sample[ID].values).all()
    assert sub[TARGET].between(0, 1).all() and sub[TARGET].notna().all()
    sub.to_csv(OUT_DIR / "submission.csv", index=False)

    engineered = [c for c in feats if c not in RAW_NUM + RAW_CAT]
    metrics = {
        "run_tag": cfg["run_tag"], "learner": lr, f"{lr}_oof_auc": round(final_auc, 6),
        "final_oof_auc": round(final_auc, 6),
        "fold_auc_mean": round(float(np.mean(fold_aucs)), 6),
        "fold_auc_std": round(float(np.std(fold_aucs)), 6),
        "fold_aucs": [round(a, 6) for a in fold_aucs], "best_iters": best_iters,
        "n_features": int(n_model_features), "n_folds": cfg["n_folds"], "cv_seed": cfg["cv_seed"],
        "model_seed": cfg["model_seed"], "te_cols": ",".join(cfg["te_cols"]),
        "engineered": engineered,
        "notebook_runtime_sec": round(time.time() - t0, 1),
    }
    print("RUN_METRICS_JSON:" + json.dumps(metrics))


if __name__ == "__main__":
    main()
