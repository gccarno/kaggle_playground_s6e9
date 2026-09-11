#!/usr/bin/env python3
"""Phase 15 (X3): search LightGBM hyperparameters on the Phase-14 (R0) representation.

R0's hyperparameters are the public frontier notebook's, tuned for ITS 320-feature,
10-fold setup -- not our 154-feature, 5-fold one (its colsample_bytree=0.303 samples
~47 of our 154 columns per tree, not ~97 of its own 320). Phase 5 tested learning_rate
and min_child_samples and found them saturated, but that was on E1's 16-feature
recipe; it says nothing about this one. There is no Optuna in src/pipeline.py, so this
is new, narrowly-scoped code: it reuses pipeline.py's own base_features /
apply_target_encoding / apply_shape_target_encoding / fit_predict rather than
reimplementing any of them, and it searches ONLY cfg["params"] (LightGBM
hyperparameters) -- feature engineering is X1/X2's job, not this script's.

LEAKAGE DISCIPLINE (playbook section 2: "the same discipline has to be repeated
identically in every usage site: feature selection, HPO, and the final stack"):
every supervised encoder here is fit exactly as main() fits it -- training-fold-only,
inner-K-fold-protected for the training rows' own encoding. The only shortcut taken is
SPEED, not leakage: the fold-0 feature build (base_features + supervised encoders) is
computed ONCE before the study starts, since it does not depend on the LightGBM
hyperparameters being searched, and every trial reuses it unchanged.

SINGLE FOLD, ON PURPOSE. This is a screen, not a measurement. Selection happens on
fold 0 only; the OUTPUT of this script is a shortlist of candidate params, and the
actual gate (+0.0000949 OOF vs R0's 0.945989) is checked with a full 5-fold
scripts/run_local.py run locally, not with anything printed here. A single-fold AUC
is therefore logged under a "lgb_hpo_oof_auc" column (n_folds=1), never
"lgb_oof_auc" (n_folds=5) -- collapsing the two would silently corrupt the OOF->LB
instrument (README section 4), which is fit on 5-fold runs only.

Usage (matches the notebook cell that runs this on Kaggle):
    python scripts/optuna_probe.py --n-trials 40 --sampler-seed 1
"""
import argparse, importlib.util, json, sys, time
from pathlib import Path

import numpy as np
import optuna
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split

optuna.logging.set_verbosity(optuna.logging.WARNING)

REPO_ROOT = Path(__file__).resolve().parent.parent
PIPELINE = REPO_ROOT / "src" / "pipeline.py"


def _load_pipeline():
    spec = importlib.util.spec_from_file_location("s6e9_pipeline", PIPELINE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


P = _load_pipeline()

# The Phase 14 champion's resolved config (experiments/preds/bdc92691/cfg.json), copied
# here literally rather than read off disk: experiments/preds/ is gitignored, so a
# fresh `git clone` on a Kaggle kernel would not have it. Every field except "params" is
# held fixed -- this script never touches feature engineering.
BASE_CFG = {
    **P.DEFAULTS,
    "run_tag": "R0_hpo_base",
    "fe_recipe_score": True,
    "freq_cols": list(P.RAW_NUM) + list(P.RAW_CAT),
    "fe_digit_cols": list(P.RAW_NUM),
    "fe_digit_powers": [-4, -3, -2, -1, 0, 1, 2, 3],
    "freq_digit_cols": ["Age", "Annual_Income_USD", "Daily_Commute_km",
                        "Charging_Stations_Near_Home", "Charging_Stations_Near_Work",
                        "Environmental_Concern_Level"],
    "te_shape_cols": ["Annual_Income_USD"],
    "te_shape_bins": [8192, 16384],
    "te_shape_smooth": 10.0,
    "te_cols": ["Annual_Income_USD", "Daily_Commute_km", "Age"],
    "te_smooth": 5,
    "te_multi_smooth": [10, "auto"],
    "te_backoff": "neighborhood",
    "te_backoff_bins": 200,
    "te_backoff_smooth": 50.0,
    "params": {
        "n_estimators": 3500, "learning_rate": 0.02, "num_leaves": 31, "max_depth": 5,
        "min_child_samples": 10, "colsample_bytree": 0.303, "reg_alpha": 0.07,
        "reg_lambda": 2.03, "max_bin": 1024,
    },
    "early_stopping_rounds": 120,
}


def build_fold0(cfg):
    """Replicates pipeline.main()'s fold loop for fold 0 only -- same split, same
    encoder call sequence, same leakage protocol. Returns everything fit_predict needs,
    computed once so every trial only pays for the LightGBM fit itself."""
    train = P.pd.read_csv(P.DATA_DIR / "train.csv")
    test = P.pd.read_csv(P.DATA_DIR / "test.csv")
    y = (train[P.TARGET] == "Yes").astype(int).values

    train, test, feats_num, feats_cat = P.base_features(train, test, cfg)
    feats = feats_num + feats_cat

    skf = StratifiedKFold(cfg["n_folds"], shuffle=True, random_state=cfg["cv_seed"])
    i, j = next(iter(skf.split(train[feats], y)))
    Xtr, Xva, Xte = train[feats].iloc[i].copy(), train[feats].iloc[j].copy(), test[feats].copy()
    ytr, yva = y[i], y[j]

    if cfg["te_cols"]:
        P.apply_target_encoding(Xtr, ytr, Xva, Xte, cfg["te_cols"], cfg, cfg["cv_seed"])
    if cfg["te_bin_cols"]:
        P.apply_key_target_encoding(Xtr, ytr, Xva, Xte, cfg["te_bin_cols"],
                                    cfg["te_bin_smooth"], cfg["te_bins"], cfg,
                                    cfg["cv_seed"], "teb_")
    if cfg["te_pair_cols"]:
        P.apply_key_target_encoding(Xtr, ytr, Xva, Xte, cfg["te_pair_cols"],
                                    cfg["te_pair_smooth"], cfg["te_pair_bins"], cfg,
                                    cfg["cv_seed"], "tep_")
    if cfg["te_shape_cols"]:
        P.apply_shape_target_encoding(Xtr, ytr, Xva, Xte, cfg["te_shape_cols"], cfg,
                                      cfg["cv_seed"])

    print(f"fold-0 build: {Xtr.shape[1]} features, {len(Xtr)} train / {len(Xva)} val rows",
          flush=True)
    return Xtr, ytr, Xva, yva, Xte, feats_cat


def suggest_params(trial):
    return {
        "n_estimators": 6000,
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.06, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 15, 127),
        "max_depth": trial.suggest_int("max_depth", 4, 9),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.2, 0.9),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "subsample_freq": 1,
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 5.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 10.0),
        "max_bin": trial.suggest_categorical("max_bin", [255, 1023, 2047]),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-trials", type=int, default=40)
    ap.add_argument("--sampler-seed", type=int, required=True,
                    help="Distinct per parallel shard -- makes the 4 Kaggle kernels "
                         "independent searches rather than duplicates.")
    ap.add_argument("--timeout-sec", type=int, default=None)
    ap.add_argument("--train-frac", type=float, default=1.0,
                    help="Stratified subsample of the FOLD-0 TRAINING rows only, applied "
                         "after TE fitting (so encoder quality is unaffected) purely to "
                         "cut LightGBM fit time per trial and afford more trials within "
                         "a kernel's wall-clock budget. This is a search-speed knob for "
                         "this screen, not a leakage-relevant choice -- the actual gate is "
                         "a full-data, full-5-fold local run of whatever this finds.")
    args = ap.parse_args()

    t0 = time.time()
    cfg = dict(BASE_CFG)
    Xtr, ytr, Xva, yva, Xte, feats_cat = build_fold0(cfg)

    if args.train_frac < 1.0:
        Xtr, _, ytr, _ = train_test_split(
            Xtr, ytr, train_size=args.train_frac, stratify=ytr, random_state=0)
        print(f"subsampled fold-0 training rows to {len(Xtr)} "
              f"({args.train_frac:.0%}) for search speed", flush=True)

    def objective(trial):
        params = suggest_params(trial)
        c = {**cfg, "params": params, "model_seed": cfg["model_seed"]}
        val_proba, _, best_iter = P.fit_predict(c, Xtr, ytr, Xva, yva, Xte, feats_cat)
        auc = roc_auc_score(yva, val_proba)
        trial.set_user_attr("best_iter", best_iter)
        return auc

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=args.sampler_seed),
    )
    study.optimize(objective, n_trials=args.n_trials, timeout=args.timeout_sec,
                   show_progress_bar=False)

    top5 = sorted(study.trials, key=lambda t: t.value or -1, reverse=True)[:5]
    print(f"\n=== sampler_seed={args.sampler_seed}  {len(study.trials)} trials  "
          f"{time.time()-t0:.0f}s ===")
    for r, t in enumerate(top5):
        print(f"  #{r+1}  auc={t.value:.6f}  best_iter={t.user_attrs.get('best_iter')}  "
              f"params={json.dumps(t.params)}")

    best = study.best_trial
    metrics = {
        "notebook_runtime_sec": round(time.time() - t0, 1),
        "n_features": Xtr.shape[1],
        "n_folds": 1,  # deliberately not 5 -- see module docstring
        "cv_seed": cfg["cv_seed"],
        "fold_auc_mean": best.value,
        "fold_auc_std": 0.0,
        "final_oof_auc": best.value,
        "lgb_hpo_oof_auc": best.value,
        "learner": "lgb_hpo",
        "run_tag": f"X3_hpo_seed{args.sampler_seed}",
        "notes": (f"Optuna fold-0 HPO shard, sampler_seed={args.sampler_seed}, "
                 f"{len(study.trials)} trials. single-fold AUC, NOT comparable to a "
                 f"5-fold OOF. best_params={json.dumps(best.params)}"),
    }
    print("\nRUN_METRICS_JSON:" + json.dumps(metrics))


if __name__ == "__main__":
    main()
