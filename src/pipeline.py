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
import gc, json, os, sys, time, warnings
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


def _kaggle_data_dir():
    """The competition's mount point under /kaggle/input isn't always the slug in
    kernel-metadata.json's competition_sources -- found by direct observation (P0_smoke
    v2 hit FileNotFoundError against /kaggle/input/playground-series-s6e9/train.csv on a
    freshly pushed kernel). Search for whichever subdirectory actually holds train.csv
    instead of hardcoding a path that has already been wrong once."""
    guess = Path("/kaggle/input/playground-series-s6e9")
    if (guess / "train.csv").exists():
        return guess
    hits = list(Path("/kaggle/input").glob("**/train.csv"))
    if len(hits) == 1:
        return hits[0].parent
    raise FileNotFoundError(
        f"could not find train.csv under /kaggle/input (guessed {guess}, found {hits})")


DATA_DIR = _kaggle_data_dir() if ON_KAGGLE else REPO_ROOT / "data"
OUT_DIR = Path(os.environ.get("S6E9_OUT", "/kaggle/working" if ON_KAGGLE else
                              REPO_ROOT / ".kaggle_output" / "scratch"))


def _origin_data_path():
    """The 10,000-row source dataset the competition's generator was fit to
    (itzzomkar/ev-adoption-behavior-and-range-anxiety on Kaggle). Only resolved when
    use_origin_extra asks for it."""
    guess = REPO_ROOT / "data" / "EV_Adoption_and_Range_Anxiety_Dataset.csv"
    if guess.exists():
        return guess
    if ON_KAGGLE:
        hits = list(Path("/kaggle/input").glob("**/EV_Adoption_and_Range_Anxiety_Dataset.csv"))
        if hits:
            return hits[0]
    raise FileNotFoundError(
        "origin dataset not found -- `kaggle datasets download -d "
        "itzzomkar/ev-adoption-behavior-and-range-anxiety -p data --unzip` locally, or "
        "attach it to the kernel")

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
    "learner": "lgb",                # lgb | xgb | cat | glm | emb | ebm | tabicl
    "model_seed": SEED,
    "seed_bag": 1,                   # average predictions over this many model seeds per fold
    "n_folds": N_FOLDS,
    "cv_seed": CV_SEED,

    # -- feature engineering, each a single independently-togglable field so that
    #    run_local.py --diff-vs can enforce strict-twin ablations.
    "drop_noise": False,             # drop NOISE_CANDIDATES
    "fe_clip_flags": False,          # income==30000 / commute==5.0 censoring indicators
    "fe_log_income": False,          # log(Annual_Income_USD) alongside the raw column
    # UNSUPERVISED: fixed-coefficient buy_score / worry_score read off the ORIGIN
    # dataset's recovered generator (cdeotte/fable-5-1-eda-original-data-insights
    # reproduces the source's random seed exactly and recovers the linear-plus-wobble
    # rule the source script used). Not fit on this competition's target at all -- the
    # coefficients are the generator's, not ours -- so this is the cheapest possible test
    # of "does the true generating formula, not a guessed proxy, help the tree resolve
    # its additive shapes faster."
    "fe_recipe_score": False,
    # UNSUPERVISED frequency encoding (train union test value counts, log1p-scaled) for
    # an arbitrary column list -- distinct from te_count_feature, which only ever ran
    # alongside a te_cols entry.
    "freq_cols": [],
    # Append the ORIGIN dataset (10,000 rows, itzzomkar/ev-adoption-behavior-and-range-
    # anxiety) to every fold's TRAINING rows only -- never validated on, never in test.
    # Untested axis, flagged independently by two public authors as unmeasured.
    "use_origin_extra": False,
    # UNSUPERVISED digit decomposition: floor(v / 10^p) % 10 for each power in
    # fe_digit_powers, for every column named in fe_digit_cols. Read from the public
    # frontier recipe (jazivxt/single-model-zoom-zoom, forked as najiama/pure-lgbm-model,
    # 55 votes) -- its OOF scored 0.946064 on our own frozen split, +0.000422 over E1
    # (README Phase 14). Safe to compute once over train union test: no target involved.
    "fe_digit_cols": [],
    "fe_digit_powers": [-4, -3, -2, -1, 0, 1, 2, 3],
    # UNSUPERVISED frequency encoding of the digit keys built by fe_digit_cols (distinct
    # from freq_cols, which encodes the raw column values themselves).
    "freq_digit_cols": [],
    # SUPERVISED multi-scale neighbour-pooled shape encoder for a numeric column, keyed
    # on EQUAL-WIDTH bins (not quantile bins like te_bin_cols) at each width in
    # te_shape_bins. Per width, emits: bin position, smoothed central rate, a
    # gaussian-neighbour-smoothed rate, left-neighbour rate, right-neighbour rate,
    # slope (right - left), curvature (central - avg(left,right)), log1p(count).
    # MECHANISM: our existing te_cols keys on the exact value at ~50 rows/value, so the
    # rate estimate's own SE (~0.053) is close to the real per-value SD (0.0748, README
    # section 6) -- signal-to-noise near 1.4:1. Pooling adjacent bins cuts that SE
    # substantially while the slope/curvature channels hand the tree the response's
    # local DERIVATIVE, which Phase 3b named as the mechanism capacity actually buys.
    # This is a materially finer regime than te_bin_cols/H1 (100 quantile bins, smooth
    # 500 -- ~132 distinct income values per bin): shape bins here are ~10-19 dollars
    # wide. Fit on the TRAINING FOLD ONLY, inner-K-fold protected like apply_target_encoding.
    "te_shape_cols": [],
    "te_shape_bins": [8192, 16384],
    "te_shape_smooth": 10.0,
    # SUPERVISED CENTRED-WINDOW target rates: for each radius r in te_window_radii, the
    # smoothed positive rate over every TRAINING-FOLD row whose value lies in [v-r, v+r].
    # MECHANISM (see apply_window_target_encoding): te_shape_cols pools neighbours through
    # a FIXED bin grid, so a value near a bin edge pools asymmetrically and its pooling
    # radius is whatever the grid width happens to be. A centred window always puts the
    # value at the centre of its own neighbourhood, at several scales at once. Phase 15's
    # X1 measured the shape encoder as FLAT from 1024 to 16384 bins, which says the grid's
    # width was never the binding limitation -- its arbitrary origin is the candidate this
    # axis tests. Radii are in the column's own units (dollars for income, km for commute).
    "te_window_cols": [],
    "te_window_radii": [2, 5, 10, 25, 50, 200],
    "te_window_smooth": 10.0,
    "te_window_count": False,   # also emit log1p(window occupancy) per radius
    # UNSUPERVISED quantisation ladder: floor(v / d) per divisor, the public frontier's
    # "smooth keys" (flagged untested at the end of Phase 15). Built in base_features, so
    # the resulting q_{col}_{d} columns can themselves be named in te_cols / freq_cols.
    "fe_quant_cols": [],
    "fe_quant_divisors": [10, 50, 500, 5000],
    "te_cols": [],                   # SUPERVISED per-value target encoding (fold-fit)
    "te_smooth": 20.0,               # additive smoothing toward the backoff target
    # When non-empty, emit ONE TE COLUMN PER SMOOTHING VALUE instead of a single te_smooth
    # column, e.g. [10, "auto"] mirrors the public frontier recipe's sklearn TargetEncoder
    # smooth=10 and smooth="auto" side by side. "auto" is an empirical-Bayes smoothing in
    # the same spirit as sklearn's rule (target variance / mean within-category variance,
    # see _fit_encoder) -- not a bit-exact port, since the goal is testing whether a
    # SECOND, differently-smoothed view of the same key helps, not matching sklearn's
    # encoder exactly. te_smooth itself is ignored for cols in te_cols when this is
    # non-empty.
    "te_multi_smooth": [],
    "te_inner_folds": 5,             # inner K-fold used for the training rows' own TE
    "te_inner_repeats": 1,           # average the inner-fold TE over this many splits
    "te_backoff": "prior",           # "prior" | "neighborhood" -- what a rare value falls back to
    "te_backoff_bins": 200,          # quantile bins defining a neighborhood (fold-fit)
    "te_backoff_smooth": 50.0,       # smoothing of the neighborhood level toward the prior
    "te_count_feature": False,       # also emit log1p(train-fold count) per encoded column
    # -- the public frontier's encoder recipe: TE over QUANTILE-BINNED and PAIRED keys
    #    with very heavy smoothing, rather than our raw per-value keys at te_smooth=5.
    #    Default-off, so every archived run is bit-identical to what it was.
    "te_bin_cols": [],               # SUPERVISED TE keyed on the quantile-BIN index
    "te_bins": 100,                  # quantile bins for te_bin_cols (fold-fit edges)
    "te_bin_smooth": 500.0,          # frontier's TARGET_SMOOTHING
    "te_pair_cols": [],              # SUPERVISED TE keyed on a PAIR, e.g. [["A","B"], ...]
    "te_pair_bins": 20,              # bins applied to a high-cardinality pair component
    "te_pair_smooth": 1500.0,        # frontier's PAIR_TARGET_SMOOTHING
    "monotone_income": False,        # monotone increasing constraint on income
    # Forbid ALL feature interactions -> the tree ensemble becomes a boosted GAM.
    # The generator is measured additive in log-odds three independent ways (README
    # section 6 and the Phase 2c log): the GLM's 28 pairwise interactions were worth
    # +0.000033, 8 of 9 joint-key residual ratios sat at 0.85-0.95, and H2/H3 made things
    # WORSE by handing the tree five pre-computed interactions. So a tree that can still
    # interact is spending capacity on structure that is not there. lgb learner only.
    "additive_only": False,
    # -- "emb" learner only: value identity as a LEARNED EMBEDDING.
    "emb_cols": ["Annual_Income_USD", "Daily_Commute_km", "Age"],
    "emb_dim": 16,                   # embedding width per token column
    "emb_min_count": 5,              # train-fold count below which a value maps to OOV
    "emb_hidden": [256, 128],
    "emb_dropout": 0.1,
    "emb_lr": 1e-3,
    "emb_batch": 4096,
    "emb_epochs": 40,
    "emb_patience": 5,

    # -- "ebm" learner only: an Explainable Boosting Machine (GA2M). The architecture
    # most directly aligned with the confirmed-additive generator (README section 6:
    # GLM interactions +0.000033, additive_only constraint costs ~nothing, 8/9 joint
    # keys land at 0.85-0.95) -- interactions=0 makes it a pure additive GAM by
    # construction rather than a tree that merely happens not to need interactions.
    "ebm_interactions": 0,
    "ebm_max_bins": 256,
    "ebm_learning_rate": 0.02,
    "ebm_max_rounds": 20000,
    "ebm_outer_bags": 14,
    "ebm_early_stopping_rounds": 100,

    # -- "tabicl" learner only: TabICL v2, an in-context tabular foundation model
    # (github.com/soda-inria/tabicl). No gradient-descent training happens here at all --
    # "fitting" just stores the context rows, and inference attends over them per query
    # batch. tabicl_max_context caps how many TRAINING-FOLD rows are shown as context (a
    # random draw, capped for GPU memory -- not a leak, since it is an unsupervised
    # subsample of rows already restricted to the training fold). tabicl_predict_chunk
    # bounds queries-per-forward-pass independently of context size, since this library's
    # peak memory is driven by (context + query) length together, not context alone.
    "tabicl_max_context": 535000,
    "tabicl_predict_chunk": 15000,
    "tabicl_n_estimators": 4,

    "cat_cols": [],                  # columns ALSO handed to the learner as native
                                     # high-cardinality categoricals (see base_features)

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

    if cfg["fe_recipe_score"]:
        # Coefficients are the ORIGIN generator's own, recovered by reproducing its
        # random seed exactly (not fit here): buy_score = 1.2*(income/1e5) + 0.6*concern
        # + 2*subsidy - 1*(anxiety=Medium) - 3*(anxiety=High), thresholded at 5.5 in the
        # source; worry_score = commute - 5*chargers_home - 5*chargers_work -
        # 150*home_charging, cut at -25/75 into Range_Anxiety_Level in the source.
        for df in (tr, te):
            df["buy_score"] = (
                1.2 * df["Annual_Income_USD"] / 1e5
                + 0.6 * df["Environmental_Concern_Level"]
                + 2.0 * (df["Subsidy_Available"] == "Yes")
                - 1.0 * (df["Range_Anxiety_Level"] == "Medium")
                - 3.0 * (df["Range_Anxiety_Level"] == "High"))
            df["worry_score"] = (
                df["Daily_Commute_km"]
                - 5.0 * df["Charging_Stations_Near_Home"]
                - 5.0 * df["Charging_Stations_Near_Work"]
                - 150.0 * (df["Home_Charging_Possible"] == "Yes"))
        feats_num += ["buy_score", "worry_score"]

    if cfg["freq_cols"]:
        # Unsupervised: a value's count over train union test carries no target
        # information (README section 3), so this is safe to compute once here rather
        # than inside the fold loop.
        for c in cfg["freq_cols"]:
            name = f"freq_{c}"
            counts = pd.concat([tr[c], te[c]]).value_counts()
            tr[name] = np.log1p(tr[c].map(counts).astype(float).values)
            te[name] = np.log1p(te[c].map(counts).astype(float).values)
            feats_num.append(name)

    def _digit(df, c, p):
        v = df[c].to_numpy(np.float64)
        return np.floor_divide(v, 10.0 ** p) % 10

    if cfg["fe_digit_cols"]:
        # UNSUPERVISED: floor(v / 10**p) % 10 at each configured power. Ported from the
        # public frontier recipe (jazivxt/single-model-zoom-zoom -- README Phase 14).
        # NUMERIC columns only -- a categorical column's "digits" would just be its
        # already-compact level index, so fe_digit_cols is meant to be a subset of
        # RAW_NUM. Safe to compute once over train union test: no target read.
        for c in cfg["fe_digit_cols"]:
            for p in cfg["fe_digit_powers"]:
                name = f"digit_{c}_{p}"
                tr[name] = _digit(tr, c, p)
                te[name] = _digit(te, c, p)
                feats_num.append(name)

    if cfg["freq_digit_cols"]:
        # UNSUPERVISED frequency encoding of the digit keys above (train union test
        # counts, log1p-scaled) -- distinct from freq_cols, which encodes the raw
        # column values themselves, not their per-power digits.
        for c in cfg["freq_digit_cols"]:
            for p in cfg["fe_digit_powers"]:
                dname = f"digit_{c}_{p}"
                if dname not in tr.columns:
                    tr[dname] = _digit(tr, c, p)
                    te[dname] = _digit(te, c, p)
                name = f"freqdig_{c}_{p}"
                counts = pd.concat([tr[dname], te[dname]]).value_counts()
                tr[name] = np.log1p(tr[dname].map(counts).astype(float).values)
                te[name] = np.log1p(te[dname].map(counts).astype(float).values)
                feats_num.append(name)

    if cfg["fe_quant_cols"]:
        # UNSUPERVISED quantisation ladder: floor(v / d) for each divisor in
        # fe_quant_divisors -- the public frontier's "smooth keys", flagged as untested
        # at the end of Phase 15 and never run. Distinct from fe_digit_cols (which takes
        # a value's DIGIT, floor(v/10**p) % 10, discarding the magnitude) and from
        # te_shape_cols (equal-width bins fit per fold): a ladder key keeps the ordering
        # and the magnitude, just at a coarser resolution, and -- because it is computed
        # here rather than inside the fold -- it can be named in te_cols/freq_cols like
        # any other column. Safe to compute once over train union test: no target read.
        for c in cfg["fe_quant_cols"]:
            for d in cfg["fe_quant_divisors"]:
                name = f"q_{c}_{d}"
                tr[name] = np.floor_divide(tr[c].to_numpy(np.float64), float(d))
                te[name] = np.floor_divide(te[c].to_numpy(np.float64), float(d))
                feats_num.append(name)

    if cfg["cat_cols"]:
        # The cheap version of the "tokens" idea the public frontier is using: instead of
        # compressing a value to one number (its target rate, as TE does), hand the value
        # IDENTITY to the learner and let it group levels itself. LightGBM's categorical
        # split sorts levels by accumulated gradient/hessian and cuts the sorted order,
        # which is a lookup mechanism the numeric path does not have at all.
        #
        # NOT a leak: the sort is computed by LightGBM from the training rows it is given,
        # inside the fold, exactly like any other split statistic. The vocabulary below is
        # an unsupervised label mapping over train union test, which is also not a leak
        # (README section 3) and is what keeps an unseen test level from raising.
        for c in cfg["cat_cols"]:
            name = f"cat_{c}"
            vocab = pd.api.types.CategoricalDtype(sorted(set(tr[c]) | set(te[c])))
            tr[name] = tr[c].astype(vocab)
            te[name] = te[c].astype(vocab)
            feats_cat.append(name)

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


def _fit_encoder(values, y, bins, cfg, prior, smooth=None):
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
    if smooth == "auto":
        # Empirical-Bayes smoothing in the spirit of sklearn's TargetEncoder(smooth=
        # "auto") -- see te_multi_smooth in DEFAULTS: not a bit-exact port, but the same
        # idea of letting the data set the smoothing rather than a fixed constant. A
        # single scalar tau = target variance / mean within-value variance, applied to
        # every value in this column (sklearn's own "auto" is also one global tau, not
        # per-category).
        y_var = float(np.var(y)) if len(y) > 1 else 0.0
        cnt = st["count"].clip(lower=1)
        rate = st["sum"] / cnt
        within = rate * (1 - rate)
        mean_within = float((within * st["count"]).sum() / max(st["count"].sum(), 1))
        sm = y_var / mean_within if mean_within > 1e-9 else cfg["te_smooth"]
    else:
        sm = cfg["te_smooth"] if smooth is None else float(smooth)
    enc = (st["sum"] + target * sm) / (st["count"] + sm)
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

    cfg["te_multi_smooth"], when non-empty, emits one te_{c}_s{smooth} column PER
    smoothing value instead of a single te_{c} column at cfg["te_smooth"] -- the public
    frontier recipe's "smooth=10 and smooth='auto' side by side" idea (README Phase 14).
    Empty (the default) reproduces the original single-column behaviour exactly, so every
    archived run stays bit-identical.
    """
    prior = float(ytr.mean())
    new_cols = []
    smooths = list(cfg["te_multi_smooth"]) if cfg["te_multi_smooth"] else [None]
    for c in cols:
        # A local de-categorized VIEW for the encoding arithmetic only -- Xtr[c]/Xva[c]/
        # Xte[c] themselves are left untouched (still native categorical, still handed to
        # the learner as such). pandas' Categorical.map() maps the .categories and
        # reconstructs a Categorical result, which then breaks plain float arithmetic
        # (`Categorical / int`) a few lines down -- str/object side-steps that entirely.
        cat_src = isinstance(Xtr[c].dtype, pd.CategoricalDtype)
        ctr = Xtr[c].astype(str) if cat_src else Xtr[c]
        cva = Xva[c].astype(str) if cat_src else Xva[c]
        cte = Xte[c].astype(str) if cat_src else Xte[c]

        # Quantile bin edges are fit on the TRAINING FOLD ONLY, per the leakage rule.
        # A neighborhood needs an ORDERING to be a neighbor in -- np.quantile has none
        # for a categorical column, so a non-numeric column always falls back to the
        # prior regardless of the global te_backoff setting, rather than erroring.
        btr = bva = bte = None
        if cfg["te_backoff"] == "neighborhood" and not cat_src:
            edges = np.unique(np.quantile(ctr.values,
                                          np.linspace(0, 1, cfg["te_backoff_bins"] + 1)))
            btr = pd.Series(np.searchsorted(edges, ctr.values), index=Xtr.index)
            bva = pd.Series(np.searchsorted(edges, cva.values), index=Xva.index)
            bte = pd.Series(np.searchsorted(edges, cte.values), index=Xte.index)

        reps = max(1, int(cfg["te_inner_repeats"]))
        last_n, last_cnt_tr = None, None
        for sm in smooths:
            name = f"te_{c}" if sm is None else f"te_{c}_s{sm}"
            new_cols.append(name)

            # The inner-fold TE a training row receives is itself noisy: it is built
            # from 4/5 of the fold, and WHICH 4/5 is an arbitrary draw. Averaging over
            # several independent inner splits cancels that draw without touching the
            # leakage property -- every split still excludes the row's own label from
            # its encoding. Pure variance reduction, which is what an additive
            # generator rewards.
            enc_tr = np.zeros(len(Xtr), dtype=np.float64)
            cnt_tr = np.zeros(len(Xtr), dtype=np.float64)
            for rep in range(reps):
                inner = StratifiedKFold(cfg["te_inner_folds"], shuffle=True,
                                        random_state=rng_seed + 1000 * rep)
                for i, j in inner.split(Xtr, ytr):
                    e, n, br = _fit_encoder(ctr.iloc[i].values, ytr[i],
                                            None if btr is None else btr.iloc[i], cfg,
                                            prior, sm)
                    fb = (pd.Series(prior, index=Xtr.index[j]) if br is None
                          else btr.iloc[j].map(br).fillna(prior))
                    enc_tr[j] += ctr.iloc[j].map(e).fillna(fb).astype(np.float64).values / reps
                    if rep == 0:
                        cnt_tr[j] = ctr.iloc[j].map(n).fillna(0.0).astype(np.float64).values
            Xtr[name] = enc_tr
            last_cnt_tr = cnt_tr

            e, n, br = _fit_encoder(ctr.values, ytr, btr, cfg, prior, sm)
            last_n = n
            for X, c_, b in ((Xva, cva, bva), (Xte, cte, bte)):
                fb = prior if br is None else b.map(br).fillna(prior)
                X[name] = c_.map(e).fillna(pd.Series(fb, index=X.index) if br is not None
                                           else prior).astype(np.float64).values

        if cfg["te_count_feature"]:
            # Counts don't depend on smoothing, so any smoothing pass's (n, cnt_tr)
            # gives the same values -- last_n/last_cnt_tr are just whichever ran last.
            cname = f"cnt_{c}"
            new_cols.append(cname)
            Xtr[cname] = np.log1p(last_cnt_tr)
            Xva[cname] = np.log1p(cva.map(last_n).fillna(0.0).astype(np.float64).values)
            Xte[cname] = np.log1p(cte.map(last_n).fillna(0.0).astype(np.float64).values)
    return new_cols


def _key_codes(Xtr, Xva, Xte, cols, nbins):
    """Integer key codes for one or more columns, aligned across the three splits.

    A high-cardinality NUMERIC component is replaced by its quantile-BIN index, with the
    edges fit on the TRAINING FOLD ONLY per the leakage rule. A low-cardinality or
    categorical component is label-encoded over train union test, which is an UNSUPERVISED
    mapping and therefore not a leak (README section 3) -- it exists only so that a level
    absent from one split cannot shift another split's codes.

    Multiple components are combined into one composite code by mixed-radix packing, which
    is what makes a PAIR a single lookup key rather than two independent columns.
    """
    codes = {"tr": None, "va": None, "te": None}
    src = {"tr": Xtr, "va": Xva, "te": Xte}
    for c in cols:
        col = {k: X[c] for k, X in src.items()}
        if pd.api.types.is_numeric_dtype(col["tr"]) and col["tr"].nunique() > nbins:
            edges = np.unique(np.quantile(col["tr"].values, np.linspace(0, 1, nbins + 1)))
            part = {k: np.searchsorted(edges, v.values) for k, v in col.items()}
            n_levels = len(edges) + 1
        else:
            cats = pd.Index(sorted(set().union(*(set(v.astype(str)) for v in col.values()))))
            part = {k: cats.get_indexer(v.astype(str)) + 1 for k, v in col.items()}
            n_levels = len(cats) + 1
        for k in codes:
            codes[k] = part[k] if codes[k] is None else codes[k] * n_levels + part[k]
    return codes


def apply_key_target_encoding(Xtr, ytr, Xva, Xte, keys, smooth, nbins, cfg, rng_seed,
                              prefix):
    """SUPERVISED TE keyed on a quantile BIN or on a PAIR of columns, fit on the fold only.

    Why this is a different regime from `apply_target_encoding` rather than a contradiction
    of C3 (raw-value keys at smoothing 100 lost 0.000667): the key here is COARSE. A bin
    holds thousands of rows instead of ~50, so heavy smoothing shrinks a well-estimated
    rate only slightly, and what the feature carries is the denoised local trend rather
    than the per-value lookup. Training rows get their encoding from an inner K-fold, as
    everywhere else, so no row contributes to its own key's rate.

    Backoff is the global prior, not a neighborhood: a bin key HAS no neighborhood beyond
    itself, and a pair key's neighborhood is the very interaction the feature is testing.
    """
    prior = float(ytr.mean())
    kcfg = {**cfg, "te_backoff": "prior"}
    new_cols = []
    for key in keys:
        cols = [key] if isinstance(key, str) else list(key)
        name = prefix + "__".join(cols)
        new_cols.append(name)
        codes = _key_codes(Xtr, Xva, Xte, cols, nbins)
        ktr = pd.Series(codes["tr"], index=Xtr.index)

        enc_tr = np.zeros(len(Xtr), dtype=np.float64)
        inner = StratifiedKFold(cfg["te_inner_folds"], shuffle=True, random_state=rng_seed)
        for i, j in inner.split(Xtr, ytr):
            e, _, _ = _fit_encoder(ktr.iloc[i].values, ytr[i], None, kcfg, prior, smooth)
            enc_tr[j] = ktr.iloc[j].map(e).fillna(prior).values
        Xtr[name] = enc_tr

        e, _, _ = _fit_encoder(ktr.values, ytr, None, kcfg, prior, smooth)
        Xva[name] = pd.Series(codes["va"], index=Xva.index).map(e).fillna(prior).values
        Xte[name] = pd.Series(codes["te"], index=Xte.index).map(e).fillna(prior).values
    return new_cols


def _shape_bin_stats(codes, y, n_bins, prior, smooth):
    """Per-bin (sum, count) plus derived CENTRAL / NEIGHBOUR-POOLED rate channels.

    Ported from the public frontier recipe's `bin_statistics`
    (jazivxt/single-model-zoom-zoom, forked as najiama/pure-lgbm-model -- README Phase
    14): central = additive-smoothed rate of the bin itself; symmetric = a 3-bin
    gaussian-weighted neighbourhood rate (kernel sigma=0.8 bins); left/right = the
    ADJACENT bin's smoothed rate; slope = right - left; curvature = how far the bin's
    own rate sits from the average of its neighbours. All smoothed toward `prior` by
    `smooth`, same additive-smoothing arithmetic as _fit_encoder.

    Returns an (n_bins, 7) array: [central, symmetric, left, right, slope, curvature,
    log1p(count)].
    """
    sums = np.bincount(codes, weights=y, minlength=n_bins).astype(np.float64)
    counts = np.bincount(codes, minlength=n_bins).astype(np.float64)
    central = (sums + smooth * prior) / (counts + smooth)
    left_sums, left_counts = np.r_[0.0, sums[:-1]], np.r_[0.0, counts[:-1]]
    right_sums, right_counts = np.r_[sums[1:], 0.0], np.r_[counts[1:], 0.0]
    left = (left_sums + smooth * prior) / (left_counts + smooth)
    right = (right_sums + smooth * prior) / (right_counts + smooth)
    kernel = np.exp(-0.5 * (np.arange(-1, 2) / 0.8) ** 2)
    neighbor_sums = np.convolve(sums, kernel, mode="same")
    neighbor_counts = np.convolve(counts, kernel, mode="same")
    symmetric = ((neighbor_sums + smooth * kernel.sum() * prior) /
                (neighbor_counts + smooth * kernel.sum()))
    slope = right - left
    curvature = central - 0.5 * (left + right)
    return np.column_stack([central, symmetric, left, right, slope, curvature,
                            np.log1p(counts)])


def apply_shape_target_encoding(Xtr, ytr, Xva, Xte, cols, cfg, rng_seed):
    """SUPERVISED multi-scale neighbour-pooled shape encoder for a NUMERIC column, fit
    on the TRAINING FOLD ONLY, one EQUAL-WIDTH bin grid per width in
    cfg["te_shape_bins"] (distinct from te_bin_cols, which uses QUANTILE bins).

    MECHANISM (README Phase 14, DEFAULTS): apply_target_encoding keys on the exact
    value at ~50 rows/value for Annual_Income_USD, so the per-value rate estimate's own
    SE (~0.053) is close to the real per-value SD (0.0748, README section 6) --
    signal-to-noise near 1.4:1. Pooling adjacent bins here cuts that SE substantially
    while the slope/curvature channels hand the tree the response's local DERIVATIVE.
    This is a materially finer regime than te_bin_cols/H1 (100 QUANTILE bins, smoothing
    500, ~132 distinct income values per bin): at te_shape_bins=[8192, 16384] over
    income's ~$158k range, each bin is roughly $10-19 wide.

    Leakage protocol mirrors apply_target_encoding: bin edges are fit on Xtr only;
    training rows get their channels from an inner K-fold so no row informs its own
    bin's statistic (the label of a $50,003 row can otherwise leak into that bin's rate
    via a wide, coarse bin, exactly like the per-value case); val/test use the full
    training-fold statistics.
    """
    prior = float(ytr.mean())
    smooth = cfg["te_shape_smooth"]
    channel_names = ["pos", "central", "symmetric", "left", "right", "slope",
                     "curvature", "logcount"]
    new_cols = []
    for c in cols:
        v_tr = Xtr[c].to_numpy(np.float64)
        v_va = Xva[c].to_numpy(np.float64)
        v_te = Xte[c].to_numpy(np.float64)
        lo, hi = v_tr.min(), v_tr.max()
        for width in cfg["te_shape_bins"]:
            edges = np.linspace(lo, hi, width + 1)

            def code(v, edges=edges, width=width):
                return np.clip(np.searchsorted(edges[1:-1], v), 0, width - 1)

            codes_tr, codes_va, codes_te = code(v_tr), code(v_va), code(v_te)
            prefix = f"teshape_{c}_{width}_"

            feat_tr = np.zeros((len(v_tr), len(channel_names)), dtype=np.float64)
            inner = StratifiedKFold(cfg["te_inner_folds"], shuffle=True,
                                    random_state=rng_seed)
            for i, j in inner.split(codes_tr, ytr):
                stats = _shape_bin_stats(codes_tr[i], ytr[i], width,
                                         float(ytr[i].mean()), smooth)
                feat_tr[j, 0] = codes_tr[j] / max(width - 1, 1)
                feat_tr[j, 1:] = stats[codes_tr[j]]
            for k, cn in enumerate(channel_names):
                Xtr[prefix + cn] = feat_tr[:, k]

            stats = _shape_bin_stats(codes_tr, ytr, width, prior, smooth)
            for X, codes in ((Xva, codes_va), (Xte, codes_te)):
                X[prefix + "pos"] = codes / max(width - 1, 1)
                block = stats[codes]
                for k, cn in enumerate(channel_names[1:]):
                    X[prefix + cn] = block[:, k]

            new_cols += [prefix + cn for cn in channel_names]
    return new_cols


def _window_rates(v_fit, y_fit, v_query, radii, prior, smooth):
    """Smoothed positive rate over every fit row whose value lies in [q-r, q+r].

    Supervised -- callers must pass a TRAINING-FOLD slice as (v_fit, y_fit).

    Sort the fit values once, take a prefix sum of the labels, and each radius is then
    two np.searchsorted calls per query row: O(n log n) once plus O(n log n) per radius,
    which keeps a 6-radius encoder inside the same wall-clock budget as the bin encoder.
    Smoothing arithmetic is _fit_encoder's: (sum + smooth*prior) / (count + smooth).

    Returns (rates, counts), each (len(v_query), len(radii)).
    """
    order = np.argsort(v_fit, kind="mergesort")
    vs, ys = v_fit[order], y_fit[order].astype(np.float64)
    cs_sum = np.r_[0.0, np.cumsum(ys)]
    cs_cnt = np.arange(len(vs) + 1, dtype=np.float64)
    rates = np.empty((len(v_query), len(radii)), dtype=np.float64)
    counts = np.empty_like(rates)
    for k, r in enumerate(radii):
        lo = np.searchsorted(vs, v_query - r, side="left")
        hi = np.searchsorted(vs, v_query + r, side="right")
        s, n = cs_sum[hi] - cs_sum[lo], cs_cnt[hi] - cs_cnt[lo]
        rates[:, k] = (s + smooth * prior) / (n + smooth)
        counts[:, k] = n
    return rates, counts


def apply_window_target_encoding(Xtr, ytr, Xva, Xte, cols, cfg, rng_seed):
    """SUPERVISED CENTRED-WINDOW target rates for a NUMERIC column, one column per
    radius in cfg["te_window_radii"], fit on the TRAINING FOLD ONLY.

    MECHANISM, and why this is not te_shape_cols again (README Phase 14/15). The shape
    encoder pools neighbours through a FIXED equal-width bin grid: which rows a value
    pools with depends on where the value falls inside its bin, so a value sitting at a
    bin edge pools asymmetrically, and the pooling radius is whatever the grid width
    happens to be. A centred window pools SYMMETRICALLY around the value itself, at
    several radii at once, so the same value always sits at the centre of its own
    neighbourhood. Phase 15's X1 found the shape encoder flat from 1024 to 16384 bins --
    the grid resolution did not matter -- which is consistent with the grid's real
    limitation being its ARBITRARY ORIGIN rather than its width, and that is exactly what
    a centred window removes. Read as an idea (not as code or artifact) from the public
    frontier's "centred-window target rates" description; reimplemented here under this
    repo's own leakage protocol.

    Leakage protocol mirrors apply_target_encoding / apply_shape_target_encoding: the
    windows are built from Xtr only; training rows get theirs from an inner K-fold, so a
    row's own label never enters its own window (a window ALWAYS contains the query value
    itself, so without this the encoder memorizes outright -- the same failure per-value
    TE has); val and test rows use the full training-fold statistics.
    """
    prior = float(ytr.mean())
    radii = list(cfg["te_window_radii"])
    smooth = float(cfg["te_window_smooth"])
    new_cols = []
    for c in cols:
        v_tr = Xtr[c].to_numpy(np.float64)
        v_va = Xva[c].to_numpy(np.float64)
        v_te = Xte[c].to_numpy(np.float64)
        names = [f"tewin_{c}_r{r}" for r in radii]

        rate_tr = np.zeros((len(v_tr), len(radii)), dtype=np.float64)
        cnt_tr = np.zeros_like(rate_tr)
        inner = StratifiedKFold(cfg["te_inner_folds"], shuffle=True,
                                random_state=rng_seed)
        for i, j in inner.split(v_tr.reshape(-1, 1), ytr):
            rate_tr[j], cnt_tr[j] = _window_rates(v_tr[i], ytr[i], v_tr[j], radii,
                                                  float(ytr[i].mean()), smooth)
        for k, nm in enumerate(names):
            Xtr[nm] = rate_tr[:, k]

        out = {}
        for key, X, v in (("va", Xva, v_va), ("te", Xte, v_te)):
            out[key] = _window_rates(v_tr, ytr, v, radii, prior, smooth)
            for k, nm in enumerate(names):
                X[nm] = out[key][0][:, k]
        new_cols += names

        if cfg["te_window_count"]:
            # Window occupancy: how many training-fold rows the rate was built from, so
            # the tree can discount a radius that is empty for this value. Counts do not
            # depend on the labels, but they are still fold-fit, so they ride along here
            # rather than in base_features.
            cnames = [f"tewin_{c}_r{r}_n" for r in radii]
            for k, nm in enumerate(cnames):
                Xtr[nm] = np.log1p(cnt_tr[:, k])
                Xva[nm] = np.log1p(out["va"][1][:, k])
                Xte[nm] = np.log1p(out["te"][1][:, k])
            new_cols += cnames
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
        # One group per column = every tree may split on exactly one feature, so the
        # ensemble is a sum of univariate functions. Note this DECOUPLES capacity from
        # interaction order: num_leaves now buys per-feature shape resolution only, which
        # is why the capacity bracket has to be re-run once this is on (B7 cut capacity
        # 63 -> 7 precisely BECAUSE interactions were harmful).
        inter = [[i] for i in range(Xtr.shape[1])] if cfg["additive_only"] else None
        m = lgb.LGBMClassifier(random_state=cfg["model_seed"], n_jobs=-1, verbose=-1,
                               monotone_constraints=mono, interaction_constraints=inter, **p)
        # eval_set=[(Xva, yva)], not the newer eval_X=/eval_y= keyword form: the latter
        # is only present in LightGBM 4.7+ (X3's actual failure on Kaggle's pinned
        # older LightGBM -- "unexpected keyword argument 'eval_X'" -- since no lgb-
        # learner run had ever previously been pushed to a Kaggle kernel; every prior
        # lgb champion was screened locally). eval_set is the universal form present in
        # every LightGBM version and is semantically identical to eval_X/eval_y.
        m.fit(Xtr, ytr, eval_set=[(Xva, yva)], eval_metric="auc",
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

    if name == "emb":
        return _fit_emb(cfg, Xtr, ytr, Xva, yva, Xte, feats_cat)

    if name == "ebm":
        from interpret.glassbox import ExplainableBoostingClassifier
        A, B, C = Xtr.copy(), Xva.copy(), Xte.copy()
        for c in feats_cat:
            if c in A.columns:
                A[c], B[c], C[c] = A[c].astype(str), B[c].astype(str), C[c].astype(str)
        m = ExplainableBoostingClassifier(
            random_state=cfg["model_seed"], n_jobs=-1,
            interactions=cfg["ebm_interactions"], max_bins=cfg["ebm_max_bins"],
            learning_rate=cfg["ebm_learning_rate"], max_rounds=cfg["ebm_max_rounds"],
            outer_bags=cfg["ebm_outer_bags"],
            early_stopping_rounds=cfg["ebm_early_stopping_rounds"])
        m.fit(A, ytr)
        return m.predict_proba(B)[:, 1], m.predict_proba(C)[:, 1], 0

    if name == "tabicl":
        from tabicl import TabICLClassifier
        A, B, C = Xtr.copy(), Xva.copy(), Xte.copy()

        cap = int(cfg["tabicl_max_context"])
        if len(A) > cap:
            keep = np.random.default_rng(cfg["model_seed"]).choice(len(A), size=cap, replace=False)
            A, ytr = A.iloc[keep], ytr[keep]

        m = TabICLClassifier(random_state=cfg["model_seed"], n_estimators=cfg["tabicl_n_estimators"])
        m.fit(A, ytr)

        def _chunked_proba(X):
            bs = int(cfg["tabicl_predict_chunk"])
            out = np.zeros(len(X))
            for k in range(0, len(X), bs):
                out[k:k + bs] = m.predict_proba(X.iloc[k:k + bs])[:, 1]
            return out

        return _chunked_proba(B), _chunked_proba(C), 0

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



def _fit_emb(cfg, Xtr, ytr, Xva, yva, Xte, feats_cat):
    """Token-embedding MLP: each value of a high-cardinality column gets a LEARNED VECTOR.

    This is the regularized form of the "tokens" idea. F1 tested the unregularized form --
    LightGBM's native categorical split, which regroups 13,214 levels freely at every node
    -- and it lost 0.000999 with best_iter collapsing from ~1000 to ~182, i.e. it memorized
    instantly. An embedding differs in exactly the way that matters: the representation is
    LOW-RANK (emb_dim=16 numbers per value, not an arbitrary partition), SHARED across the
    whole model, and shrunk by weight decay and dropout. It can express more about a value
    than target encoding's single scalar without being able to memorize a fold.

    Leakage: the vocabulary is built over train union test, which is an unsupervised label
    mapping and not a leak (README section 3). The COUNT threshold that decides which values
    collapse to OOV is computed on the TRAINING FOLD ONLY, so which values are rare is never
    learned from held-out rows. Embeddings themselves are fit inside the fold like any other
    parameter.
    """
    import torch
    import torch.nn as nn

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok_cols = [c for c in cfg["emb_cols"] if c in Xtr.columns]
    # Dense side: everything that is not a token column, one-hot for the categoricals.
    dense_src = [c for c in Xtr.columns if c not in tok_cols]
    A = pd.get_dummies(Xtr[dense_src], columns=[c for c in feats_cat if c in dense_src],
                       drop_first=True).astype(np.float32)
    B = pd.get_dummies(Xva[dense_src], columns=[c for c in feats_cat if c in dense_src],
                       drop_first=True).astype(np.float32).reindex(columns=A.columns, fill_value=0)
    C = pd.get_dummies(Xte[dense_src], columns=[c for c in feats_cat if c in dense_src],
                       drop_first=True).astype(np.float32).reindex(columns=A.columns, fill_value=0)
    mu, sd = A.values.mean(0), A.values.std(0) + 1e-6      # fit on the training fold only
    Ad, Bd, Cd = ((X.values - mu) / sd for X in (A, B, C))

    # Token indices. 0 is reserved for OOV, which is where every value too rare in the
    # TRAINING fold is sent -- so the OOV embedding is actually trained rather than being
    # a random vector that only unseen rows ever hit.
    vocabs, Ti = [], {}
    for split, X in (("tr", Xtr), ("va", Xva), ("te", Xte)):
        Ti[split] = []
    for c in tok_cols:
        vc = Xtr[c].value_counts()
        keep = vc[vc >= cfg["emb_min_count"]].index
        mapping = {v: i + 1 for i, v in enumerate(keep)}
        vocabs.append(len(mapping) + 1)
        for split, X in (("tr", Xtr), ("va", Xva), ("te", Xte)):
            Ti[split].append(X[c].map(mapping).fillna(0).astype(np.int64).values)
    # emb_cols=[] (a plain dense MLP, no token embeddings at all) leaves Ti[k] empty for
    # every split -- the placeholder must still carry the split's real ROW COUNT so that
    # xt[idx] below indexes correctly, not a (0, 0) array shared by every split.
    n_rows = {"tr": len(Xtr), "va": len(Xva), "te": len(Xte)}
    Tt = {k: (np.stack(v, 1) if v else np.zeros((n_rows[k], 0), np.int64))
          for k, v in Ti.items()}

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.embs = nn.ModuleList([nn.Embedding(n, cfg["emb_dim"]) for n in vocabs])
            for e in self.embs:
                nn.init.normal_(e.weight, 0, 0.01)
            d = Ad.shape[1] + cfg["emb_dim"] * len(vocabs)
            layers, prev = [], d
            for h in cfg["emb_hidden"]:
                layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.GELU(),
                           nn.Dropout(cfg["emb_dropout"])]
                prev = h
            layers.append(nn.Linear(prev, 1))
            self.mlp = nn.Sequential(*layers)

        def forward(self, xd, xt):
            parts = [xd] + [e(xt[:, k]) for k, e in enumerate(self.embs)]
            return self.mlp(torch.cat(parts, 1)).squeeze(1)

    torch.manual_seed(cfg["model_seed"])
    net = Net().to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=cfg["emb_lr"], weight_decay=1e-4)
    lossf = nn.BCEWithLogitsLoss()

    def tens(d, tk):
        return (torch.tensor(d, dtype=torch.float32, device=dev),
                torch.tensor(tk, dtype=torch.long, device=dev))
    xdtr, xttr = tens(Ad, Tt["tr"])
    xdva, xtva = tens(Bd, Tt["va"])
    xdte, xtte = tens(Cd, Tt["te"])
    ytr_t = torch.tensor(ytr, dtype=torch.float32, device=dev)

    n, bs = len(ytr), cfg["emb_batch"]
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=cfg["emb_lr"], total_steps=cfg["emb_epochs"] * ((n + bs - 1) // bs))

    @torch.no_grad()
    def predict(xd, xt):
        net.eval()
        out = [torch.sigmoid(net(xd[i:i + 65536], xt[i:i + 65536])).cpu().numpy()
               for i in range(0, len(xd), 65536)]
        return np.concatenate(out)

    best, best_va, best_te, bad, best_ep = -1.0, None, None, 0, 0
    g = torch.Generator(device="cpu").manual_seed(cfg["model_seed"])
    for ep in range(cfg["emb_epochs"]):
        net.train()
        perm = torch.randperm(n, generator=g).to(dev)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            loss = lossf(net(xdtr[idx], xttr[idx]), ytr_t[idx])
            loss.backward()
            opt.step()
            sched.step()
        pv = predict(xdva, xtva)
        auc = roc_auc_score(yva, pv)          # AUC is the objective everywhere (README s3)
        if auc > best + 1e-7:
            best, best_va, best_te, bad, best_ep = auc, pv, predict(xdte, xtte), 0, ep
        else:
            bad += 1
            if bad >= cfg["emb_patience"]:
                break
    print(f"    emb: best val auc {best:.6f} at epoch {best_ep}", flush=True)
    return best_va, best_te, best_ep


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

    origin_X, origin_y = None, None
    if cfg["use_origin_extra"]:
        # The 10,000-row ORIGIN dataset the competition's generator was fit to. Its raw
        # columns share this competition's names and value vocabulary exactly (same
        # generator lineage), so no supervised transform is needed here -- only an
        # unsupervised category-dtype alignment so concat cannot silently upcast a
        # column to object. Missing values (source has ~2% NaN on 3 columns) are left as
        # NaN: LightGBM learns a split direction for them like any other missing value,
        # and a TE groupby simply drops a NaN row from that one column's stats.
        origin_raw = pd.read_csv(_origin_data_path())
        origin_X = pd.DataFrame({c: origin_raw[c].values for c in feats if c in origin_raw.columns})
        for c in feats_cat:
            if c in origin_X.columns:
                origin_X[c] = pd.Categorical(origin_X[c], categories=train[c].cat.categories)
        origin_X = origin_X[feats]
        origin_y = (origin_raw[TARGET] == "Yes").astype(int).values
        print(f"origin extra rows: {len(origin_X)} from {_origin_data_path().name}, "
              f"never validated on, appended to every fold's training rows only")

    skf = StratifiedKFold(cfg["n_folds"], shuffle=True, random_state=cfg["cv_seed"])
    oof = np.zeros(len(train))
    fold_id = np.full(len(train), -1, dtype=np.int8)
    test_proba = np.zeros(len(test))
    fold_aucs, best_iters = [], []

    for f, (i, j) in enumerate(skf.split(train[feats], y)):
        Xtr, Xva, Xte = train[feats].iloc[i].copy(), train[feats].iloc[j].copy(), test[feats].copy()
        ytr, yva = y[i], y[j]

        if origin_X is not None:
            # Training rows only: origin rows are never in i/j, so they can never land
            # in Xva/oof, and they are never in Xte. Reset the index so the TE
            # functions' internal Xtr.index bookkeeping (positional, rebuilt fresh each
            # fold) stays consistent -- nothing downstream keys on train's original ids.
            Xtr = pd.concat([Xtr.reset_index(drop=True), origin_X], ignore_index=True)
            ytr = np.concatenate([ytr, origin_y])

        # Supervised transforms live INSIDE the fold loop. This is the load-bearing line.
        if cfg["te_cols"]:
            apply_target_encoding(Xtr, ytr, Xva, Xte, cfg["te_cols"], cfg,
                                  cfg["cv_seed"] + f)
        if cfg["te_bin_cols"]:
            apply_key_target_encoding(Xtr, ytr, Xva, Xte, cfg["te_bin_cols"],
                                      cfg["te_bin_smooth"], cfg["te_bins"], cfg,
                                      cfg["cv_seed"] + f, "teb_")
        if cfg["te_pair_cols"]:
            apply_key_target_encoding(Xtr, ytr, Xva, Xte, cfg["te_pair_cols"],
                                      cfg["te_pair_smooth"], cfg["te_pair_bins"], cfg,
                                      cfg["cv_seed"] + f, "tep_")
        if cfg["te_shape_cols"]:
            apply_shape_target_encoding(Xtr, ytr, Xva, Xte, cfg["te_shape_cols"], cfg,
                                        cfg["cv_seed"] + f)
        if cfg["te_window_cols"]:
            apply_window_target_encoding(Xtr, ytr, Xva, Xte, cfg["te_window_cols"], cfg,
                                         cfg["cv_seed"] + f)

        n_model_features = Xtr.shape[1]
        # Seed-bagging INSIDE a fold averages several models trained on the same rows,
        # which cancels the fitting randomness (feature/row subsampling, tie-breaking)
        # without touching the split. Playbook s6: this is the one seed manipulation that
        # paid in S6E7 -- re-running at a new OUTER SPLIT seed is the one that did not,
        # because test predictions are already averaged over the 5 fold-models.
        bag = max(1, int(cfg["seed_bag"]))
        pv = np.zeros(len(Xva)); pt = np.zeros(len(Xte)); bi = 0
        for b in range(bag):
            c = {**cfg, "model_seed": cfg["model_seed"] + 100 * b}
            a, t_, i_ = fit_predict(c, Xtr, ytr, Xva, yva, Xte, feats_cat)
            pv += a / bag; pt += t_ / bag; bi = max(bi, i_)
        oof[j], fold_id[j] = pv, f
        test_proba += pt / cfg["n_folds"]
        fold_aucs.append(roc_auc_score(yva, pv))
        best_iters.append(bi)
        print(f"fold {f}  auc={fold_aucs[-1]:.6f}  best_iter={bi}", flush=True)

        # Release this fold's encoded frames BEFORE the next iteration builds its own.
        # Without this, `train[feats].iloc[i].copy()` for fold f+1 is evaluated while
        # fold f's Xtr/Xva/Xte are still referenced, so peak RSS is two folds' worth of
        # a (535k x n_features) float64 matrix rather than one. At R0's 154 features
        # that is ~1.3GB of avoidable peak, which is the difference between running and
        # being OOM-killed on a 16GB machine once a probe pushes past ~165 features.
        # Purely a memory fix: nothing here touches a computed value.
        del Xtr, Xva, Xte, pv, pt
        gc.collect()

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
