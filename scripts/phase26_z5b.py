"""Phase 26 Z5b -- pseudo-label TE, with the diagnostic's own leak removed.

Z5 used experiments/preds/*/test_proba_lgb.csv as the pseudo-labels. src/pipeline.py:1176
builds that file as `test_proba += pt / n_folds` -- the 5-fold AVERAGE -- so it carries
information from every train row's label, including the held-out fold's. The uniform
+0.0058 it produced is that leak, not the mechanism.

Here the pseudo-labels for fold f come from a lean LightGBM trained ONLY on folds != f,
so they are honest with respect to fold f. Everything else is identical.
"""
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

tr = pd.read_csv("data/train.csv"); te = pd.read_csv("data/test.csv")
y = (tr["Will_Buy_EV"] == "Yes").astype(int).values
PRIOR, SMOOTH = y.mean(), 5.0
CATS = ["Gender", "City_Type", "Current_Car_Type", "Home_Charging_Possible",
        "Subsidy_Available", "Range_Anxiety_Level"]
FEATS = [c for c in tr.columns if c not in ("id", "Will_Buy_EV")]
for c in CATS:
    m = {v: i for i, v in enumerate(sorted(set(tr[c]) | set(te[c])))}   # unsupervised: not a leak
    tr[c] = tr[c].map(m).astype("category"); te[c] = te[c].map(m).astype("category")
vtr, vte = tr["Annual_Income_USD"].values, te["Annual_Income_USD"].values

def enc(vals, tgts):
    d = pd.DataFrame({"v": vals, "t": tgts})
    g = d.groupby("v")["t"].agg(["sum", "size"])
    return (g["sum"] + SMOOTH * PRIOR) / (g["size"] + SMOOTH)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
out = {"train-only": {"rmse": [], "auc": []}, "pseudo-aug": {"rmse": [], "auc": []}}
for f, (itr, iva) in enumerate(skf.split(vtr, y)):
    e_tr = enc(vtr[itr], y[itr].astype(float))             # fold-only fit, per README section 3
    Xtr = tr.iloc[itr][FEATS].copy(); Xva = tr.iloc[iva][FEATS].copy(); Xte = te[FEATS].copy()
    for X, v in ((Xtr, vtr[itr]), (Xva, vtr[iva]), (Xte, vte)):
        X["inc_te"] = pd.Series(v).map(e_tr).fillna(PRIOR).values
    m = lgb.train({"objective": "binary", "metric": "auc", "learning_rate": 0.05,
                   "num_leaves": 31, "verbose": -1, "seed": 42, "num_threads": 0},
                  lgb.Dataset(Xtr, y[itr]), num_boost_round=400)
    pt_f = m.predict(Xte)                                   # HONEST wrt fold f: never saw it
    print(f"fold {f}: lean model val AUC {roc_auc_score(y[iva], m.predict(Xva)):.6f}  "
          f"test pseudo-label mean {pt_f.mean():.6f}")
    e_ps = enc(np.concatenate([vtr[itr], vte]),
               np.concatenate([y[itr].astype(float), pt_f]))
    hv = pd.DataFrame({"v": vtr[iva], "y": y[iva]}).groupby("v").agg(r=("y","mean"), n=("y","size"))
    hv = hv[hv["n"] >= 20]; w = hv["n"].values
    for name, e in (("train-only", e_tr), ("pseudo-aug", e_ps)):
        a = e.reindex(hv.index).fillna(PRIOR).values
        out[name]["rmse"].append(np.sqrt(np.average((a - hv["r"].values) ** 2, weights=w)))
        out[name]["auc"].append(roc_auc_score(y[iva], e.reindex(pd.Index(vtr[iva])).fillna(PRIOR).values))

print(f"\n{'encoder':12s} {'wRMSE vs held-out rate':>24} {'standalone AUC':>16}")
for k in out: print(f"{k:12s} {np.mean(out[k]['rmse']):24.6f} {np.mean(out[k]['auc']):16.6f}")
dr = np.mean(out['pseudo-aug']['rmse']) - np.mean(out['train-only']['rmse'])
da = np.mean(out['pseudo-aug']['auc']) - np.mean(out['train-only']['auc'])
print(f"\ndelta (pseudo - train-only):  wRMSE {dr:+.6f} (negative = better)   AUC {da:+.6f}")
print("per-fold AUC delta: " + " ".join(f"{b-a:+.6f}" for a, b in
      zip(out['train-only']['auc'], out['pseudo-aug']['auc'])))
print("\nLEAKED version (Z5, 5-fold-average pseudo-labels) reported wRMSE -0.003190 / AUC +0.005782")
