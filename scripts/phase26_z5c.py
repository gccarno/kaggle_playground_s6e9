"""Phase 26 Z5c -- does the pseudo-augmented encoder add END TO END?

Z5b showed the pseudo-augmented income encoder predicts held-out per-value rates better
(wRMSE -0.001039, standalone AUC +0.001963, all five folds same sign). But it is better
partly BECAUSE the soft label for a test row at income v is informed by that row's other
12 features -- and the model already has those 12 features. README Phase 1b/1c measured
this redundancy three times: three routes to one problem shrink each other.

So: same lean model, same frozen split, three feature sets. Arm C is the ADD test.
Pseudo-labels are per-fold and honest (trained on folds != f only), as in Z5b.
"""
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

tr = pd.read_csv("data/train.csv"); te = pd.read_csv("data/test.csv")
y = (tr["Will_Buy_EV"] == "Yes").astype(int).values
PRIOR, SMOOTH = y.mean(), 5.0
TECOLS = ["Annual_Income_USD", "Daily_Commute_km", "Age"]
CATS = ["Gender", "City_Type", "Current_Car_Type", "Home_Charging_Possible",
        "Subsidy_Available", "Range_Anxiety_Level"]
FEATS = [c for c in tr.columns if c not in ("id", "Will_Buy_EV")]
for c in CATS:
    m = {v: i for i, v in enumerate(sorted(set(tr[c]) | set(te[c])))}
    tr[c] = tr[c].map(m).astype("category"); te[c] = te[c].map(m).astype("category")

def enc(vals, tgts):
    g = pd.DataFrame({"v": vals, "t": tgts}).groupby("v")["t"].agg(["sum", "size"])
    return (g["sum"] + SMOOTH * PRIOR) / (g["size"] + SMOOTH)

P = {"objective": "binary", "metric": "auc", "learning_rate": 0.05, "num_leaves": 31,
     "verbose": -1, "seed": 42, "num_threads": 0}
ARMS = ["A train-only TE", "B pseudo-aug TE", "C both"]
oof = {a: np.zeros(len(y)) for a in ARMS}
skf = StratifiedKFold(5, shuffle=True, random_state=42)
for f, (i, j) in enumerate(skf.split(tr[FEATS], y)):
    e_tr = {c: enc(tr[c].values[i], y[i].astype(float)) for c in TECOLS}
    X = {k: v[FEATS].copy() for k, v in (("tr", tr.iloc[i]), ("va", tr.iloc[j]), ("te", te))}
    src = {"tr": tr.iloc[i], "va": tr.iloc[j], "te": te}
    for k in X:
        for c in TECOLS:
            X[k][f"te_{c}"] = src[k][c].map(e_tr[c]).fillna(PRIOR).values
    m = lgb.train(P, lgb.Dataset(X["tr"], y[i]), num_boost_round=400)
    pt_f = m.predict(X["te"])                          # honest wrt fold f
    e_ps = {c: enc(np.concatenate([tr[c].values[i], te[c].values]),
                   np.concatenate([y[i].astype(float), pt_f])) for c in TECOLS}
    for k in X:
        for c in TECOLS:
            X[k][f"ps_{c}"] = src[k][c].map(e_ps[c]).fillna(PRIOR).values
    cols = {"A train-only TE": [c for c in X["tr"].columns if not c.startswith("ps_")],
            "B pseudo-aug TE": [c for c in X["tr"].columns if not c.startswith("te_")],
            "C both":          list(X["tr"].columns)}
    line = f"fold {f}: "
    for a in ARMS:
        mm = lgb.train(P, lgb.Dataset(X["tr"][cols[a]], y[i]), num_boost_round=400)
        oof[a][j] = mm.predict(X["va"][cols[a]])
        line += f"{a[0]}={roc_auc_score(y[j], oof[a][j]):.6f}  "
    print(line, flush=True)

print()
base = roc_auc_score(y, oof["A train-only TE"])
for a in ARMS:
    s = roc_auc_score(y, oof[a])
    print(f"{a:18s} OOF {s:.6f}   vs A {s-base:+.6f}")
print(f"\nseed-noise floor 0.000038   shipping gate +0.0000949 (README section 4)")
