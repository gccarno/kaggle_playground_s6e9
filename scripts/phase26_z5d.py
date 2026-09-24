"""Phase 26 Z5d -- the control that decides whether Z5c's +0.000430 is INFORMATION or SMOOTHING.

The pseudo-augmented encoder sees 42.9% more rows per income value than the train-only one.
More support at fixed te_smooth = less shrinkage = a different encoder even if the soft
labels carry nothing. So Z5c's gain has a null explanation that must be killed first.

  A  train-only TE                                    (Z5c baseline, 0.944001)
  B  aug with the model's per-fold test predictions    (Z5c arm B, 0.944431)
  D  aug with the CONSTANT PRIOR as the soft label     -- same support, zero information
  E  aug with pt_f SHUFFLED across test rows           -- same support, same marginal
                                                          distribution, association destroyed

If B ~ D ~ E the gain is shrinkage and is reachable by tuning te_smooth instead.
If B > D, E the soft labels carry per-value information and the mechanism is real.
"""
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

RNG = np.random.default_rng(42)
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
ARMS = ["A train-only", "B aug model", "D aug prior", "E aug shuffled"]
oof = {a: np.zeros(len(y)) for a in ARMS}
skf = StratifiedKFold(5, shuffle=True, random_state=42)
for f, (i, j) in enumerate(skf.split(tr[FEATS], y)):
    src = {"tr": tr.iloc[i], "va": tr.iloc[j], "te": te}
    e0 = {c: enc(tr[c].values[i], y[i].astype(float)) for c in TECOLS}
    X0 = {k: v[FEATS].copy() for k, v in src.items()}
    for k in X0:
        for c in TECOLS: X0[k][f"te_{c}"] = src[k][c].map(e0[c]).fillna(PRIOR).values
    pt = lgb.train(P, lgb.Dataset(X0["tr"], y[i]), num_boost_round=400).predict(X0["te"])
    soft = {"B aug model": pt,
            "D aug prior": np.full(len(pt), PRIOR),
            "E aug shuffled": RNG.permutation(pt)}
    line = f"fold {f}: "
    for a in ARMS:
        if a == "A train-only":
            e = e0
        else:
            e = {c: enc(np.concatenate([tr[c].values[i], te[c].values]),
                        np.concatenate([y[i].astype(float), soft[a]])) for c in TECOLS}
        X = {k: src[k][FEATS].copy() for k in src}
        for k in X:
            for c in TECOLS: X[k][f"te_{c}"] = src[k][c].map(e[c]).fillna(PRIOR).values
        mm = lgb.train(P, lgb.Dataset(X["tr"], y[i]), num_boost_round=400)
        oof[a][j] = mm.predict(X["va"])
        line += f"{a.split()[0]}={roc_auc_score(y[j], oof[a][j]):.6f} "
    print(line, flush=True)

print()
base = roc_auc_score(y, oof["A train-only"])
for a in ARMS:
    s = roc_auc_score(y, oof[a])
    print(f"{a:16s} OOF {s:.6f}   vs A {s-base:+.6f}")
print("\nB - D = %+.6f   B - E = %+.6f" % (
    roc_auc_score(y, oof["B aug model"]) - roc_auc_score(y, oof["D aug prior"]),
    roc_auc_score(y, oof["B aug model"]) - roc_auc_score(y, oof["E aug shuffled"])))
print("seed-noise floor 0.000038   shipping gate +0.0000949")
