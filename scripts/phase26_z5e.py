"""Phase 26 Z5e -- the redundancy test, which decides whether the pipeline surgery is worth it.

Z5c/Z5d measured the pseudo-label gain on a LEAN representation: raw features plus a single
per-value TE of the three numerics. The champion carries eleven TE columns -- the three raw
numerics PLUS eight multi-scale quantile-bucket encoders (q_income_10/50/500/5000,
q_commute_1/2/5/10) -- and those are themselves neighbourhood-pooled estimates of the same
per-value rate at coarser support. README Phase 1b/1c measured this exact redundancy three
times: several routes to one problem shrink each other's delta.

So the question the expensive champion run would answer is asked here for a tenth of the
cost: does the pseudo-label gain survive once the multi-scale pooled encoders are present?

  A  train-only TE, 11 columns (champion-shaped)
  B  pseudo-augmented TE, the same 11 columns
  A1 train-only TE, 3 columns (Z5c's lean shape, for the shrink factor)
  B1 pseudo-augmented TE, 3 columns
"""
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

tr = pd.read_csv("data/train.csv"); te = pd.read_csv("data/test.csv")
y = (tr["Will_Buy_EV"] == "Yes").astype(int).values
PRIOR, SMOOTH = y.mean(), 5.0
CATS = ["Gender", "City_Type", "Current_Car_Type", "Home_Charging_Possible",
        "Subsidy_Available", "Range_Anxiety_Level"]
for c in CATS:
    m = {v: i for i, v in enumerate(sorted(set(tr[c]) | set(te[c])))}
    tr[c] = tr[c].map(m).astype("category"); te[c] = te[c].map(m).astype("category")
# champion's fe_quant_cols / fe_quant_divisors, rebuilt: unsupervised, not a leak
QUANT = {"Annual_Income_USD": [10, 50, 500, 5000], "Daily_Commute_km": [1, 2, 5, 10]}
for c, divs in QUANT.items():
    for d in divs:
        for df in (tr, te): df[f"q_{c}_{d}"] = (df[c] / d).astype(np.int64)
LEAN3 = ["Annual_Income_USD", "Daily_Commute_km", "Age"]
FULL11 = LEAN3 + [f"q_{c}_{d}" for c, divs in QUANT.items() for d in divs]
FEATS = [c for c in tr.columns if c not in ("id", "Will_Buy_EV")]
print(f"{len(FEATS)} base features; TE sets: lean {len(LEAN3)}, champion-shaped {len(FULL11)}")

def enc(vals, tgts):
    g = pd.DataFrame({"v": vals, "t": tgts}).groupby("v")["t"].agg(["sum", "size"])
    return (g["sum"] + SMOOTH * PRIOR) / (g["size"] + SMOOTH)

P = {"objective": "binary", "metric": "auc", "learning_rate": 0.05, "num_leaves": 31,
     "verbose": -1, "seed": 42, "num_threads": 0}
ARMS = [("A  11col train-only", FULL11, False), ("B  11col pseudo-aug", FULL11, True),
        ("A1  3col train-only", LEAN3, False), ("B1  3col pseudo-aug", LEAN3, True)]
oof = {a[0]: np.zeros(len(y)) for a in ARMS}
skf = StratifiedKFold(5, shuffle=True, random_state=42)
for f, (i, j) in enumerate(skf.split(tr[FEATS], y)):
    src = {"tr": tr.iloc[i], "va": tr.iloc[j], "te": te}
    # pass 1: honest per-fold test predictions from the champion-shaped train-only arm
    e0 = {c: enc(tr[c].values[i], y[i].astype(float)) for c in FULL11}
    X0 = {k: src[k][FEATS].copy() for k in src}
    for k in X0:
        for c in FULL11: X0[k][f"te_{c}"] = src[k][c].map(e0[c]).fillna(PRIOR).values
    pt = lgb.train(P, lgb.Dataset(X0["tr"], y[i]), num_boost_round=400).predict(X0["te"])
    line = f"fold {f}: "
    for name, cols, aug in ARMS:
        e = ({c: enc(np.concatenate([tr[c].values[i], te[c].values]),
                     np.concatenate([y[i].astype(float), pt])) for c in cols} if aug
             else {c: enc(tr[c].values[i], y[i].astype(float)) for c in cols})
        X = {k: src[k][FEATS].copy() for k in src}
        for k in X:
            for c in cols: X[k][f"te_{c}"] = src[k][c].map(e[c]).fillna(PRIOR).values
        mm = lgb.train(P, lgb.Dataset(X["tr"], y[i]), num_boost_round=400)
        oof[name][j] = mm.predict(X["va"])
        line += f"{name.split()[0]}={roc_auc_score(y[j], oof[name][j]):.6f} "
    print(line, flush=True)

print()
s = {n: roc_auc_score(y, oof[n]) for n, _, _ in ARMS}
for n in s: print(f"{n:22s} OOF {s[n]:.6f}")
print(f"\nlean  (3 col):  B1 - A1 = {s['B1  3col pseudo-aug'] - s['A1  3col train-only']:+.6f}")
print(f"champion-shaped (11 col):  B - A  = {s['B  11col pseudo-aug'] - s['A  11col train-only']:+.6f}")
print(f"multi-scale pooling alone: A - A1 = {s['A  11col train-only'] - s['A1  3col train-only']:+.6f}")
print("\nshipping gate +0.0000949   seed-noise floor 0.000038")
