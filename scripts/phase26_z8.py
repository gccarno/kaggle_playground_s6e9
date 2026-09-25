"""Phase 26 Z8 -- does the pseudo-label mechanism ITERATE?

Z5e measured one round: soft labels from a pass-1 model sharpen the encoders by +0.000334
on the champion-shaped 11-column TE set. If the sharpened encoder makes a better model, that
model makes better soft labels, which make a better encoder. Either that compounds or it
saturates at round 1, and the answer changes how much the mechanism is worth.

Rounds are honest at every step: every soft-label-producing model for fold f is trained on
folds != f only, with fixed rounds and no eval_set. Champion-shaped feature set so the
numbers are directly comparable to Z5e.
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
QUANT = {"Annual_Income_USD": [10, 50, 500, 5000], "Daily_Commute_km": [1, 2, 5, 10]}
for c, divs in QUANT.items():
    for d in divs:
        for df in (tr, te): df[f"q_{c}_{d}"] = (df[c] / d).astype(np.int64)
TECOLS = ["Annual_Income_USD", "Daily_Commute_km", "Age"] + \
         [f"q_{c}_{d}" for c, divs in QUANT.items() for d in divs]
FEATS = [c for c in tr.columns if c not in ("id", "Will_Buy_EV")]

def enc(vals, tgts):
    g = pd.DataFrame({"v": vals, "t": tgts}).groupby("v")["t"].agg(["sum", "size"])
    return (g["sum"] + SMOOTH * PRIOR) / (g["size"] + SMOOTH)

P = {"objective": "binary", "metric": "auc", "learning_rate": 0.05, "num_leaves": 31,
     "verbose": -1, "seed": 42, "num_threads": 0}
ROUNDS = ["r0 train-only", "r1 one pass", "r2 iterated", "r3 iterated x2"]
oof = {r: np.zeros(len(y)) for r in ROUNDS}
skf = StratifiedKFold(5, shuffle=True, random_state=42)
for f, (i, j) in enumerate(skf.split(tr[FEATS], y)):
    src = {"tr": tr.iloc[i], "va": tr.iloc[j], "te": te}
    soft = None
    line = f"fold {f}: "
    for r in ROUNDS:
        e = ({c: enc(tr[c].values[i], y[i].astype(float)) for c in TECOLS} if soft is None
             else {c: enc(np.concatenate([tr[c].values[i], te[c].values]),
                          np.concatenate([y[i].astype(float), soft])) for c in TECOLS})
        X = {k: src[k][FEATS].copy() for k in src}
        for k in X:
            for c in TECOLS: X[k][f"te_{c}"] = src[k][c].map(e[c]).fillna(PRIOR).values
        m = lgb.train(P, lgb.Dataset(X["tr"], y[i]), num_boost_round=400)
        oof[r][j] = m.predict(X["va"])
        soft = m.predict(X["te"])              # honest wrt fold f at every round
        line += f"{r.split()[0]}={roc_auc_score(y[j], oof[r][j]):.6f} "
    print(line, flush=True)

print()
s = {r: roc_auc_score(y, oof[r]) for r in ROUNDS}
prev = None
for r in ROUNDS:
    step = "" if prev is None else f"   step {s[r]-prev:+.6f}"
    print(f"{r:16s} OOF {s[r]:.6f}   vs r0 {s[r]-s['r0 train-only']:+.6f}{step}")
    prev = s[r]
print("\nZ5e measured r1 - r0 = +0.000334 on this same feature set")
print("shipping gate +0.0000949   seed-noise floor 0.000038")
