"""Phase 27 G3 -- the parent block as FEATURES, upper-bound screen.

Phase 26 Z7 established the rule: a lean stand-in can only UPPER-BOUND a representation
gain, never estimate it. That cuts both ways and makes it a cheap KILL test -- if the gain
is already below the shipping gate on a lean recipe, it cannot be above it on the champion.

Features added (all derived from the income -> unique-parent join, which is a join onto the
competition's own acknowledged source dataset, not another competitor's predictions):
  par_label        the parent's own Will_Buy_EV, NaN where there is no unique parent
  par_match        how many of 12 columns the child shares with its parent, -1 if none
  par_lab_x_match  the product, so a tree needs one split rather than two to use it
  par_has          whether a unique parent exists at all

G2 measured the parent-label residual rising with agreement (+0.0119 at match>=7, z=+2.57;
+0.0235 at match>=8) with a ceiling of 0.000701 at match>=6. If a model can read it, this is
where it shows up.

GATE: +0.0000949 (shipping gate) on the lean arm, as an UPPER BOUND on the champion.
"""
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

tr = pd.read_csv("data/train.csv"); te = pd.read_csv("data/test.csv")
orig = pd.read_csv("data/EV_Adoption_and_Range_Anxiety_Dataset.csv")
y = (tr["Will_Buy_EV"] == "Yes").astype(int).values
PRIOR, SMOOTH = y.mean(), 5.0
o = orig.dropna(subset=["Annual_Income_USD", "Will_Buy_EV"]).copy()
o["olab"] = (o["Will_Buy_EV"] == "Yes").astype(int)
cnt = o.groupby("Annual_Income_USD").size()
par = o[o["Annual_Income_USD"].isin(cnt[cnt == 1].index)].set_index("Annual_Income_USD")
PCOLS = ["Age", "Gender", "City_Type", "Daily_Commute_km", "Number_of_Cars_Owned",
         "Current_Car_Type", "Charging_Stations_Near_Home", "Charging_Stations_Near_Work",
         "Home_Charging_Possible", "Environmental_Concern_Level", "Subsidy_Available",
         "Range_Anxiety_Level"]

def parent_block(df):
    m = df["Annual_Income_USD"].isin(par.index).values
    pj = par.reindex(df.loc[m, "Annual_Income_USD"].values)
    tot = np.zeros(m.sum())
    for c in PCOLS:
        a, b = df.loc[m, c].values, pj[c].values
        tot += (np.isclose(a.astype(float), b.astype(float), equal_nan=True)
                if df[c].dtype.kind in "fi"
                else pd.Series(a).astype(str).values == pd.Series(b).astype(str).values)
    lab = np.full(len(df), -1.0); mch = np.full(len(df), -1.0)
    lab[m] = pj["olab"].values; mch[m] = tot
    return pd.DataFrame({"par_label": lab, "par_match": mch,
                         "par_lab_x_match": np.where(m, lab * mch, -1.0),
                         "par_has": m.astype(float)}, index=df.index)

CATS = ["Gender", "City_Type", "Current_Car_Type", "Home_Charging_Possible",
        "Subsidy_Available", "Range_Anxiety_Level"]
for c in CATS:
    mm = {v: i for i, v in enumerate(sorted(set(tr[c]) | set(te[c])))}
    tr[c] = tr[c].map(mm).astype("category"); te[c] = te[c].map(mm).astype("category")
QUANT = {"Annual_Income_USD": [10, 50, 500, 5000], "Daily_Commute_km": [1, 2, 5, 10]}
for c, divs in QUANT.items():
    for d in divs:
        for df in (tr, te): df[f"q_{c}_{d}"] = (df[c] / d).astype(np.int64)
TECOLS = ["Annual_Income_USD", "Daily_Commute_km", "Age"] + \
         [f"q_{c}_{d}" for c, divs in QUANT.items() for d in divs]
FEATS = [c for c in tr.columns if c not in ("id", "Will_Buy_EV")]
PB = {"tr": parent_block(tr), "te": parent_block(te)}
print(f"parent block built: train par_has {PB['tr']['par_has'].mean():.4f}  "
      f"test {PB['te']['par_has'].mean():.4f}")

def enc(v, t):
    g = pd.DataFrame({"v": v, "t": t}).groupby("v")["t"].agg(["sum", "size"])
    return (g["sum"] + SMOOTH * PRIOR) / (g["size"] + SMOOTH)

P = {"objective": "binary", "metric": "auc", "learning_rate": 0.05, "num_leaves": 31,
     "verbose": -1, "seed": 42, "num_threads": 0}
ARMS = ["A  no parent block", "B  + parent block"]
oof = {a: np.zeros(len(y)) for a in ARMS}
skf = StratifiedKFold(5, shuffle=True, random_state=42)
for f, (i, j) in enumerate(skf.split(tr[FEATS], y)):
    src = {"tr": tr.iloc[i], "va": tr.iloc[j], "te": te}
    pbs = {"tr": PB["tr"].iloc[i], "va": PB["tr"].iloc[j], "te": PB["te"]}
    e = {c: enc(tr[c].values[i], y[i].astype(float)) for c in TECOLS}
    X = {k: src[k][FEATS].copy() for k in src}
    for k in X:
        for c in TECOLS: X[k][f"te_{c}"] = src[k][c].map(e[c]).fillna(PRIOR).values
    line = f"fold {f}: "
    for a in ARMS:
        XX = {k: (X[k] if a.startswith("A") else
                  pd.concat([X[k], pbs[k].set_axis(X[k].index)], axis=1)) for k in X}
        m = lgb.train(P, lgb.Dataset(XX["tr"], y[i]), num_boost_round=400)
        oof[a][j] = m.predict(XX["va"])
        line += f"{a[0]}={roc_auc_score(y[j], oof[a][j]):.6f} "
    print(line, flush=True)

print()
s = {a: roc_auc_score(y, oof[a]) for a in ARMS}
for a in ARMS: print(f"{a:20s} OOF {s[a]:.6f}")
d = s["B  + parent block"] - s["A  no parent block"]
print(f"\nB - A = {d:+.6f}     gate +0.0000949   seed floor 0.000038")
print("G2 ceiling at match>=6 was 0.000701; this is a lean UPPER BOUND on the champion.")
