"""Phase 26 Z6b -- where, if anywhere, the parent label is NOT already extracted.

Z6: the unique-parent original label shifts the per-income-value train rate by +0.094
(z=+16.8) -- it IS the lookup table README section 6 describes -- but at row level, against
the champion's OOF, only +0.004463 of residual survives (z=+4.32), and adding it as an
additive bump LOWERS AUC at every dose. So it is ~95% absorbed by the income TE.

The residual 5% should live wherever the TE is weakest: income values with little train
support, where the encoder shrinks to its bin rate while the parent label stays exact.
This stratifies by train support to find out whether there is a pocket worth a feature.
Analysis only, no model. No slot.
"""
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score

tr = pd.read_csv("data/train.csv"); te = pd.read_csv("data/test.csv")
orig = pd.read_csv("data/EV_Adoption_and_Range_Anxiety_Dataset.csv")
y = (tr["Will_Buy_EV"] == "Yes").astype(int).values
p = pd.read_csv("experiments/preds/50d84171/oof_proba_lgb.csv").sort_values("id")["proba"].values

o = orig.dropna(subset=["Annual_Income_USD", "Will_Buy_EV"]).copy()
o["olab"] = (o["Will_Buy_EV"] == "Yes").astype(int)
g = o.groupby("Annual_Income_USD")["olab"].agg(["size", "mean"])
uniq = g[g["size"] == 1]["mean"]

cnt = tr["Annual_Income_USD"].value_counts()
sup = tr["Annual_Income_USD"].map(cnt).values
par = tr["Annual_Income_USD"].map(uniq).values
res = y - p
print(f"train rows with a unique parent: {np.isfinite(par).mean():.4f}")
print(f"\n{'train support of income value':32s} {'n rows':>8} {'%par':>6} "
      f"{'resid Yes':>10} {'resid No':>10} {'diff':>9} {'z':>7}")
BUCKETS = [(0, 5), (5, 10), (10, 25), (25, 50), (50, 100), (100, 10**9)]
for lo, hi in BUCKETS:
    m = (sup >= lo) & (sup < hi) & np.isfinite(par)
    if m.sum() < 200: print(f"  support {lo}-{hi}: too few rows ({m.sum()})"); continue
    g1, g0 = res[m & (par == 1)], res[m & (par == 0)]
    if len(g1) < 30 or len(g0) < 30: print(f"  support {lo}-{hi}: too few in a group"); continue
    d = g1.mean() - g0.mean()
    se = np.sqrt(g1.var(ddof=1)/len(g1) + g0.var(ddof=1)/len(g0))
    lab = f"  support {lo}-{'inf' if hi > 10**8 else hi}"
    print(f"{lab:32s} {m.sum():8d} {np.isfinite(par[(sup>=lo)&(sup<hi)]).mean():6.3f} "
          f"{g1.mean():+10.6f} {g0.mean():+10.6f} {d:+9.6f} {d/se:+7.2f}")

# How much of TEST is in a pocket where the parent could help: low train support AND a parent
tsup = te["Annual_Income_USD"].map(cnt).fillna(0).values
tpar = te["Annual_Income_USD"].map(uniq).values
print(f"\nTEST rows by regime:")
print(f"  income value unseen in train            {(tsup == 0).mean():.4f}   "
      f"of which have a unique parent {np.isfinite(tpar[tsup == 0]).mean():.4f}")
for t in (5, 10, 25):
    m = tsup < t
    print(f"  train support < {t:2d}                      {m.mean():.4f}   "
          f"of which have a unique parent {np.isfinite(tpar[m]).mean():.4f}")
print(f"  any unique parent                       {np.isfinite(tpar).mean():.4f}")

# ceiling on what a perfect use of the parent could buy, given the measured effect size
print("\nceiling estimate: if the parent label were used PERFECTLY on the low-support pocket")
for t in (5, 10, 25):
    frac = ((tsup < t) & np.isfinite(tpar)).mean()
    m = (sup < t) & np.isfinite(par)
    if m.sum() < 200: continue
    g1, g0 = res[m & (par == 1)], res[m & (par == 0)]
    if len(g1) < 30 or len(g0) < 30: continue
    print(f"  support < {t:2d}: {frac:.4f} of test rows, residual separation "
          f"{g1.mean()-g0.mean():+.6f}  -> order {frac*abs(g1.mean()-g0.mean()):.6f} of AUC at most")
