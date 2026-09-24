"""Phase 26 Z6 -- the join Z2 should have made: EXACT income value -> original row.

Z2 matched nearest-neighbour in 13-d standardised space and found a clean null. That was
the wrong key. `Annual_Income_USD` is a float that is near-unique in the 10,000-row source
dataset, so an exact income match identifies the original row directly. This is the join
`marcmaldonado/s6e9-generator-fingerprints` publishes; it is rebuilt here from
data/EV_Adoption_and_Range_Anxiety_Dataset.csv rather than ingested, so the provenance is
ours and the numbers are verified rather than trusted.

Read-only. No slot.
"""
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score

tr = pd.read_csv("data/train.csv"); te = pd.read_csv("data/test.csv")
orig = pd.read_csv("data/EV_Adoption_and_Range_Anxiety_Dataset.csv")
y = (tr["Will_Buy_EV"] == "Yes").astype(int).values
print(f"original rows {len(orig)}   income notna {orig['Annual_Income_USD'].notna().sum()}")

o = orig.dropna(subset=["Annual_Income_USD", "Will_Buy_EV"]).copy()
o["olab"] = (o["Will_Buy_EV"] == "Yes").astype(int)
g = o.groupby("Annual_Income_USD")["olab"].agg(["size", "mean"])
print(f"distinct original income values {len(g)}")
print("original rows per income value:")
print(g["size"].value_counts().sort_index().head(8).to_string())
uniq = g[g["size"] == 1]
print(f"\nincome values mapping to EXACTLY ONE original row: {len(uniq)}")

for nm, df in (("train", tr), ("test", te)):
    hit = df["Annual_Income_USD"].isin(g.index)
    hit1 = df["Annual_Income_USD"].isin(uniq.index)
    print(f"{nm}: rows whose income appears in original {hit.mean():.4f}   "
          f"maps to a UNIQUE original row {hit1.mean():.4f}")

# Does the original row's label explain the per-income-value target rate in TRAIN?
vt = pd.DataFrame({"v": tr["Annual_Income_USD"].values, "y": y}).groupby("v")["y"].agg(["size","mean"])
j = vt.join(g.rename(columns={"size": "on", "mean": "olab"}), how="inner")
j = j[(j["size"] >= 20) & (j["on"] == 1)]
print(f"\n{len(j)} income values with >=20 train rows and exactly 1 original row")
a, b = j[j["olab"] == 1]["mean"], j[j["olab"] == 0]["mean"]
se = np.sqrt(a.var(ddof=1)/len(a) + b.var(ddof=1)/len(b))
print(f"  train per-value rate: original label Yes {a.mean():.6f} (n={len(a)})   "
      f"No {b.mean():.6f} (n={len(b)})")
print(f"  difference {a.mean()-b.mean():+.6f}   SE {se:.6f}   z={(a.mean()-b.mean())/se:+.2f}")
print(f"  (global rate {y.mean():.6f}; for reference the per-value rate SD is ~0.075)")

# row-level: the ADD test against the champion's own OOF
oof = pd.read_csv("experiments/preds/50d84171/oof_proba_lgb.csv").sort_values("id")
p = oof["proba"].values
ol = tr["Annual_Income_USD"].map(uniq["mean"])
m = ol.notna().values
print(f"\nrow-level ADD test on the {m.sum()} train rows ({m.mean():.3f}) with a unique parent")
print(f"  champion OOF AUC on those rows            {roc_auc_score(y[m], p[m]):.6f}")
res = y[m] - p[m]
g1, g0 = res[ol[m].values == 1], res[ol[m].values == 0]
se2 = np.sqrt(g1.var(ddof=1)/len(g1) + g0.var(ddof=1)/len(g0))
print(f"  mean residual  parentYes {g1.mean():+.6f} (n={len(g1)})  "
      f"parentNo {g0.mean():+.6f} (n={len(g0)})")
print(f"  difference {g1.mean()-g0.mean():+.6f}  SE {se2:.6f}  z={(g1.mean()-g0.mean())/se2:+.2f}")
for eps in (0.003, 0.01, 0.03):
    print(f"  AUC(p + {eps}*parentlabel) on those rows  "
          f"{roc_auc_score(y[m], p[m] + eps*ol[m].values):.6f}")
