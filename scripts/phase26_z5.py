"""Phase 26 Z5 -- test-augmented (pseudo-label) target encoding, tested at the MECHANISM.

The expensive form of this probe is a pipeline change. The cheap form asks the question the
pipeline change would answer: does a per-value income encoder built from
train-fold + pseudo-labelled test rows predict the HELD-OUT fold's realized per-value rate
better than the train-fold-only encoder? If it does not, no amount of plumbing helps.

Held-out realized rate is the ground truth here; both encoders are scored against it by
weighted RMSE and by the AUC of the encoding as a standalone ranker on the held-out fold.
Leakage: pseudo-labels are TEST-row probabilities from a model whose folds never saw the
held-out train fold's labels... they saw 4/5 of train. That makes this test OPTIMISTIC for
pseudo-labelling, which is the safe direction for a null result.
"""
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

tr = pd.read_csv("data/train.csv", usecols=["id", "Annual_Income_USD", "Will_Buy_EV"])
te = pd.read_csv("data/test.csv", usecols=["id", "Annual_Income_USD"])
y = (tr["Will_Buy_EV"] == "Yes").astype(int).values
vtr = tr["Annual_Income_USD"].values
vte = te["Annual_Income_USD"].values
ptest = pd.read_csv("experiments/preds/50d84171/test_proba_lgb.csv").sort_values("id")["proba"].values
assert len(ptest) == len(vte)
PRIOR, SMOOTH = y.mean(), 5.0
print(f"train {len(y)}  test {len(vte)}  prior {PRIOR:.6f}")
print(f"income values: train {len(np.unique(vtr))}  test {len(np.unique(vte))}  "
      f"test-in-train {np.isin(vte, vtr).mean():.4f}")
print(f"rows per train income value: mean {len(vtr)/len(np.unique(vtr)):.1f}")
print(f"pseudo-label support added per value: {len(vte)/len(vtr)*100:.1f}% more rows\n")

def enc(vals, tgts, weights=None):
    d = pd.DataFrame({"v": vals, "t": tgts, "w": 1.0 if weights is None else weights})
    g = d.groupby("v").agg(s=("t", lambda x: np.nan), n=("w", "sum"))
    g["s"] = d.assign(wt=d["t"] * d["w"]).groupby("v")["wt"].sum()
    return (g["s"] + SMOOTH * PRIOR) / (g["n"] + SMOOTH)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
res = {"train-only": [], "pseudo-aug": []}
aucs = {"train-only": [], "pseudo-aug": []}
for f, (itr, iva) in enumerate(skf.split(vtr, y)):
    # held-out truth: realized rate per income value in the validation fold
    hv = pd.DataFrame({"v": vtr[iva], "y": y[iva]}).groupby("v").agg(r=("y", "mean"), n=("y", "size"))
    hv = hv[hv["n"] >= 20]
    for name in res:
        if name == "train-only":
            e = enc(vtr[itr], y[itr].astype(float))
        else:
            e = enc(np.concatenate([vtr[itr], vte]),
                    np.concatenate([y[itr].astype(float), ptest]))
        aligned = e.reindex(hv.index).fillna(PRIOR).values
        w = hv["n"].values
        res[name].append(np.sqrt(np.average((aligned - hv["r"].values) ** 2, weights=w)))
        full = e.reindex(pd.Index(vtr[iva])).fillna(PRIOR).values
        aucs[name].append(roc_auc_score(y[iva], full))
    if f == 0:
        print(f"fold 0: {len(hv)} held-out income values with n>=20")

print(f"\n{'encoder':12s} {'wRMSE vs held-out rate':>24} {'standalone AUC on held-out':>28}")
for name in res:
    print(f"{name:12s} {np.mean(res[name]):24.6f} {np.mean(aucs[name]):28.6f}")
d_rmse = np.mean(res['pseudo-aug']) - np.mean(res['train-only'])
d_auc = np.mean(aucs['pseudo-aug']) - np.mean(aucs['train-only'])
print(f"\ndelta (pseudo - train-only):  wRMSE {d_rmse:+.6f} (negative = better)"
      f"   AUC {d_auc:+.6f}")
print("per-fold AUC delta: " + " ".join(f"{b-a:+.6f}" for a, b in zip(aucs['train-only'], aucs['pseudo-aug'])))

# where it could possibly matter: values with little train support
print("\nrestricted to held-out values with LOW train support (the backoff regime):")
for thresh in (10, 25, 50):
    dd = []
    for itr, iva in skf.split(vtr, y):
        cnt = pd.Series(vtr[itr]).value_counts()
        hv = pd.DataFrame({"v": vtr[iva], "y": y[iva]}).groupby("v").agg(r=("y","mean"), n=("y","size"))
        hv = hv[(hv["n"] >= 20) & (hv.index.map(cnt).fillna(0) < thresh)]
        if len(hv) < 5: dd.append(np.nan); continue
        a = enc(vtr[itr], y[itr].astype(float)).reindex(hv.index).fillna(PRIOR).values
        b = enc(np.concatenate([vtr[itr], vte]),
                np.concatenate([y[itr].astype(float), ptest])).reindex(hv.index).fillna(PRIOR).values
        w = hv["n"].values
        dd.append(np.sqrt(np.average((b-hv['r'].values)**2, weights=w))
                  - np.sqrt(np.average((a-hv['r'].values)**2, weights=w)))
    print(f"  train support < {thresh:3d}: mean wRMSE delta {np.nanmean(dd):+.6f}"
          f"  (n values/fold ~{len(hv)})")
