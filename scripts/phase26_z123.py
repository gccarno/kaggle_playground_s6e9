"""Phase 26 probes Z1 (id axis), Z2 (nearest-parent leak), Z3 (the ceiling).

Read-only diagnostics. Touches no OOF/LB artifact, spends no slot.
Gates are pre-registered in README.md Phase 26 and are NOT restated from the results.
"""
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score

RNG = np.random.default_rng(42)
tr = pd.read_csv("data/train.csv")
y = (tr["Will_Buy_EV"] == "Yes").astype(int).values
n = len(y)
print(f"train {n} rows, rate {y.mean():.6f}")

# champion ours-only OOF (Phase 25 Final A, d8ef7c11)
oof = pd.read_csv("experiments/preds/50d84171/oof_proba_lgb.csv")
oof = oof.sort_values("id"); pcol = "proba"
p = oof[pcol].values
assert len(p) == n
print(f"champion OOF AUC  {roc_auc_score(y, p):.6f}   (expect 0.946137 TEXbag)")

print("\n================ Z1  the id axis ================")
ids = tr["id"].values
print(f"id range {ids.min()}..{ids.max()}  monotone={bool(np.all(np.diff(ids) > 0))}")
print(f"single-feature AUC(id)        {roc_auc_score(y, ids):.6f}")
for nb in (10, 50, 200, 1000):
    edges = np.linspace(0, n, nb + 1).astype(int)
    rates = np.array([y[edges[i]:edges[i+1]].mean() for i in range(nb)])
    sizes = np.array([edges[i+1] - edges[i] for i in range(nb)])
    exp_sd = np.sqrt(y.mean() * (1 - y.mean()) / sizes.mean())
    print(f"  {nb:5d} id-blocks: rate SD {rates.std():.6f}  binomial exp {exp_sd:.6f}"
          f"  ratio {rates.std()/exp_sd:.3f}")
# does id add on top of the champion? residual-vs-id trend
res = y - p
sl = np.polyfit(ids / n, res, 1)
boot = np.array([np.polyfit(ids[i] / n, res[i], 1)[0]
                 for i in (RNG.integers(0, n, n) for _ in range(30))])
print(f"  residual~id slope {sl[0]:+.6f}   bootstrap SD {boot.std():.6f}"
      f"   z={sl[0]/boot.std():+.2f}")

print("\n================ Z2  the nearest-parent leak ================")
orig = pd.read_csv("data/EV_Adoption_and_Range_Anxiety_Dataset.csv")
orig = orig.dropna().reset_index(drop=True)
yo = (orig["Will_Buy_EV"] == "Yes").astype(int).values
print(f"original {len(orig)} rows, rate {yo.mean():.6f}")

NUM = ["Age", "Annual_Income_USD", "Daily_Commute_km", "Number_of_Cars_Owned",
       "Charging_Stations_Near_Home", "Charging_Stations_Near_Work",
       "Environmental_Concern_Level"]
CAT = ["Gender", "City_Type", "Current_Car_Type", "Home_Charging_Possible",
       "Subsidy_Available", "Range_Anxiety_Level"]

# standardize numerics on the ORIGINAL's scale; one-hot cats with a large weight so a
# cat mismatch cannot be traded for a numeric one.
def design(df):
    Z = np.column_stack([df[c].astype(float).values for c in NUM])
    C = pd.get_dummies(df[CAT].astype(str), columns=CAT).reindex(columns=cat_cols,
                                                                fill_value=0).values.astype(float)
    return Z, C
cat_cols = sorted(pd.get_dummies(orig[CAT].astype(str), columns=CAT).columns)
Zo, Co = design(orig)
mu, sd = Zo.mean(0), Zo.std(0)
from sklearn.neighbors import NearestNeighbors
Ao = np.hstack([(Zo - mu) / sd, Co * 4.0])
nn = NearestNeighbors(n_neighbors=1, algorithm="kd_tree").fit(Ao)

SUB = 200_000                     # subsample: the test is about group means, not coverage
idx = RNG.choice(n, SUB, replace=False)
Zt, Ct = design(tr.iloc[idx])
At = np.hstack([(Zt - mu) / sd, Ct * 4.0])
dist, par = nn.kneighbors(At, return_distance=True)
dist, par = dist.ravel(), par.ravel()
print(f"nearest-parent distance: median {np.median(dist):.4f}  "
      f"p10 {np.percentile(dist,10):.4f}  p90 {np.percentile(dist,90):.4f}")
print(f"distinct parents used {len(np.unique(par))} / {len(orig)}")

rs, ps_, ys_ = res[idx], p[idx], y[idx]
plab = yo[par]
for name, mask in (("ALL", np.ones(SUB, bool)),
                   ("nearest 25% by distance", dist <= np.percentile(dist, 25))):
    m = mask
    g1, g0 = rs[m & (plab == 1)], rs[m & (plab == 0)]
    diff = g1.mean() - g0.mean()
    se = np.sqrt(g1.var(ddof=1) / len(g1) + g0.var(ddof=1) / len(g0))
    print(f"  {name:24s} n={m.sum():6d}  mean resid  parentYes {g1.mean():+.6f} (n={len(g1)})"
          f"  parentNo {g0.mean():+.6f} (n={len(g0)})  diff {diff:+.6f}  SE {se:.6f}"
          f"  z={diff/se:+.2f}")
    a0 = roc_auc_score(ys_[m], ps_[m])
    a1 = roc_auc_score(ys_[m], ps_[m] + 0.01 * plab[m])
    print(f"    {'':22s}  AUC {a0:.6f} -> +0.01*parentlabel {a1:.6f}  ({a1-a0:+.6f})")

print("\n================ Z3  the ceiling ================")
a_real = roc_auc_score(y, p)
sims = []
for _ in range(5):
    ysim = (RNG.random(n) < p).astype(int)
    sims.append(roc_auc_score(ysim, p))
sims = np.array(sims)
print(f"observed AUC(y, p)                   {a_real:.6f}")
print(f"self-consistent ceiling AUC(ysim, p) {sims.mean():.6f} +- {sims.std():.6f}")
print(f"  => modelling shortfall against our own probability field: {sims.mean()-a_real:+.6f}")

lg = np.log(np.clip(p, 1e-9, 1 - 1e-9) / (1 - np.clip(p, 1e-9, 1 - 1e-9)))
print("\n  log-odds dispersion needed to REACH a target AUC (perfect model, rescaled field):")
print(f"  {'scale s':>8} {'logit SD':>9} {'pos rate':>9} {'ceiling AUC':>12}")
base_sd = lg.std()
for s in (1.0, 1.1, 1.25, 1.5, 1.75, 2.0, 2.5):
    lg2 = lg.mean() + s * (lg - lg.mean())
    p2 = 1 / (1 + np.exp(-lg2))
    ysim = (RNG.random(n) < p2).astype(int)
    print(f"  {s:8.2f} {lg2.std():9.4f} {p2.mean():9.6f} {roc_auc_score(ysim, p2):12.6f}")
print(f"  (our field: logit SD {base_sd:.4f}, mean p {p.mean():.6f})")
