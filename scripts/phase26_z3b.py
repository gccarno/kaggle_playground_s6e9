"""Phase 26 Z3b -- the ceiling, with the positive rate held fixed.

Z3's first pass rescaled the logit about its mean, which moved the positive rate
0.1746 -> 0.30 and made AUC rise for two reasons at once. Here every field is
re-centred by bisection so mean(p) == the true 0.174645 exactly, so the only thing
varying is DISPERSION. Read-only; no slot.
"""
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score

RNG = np.random.default_rng(42)
tr = pd.read_csv("data/train.csv", usecols=["id", "Will_Buy_EV"])
y = (tr["Will_Buy_EV"] == "Yes").astype(int).values
n, RATE = len(y), (tr["Will_Buy_EV"] == "Yes").mean()
oof = pd.read_csv("experiments/preds/50d84171/oof_proba_lgb.csv").sort_values("id")
p = oof["proba"].values
lg = np.log(p / (1 - p))
print(f"TEXbag OOF AUC {roc_auc_score(y, p):.6f}   logit SD {lg.std():.4f}   mean p {p.mean():.6f}")

def recentre(z):
    lo, hi = -30.0, 30.0
    for _ in range(80):
        mid = (lo + hi) / 2
        if (1 / (1 + np.exp(-(z + mid)))).mean() > RATE: hi = mid
        else: lo = mid
    return z + (lo + hi) / 2

def ceiling(z, reps=3):
    q = 1 / (1 + np.exp(-recentre(z)))
    return np.mean([roc_auc_score((RNG.random(n) < q).astype(int), q) for _ in range(reps)]), q

print("\n(A) pure rescale of OUR logit field, positive rate pinned at 0.174645")
print(f"  {'s':>6} {'logit SD':>9} {'mean p':>9} {'ceiling AUC':>12}")
rows = []
for s in (0.9, 1.0, 1.02, 1.05, 1.10, 1.20, 1.40):
    a, q = ceiling(lg.mean() + s * (lg - lg.mean()))
    rows.append((s, a)); print(f"  {s:6.2f} {np.log(q/(1-q)).std():9.4f} {q.mean():9.6f} {a:12.6f}")

print("\n(B) our field PLUS an independent missing signal of log-odds SD sigma_extra")
print("    (rate pinned; this is 'how strong a feature we cannot see would explain Alicia')")
print(f"  {'sigma_extra':>11} {'logit SD':>9} {'ceiling AUC':>12} {'vs ours':>9}")
base = None
for se in (0.0, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0):
    z = lg + se * RNG.standard_normal(n)
    a, q = ceiling(z)
    if base is None: base = a
    print(f"  {se:11.2f} {np.log(q/(1-q)).std():9.4f} {a:12.6f} {a-base:+9.6f}")

obs = roc_auc_score(y, p)
print(f"\nobserved   {obs:.6f}")
print(f"ceiling at sigma_extra=0 (our own field, rate-pinned)  {base:.6f}")
print(f"  observed - self-consistent ceiling  {obs-base:+.6f}")
print("\ntarget to explain: Team Alicia 0.94945  => needs  %+.6f  over our observed" % (0.94945 - obs))
