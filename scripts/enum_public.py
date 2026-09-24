"""Phase 25 zero-slot enumeration: the public axis at LEG level, not library level.

Phase 24 concluded the public axis is "one bit" from 4 shares x 2 fold counts x 2 combiners
-- but every one of those used megayak's pre-made `ensemble` column as a single unit. The six
individual views A-F were never enumerated, and megayak's own card says D and F add despite
being weaker ("they see the data differently", F has the loosest correlation at 0.99761).
najiama's V-series was likewise never mixed with the 10-fold legs. This closes the axis by
exhaustive measurement rather than by extrapolation from five points. Zero slots.

Combiner is rank_mean throughout (Phase 24 finding 3: the legs are not in the same space).
"""
import glob
import itertools
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

R = lambda v: (rankdata(v) / (len(v) + 1.0)).astype(np.float32)

tr = pd.read_csv("data/train.csv", usecols=["id", "Will_Buy_EV"])
y = (tr["Will_Buy_EV"] == "Yes").astype(int).values
ids = tr["id"].values

ours = R(pd.read_csv(sorted(glob.glob("experiments/preds/61fb5598/oof_proba_*.csv"))[0])
         .sort_values("id")["proba"].values)
A_OURS = roc_auc_score(y, ours)

sv = pd.read_csv("data/public/s6e9-six-feature-views-oof-library/oof_six_views.csv").sort_values("id")
rm = pd.read_csv("data/public/s6e9-six-feature-views-oof-library/oof_realmlp_g.csv").sort_values("id")
assert (sv["id"].values == ids).all() and (rm["id"].values == ids).all()

legs = {}
for short, col in [("A", "A_lgbm_triple_te_digits_3seed"), ("B", "B_xgb_on_A_features"),
                   ("C", "C_no_digits_windows_lift_sm2_30_300"),
                   ("D", "D_no_exact_key_ladder_windows"),
                   ("E", "E_ladder25_250_2500_lift_sm5_50_500"),
                   ("F", "F_exact_rate_as_init_score"), ("ENS", "ensemble")]:
    legs[short] = R(sv[col].values)
legs["rmlp"] = R(rm["G_realmlp_3seed"].values)
for short, f in [("xgb5f", "XGBoost_Triple_TE_5folds"), ("xgb10f", "XGBoost_Triple_TE_10folds"),
                 ("V1", "Pure LGBM_V1"), ("V3", "Pure LGBM_V3"),
                 ("V5", "Pure LGBM_V5"), ("V6", "Pure LGBM_V6")]:
    d = pd.read_csv(f"data/public/s6e9-oof/{f}_oof.csv").sort_values("id")
    assert (d["id"].values == ids).all()
    legs[short] = R(d["OOF_Pred"].values)

FOLDS = dict(A=10, B=10, C=10, D=10, E=10, F=10, ENS=10, rmlp=10,
             xgb5f=5, xgb10f=10, V1=5, V3=5, V5=5, V6=5)

print(f"OURS champion 61fb5598  solo OOF {A_OURS:.6f}")
for n, v in sorted(legs.items(), key=lambda kv: -roc_auc_score(y, kv[1])):
    print(f"  {n:7s} {FOLDS[n]:2d}f  solo {roc_auc_score(y, v):.6f}")

# ENS is a fixed combination of A-F, so never enumerate it alongside its own components.
pool = [n for n in legs if n != "ENS"]
res = []
for k in range(1, 5):
    for combo in itertools.combinations(pool, k):
        P = np.mean([legs[n] for n in combo], axis=0)
        for w in (0.50, 0.34, 0.25):
            res.append((roc_auc_score(y, w * ours + (1 - w) * P), w, combo))
# and the Phase 23/24 shipped shapes, for reference
for combo in [("ENS", "rmlp"), ("ENS", "rmlp", "xgb5f", "V5")]:
    P = np.mean([legs[n] for n in combo], axis=0)
    for w in (0.50, 0.34, 0.25):
        res.append((roc_auc_score(y, w * ours + (1 - w) * P), w, combo))

res.sort(reverse=True)
print(f"\n{len(res)} blends enumerated.  Champion 61fb5598 = {A_OURS:.6f}")
print("TOP 25 by OOF:")
for a, w, c in res[:25]:
    n5 = sum(FOLDS[n] == 5 for n in c)
    print(f"  {a:.6f} (+{a - A_OURS:.6f})  w={w:.2f}  {n5}/{len(c)} 5-fold  {'+'.join(c)}")

print("\nBEST AT EACH 5-FOLD PURITY (how much of the public side is fold-honest):")
for need in (1.0, 0.75, 0.5, 0.0):
    hits = [r for r in res if len(r[2]) and
            abs(sum(FOLDS[n] == 5 for n in r[2]) / len(r[2]) - need) < 1e-9]
    if hits:
        a, w, c = hits[0]
        print(f"  {need:.0%} fold-honest: {a:.6f} (+{a - A_OURS:.6f})  w={w:.2f}  {'+'.join(c)}")
