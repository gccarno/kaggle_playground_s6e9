"""Phase 26 Z4 -- joint-key lookup, at champion strength, with a PERMUTATION null.

Phase 1 ran this on C2 (OOF 0.945225) with a binomial null and correctly read its
largest-of-nine as selection. Two changes: (1) the pool is TEXbag (0.946137), (2) the
null comes from permuting key ASSIGNMENT among rows with matched predicted probability,
which absorbs the OOF anticorrelation that pushed Phase 1's ratios below 1.0.

Keys are fixed in this list before any number is read. Read-only; no slot.
"""
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score

RNG = np.random.default_rng(42)
tr = pd.read_csv("data/train.csv")
y = (tr["Will_Buy_EV"] == "Yes").astype(int).values
oof = pd.read_csv("experiments/preds/50d84171/oof_proba_lgb.csv").sort_values("id")
p = oof["proba"].values
print(f"pool TEXbag OOF {roc_auc_score(y, p):.6f}\n")

inc_b = (tr["Annual_Income_USD"] // 5000).astype(int)     # 5k income bands
com_b = (tr["Daily_Commute_km"] // 5).astype(int)
age_b = (tr["Age"] // 5).astype(int)
KEYS = {
    "income5k x ECL":          [inc_b, tr["Environmental_Concern_Level"]],
    "income5k x Subsidy":      [inc_b, tr["Subsidy_Available"]],
    "income5k x HomeCharge":   [inc_b, tr["Home_Charging_Possible"]],
    "income5k x CityType":     [inc_b, tr["City_Type"]],
    "income5k x commute5":     [inc_b, com_b],
    "income5k x age5":         [inc_b, age_b],
    "ECL x Subsidy x HomeCh":  [tr["Environmental_Concern_Level"], tr["Subsidy_Available"],
                                tr["Home_Charging_Possible"]],
    "ECL x RangeAnx x nHome":  [tr["Environmental_Concern_Level"], tr["Range_Anxiety_Level"],
                                tr["Charging_Stations_Near_Home"]],
    "nHome x nWork":           [tr["Charging_Stations_Near_Home"], tr["Charging_Stations_Near_Work"]],
    "nCars x CurrentCar x ECL":[tr["Number_of_Cars_Owned"], tr["Current_Car_Type"],
                                tr["Environmental_Concern_Level"]],
    "CONTROL random 300 keys": None,
}
MINN = 200
# strata of predicted probability, for the permutation null
strata = np.searchsorted(np.quantile(p, np.linspace(0, 1, 101)[1:-1]), p)

def ratio(codes):
    """sum over keys of (obs residual sum)^2 / (expected variance), as a ratio."""
    df = pd.DataFrame({"k": codes, "r": y - p, "v": p * (1 - p)})
    g = df.groupby("k").agg(n=("r", "size"), rs=("r", "sum"), vs=("v", "sum"))
    g = g[g["n"] >= MINN]
    return (g["rs"] ** 2 / g["vs"]).sum() / len(g), len(g)

print(f"{'key':26s} {'nkeys':>6} {'ratio':>8} {'null mean':>10} {'null p99':>9} {'z':>7}  verdict")
for name, cols in KEYS.items():
    if cols is None:
        codes = RNG.integers(0, 300, len(y))
    else:
        codes = pd.Series(list(zip(*[c.astype(str) for c in cols]))).factorize()[0]
    obs, nk = ratio(codes)
    null = []
    for _ in range(40):                      # permute key labels WITHIN probability strata
        perm = np.empty(len(y), int)
        order = np.argsort(strata, kind="stable")
        shuffled = codes[order].copy()
        b = np.searchsorted(strata[order], np.arange(strata.max() + 2))
        for i in range(len(b) - 1):
            seg = shuffled[b[i]:b[i+1]]
            RNG.shuffle(seg); shuffled[b[i]:b[i+1]] = seg
        perm[order] = shuffled
        null.append(ratio(perm)[0])
    null = np.array(null)
    z = (obs - null.mean()) / null.std()
    v = "SIGNAL" if obs > np.percentile(null, 99) else "null"
    print(f"{name:26s} {nk:6d} {obs:8.4f} {null.mean():10.4f} "
          f"{np.percentile(null,99):9.4f} {z:+7.2f}  {v}")
