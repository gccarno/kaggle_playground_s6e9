"""Phase 27 G2 -- is the parent label's residual effect MODULATED by feature agreement?

G1: the generator does not copy columns (match ratios 1.00-1.36 vs chance). But descent is
real -- Z6 measured the parent label shifting the child's per-value rate by +0.094, z=+16.8 --
so the mechanism is a generative model MEMORISING a 10,000-row source: when it emits an income
value seen in exactly one source row, it tends to emit a row resembling that source row,
label included. Soft descent, not copying.

If that is the mechanism, agreement is a CONFIDENCE weight: a child matching its parent on 8
of 12 columns is far closer to a memorised copy than one matching on 2, so its parent's label
should predict its own much more strongly. Phase 26's Z6b found the parent-label residual flat
across TRAIN SUPPORT and closed the axis at a 0.00027 ceiling -- match count is a different
axis and was never tested.

Pre-registered gate: the residual difference must RISE with match count, reaching z >= 3 in a
stratum holding >= 1% of test rows. A flat profile closes the parent axis for good.
Read-only. No slot.
"""
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score

tr = pd.read_csv("data/train.csv"); te = pd.read_csv("data/test.csv")
orig = pd.read_csv("data/EV_Adoption_and_Range_Anxiety_Dataset.csv")
y = (tr["Will_Buy_EV"] == "Yes").astype(int).values
p = pd.read_csv("experiments/preds/50d84171/oof_proba_lgb.csv").sort_values("id")["proba"].values
res = y - p

o = orig.dropna(subset=["Annual_Income_USD", "Will_Buy_EV"]).copy()
o["olab"] = (o["Will_Buy_EV"] == "Yes").astype(int)
cnt = o.groupby("Annual_Income_USD").size()
par = o[o["Annual_Income_USD"].isin(cnt[cnt == 1].index)].set_index("Annual_Income_USD")
COLS = ["Age", "Gender", "City_Type", "Daily_Commute_km", "Number_of_Cars_Owned",
        "Current_Car_Type", "Charging_Stations_Near_Home", "Charging_Stations_Near_Work",
        "Home_Charging_Possible", "Environmental_Concern_Level", "Subsidy_Available",
        "Range_Anxiety_Level"]

def match_count(df):
    m = df["Annual_Income_USD"].isin(par.index).values
    mc = np.full(len(df), -1.0); pl = np.full(len(df), np.nan)
    pj = par.reindex(df.loc[m, "Annual_Income_USD"].values)
    tot = np.zeros(m.sum())
    for c in COLS:
        a, b = df.loc[m, c].values, pj[c].values
        if df[c].dtype.kind in "fi":
            tot += np.isclose(a.astype(float), b.astype(float), equal_nan=True)
        else:
            tot += (pd.Series(a).astype(str).values == pd.Series(b).astype(str).values)
    mc[m] = tot; pl[m] = pj["olab"].values
    return mc, pl

mc_tr, pl_tr = match_count(tr)
mc_te, _ = match_count(te)
print(f"train with parent {np.isfinite(pl_tr).mean():.4f}   test {(mc_te >= 0).mean():.4f}\n")
print(f"{'match count':>12} {'train rows':>11} {'% of test':>10} {'resid Yes':>10} "
      f"{'resid No':>10} {'diff':>9} {'z':>7}")
rows = []
for k in range(0, 10):
    m = (mc_tr == k) & np.isfinite(pl_tr)
    if m.sum() < 150: continue
    g1, g0 = res[m & (pl_tr == 1)], res[m & (pl_tr == 0)]
    if len(g1) < 30 or len(g0) < 30: continue
    d = g1.mean() - g0.mean()
    se = np.sqrt(g1.var(ddof=1)/len(g1) + g0.var(ddof=1)/len(g0))
    rows.append((k, m.sum(), (mc_te == k).mean(), d, d/se))
    print(f"{k:12d} {m.sum():11d} {(mc_te==k).mean():10.4f} {g1.mean():+10.6f} "
          f"{g0.mean():+10.6f} {d:+9.6f} {d/se:+7.2f}")

# is there a TREND in the difference with match count, weighted by row count?
ks = np.array([r[0] for r in rows]); ds = np.array([r[3] for r in rows])
ws = np.array([r[1] for r in rows], float)
slope = np.polyfit(ks, ds, 1, w=np.sqrt(ws))[0]
print(f"\nweighted slope of (residual difference) on match count: {slope:+.6f} per column")

# the high-agreement pocket, taken together
print("\npooled high-agreement strata:")
for thr in (6, 7, 8):
    m = (mc_tr >= thr) & np.isfinite(pl_tr)
    g1, g0 = res[m & (pl_tr == 1)], res[m & (pl_tr == 0)]
    if len(g1) < 30 or len(g0) < 30: continue
    d = g1.mean() - g0.mean()
    se = np.sqrt(g1.var(ddof=1)/len(g1) + g0.var(ddof=1)/len(g0))
    ft = (mc_te >= thr).mean()
    a0 = roc_auc_score(y[m], p[m]); a1 = roc_auc_score(y[m], p[m] + 0.01*pl_tr[m])
    print(f"  match>={thr}: {m.sum():6d} train rows, {ft:.4f} of test, diff {d:+.6f} "
          f"z={d/se:+.2f}   AUC {a0:.6f} -> {a1:.6f} ({a1-a0:+.6f})")
    print(f"           ceiling if used perfectly: order {ft*abs(d):.6f} of AUC")
