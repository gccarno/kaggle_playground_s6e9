"""Phase 27 G1 -- HOW does the generator build a child from its parent?

Z6 established that Annual_Income_USD is a near-unique key into the 10,000-row source:
8,571 values map to exactly one original row, covering 75.8% of train AND test, and the
parent's label shifts the child's per-value target rate by +0.094 (z=+16.8). Phase 26 then
tested the parent's LABEL and closed it at a 0.00027 ceiling.

It never tested the parent's other TWELVE FEATURES. If the generator copies columns from the
parent rather than resampling them independently, then per-column exact-match rates run far
above chance, and two new channels open that nothing in this repo has touched:

  (a) match COUNT is a per-row confidence weight on the parent label -- which would turn
      Phase 26's flat z=+4.32 residual into something a model can condition on;
  (b) for a column the child CLIPS (Daily_Commute_km at 5.0, 21.6% of rows) the parent may
      carry the TRUE pre-clip value, which is information erased from the child entirely.

Chance level is computed per column from the marginal distributions, not assumed.
Read-only structural measurement. No slot.
"""
import numpy as np, pandas as pd

tr = pd.read_csv("data/train.csv"); te = pd.read_csv("data/test.csv")
orig = pd.read_csv("data/EV_Adoption_and_Range_Anxiety_Dataset.csv")
o = orig.dropna(subset=["Annual_Income_USD"]).copy()
cnt = o.groupby("Annual_Income_USD").size()
uniq_vals = cnt[cnt == 1].index
par = o[o["Annual_Income_USD"].isin(uniq_vals)].set_index("Annual_Income_USD")
print(f"original {len(orig)} rows; {len(par)} unique-income parent rows")

COLS = ["Age", "Gender", "City_Type", "Daily_Commute_km", "Number_of_Cars_Owned",
        "Current_Car_Type", "Charging_Stations_Near_Home", "Charging_Stations_Near_Work",
        "Home_Charging_Possible", "Environmental_Concern_Level", "Subsidy_Available",
        "Range_Anxiety_Level"]

for nm, df in (("TRAIN", tr), ("TEST", te)):
    m = df["Annual_Income_USD"].isin(uniq_vals).values
    sub = df[m]
    pj = par.reindex(sub["Annual_Income_USD"].values)
    print(f"\n=== {nm}: {m.sum()} rows with a unique parent ({m.mean():.4f}) ===")
    print(f"{'column':32s} {'match rate':>11} {'chance':>9} {'ratio':>7}")
    tot = np.zeros(len(sub))
    for c in COLS:
        a = sub[c].values
        b = pj[c].values
        if sub[c].dtype.kind in "fi":
            eq = np.isclose(a.astype(float), b.astype(float), equal_nan=True)
        else:
            eq = (pd.Series(a).astype(str).values == pd.Series(b).astype(str).values)
        # chance = sum_v p_child(v) * p_parent(v), from the two marginals
        pc = pd.Series(a).astype(str).value_counts(normalize=True)
        pp = pd.Series(b).astype(str).value_counts(normalize=True)
        ch = float((pc * pp.reindex(pc.index).fillna(0)).sum())
        print(f"{c:32s} {eq.mean():11.4f} {ch:9.4f} {eq.mean()/ch if ch else np.nan:7.2f}")
        tot += eq
    print(f"{'--- columns matched, mean':32s} {tot.mean():11.3f}  of {len(COLS)}")
    print("distribution of match count:")
    vc = pd.Series(tot).value_counts().sort_index()
    print("  " + "  ".join(f"{int(k)}:{v/len(sub):.3f}" for k, v in vc.items()))

print("\n=== the clipping question ===")
for c, clip in (("Annual_Income_USD", 30000.0), ("Daily_Commute_km", 5.0)):
    print(f"{c}: train at clip {np.isclose(tr[c], clip).mean():.4f}   "
          f"original min {orig[c].min()}   original below clip "
          f"{(orig[c] < clip).mean():.4f}")
