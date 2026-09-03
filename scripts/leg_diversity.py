#!/usr/bin/env python3
"""Why the pack's 0.9947 median max-correlation is NOT the constraint. Read-only.

Phase 2 closed with an open worry: every leg in the 22-leg stack has a near-twin, and the one
leg that broke the pattern (L1lookupt, max corr 0.9796) bought most of the gain. The natural
reading was "redundancy is the constraint -- reduce it and the stack improves". This script is
the measurement that falsified that reading, kept here so the negative survives the session it
was produced in. Three sections, each a separate attack on the redundancy:

  1. corr_table()         -- solo AUC, max correlation and nearest neighbour for all 22 legs
                             plus the 9 orphans, in logit space.
  2. segment_table()      -- is the redundancy uniform, or an average over heterogeneous parts?
                             If tree legs and neural legs disagreed more in some regime, a
                             combiner that switched on that regime would beat a fixed one.
  3. meta_table()         -- can a NONLINEAR combiner extract more from the same 22 legs?
  4. contribution_table() -- what do the 9 orphan legs, all BELOW the 0.9947 median, add?

RESULT, and the reason this phase stopped probing (reproduce with `python scripts/leg_diversity.py`):

    LightGBM meta over the 22 logits          -0.000042    the linear logit stack is already right
    ... plus raw features and missing-count   -0.000065    nothing to gate on
    tree~neural corr across all 5 regimes     0.9785-0.9811  redundancy is uniform
    all nine orphan legs admitted             +0.000047    a quarter of the +0.0002 gate
    K3tabtf alone (max corr 0.9251)           -0.000001    the most decorrelated leg we own

MAX-CORRELATION DOES NOT PREDICT STACK CONTRIBUTION. A leg at 0.9251 contributes nothing because
it is 0.015 AUC of noise. L1lookupt paid because it was simultaneously our STRONGEST leg (0.968626)
and decorrelated. Strength is the binding constraint; decorrelation only pays on top of it. Do not
justify a future probe with "it will be different" -- that justification is measured insufficient.

Writes nothing. Nothing may be shipped from it. Runtime ~10 minutes, dominated by section 3.
"""
import argparse
import time

import numpy as np
import pandas as pd
from scipy.special import logit
from sklearn.metrics import roc_auc_score

from stack_logit import EPS, REPO_ROOT, honest_oof, load_legs
from subset_ceiling import LEGS

# The nine legs from Phase 1 that are NOT in the pool. They are excluded BY MEASUREMENT, not by
# oversight: see contribution_table() below. Each has complete oof+test artifacts on the frozen
# partition, so admitting one is free -- which is exactly why it is worth having measured.
ORPHANS = {
    "v1anchor":  "ba6c676a",   # raw-feature LightGBM, the v1 anchor
    "A1seed":    "22de9888",   # seed twin of the anchor -- included only to price the seed axis
    "B1inter":   "66e3dede",   # fe_interaction, pre-target-encoding era
    "A2es400":   "6b671f87",   # early_stop 100 -> 400 on the anchor
    "D2numcat":  "b4326a68",   # the 9 numerics declared categorical
    "D2blookup": "2b876e0b",   # categorical for the two lookup-table columns only
    "E3raw":     "a594ffe2",   # MLP on RAW features, no target encoding
    "G2tabtf":   "f1424102",   # TabTransformer
    "K3tabtf":   "9527c2d0",   # TabTransformer at full capacity on a T4
}

# Groupings for section 2. Deliberately coarse: the question is whether the TREE axis and the
# NEURAL axis disagree more in some regime, not how individual legs behave.
TREE = ["B10x", "C3x", "C4x", "D1cat", "B2xgb", "B2ccat", "B2blgb", "B6", "C1"]
NEURAL = ["E1mlp", "E2emb", "F1real", "K1real", "G1ftt", "K2ftt", "K4node"]
RAW_NUM = ["age", "daily_screen_time_hours", "social_media_hours", "gaming_hours",
           "work_study_hours", "sleep_hours", "notifications_per_day",
           "app_opens_per_day", "weekend_screen_time"]
STACK_C = 0.1          # the shipped stack's C, so every delta here is comparable to 0a3c852e
GATE = 0.0002


def to_logit(P):
    return logit(np.clip(P, EPS, 1 - EPS))


def corr_table(names, L, orph_names, LO):
    """Solo AUC, max |corr| and nearest neighbour, for the pool and for each orphan."""
    y = corr_table.y
    C = np.corrcoef(L.T)
    np.fill_diagonal(C, 0.0)
    rows = []
    for i, n in enumerate(names):
        j = C[i].argmax()
        rows.append((n, "pool", roc_auc_score(y, L[:, i]), C[i, j], names[j]))
    # An orphan's max-corr is measured against the POOL only -- orphan-to-orphan correlation is
    # not the question, since none of them is in the stack to begin with.
    for i, n in enumerate(orph_names):
        c = np.array([np.corrcoef(LO[:, i], L[:, k])[0, 1] for k in range(len(names))])
        j = c.argmax()
        rows.append((n, "ORPHAN", roc_auc_score(y, LO[:, i]), c[j], names[j]))
    df = pd.DataFrame(rows, columns=["leg", "status", "solo", "max_corr", "nearest"])
    df = df.sort_values("max_corr")
    print(df.to_string(index=False, float_format="%.4f"))
    med = df.loc[df.status == "pool", "max_corr"].median()
    print(f"\npool median max_corr {med:.4f}   "
          f"-- every orphan sits BELOW it, and section 4 prices what that is worth.")


def segment_table(idx, L):
    """Is the redundancy uniform, or an average over heterogeneous parts?

    If the tree axis and the neural axis disagreed more on, say, rows where the composition
    constraint is unavailable, then a combiner that switched on missingness would beat a fixed
    linear one -- and section 3 would find it. They do not, and it does not.
    """
    y, tr = corr_table.y, segment_table.train
    nmiss = tr[RAW_NUM].isna().sum(1).to_numpy()
    budget = ["daily_screen_time_hours", "social_media_hours", "gaming_hours", "work_study_hours"]
    comp_ok = tr[budget].notna().all(1).to_numpy()

    t = L[:, [idx[n] for n in TREE]].mean(1)
    nn = L[:, [idx[n] for n in NEURAL]].mean(1)
    lk = L[:, idx["L1lookupt"]]

    segs = [("ALL", np.ones(len(y), bool)),
            ("composition complete", comp_ok),
            ("composition broken", ~comp_ok)]
    segs += [(f"n_missing == {k}", nmiss == k) for k in range(5)]

    print(f"{'segment':<24}{'n':>8}{'tree':>9}{'neural':>9}{'lookupT':>9}{'corr t~n':>10}")
    for nm, m in segs:
        if m.sum() < 5000:
            continue
        print(f"{nm:<24}{m.sum():>8}{roc_auc_score(y[m], t[m]):>9.4f}"
              f"{roc_auc_score(y[m], nn[m]):>9.4f}{roc_auc_score(y[m], lk[m]):>9.4f}"
              f"{np.corrcoef(t[m], nn[m])[0, 1]:>10.4f}")
    print("\nThe correlation column is FLAT. There is no regime in which the two axes disagree\n"
          "more, so there is no gating signal for a combiner to exploit -- confirmed in section 3.")


def meta_table(L, folds, base):
    """Can a NONLINEAR combiner extract more from the same 22 legs? No: it extracts less."""
    import lightgbm as lgb
    y, tr = corr_table.y, segment_table.train
    X2 = np.column_stack([L, tr[RAW_NUM].to_numpy(), tr[RAW_NUM].isna().sum(1).to_numpy()])

    def honest_gbm(X):
        p = np.zeros(len(y))
        for k in sorted(set(folds)):
            a, b = folds != k, folds == k
            m = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.03, num_leaves=15,
                                   min_child_samples=200, colsample_bytree=0.7, subsample=0.8,
                                   subsample_freq=1, reg_lambda=5, verbose=-1, random_state=42)
            m.fit(X[a], y[a])
            p[b] = m.predict_proba(X[b])[:, 1]
        return roc_auc_score(y, p)

    print(f"logistic meta, 22 logits            {base:.6f}   (the shipped stack, LB 0.97056)")
    for label, X in [("LightGBM meta, 22 logits", L),
                     ("LightGBM meta, + raw feats", X2)]:
        t0 = time.time()
        a = honest_gbm(X)
        print(f"{label:<36}{a:.6f}  {a - base:+.6f}   [{time.time() - t0:.0f}s]")
    print("\nBoth are NEGATIVE. The linear logit stack is already the right combiner: the legs\n"
          "differ in scale and sign, which a linear model handles, not in WHERE they are right.")


def contribution_table(L, LO, orph_names, folds, base):
    """What do the nine orphans -- every one BELOW the pool median max-corr -- actually add?"""
    y = corr_table.y
    allin = roc_auc_score(y, honest_oof(np.hstack([L, LO]), y, folds, STACK_C))
    keep = [i for i, n in enumerate(orph_names) if n != "A1seed"]
    noseed = roc_auc_score(y, honest_oof(np.hstack([L, LO[:, keep]]), y, folds, STACK_C))
    print(f"22 legs (shipped, LB 0.97056)          {base:.6f}")
    print(f"31 legs (all orphans admitted)         {allin:.6f}  {allin - base:+.6f}")
    print(f"30 legs (no seed twin)                 {noseed:.6f}  {noseed - base:+.6f}")
    print(f"\nadd-one, each orphan against the 22-leg stack:")
    for i, n in enumerate(orph_names):
        a = roc_auc_score(y, honest_oof(np.hstack([L, LO[:, [i]]]), y, folds, STACK_C))
        print(f"  22 + {n:<12} {a:.6f}  {a - base:+.6f}")
    print(f"\nNothing here approaches the +{GATE} gate. Note especially K3tabtf: at max corr\n"
          "0.9251 it is the most decorrelated leg we own and it contributes -0.000001, because\n"
          "0.015 AUC below the pack is noise however uncorrelated the noise happens to be.")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--skip-slow", action="store_true",
                    help="sections 1-2 only (they take seconds; 3-4 take ~10 minutes)")
    args = ap.parse_args()

    # load_legs carries the assertion that every artifact shares the frozen partition. That
    # assertion is the halt condition for this whole file: a mismatch makes the matrices
    # incomparable and every number below meaningless.
    names, P, _, folds, _, truth = load_legs(list(LEGS.values()))
    onames, PO, _, folds_o, _, _ = load_legs(list(ORPHANS.values()))
    assert (folds == folds_o).all(), "orphans are on a DIFFERENT CV partition"
    names = [n.split(":")[0] for n in names]
    rev = {v: k for k, v in LEGS.items()}
    names = [rev[n] for n in names]
    orev = {v: k for k, v in ORPHANS.items()}
    onames = [orev[n.split(":")[0]] for n in onames]

    y = truth["addicted_label"].to_numpy()
    L, LO = to_logit(P), to_logit(PO)
    idx = {n: i for i, n in enumerate(names)}
    corr_table.y = y
    segment_table.train = pd.read_csv(REPO_ROOT / "data" / "train.csv", usecols=RAW_NUM)
    print(f"{len(names)} pool legs + {len(onames)} orphans, {len(y):,} rows, "
          "all on the frozen partition\n")

    print("=" * 78 + "\n1. CORRELATION STRUCTURE\n" + "=" * 78)
    corr_table(names, L, onames, LO)
    print("\n" + "=" * 78 + "\n2. IS THE REDUNDANCY UNIFORM?\n" + "=" * 78)
    segment_table(idx, L)
    if args.skip_slow:
        return
    base = roc_auc_score(y, honest_oof(L, y, folds, STACK_C))
    print("\n" + "=" * 78 + "\n3. CAN A NONLINEAR COMBINER DO BETTER?\n" + "=" * 78)
    meta_table(L, folds, base)
    print("\n" + "=" * 78 + "\n4. WHAT ARE THE NINE ORPHAN LEGS WORTH?\n" + "=" * 78)
    contribution_table(L, LO, onames, folds, base)


if __name__ == "__main__":
    main()
