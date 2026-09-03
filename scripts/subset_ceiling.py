#!/usr/bin/env python3
"""Ceiling diagnostic: does ANY equal-weight subset of our legs clear the gate?

This exists to answer one question honestly -- "is the +0.0002 gate reachable at all by
recombining what we already own, or has the ensemble axis genuinely closed?" -- after three
separate hand-picked combinations landed at 0.9x the gate.

IT IS A DIAGNOSTIC AND NOTHING MAY BE SHIPPED FROM IT. The reported maximum is a maximum
over thousands of subsets scored on the SAME OOF rows, which is precisely the selection
bias KAGGLE_PLAYBOOK.md section 5 warns about -- a hill-climb over many models found
+0.00014 in S6E7 and it was pure noise. A high number here means "the ceiling is at least
this", not "ship this".

Two stages, because 16k subsets x 691k rows of roc_auc_score is ~40 minutes:
  1. screen every subset on a fixed 200k-row stratified subsample using a rank-based AUC;
  2. re-score the top survivors on the FULL OOF, which is the only number quoted.
"""
import glob
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

REPO = Path(__file__).resolve().parent.parent
LEGS = {"B6": "f64e2781", "B7": "6e3dd7c3", "B8": "9e68c7ce", "B10x": "7ebde432",
        "C1": "6a091632", "C2": "5daf6c12", "C3x": "0c32463c", "C4x": "fdeaa047",
        "D1cat": "7c1e9334", "E1mlp": "2fb89920", "E2emb": "8bd89dee",
        "F1real": "53c678f6", "G1ftt": "b2f35b4a",
        # newly admitted: the T4 reruns (permitted -- not treated as seed bagging) and
        # NODE, which is the only leg roughly equidistant from BOTH the tree and neural
        # axes (3.427% vs trees, 3.245% vs neural) and so is excluded from the 0.965
        # screening rule on purpose -- that rule was derived from three unrelated legs.
        "K1real": "cd2dbce4", "K2ftt": "46859c68", "K4node": "976c2703",
        # Phase 2. The composition legs carry the generator's budget constraint, which
        # cleared on all three tree engines at 5-6x the gate; L1lookupt is the
        # Lookup-Transformer, the first leg here that is BOTH the strongest solo and the
        # most decorrelated (max corr 0.9796 against a pack whose own median max-corr is
        # 0.9941), which is the counter-example to the section 6 wall.
        "B2xgb": "f047412f", "B2blgb": "7d1de67e", "B2ccat": "e73f6257",
        "L1lookupt": "dada9e2d", "M1tabm": "f1d249bd", "B3imp": "54db2990",
        # Phase 4. R1mnca is Final A's 23rd leg (7f69fcf6, OOF 0.969434, LB 0.97060).
        # It was left out of this dict when it was run, so every gate-2 measurement
        # since Phase 4 has been taken against a 22-leg pool that is NOT what shipped.
        "R1mnca": "9107a864"}
# G2/K3 TabTransformer (0.9533) and E3 raw-MLP (0.9405) stay out. The original reason given
# here -- "measured strictly harmful" -- was measured under an EQUAL-WEIGHT MEAN, where a weak
# leg can only dilute; it is not evidence about a fitted combiner, which can subtract. They
# have since been re-measured under the fitted logit stack by scripts/leg_diversity.py, along
# with the other six Phase 1 orphans, and they still do not belong: G2 and K3 each contribute
# -0.000001 and E3 +0.000008, all nine together +0.000047, a quarter of the gate. K3 is worth
# singling out because it has max corr 0.9251 against a pool whose median is 0.9947 -- the most
# decorrelated leg we own, contributing nothing. Decorrelation is not sufficient; strength is
# the binding constraint. Keeping them out ALSO keeps this search tractable: each leg doubles it.
CHAMPION = ["B10x", "C4x", "D1cat", "E2emb", "F1real"]   # Final A, ffb65555, LB 0.96892
GATE = 0.0002


def fast_auc(y, s):
    """Mann-Whitney AUC. Ranks via scipy (C-level tie averaging), counts in FLOAT.

    Both details are load-bearing and were wrong in the first version:
      * n_pos * (n_pos + 1) and n_pos * n_neg overflow numpy int32 on Windows once n is
        a few hundred thousand (142,000 * 142,001 = 2.0e10 vs int32 max 2.1e9). It wraps
        SILENTLY and returns a wrong number. Casting to float first is the fix.
      * the original averaged ties in a Python loop over tie groups, which is O(n)
        interpreted iterations per call -- it made this no faster than sklearn.

    Validated against roc_auc_score at 200k and 691k rows, i.e. the sizes actually used,
    not the small ones the first check used.
    """
    ranks = rankdata(s)                       # average ranks within ties, in C
    n_pos = float(y.sum())
    n_neg = float(len(y)) - n_pos
    return (float(ranks[y == 1].sum()) - n_pos * (n_pos + 1.0) / 2.0) / (n_pos * n_neg)


def main():
    y = pd.read_csv(REPO / "data" / "train.csv",
                    usecols=["addicted_label"]).addicted_label.to_numpy()
    names, cols, folds = [], [], None
    for n, r in LEGS.items():
        f = glob.glob(str(REPO / "experiments" / "preds" / r / "oof_proba_*.csv"))
        if not f:
            print(f"  (skip {n}: no artifact)"); continue
        d = pd.read_csv(f[0])
        if folds is None:
            folds = d.fold.to_numpy()
        else:
            assert (d.fold.to_numpy() == folds).all(), f"{n}: different CV partition"
        names.append(n); cols.append(d.proba.to_numpy())
    A = np.vstack(cols)
    idx = {n: i for i, n in enumerate(names)}
    champ_full = roc_auc_score(y, A[[idx[n] for n in CHAMPION]].mean(0))
    print(f"{len(names)} legs; champion {'+'.join(CHAMPION)} = {champ_full:.6f}; gate +{GATE}")

    rng = np.random.default_rng(42)
    sub = np.sort(rng.choice(len(y), 200_000, replace=False))
    ys, As = y[sub], A[:, sub]

    combos = [c for r in range(2, len(names) + 1)
              for c in itertools.combinations(range(len(names)), r)]
    print(f"stage 1: screening {len(combos):,} equal-weight subsets on 200k rows", flush=True)
    scored = [(fast_auc(ys, As[list(c)].mean(0)), c) for c in combos]
    scored.sort(reverse=True)

    print("stage 2: re-scoring the top 25 on the FULL OOF\n", flush=True)
    final = sorted(((roc_auc_score(y, A[list(c)].mean(0)), c) for _, c in scored[:25]),
                   reverse=True)
    print(f"{'full-OOF':>10}{'vs champ':>11}  legs")
    for s, c in final:
        flag = "  <-- clears (DIAGNOSTIC ONLY)" if s - champ_full >= GATE else ""
        print(f"{s:>10.6f}{s - champ_full:>+11.6f}  {'+'.join(names[i] for i in c)}{flag}")

    n_clear = sum(1 for s, _ in final if s - champ_full >= GATE)
    print(f"\n{n_clear} of the top 25 clear +{GATE} on full OOF.")
    print("Selection bias warning: this is a maximum over "
          f"{len(combos):,} subsets scored on the same rows. Section 5 of the playbook")
    print("says a hill-climb at this scale is noise. Nothing ships from this table.")


if __name__ == "__main__":
    main()
