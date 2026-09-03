#!/usr/bin/env python3
"""Gate 2 for a single new leg: what does it add to the shipped 22-leg logit stack?

This repo has now needed this exact measurement eight times (TabM, fe_impute, the nine
orphans, N1, N2, N3, ...) and has computed it ad hoc every time, which is why the answers
lived in session transcripts instead of in a file. It is one command now.

Gate 1 (solo strength) is necessary but has twice been passed by a leg that then
contributed nothing, so the number that decides anything is the LAST column:

    python scripts/leg_probe.py <run_id> [--gate 0.0002]

Read-only. Writes nothing, ships nothing.
"""
import argparse
import glob

import numpy as np
import pandas as pd
from scipy.special import logit
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from stack_logit import EPS, PREDS, REPO_ROOT, honest_oof, load_legs
from subset_ceiling import LEGS

STACK_C = 0.1          # the shipped stack's C, so the delta is comparable to 7f69fcf6
SHIPPED = 0.969434     # 7f69fcf6, 23 legs, LB 0.97060 -- Final A


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_id", help="run_id of the new leg under experiments/preds/")
    ap.add_argument("--gate", type=float, default=0.0002)
    ap.add_argument("--name", default=None, help="label for the probe in the output")
    args = ap.parse_args()

    if not glob.glob(str(PREDS / args.run_id / "oof_proba_*.csv")):
        raise SystemExit(f"no oof artifact under {PREDS / args.run_id}")

    names, P, _, folds, _, truth = load_legs(list(LEGS.values()))
    rev = {v: k for k, v in LEGS.items()}
    names = [rev[n.split(":")[0]] for n in names]

    # Loading the probe THROUGH load_legs is what guarantees it sits on the frozen
    # partition: the assertion lives in there and raises rather than silently producing an
    # incomparable number. Passing the pool alongside it is what gives it something to
    # assert against (load_legs needs >= 2 legs anyway).
    _, P2, _, folds2, _, _ = load_legs([args.run_id] + list(LEGS.values()))
    assert (folds2 == folds).all(), "probe is on a DIFFERENT CV partition"
    y = truth["addicted_label"].to_numpy()

    L = logit(np.clip(P, EPS, 1 - EPS))            # the 22-leg pool
    lp = logit(np.clip(P2[:, 0], EPS, 1 - EPS))    # the probe
    label = args.name or args.run_id

    solo = roc_auc_score(y, lp)
    c = np.array([np.corrcoef(lp, L[:, k])[0, 1] for k in range(L.shape[1])])
    j = int(c.argmax())
    CC = np.corrcoef(L.T)
    np.fill_diagonal(CC, 0.0)
    pool_med = float(np.median(CC.max(1)))

    base = roc_auc_score(y, honest_oof(L, y, folds, STACK_C))
    with_it = roc_auc_score(y, honest_oof(np.column_stack([L, lp]), y, folds, STACK_C))

    # The fitted weight, on standardised logits so it is comparable across legs.
    X = np.column_stack([L, lp])
    sc = StandardScaler().fit(X)
    w = LogisticRegression(C=STACK_C, max_iter=5000, tol=1e-5).fit(sc.transform(X), y).coef_[0]
    order = np.argsort(-np.abs(w))
    rank = int(np.where(order == len(w) - 1)[0][0]) + 1

    print(f"\n{'':<22}{'value':>12}{'gate':>12}{'verdict':>10}")
    print("-" * 56)
    print(f"{'solo OOF':<22}{solo:>12.6f}{0.9655:>12.4f}"
          f"{('PASS' if solo >= 0.9655 else 'miss'):>10}")
    print(f"{'max |corr| vs pool':<22}{c[j]:>12.4f}{'':>12}{'  vs ' + names[j]:>10}")
    print(f"{'  pool median max-corr':<22}{pool_med:>12.4f}")
    if "L1lookupt" in names:
        print(f"{'  corr vs L1lookupt':<22}{c[names.index('L1lookupt')]:>12.4f}")
    print(f"{'|weight| rank in stack':<22}{rank:>9} of {len(w)}")
    print("-" * 56)
    print(f"{'23-leg stack (shipped)':<22}{base:>12.6f}")
    print(f"{'24-leg stack':<22}{with_it:>12.6f}")
    d = with_it - base
    print(f"{'CONTRIBUTION':<22}{d:>+12.6f}{args.gate:>12.4f}"
          f"{('CLEARS' if d >= args.gate else 'MISS'):>10}")
    if abs(base - SHIPPED) > 5e-6:
        print(f"\nnote: recomputed base {base:.6f} vs the logged {SHIPPED:.6f}")
    verdict = ("gate 2 CLEARED -- this is submittable" if d >= args.gate
               else "misses gate 2. Nothing ships. Record the mechanism.")
    print(f"\n{label}: {verdict}")


if __name__ == "__main__":
    main()
