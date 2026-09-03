#!/usr/bin/env python3
"""Compare two archived OOF probability files.

Two jobs, one tool:

  * NOW -- the gate instrument. Reports delta-AUC between two models together with a
    PAIRED BOOTSTRAP confidence interval on that delta. Paired matters: both models are
    scored on identical rows, so the shared row-difficulty cancels and the CI on the
    difference is far tighter than the CI on either AUC alone. Comparing two models by
    their separate AUC error bars would be the wrong test and would hide real effects.

  * LATER -- the strength/decorrelation tooling for KAGGLE_PLAYBOOK.md section 6. Reports
    disagreement rate alongside solo strength, which is the plot that tells you whether the
    ensemble axis is still open (high disagreement WITHOUT competitive strength is a model
    being wrong in new places, not diversity).

Also verifies the two runs share a byte-identical fold assignment -- if they don't, their
OOF matrices are not comparable and nothing downstream (blending, gating) is valid.

Usage:
    python scripts/compare_oof.py experiments/preds/<champ>/oof_proba_lgb.csv \
                                 experiments/preds/<probe>/oof_proba_lgb.csv
    python scripts/compare_oof.py --all          # every archived OOF vs the best one
"""
import argparse, sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

REPO_ROOT = Path(__file__).resolve().parent.parent
PREDS = REPO_ROOT / "experiments" / "preds"
TRAIN = REPO_ROOT / "data" / "train.csv"


def load_truth():
    tr = pd.read_csv(TRAIN, usecols=["id", "addicted_label"])
    return tr["id"].to_numpy(), tr["addicted_label"].to_numpy()


def load_oof(path, ids):
    df = pd.read_csv(path)
    assert (df["id"].to_numpy() == ids).all(), f"{path}: ids not aligned to train.csv order"
    fold = df["fold"].to_numpy() if "fold" in df else None
    return df["proba"].to_numpy(), fold


def paired_bootstrap(y, pa, pb, n_boot=1000, seed=0):
    """Bootstrap the DIFFERENCE in AUC, resampling rows once and scoring both models on
    the same resample. Returns (mean delta, lo, hi, se) for a 95% interval."""
    rng = np.random.default_rng(seed)
    n = len(y)
    deltas = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        yy = y[idx]
        if yy.min() == yy.max():          # degenerate resample, both classes needed
            deltas[i] = np.nan
            continue
        deltas[i] = roc_auc_score(yy, pb[idx]) - roc_auc_score(yy, pa[idx])
    deltas = deltas[~np.isnan(deltas)]
    return deltas.mean(), np.percentile(deltas, 2.5), np.percentile(deltas, 97.5), deltas.std()


def compare(path_a, path_b, n_boot, seed):
    ids, y = load_truth()
    pa, fa = load_oof(path_a, ids)
    pb, fb = load_oof(path_b, ids)

    if fa is not None and fb is not None:
        same = bool((fa == fb).all())
        print(f"fold assignment identical : {same}")
        if not same:
            print("  *** HALT: different CV partitions. These OOF matrices are NOT comparable,")
            print("      and any blend built from them is invalid. Check the frozen split.")
            return 1
    auc_a, auc_b = roc_auc_score(y, pa), roc_auc_score(y, pb)
    delta = auc_b - auc_a

    print(f"\nA {Path(path_a).parent.name:>12}  AUC = {auc_a:.6f}")
    print(f"B {Path(path_b).parent.name:>12}  AUC = {auc_b:.6f}")
    print(f"delta (B - A)              = {delta:+.6f}")

    print(f"\npaired bootstrap ({n_boot} resamples)...", flush=True)
    m, lo, hi, se = paired_bootstrap(y, pa, pb, n_boot, seed)
    print(f"  delta 95% CI  = [{lo:+.6f}, {hi:+.6f}]")
    print(f"  paired SE     = {se:.6f}   (3*SE = {3*se:.6f})")
    crosses_zero = lo <= 0 <= hi
    print(f"  distinguishable = {not crosses_zero}  "
          f"({'CI crosses zero' if crosses_zero else 'CI excludes zero'})")
    print("  ^ this answers 'are these two models different ON THESE ROWS', NOT 'is B a real")
    print("    improvement'. Two seeds of the SAME config score a 'distinguishable' delta of")
    print("    0.000066 (run 22de9888) -- genuinely different models, but rerunning would move")
    print("    it again. Only the GATE decides shipping; this CI never does.")

    disagree = float(np.mean((pa >= 0.5) != (pb >= 0.5)))
    print(f"\ndisagreement rate (at 0.5)  = {disagree*100:.3f}%")
    print(f"rank correlation (spearman) = {pd.Series(pa).corr(pd.Series(pb), method='spearman'):.5f}")
    print("\nsection 6 reminder: high disagreement WITHOUT competitive solo strength is the")
    print("model being wrong in new places, not diversity worth blending.")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("a", nargs="?", help="baseline oof_proba csv")
    ap.add_argument("b", nargs="?", help="probe oof_proba csv")
    ap.add_argument("--all", action="store_true", help="list every archived OOF with its AUC")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.all:
        ids, y = load_truth()
        rows = []
        for p in sorted(PREDS.glob("*/oof_proba_*.csv")):
            try:
                proba, _ = load_oof(p, ids)
                rows.append({"run": p.parent.name, "file": p.name,
                             "oof_auc": round(roc_auc_score(y, proba), 6)})
            except Exception as e:
                print(f"  skip {p}: {e}", file=sys.stderr)
        print(pd.DataFrame(rows).sort_values("oof_auc", ascending=False).to_string(index=False))
        return 0

    if not (args.a and args.b):
        ap.error("give two oof_proba paths, or --all")
    return compare(args.a, args.b, args.n_boot, args.seed)


if __name__ == "__main__":
    sys.exit(main())
