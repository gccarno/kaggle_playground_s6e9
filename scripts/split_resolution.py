#!/usr/bin/env python3
"""Measure the leaderboard's own resolution, before reading anything into it.

KAGGLE_PLAYBOOK.md section 5 exists because S6E8 built an entire phase on a four-point LB
"trend" whose range was smaller than the split's own paired error bar. The defence is to
compute that error bar on day one and keep it on the wall:

  * PAIRED bootstrap SD of dAUC between two models, at a given row count. This is the
    resolution for "is model B better than model A on the leaderboard?".
  * ABSOLUTE bootstrap SD of a single model's AUC, at the same row count. This is the
    (much larger) spread of a raw score, and it is NOT the right yardstick for comparing
    two of your own submissions. Confusing the two is how a leaderboard eats a month.

Both are computed by resampling our own OOF rows, which is legitimate because the test
split is an i.i.d. draw from the same generator (verified: no train/test drift).

Usage:
    python scripts/split_resolution.py experiments/preds/<runA> experiments/preds/<runB>
    python scripts/split_resolution.py <runA> <runB> --public-frac 0.2 --n-boot 400
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

REPO_ROOT = Path(__file__).resolve().parent.parent
N_TEST = 286_571          # test.csv row count


def load_oof(run_dir):
    run_dir = Path(run_dir)
    if not run_dir.is_absolute():
        run_dir = REPO_ROOT / run_dir
    files = sorted(run_dir.glob("oof_proba_*.csv"))
    if not files:
        raise SystemExit(f"no oof_proba_*.csv in {run_dir}")
    df = pd.read_csv(files[0]).sort_values("id").reset_index(drop=True)
    return df["id"].values, df["proba"].values, files[0].name


def bootstrap(y, pa, pb, n_rows, n_boot, rng):
    """Return (paired dAUC SD, single-score SD) from `n_boot` resamples of `n_rows` rows."""
    d, s = np.empty(n_boot), np.empty(n_boot)
    n = len(y)
    for k in range(n_boot):
        idx = rng.integers(0, n, size=n_rows)
        yy = y[idx]
        if yy.min() == yy.max():        # degenerate draw; vanishingly rare at these sizes
            d[k] = s[k] = np.nan
            continue
        aa, bb = roc_auc_score(yy, pa[idx]), roc_auc_score(yy, pb[idx])
        d[k], s[k] = bb - aa, aa
    return float(np.nanstd(d)), float(np.nanstd(s))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_a"); ap.add_argument("run_b")
    ap.add_argument("--public-frac", type=float, default=0.20,
                    help="assumed public split fraction (Playground default is 0.20)")
    ap.add_argument("--n-boot", type=int, default=400)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    train = pd.read_csv(REPO_ROOT / "data" / "train.csv")[["id", "Will_Buy_EV"]]
    train = train.sort_values("id").reset_index(drop=True)
    y = (train["Will_Buy_EV"] == "Yes").astype(int).values

    ida, pa, na = load_oof(args.run_a)
    idb, pb, nb = load_oof(args.run_b)
    assert (ida == train["id"].values).all() and (idb == train["id"].values).all(), \
        "OOF ids are not aligned to train.csv -- the frozen split has been violated"

    auc_a, auc_b = roc_auc_score(y, pa), roc_auc_score(y, pb)
    print(f"A = {args.run_a} ({na})   OOF AUC {auc_a:.6f}")
    print(f"B = {args.run_b} ({nb})   OOF AUC {auc_b:.6f}")
    print(f"observed dAUC (B-A) on the full {len(y):,} OOF rows: {auc_b - auc_a:+.6f}\n")

    n_pub = int(round(N_TEST * args.public_frac))
    n_pri = N_TEST - n_pub
    sizes = [("public split", n_pub), ("private split", n_pri),
             ("full OOF", len(y))]

    rng = np.random.default_rng(args.seed)
    print(f"{'slice':<16}{'rows':>10}{'paired dAUC SD':>18}{'single-score SD':>18}")
    print("-" * 62)
    for name, n_rows in sizes:
        sd_d, sd_s = bootstrap(y, pa, pb, n_rows, args.n_boot, rng)
        print(f"{name:<16}{n_rows:>10,}{sd_d:>18.6f}{sd_s:>18.6f}")

    print(f"\n({args.n_boot} bootstrap resamples; public split assumed to be "
          f"{args.public_frac:.0%} of the {N_TEST:,} test rows.)")
    print("Read: nothing smaller than the paired SD is a leaderboard result. The private "
          "paired SD is how much room the shakeup has.")


if __name__ == "__main__":
    main()
