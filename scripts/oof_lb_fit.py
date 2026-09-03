#!/usr/bin/env python3
"""Fit the OOF -> public-LB relationship over every run that has both, and report it as
a SLOPE plus a RESIDUAL SIGMA -- never as an offset.

KAGGLE_PLAYBOOK.md section 4: the offset (LB - OOF) is a property of which rows landed in
which split, not of your models. S6E8's was 0.00123 on the public 20% and 0.00089 on the
private 80% for the very same submissions, purely because the public slice was easier.
The SLOPE is the part that transferred (0.907 public vs 0.893 private). So anything read
off the offset's level or its drift is inference about the split, and section 5 is the
post-mortem of doing exactly that.

The residual sigma this prints is what sets the shipping gate: convert ~1 sigma into OOF
units by dividing by the slope, and set the gate there or a little above. Once set, it
stays set for the competition -- do NOT re-derive it later from marginal LB deltas.

Usage:
    python scripts/oof_lb_fit.py
    python scripts/oof_lb_fit.py --min-pairs 10
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
RUNS_CSV = REPO_ROOT / "experiments" / "runs.csv"

# README.md section 4, measured in Phase 0.
PUBLIC_PAIRED_SD = 0.000116
SEED_NOISE_FLOOR = 0.000038


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--min-pairs", type=int, default=10,
                    help="refuse to report a slope with fewer paired points than this")
    args = ap.parse_args()

    runs = pd.read_csv(RUNS_CSV)
    d = runs.dropna(subset=["final_oof_auc", "public_lb_score"]).copy()
    d["public_lb_score"] = pd.to_numeric(d["public_lb_score"], errors="coerce")
    d = d.dropna(subset=["public_lb_score"])
    if d.empty:
        raise SystemExit("no runs have both an OOF score and a public LB score yet")

    d = d.sort_values("final_oof_auc")
    d["offset"] = d["public_lb_score"] - d["final_oof_auc"]
    print(f"{len(d)} paired runs\n")
    print(d[["run_id", "run_tag", "final_oof_auc", "public_lb_score", "offset"]]
          .to_string(index=False))

    if len(d) < 3:
        print(f"\nToo few pairs ({len(d)}) to fit anything. Keep submitting: every spent "
              "daily slot is a paired point, and the paired points are the whole "
              "instrument (playbook section 1).")
        return

    x, y = d["final_oof_auc"].values, d["public_lb_score"].values
    slope, intercept = np.polyfit(x, y, 1)
    resid = y - (slope * x + intercept)
    sigma = float(np.std(resid, ddof=2)) if len(d) > 2 else float("nan")
    rho = pd.Series(x).corr(pd.Series(y), method="spearman")

    print(f"\nSpearman(OOF, public LB) = {rho:.4f}   over {len(d)} runs")
    print(f"slope  dLB/dOOF          = {slope:.3f}")
    print(f"residual sigma (LB units)= {sigma:.6f}")
    print(f"offset  mean {d['offset'].mean():+.6f}  range {d['offset'].max()-d['offset'].min():.6f}"
          "   <- a property of the split, not of the models. Do not fit a theory to it.")

    if len(d) < args.min_pairs:
        print(f"\nWARNING: {len(d)} pairs is below --min-pairs={args.min_pairs}. The slope "
              "and sigma above are provisional; do not freeze the shipping gate on them.")
        return

    gate = sigma / slope if slope > 0 else float("nan")
    print(f"\n1 sigma in OOF units = sigma/slope = {gate:.6f}")
    print(f"Suggested shipping gate: {max(gate, SEED_NOISE_FLOOR):.6f} OOF "
          f"(floored at the {SEED_NOISE_FLOOR:.6f} seed-noise floor).")
    print(f"For reference, the public split's own paired dAUC SD is {PUBLIC_PAIRED_SD:.6f}: "
          "a probe below that is invisible on the leaderboard even when it is real.")


if __name__ == "__main__":
    main()
