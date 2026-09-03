#!/usr/bin/env python3
"""Gate 2 for a leg that is a BETTER VERSION of a leg already in the pool.

leg_probe.py answers "what does adding this leg buy?", which is the right question for a
new family. It is the WRONG question for a strict twin of an existing member: the twin
correlates ~0.995 with its own parent, so an add-only measurement reports the redundancy
and hides the upgrade. Every fe_composition probe in the Phase-7 round is of that shape --
the same recipe, one feature family richer -- so all three numbers are reported together:

    base    the shipped 23-leg pool (7f69fcf6, Final A, OOF 0.969434)
    add     base + twin              -- what leg_probe.py reports
    swap    base - parent(s) + twin  -- the upgrade, pool size held constant
    drop    base - parent(s)         -- so the swap can be read against the right control

The swap is the honest one at constant pool size: Phase 5 measured that the CV->LB offset
decays as members are added, i.e. a growing pool inflates the stack's OOF, so "add" and
"swap" are not on the same scale.

    python scripts/leg_swap.py <new_run_id> --replace K1real --replace F1real

Read-only. Writes nothing, ships nothing.
"""
import argparse

import numpy as np
from scipy.special import logit
from sklearn.metrics import roc_auc_score

from stack_logit import EPS, honest_oof, load_legs
from subset_ceiling import LEGS

STACK_C = 0.1
GATE = 0.0002


def score(run_ids):
    _, P, _, folds, _, truth = load_legs(run_ids)
    y = truth["addicted_label"].to_numpy()
    L = logit(np.clip(P, EPS, 1 - EPS))
    return roc_auc_score(y, honest_oof(L, y, folds, STACK_C))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_id")
    ap.add_argument("--replace", action="append", default=[],
                    help="LEGS key(s) this leg supersedes; repeatable")
    ap.add_argument("--gate", type=float, default=GATE)
    ap.add_argument("--name", default=None)
    args = ap.parse_args()

    unknown = [k for k in args.replace if k not in LEGS]
    if unknown:
        raise SystemExit(f"not LEGS keys: {unknown}")

    pool = list(LEGS.values())
    kept = [LEGS[k] for k in LEGS if k not in args.replace]

    base = score(pool)
    add = score(pool + [args.run_id])
    swap = score(kept + [args.run_id])
    drop = score(kept) if args.replace else base

    label = args.name or args.run_id
    rep = "+".join(args.replace) or "(nothing)"
    print(f"\n{label}   superseding {rep}")
    print("-" * 62)
    print(f"{'base  23-leg Final A':<34}{len(pool):>3} legs{base:>13.6f}")
    print(f"{'drop  parent(s) removed':<34}{len(kept):>3} legs{drop:>13.6f}"
          f"   {drop - base:+.6f}")
    print(f"{'add   base + twin':<34}{len(pool)+1:>3} legs{add:>13.6f}"
          f"   {add - base:+.6f}")
    print(f"{'swap  parent(s) -> twin':<34}{len(kept)+1:>3} legs{swap:>13.6f}"
          f"   {swap - base:+.6f}")
    print("-" * 62)
    best = max(add - base, swap - base)
    print(f"{'BEST vs Final A':<34}{'':>8}{best:>+13.6f}   gate {args.gate:+.4f}"
          f"   {'CLEARS' if best >= args.gate else 'MISS'}")
    print(f"{'upgrade over the parent slot':<34}{'':>8}{swap - drop:>+13.6f}"
          "   (swap - drop: what the twin adds where its parent stood)")


if __name__ == "__main__":
    main()
