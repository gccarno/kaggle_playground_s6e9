#!/usr/bin/env python3
"""The S6E9 ensemble pool, in one place.

S6E8's analysis tools each carried their own hard-coded dict of run_ids plus a `SHIPPED`
constant naming the champion's OOF. That worked, and it also produced the bug recorded in
subset_ceiling.py's own comment: Final A's 23rd leg "was left out of this dict when it was
run, so every gate-2 measurement since Phase 4 has been taken against a 22-leg pool that
is NOT what shipped." Every stack contribution measured in that window was against the
wrong baseline.

So here the pool lives in ONE tracked file, experiments/pool.json, and every tool reads
it. Adding a leg is one edit in one place.

    python scripts/pool.py                       # show the pool and its floor
    python scripts/pool.py --add E3 1a2b3c4d     # append a leg
    python scripts/pool.py --set-champion 0.9457 # record the shipped stack's OOF
"""
import argparse, json
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
POOL_JSON = REPO_ROOT / "experiments" / "pool.json"
RUNS_CSV = REPO_ROOT / "experiments" / "runs.csv"

DEFAULT = {
    "legs": {},               # tag -> run_id
    "stack_C": 0.1,           # the shipped stack's regularization constant
    "champion_stack_oof": None,   # OOF of the stack currently shipped; None until one is
}


def load_pool():
    if not POOL_JSON.exists():
        return dict(DEFAULT)
    return {**DEFAULT, **json.loads(POOL_JSON.read_text(encoding="utf-8"))}


def save_pool(pool):
    POOL_JSON.parent.mkdir(parents=True, exist_ok=True)
    POOL_JSON.write_text(json.dumps(pool, indent=2) + "\n", encoding="utf-8")


def legs():
    """tag -> run_id for every member of the current pool."""
    return dict(load_pool()["legs"])


def solo_scores():
    """run_id -> solo OOF AUC, read from the run log."""
    runs = pd.read_csv(RUNS_CSV)
    runs["final_oof_auc"] = pd.to_numeric(runs["final_oof_auc"], errors="coerce")
    return dict(zip(runs["run_id"].astype(str), runs["final_oof_auc"]))


def pool_floor():
    """The weakest solo score currently IN the pool.

    Used as the display gate for a new leg. It is deliberately derived from the pool
    rather than pinned: S6E8's 0.9655 constant does not transfer to a competition whose
    legs live around 0.945, and a stale constant is worse than no constant because it
    still prints a verdict.
    """
    s, L = solo_scores(), legs()
    vals = [s[r] for r in L.values() if r in s and pd.notna(s[r])]
    if not vals:
        raise SystemExit("the pool is empty -- add legs with scripts/pool.py --add")
    return min(vals)


def champion_oof():
    return load_pool()["champion_stack_oof"]


def stack_C():
    return load_pool()["stack_C"]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--add", nargs=2, metavar=("TAG", "RUN_ID"))
    ap.add_argument("--remove", metavar="TAG")
    ap.add_argument("--set-champion", type=float, metavar="OOF")
    ap.add_argument("--set-C", type=float, metavar="C")
    args = ap.parse_args()

    pool = load_pool()
    if args.add:
        tag, rid = args.add
        if not (REPO_ROOT / "experiments" / "preds" / rid).exists():
            raise SystemExit(f"no artifacts under experiments/preds/{rid}")
        pool["legs"][tag] = rid
    if args.remove:
        pool["legs"].pop(args.remove, None)
    if args.set_champion is not None:
        pool["champion_stack_oof"] = args.set_champion
    if args.set_C is not None:
        pool["stack_C"] = args.set_C
    if any([args.add, args.remove, args.set_champion is not None, args.set_C is not None]):
        save_pool(pool)

    s = solo_scores()
    print(f"pool: {len(pool['legs'])} legs   C={pool['stack_C']}   "
          f"champion stack OOF={pool['champion_stack_oof']}")
    for tag, rid in sorted(pool["legs"].items(), key=lambda kv: -(s.get(kv[1]) or 0)):
        print(f"  {tag:<12} {rid}  solo {s.get(rid, float('nan')):.6f}")
    if pool["legs"]:
        print(f"pool floor (weakest solo): {pool_floor():.6f}")


if __name__ == "__main__":
    main()
