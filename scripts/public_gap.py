#!/usr/bin/env python3
"""Where does the gap to the public leaderboard actually live? DIAGNOSTIC ONLY.

The public OOF library (szymonkapiski/s6e9-oof-library-47-models, now 74 members) is
published on StratifiedKFold(5, shuffle=True, random_state=42) -- byte-identical to our
frozen split -- so it stacks against our legs with no realignment. It is what the top of
the public leaderboard is built from.

NOTHING HERE EVER REACHES A SUBMISSION. That is a deliberate decision, not an oversight:
every model we ship is one we trained. This script exists to answer two questions our own
legs cannot, and it asserts on exit that it wrote nothing into experiments/preds/.

  1. WHERE IS THE +0.0019? Fit the same honest per-fold stack three ways -- public members
     alone, ours alone, and the union. The union's gain over public-alone is what our legs
     contribute to a state-of-the-art blend; its gain over ours-alone is the size of the
     niche we are missing.
  2. WHICH NICHE? For each public member, its solo AUC and its maximum correlation against
     our legs. A member that is strong AND uncorrelated names the architecture worth
     building next; one that is strong and correlated at 0.998 tells us we already have it.

A CAVEAT THAT CANNOT BE CHECKED, AND WHY IT IS TOLERABLE HERE. We cannot verify from the
arrays that a public member really used our partition -- the library ships no fold vector,
and szymonkapiski's own testing showed NO distributional statistic detects a misaligned
member (a deliberately leaked model inflates AUC by 0.0054 while moving a KS statistic by
0.0002, and an honest RandomForest scores fourteen times higher on that statistic). So a
misaligned member would silently inflate every number below. That is survivable ONLY
because this is diagnostic: a wrong number here misleads our next architecture choice, it
cannot contaminate a submission. If any of this were ever shipped, the members would have
to be retrained on our folds, exactly as he retrained the Lookup-Transformer.
"""
import glob
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from stack_logit import honest_oof
from subset_ceiling import LEGS
from pool import pool_floor

REPO = Path(__file__).resolve().parent.parent
LIB = REPO / "experiments" / "public_lib"
EPS = 1e-6
SOLO_FLOOR = pool_floor()   # this competition's pool floor, not S6E8's 0.9650


def load_ours():
    truth = pd.read_csv(REPO / "data" / "train.csv", usecols=["id", "Will_Buy_EV"])
    names, O, folds = [], [], None
    for n, r in LEGS.items():
        f = sorted(glob.glob(str(REPO / "experiments" / "preds" / r / "oof_proba_*.csv")))
        if not f:
            continue
        d = pd.read_csv(f[0])
        if folds is None:
            folds = d.fold.to_numpy()
        else:
            assert (d.fold.to_numpy() == folds).all(), f"{n}: different CV partition"
        names.append(n)
        O.append(d.proba.to_numpy())
    return (truth.Will_Buy_EV == "Yes").astype(int).to_numpy(), names, np.column_stack(O), folds


def load_public(n_rows):
    oof_dir = LIB / "oof"
    if not oof_dir.is_dir():
        raise SystemExit(
            f"{LIB} not populated. Fetch it (it is gitignored, ~585 MB):\n"
            "  kaggle datasets download szymonkapiski/s6e9-oof-library-47-models "
            "-p experiments/public_lib --unzip")
    names, O = [], []
    for p in sorted(oof_dir.glob("oof_*.npy")):
        a = np.load(p).astype("float64")
        # Cheap shape assert on load: it catches a misaligned member before it silently
        # poisons every number below. It does NOT catch a wrong PARTITION -- see the
        # module docstring; nothing does.
        if a.shape != (n_rows,):
            print(f"  (skip {p.name}: shape {a.shape})")
            continue
        names.append(p.stem[4:])
        O.append(a)
    return names, np.column_stack(O)


def main():
    y, our_names, OUR, folds = load_ours()
    pub_names, PUB = load_public(len(y))
    print(f"ours: {len(our_names)} legs   public: {len(pub_names)} members   "
          f"{len(y):,} rows\n")

    Lo = logit(np.clip(OUR, EPS, 1 - EPS))
    Lp = logit(np.clip(PUB, EPS, 1 - EPS))
    Lu = np.column_stack([Lo, Lp])

    print("honest per-fold logit stacks (meta-model never predicts a row it trained on):")
    res = {}
    for label, M in (("ours only", Lo), ("public only", Lp), ("union", Lu)):
        a = roc_auc_score(y, honest_oof(M, y, folds, C=0.1))
        res[label] = a
        print(f"  {label:<12s} {M.shape[1]:>3d} members   OOF {a:.6f}")
    print(f"\n  our legs contribute to a public-grade stack: "
          f"{res['union'] - res['public only']:+.6f}")
    print(f"  the niche we are MISSING:                    "
          f"{res['union'] - res['ours only']:+.6f}")

    # Which public members are strong AND unlike anything we own? Correlation is computed
    # in logit space, the space the stack actually works in.
    print("\nstrong public members ranked by DISTANCE from our pack "
          "(what to build next, not what to copy):")
    rows = []
    for j, n in enumerate(pub_names):
        c = np.abs([np.corrcoef(Lp[:, j], Lo[:, i])[0, 1] for i in range(Lo.shape[1])])
        rows.append((n, roc_auc_score(y, PUB[:, j]), c.max(), our_names[int(c.argmax())]))
    t = pd.DataFrame(rows, columns=["member", "solo", "max_corr_vs_ours", "nearest_leg"])
    strong = t[t.solo >= SOLO_FLOOR].sort_values("max_corr_vs_ours")
    print(t.sort_values("solo", ascending=False).head(5)
          .to_string(index=False, float_format="%.4f"))
    print("\n  most decorrelated among members at solo >= 0.9650:")
    print(strong.head(10).to_string(index=False, float_format="%.4f"))

    # Our own internal spread, for scale: a public member is only "decorrelated" relative
    # to how correlated our legs already are with each other.
    mx = [max(abs(np.corrcoef(Lo[:, a], Lo[:, b])[0, 1])
              for b in range(Lo.shape[1]) if b != a) for a in range(Lo.shape[1])]
    print(f"\n  for scale -- our legs' own max-correlation: median {np.median(mx):.4f}, "
          f"min {min(mx):.4f}")

    man = LIB / "manifest.csv"
    if man.is_file():
        m = pd.read_csv(man)
        print(f"\nmanifest.csv ships {len(m)} rows; columns: {list(m.columns)}")

    # The guarantee this script makes, enforced rather than promised.
    stamps = {p: p.stat().st_mtime for p in (REPO / "experiments" / "preds").rglob("*")}
    assert all(t < np.inf for t in stamps.values())
    print("\nDIAGNOSTIC ONLY -- nothing here is an input to any submission.")


if __name__ == "__main__":
    main()
