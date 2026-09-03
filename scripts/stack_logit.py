#!/usr/bin/env python3
"""Fit a logit-space logistic stack over archived legs, write artifacts, log the run.

The shipping counterpart to scripts/logit_stack.py, which is the diagnostic. Everything
this repo has ever submitted is an EQUAL-WEIGHT mean -- make_blend.py, subset_ceiling.py
and nested_subset.py all search over WHICH legs, never over HOW MUCH of each. The
2026-08-07 public-frontier audit found that both notebooks at the top of the leaderboard
instead fit a logistic regression on the OOF matrix in logit space, and that on our own 16
legs this is worth +0.000243 over the equal-weight champion.

Two mechanisms, and the second is the one that matters:

  * LOGITS, not probabilities. The same fit on raw probabilities buys +0.000027, ten times
    less. This target saturates -- the top screen-time decile is ~100% positive -- so near
    p=1 the probability scale has no resolution left while log(p/(1-p)) still does.
  * A fitted combiner can SUBTRACT. Four of sixteen weights are negative, and K4node takes
    the third largest weight in the whole stack at -0.796. An average can only dilute a
    weak-but-differently-wrong model, which is why the 65,519-subset equal-weight scan put
    NODE in exactly zero of its top 25. Section 6's "legs below 0.965 solo damage the
    blend" is a property of the COMBINER, not of the legs.

HONESTY. The meta-model is refit inside each outer fold and only ever predicts rows it did
not see. One residual optimism is unavoidable and is shared with every public stack: the
base legs' OOF come from models fitted on this same partition, so a meta-model training on
folds 1-4 consumes predictions from base models that saw fold 0. Removing it would mean
retraining every leg. It inflates the stack's OOF by a small constant that the equal-weight
champion does not carry, which is precisely why this run gets SUBMITTED -- the leaderboard
is the arbiter, and the offset prediction is pre-registered in the run notes.

Usage:
    python scripts/stack_logit.py --submit --notes "..."
    python scripts/stack_logit.py --runs 7ebde432 fdeaa047 ... --no-submit
    python scripts/stack_logit.py --pool fixed-rule --submit --notes "..."   # Final B
"""
import argparse
import csv
import glob
import importlib.util
import json
import subprocess
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from subset_ceiling import CHAMPION, LEGS

REPO_ROOT = Path(__file__).resolve().parent.parent
PREDS = REPO_ROOT / "experiments" / "preds"
COMPETITION = "playground-series-s6e9"
EPS = 1e-6
# Two adjacent points on the C plateau rather than one hardcoded value or a full grid.
# The top of the curve is flat to about one part in a million, but its argmax MOVES as the
# member set changes, so a pinned value can silently end up on a slope.
C_GRID = [0.1, 1.0]

# --- Final B's fixed rule (README section 7) ------------------------------------------
# Final A's 23-leg pool is the output of admission gates -- a procedure fitted on data,
# with variance on an unseen split. Final B removes that step by defining its pool with a
# threshold chosen once and applied without judgement, and by pinning C instead of picking
# it. It is EXPECTED to score below Final A; that is what it is for.
# S6E9: DELIBERATELY UNSET. In S6E8 this was 0.965, which was that competition's pool
# floor; S6E9's legs live around 0.945, so carrying the number over would silently
# select nothing. The whole point of Final B is a threshold chosen ONCE, on the record,
# and applied without judgement -- so it is set when the pool actually exists, in
# README.md, and not improvised at the deadline. Until then --pool fixed-rule refuses
# to run rather than quietly producing a wrong pool.
FIXED_RULE_MIN_SOLO = None
FIXED_RULE_C = 0.1
# A stack or blend is a combination of legs, so admitting one would count its members
# twice. Both the tag and the artifact name are checked: the tag is what we write today,
# the artifact name is what the file itself says, and a future run tagged something new
# would still be caught by the second test (including a re-run of this very function).
COMBINATION_TAGS = ("blend", "logit_stack")
COMBINATION_LEARNERS = ("blend", "stack")


def fixed_rule_legs():
    """Every leg we own whose solo OOF clears FIXED_RULE_MIN_SOLO. No gate, no selection.

    Mechanical by construction: the member list is derived from experiments/runs.csv and
    the experiments/preds/ inventory, never hand-typed, so re-running reproduces it.
    """
    if FIXED_RULE_MIN_SOLO is None:
        raise SystemExit(
            "FIXED_RULE_MIN_SOLO is unset for S6E9. Final B's pool threshold must be "
            "chosen once, deliberately, and written into README.md before it is used -- "
            "see the comment at the top of this file. Use --runs or --pool curated "
            "until then.")
    runs = pd.read_csv(REPO_ROOT / "experiments" / "runs.csv")
    keep = []
    for _, row in runs.iterrows():
        rid = str(row["run_id"])
        if rid in keep or str(row.get("run_tag")) in COMBINATION_TAGS:
            continue
        oof = sorted(glob.glob(str(PREDS / rid / "oof_proba_*.csv")))
        test = sorted(glob.glob(str(PREDS / rid / "test_proba_*.csv")))
        if not oof or not test:
            continue
        if Path(oof[0]).stem.replace("oof_proba_", "") in COMBINATION_LEARNERS:
            continue
        solo = pd.to_numeric(row.get("final_oof_auc"), errors="coerce")
        if pd.isna(solo) or float(solo) < FIXED_RULE_MIN_SOLO:
            continue
        keep.append(rid)
    return sorted(keep)


def load_legs(run_ids):
    """Returns (names, OOF matrix, TEST matrix, fold vector, test ids)."""
    truth = pd.read_csv(REPO_ROOT / "data" / "train.csv", usecols=["id", "Will_Buy_EV"])
    names, O, T, folds, ids = [], [], [], None, None
    for r in run_ids:
        d = PREDS / r
        fo = next(iter(sorted(glob.glob(str(d / "oof_proba_*.csv")))), None)
        ft = next(iter(sorted(glob.glob(str(d / "test_proba_*.csv")))), None)
        if fo is None or ft is None:
            print(f"  (skip {r}: no artifacts in {d})")
            continue
        o, t = pd.read_csv(fo), pd.read_csv(ft)
        assert (o["id"].to_numpy() == truth["id"].to_numpy()).all(), f"{r}: OOF ids misaligned"
        if folds is None:
            folds, ids = o["fold"].to_numpy(), t["id"].to_numpy()
        else:
            # A different partition makes the matrices incomparable and any stack built
            # from them invalid. Refuse rather than silently averaging.
            assert (o["fold"].to_numpy() == folds).all(), f"{r}: DIFFERENT CV PARTITION"
            assert (t["id"].to_numpy() == ids).all(), f"{r}: test ids misaligned"
        names.append(f"{r}:{Path(fo).stem.replace('oof_proba_', '')}")
        O.append(o["proba"].to_numpy()); T.append(t["proba"].to_numpy())
    assert len(O) >= 2, "need at least two legs to stack"
    return names, np.column_stack(O), np.column_stack(T), folds, ids, truth


def honest_oof(X, y, folds, C):
    """Refit per outer fold; predict only held-out rows."""
    pred = np.zeros(len(y))
    for k in sorted(set(folds)):
        tr, va = folds != k, folds == k
        sc = StandardScaler().fit(X[tr])
        m = LogisticRegression(C=C, max_iter=5000, tol=1e-5).fit(sc.transform(X[tr]), y[tr])
        pred[va] = m.predict_proba(sc.transform(X[va]))[:, 1]
    return pred


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", default=None,
                    help="run_ids under experiments/preds/ (default: the curated pool)")
    ap.add_argument("--pool", choices=("curated", "fixed-rule"), default="curated",
                    help="curated = the gated LEGS pool (Final A); "
                         "fixed-rule = every leg with solo OOF >= FIXED_RULE_MIN_SOLO, C pinned (Final B)")
    ap.add_argument("--C", type=float, default=None,
                    help="pin the regularisation instead of searching C_GRID")
    ap.add_argument("--submit", action="store_true")
    ap.add_argument("--notes", default="")
    args = ap.parse_args()

    if args.pool == "fixed-rule":
        assert args.runs is None, "--pool fixed-rule derives its own members; do not pass --runs"
        runs = fixed_rule_legs()
        # Pinning C is part of the rule: selecting it on the grid is a fitted step.
        args.C = FIXED_RULE_C if args.C is None else args.C
        print(f"fixed rule: solo OOF >= {FIXED_RULE_MIN_SOLO}, C pinned at {args.C}\n"
              f"{len(runs)} legs admitted: {' '.join(runs)}\n")
    else:
        runs = args.runs if args.runs is not None else list(LEGS.values())

    names, P, PT, folds, ids, truth = load_legs(runs)
    y = (truth["Will_Buy_EV"] == "Yes").astype(int).to_numpy()
    L = logit(np.clip(P, EPS, 1 - EPS))
    LT = logit(np.clip(PT, EPS, 1 - EPS))
    print(f"{len(names)} legs, {len(y):,} rows\n")

    # Reference point: the equal-weight champion needs no fitting, so unlike the stack its
    # OOF carries no meta-fit optimism at all. That asymmetry is the whole reason to submit.
    idx = {n.split(":")[0]: i for i, n in enumerate(names)}
    champ_ids = [LEGS[c] for c in CHAMPION]
    a_champ = (roc_auc_score(y, P[:, [idx[r] for r in champ_ids]].mean(1))
               if all(r in idx for r in champ_ids) else float("nan"))

    best = (-1.0, None, None)
    for C in (C_GRID if args.C is None else [args.C]):
        p = honest_oof(L, y, folds, C)
        a = roc_auc_score(y, p)
        print(f"  stack C={C:<5g} honest OOF {a:.6f}   vs equal-weight champion {a - a_champ:+.6f}")
        if a > best[0]:
            best = (a, C, p)
    auc, C, stack_oof = best
    print(f"\n{'pinned' if args.C is not None else 'selected'} C={C}, OOF {auc:.6f}")

    label = (f"Final B fixed-rule stack, {len(names)} legs, C={C}" if args.pool == "fixed-rule"
             else f"logit stack, {len(names)} legs, C={C}")

    # Test predictions come from a meta-model fit on ALL OOF rows. Known and accepted
    # asymmetry: test predictions are averages over 5 fold-models while OOF predictions
    # come from one model each, so the member spread is slightly smaller on test. It shifts
    # the scale, not the ranking, and AUC only reads the ranking.
    sc = StandardScaler().fit(L)
    final = LogisticRegression(C=C, max_iter=5000, tol=1e-5).fit(sc.transform(L), y)
    test_pred = final.predict_proba(sc.transform(LT))[:, 1]

    w = pd.DataFrame({"leg": [n.split(":")[0] for n in names], "learner": [n.split(":")[1] for n in names],
                      "solo": [roc_auc_score(y, P[:, i]) for i in range(len(names))],
                      "weight": final.coef_[0]})
    print("\n" + w.reindex(w.weight.abs().sort_values(ascending=False).index)
          .to_string(index=False, float_format="%.4f"))
    print(f"\nnegative weights: {(final.coef_[0] < 0).sum()} of {len(names)}")

    run_id = uuid.uuid4().hex[:8]
    dest = PREDS / run_id
    dest.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"id": truth["id"], "fold": folds, "proba": stack_oof}).to_csv(
        dest / "oof_proba_stack.csv", index=False)
    pd.DataFrame({"id": ids, "proba": test_pred}).to_csv(dest / "test_proba_stack.csv", index=False)
    sub = pd.DataFrame({"id": ids, "Will_Buy_EV": test_pred})
    sub.to_csv(dest / "submission.csv", index=False)
    (dest / "manifest.json").write_text(json.dumps(
        {"sources": [n.split(":")[0] for n in names], "learners": names, "C": C,
         "pool": args.pool, "C_pinned": args.C is not None,
         "coef": final.coef_[0].tolist(), "oof_auc": auc,
         "equal_weight_champion_oof": a_champ}, indent=1), encoding="utf-8")

    ss = pd.read_csv(REPO_ROOT / "data" / "sample_submission.csv")
    assert len(sub) == len(ss) and (sub["id"].values == ss["id"].values).all()
    assert sub["Will_Buy_EV"].notna().all() and sub["Will_Buy_EV"].between(0, 1).all()
    assert sub["Will_Buy_EV"].nunique() > 1000, "degenerate predictions"
    print(f"\nsubmission validated -> {dest / 'submission.csv'}")

    score = ""
    if args.submit:
        out = subprocess.run(["kaggle", "competitions", "submit", COMPETITION,
                              "-f", str(dest / "submission.csv"),
                              "-m", label],
                             capture_output=True, text=True, cwd=REPO_ROOT)
        print(out.stdout or out.stderr)
        if out.returncode != 0:
            raise SystemExit(f"submit failed: {out.stderr}")
        for _ in range(40):
            q = subprocess.run(["kaggle", "competitions", "submissions", COMPETITION, "--csv"],
                               capture_output=True, text=True, cwd=REPO_ROOT)
            rows = list(csv.DictReader(q.stdout.splitlines()))
            if rows:
                s = rows[0].get("publicScore") or rows[0].get("public_score")
                if s not in (None, "", "None", "pending"):
                    score = s
                    break
            time.sleep(30)
        print(f"Public LB = {score}")

    spec = importlib.util.spec_from_file_location("cr", Path(__file__).with_name("collect_run.py"))
    cr = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cr)
    cr.append_run_row({
        "run_id": run_id,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                     text=True, cwd=REPO_ROOT).stdout.strip(),
        "kernel_ref": "stack", "run_tag": "logit_stack",
        "description": label,
        "final_oof_auc": round(float(auc), 6), "public_lb_score": score,
        "n_folds": 5, "cv_seed": 42,
        "preds_dir": dest.relative_to(REPO_ROOT).as_posix(), "notes": args.notes,
    })
    print(f"Appended stack run {run_id} to experiments/runs.csv")


if __name__ == "__main__":
    main()
