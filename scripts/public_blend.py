#!/usr/bin/env python3
"""Fixed weighted logit-mean blend of OUR archived champion with PUBLIC OOF artifacts.

Phase 23. The artifact-sharing policy of README section 5 was reopened on 2026-09-21, the
scheduled date, and opened: public artifacts may now enter a shipped model. The hedge is
structural -- Final A stays ours-only, Final B carries the public mix -- so every file this
script writes records exactly which outside artifacts went into it. A blend whose provenance
cannot be read back off disk is not selectable as a final.

WHY THIS IS FOLD-HONEST. The public libraries ship OOF arrays over train.csv in original row
order, and dariushafshar's `folds_seed42.npy` is bit-identical to this repo's frozen split
(checked 2026-09-21). But alignment is not what makes this honest: a FIXED blend fits nothing,
and every leg's OOF prediction for a row comes from a model that never saw that row. So the
blend's OOF is directly comparable to a single leg's, exactly as in stack_logit.py's
logit_mean mode. A FITTED stack over these same legs would NOT be, because a 10-fold leg's
prediction for a row in our fold 0 comes from a model that trained on 90% of the file,
including rows the meta-model is being scored on. That is why this script only ever averages.

THE 10-FOLD DISCOUNT, measured on our own archive and pre-registered here. Run `e1495238`
(X4_10fold) scored OOF 0.946122 -> LB 0.94617, while TEX at the IDENTICAL OOF 0.946122 scored
0.94635. A 10-fold OOF therefore reads about 0.00018 optimistic against this competition's
leaderboard. Most strong public artifacts are 10-fold, so `fold_count` is carried in the
registry and printed with every blend: a headline OOF gain that comes mostly from 10-fold legs
should be discounted before it is read as a result.

Usage:
    python scripts/public_blend.py --ours-weight 0.50 --public 6view rmlp --no-submit
    python scripts/public_blend.py --ours-weight 0.25 --public 6view rmlp \
        --notes "HYPOTHESIS ... GATE ..." --submit
"""
import argparse
import csv
import importlib.util
import json
import subprocess
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.metrics import roc_auc_score

REPO_ROOT = Path(__file__).resolve().parent.parent
PREDS = REPO_ROOT / "experiments" / "preds"
PUBLIC = REPO_ROOT / "data" / "public"
COMPETITION = "playground-series-s6e9"
EPS = 1e-6

OURS_DEFAULT = "61fb5598"  # Phase 22 champion: fixed logit-mean blend(TEX, TEXF)

# name -> (oof csv, oof column, test csv, test column, fold_count, provenance)
# fold_count is the LEG's own CV, not ours: 10 means apply the discount in the docstring.
REGISTRY = {
    "6view": ("s6e9-six-feature-views-oof-library/oof_six_views.csv", "ensemble",
              "s6e9-six-feature-views-oof-library/test_six_views.csv", "ensemble",
              10, "megayak/s6e9-six-feature-views-oof-library"),
    "rmlp": ("s6e9-six-feature-views-oof-library/oof_realmlp_g.csv", "G_realmlp_3seed",
             "s6e9-six-feature-views-oof-library/test_realmlp_g.csv", "G_realmlp_3seed",
             10, "megayak/s6e9-six-feature-views-oof-library (RealMLP, non-GBDT)"),
    "xgb5f": ("s6e9-oof/XGBoost_Triple_TE_5folds_oof.csv", None,
              "s6e9-oof/XGBoost_Triple_TE_5folds_test.csv", None,
              5, "najiama/s6e9-oof"),
    "lgbV5": ("s6e9-oof/Pure LGBM_V5_oof.csv", None,
              "s6e9-oof/Pure LGBM_V5_test.csv", None,
              5, "najiama/s6e9-oof"),
}


def _col(df, col):
    """The registry names a column only where the file has several."""
    if col is not None:
        return df[col].values
    rest = [c for c in df.columns if c != "id"]
    assert len(rest) == 1, f"ambiguous columns {rest}; name one in REGISTRY"
    return df[rest[0]].values


def load_public(name, train_ids, test_ids):
    oof_f, oof_c, test_f, test_c, nfold, src = REGISTRY[name]
    o = pd.read_csv(PUBLIC / oof_f).sort_values("id")
    t = pd.read_csv(PUBLIC / test_f).sort_values("id")
    assert (o["id"].values == train_ids).all(), f"{name}: train ids misaligned"
    assert (t["id"].values == test_ids).all(), f"{name}: test ids misaligned"
    return (logit(np.clip(_col(o, oof_c), EPS, 1 - EPS)).astype(np.float32),
            logit(np.clip(_col(t, test_c), EPS, 1 - EPS)).astype(np.float32), nfold, src)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ours", default=OURS_DEFAULT, help="archived run_id of our own side")
    ap.add_argument("--ours-weight", type=float, required=True,
                    help="weight on our logit; the public legs share (1 - w) equally")
    ap.add_argument("--public", nargs="+", required=True, choices=sorted(REGISTRY))
    ap.add_argument("--submit", action="store_true")
    ap.add_argument("--notes", default="")
    ap.add_argument("--description", default="")
    args = ap.parse_args()
    assert 0.0 < args.ours_weight <= 1.0, "--ours-weight must be in (0, 1]"

    truth = pd.read_csv(REPO_ROOT / "data" / "train.csv", usecols=["id", "Will_Buy_EV"])
    y = (truth["Will_Buy_EV"] == "Yes").astype(int).to_numpy()
    test_ids = np.sort(pd.read_csv(REPO_ROOT / "data" / "test.csv", usecols=["id"])["id"].values)

    ours_dir = PREDS / args.ours
    oof_f = sorted(ours_dir.glob("oof_proba_*.csv"))[0]
    test_f = sorted(ours_dir.glob("test_proba_*.csv"))[0]
    od = pd.read_csv(oof_f).sort_values("id")
    td = pd.read_csv(test_f).sort_values("id")
    assert (od["id"].values == truth["id"].values).all(), "our OOF ids misaligned"
    assert (td["id"].values == test_ids).all(), "our test ids misaligned"
    ours_oof = logit(np.clip(od["proba"].values, EPS, 1 - EPS)).astype(np.float32)
    ours_test = logit(np.clip(td["proba"].values, EPS, 1 - EPS)).astype(np.float32)
    a_ours = roc_auc_score(y, ours_oof)

    w = args.ours_weight
    pub_oof, pub_test, rows = [], [], []
    for n in args.public:
        lo, lt, nfold, src = load_public(n, truth["id"].values, test_ids)
        pub_oof.append(lo)
        pub_test.append(lt)
        rows.append(dict(leg=n, solo_oof=roc_auc_score(y, lo), leg_folds=nfold,
                         weight=(1 - w) / len(args.public), source=src))
    blend_oof = w * ours_oof + (1 - w) * np.mean(pub_oof, axis=0)
    blend_test = w * ours_test + (1 - w) * np.mean(pub_test, axis=0)
    auc = roc_auc_score(y, blend_oof)

    tab = pd.DataFrame([dict(leg=f"OURS:{args.ours}", solo_oof=a_ours, leg_folds=5,
                             weight=w, source="this repo")] + rows)
    print(tab.to_string(index=False, float_format="%.6f"))
    n10 = sum(r["leg_folds"] == 10 for r in rows)
    print(f"\nblend OOF {auc:.6f}   vs our side alone {auc - a_ours:+.6f}"
          f"   (shipping gate +0.0000949)")
    if n10:
        print(f"CAUTION: {n10} of {len(rows)} public legs are 10-fold OOF, which reads "
              f"~0.00018 optimistic against this LB (run e1495238). Discount before reading "
              f"this gain as a result; the LB is the arbiter on this axis, not our OOF.")

    run_id = uuid.uuid4().hex[:8]
    dest = PREDS / run_id
    dest.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"id": truth["id"], "fold": od.get("fold", pd.Series(-1, index=od.index)),
                  "proba": expit(blend_oof)}).to_csv(dest / "oof_proba_blend.csv", index=False)
    pd.DataFrame({"id": test_ids, "proba": expit(blend_test)}).to_csv(
        dest / "test_proba_blend.csv", index=False)
    sub = pd.DataFrame({"id": test_ids, "Will_Buy_EV": expit(blend_test)})
    sub.to_csv(dest / "submission.csv", index=False)

    label = args.description or (
        f"public-mix logit blend: OURS({args.ours})*{w:.2f} + "
        f"{'+'.join(args.public)}*{(1 - w):.2f}")
    (dest / "manifest.json").write_text(json.dumps(
        {"mode": "public_weighted_logit_mean", "ours_run": args.ours, "ours_weight": w,
         "ours_solo_oof": a_ours, "public_legs": rows, "oof_auc": auc,
         "n_public_legs_10fold": n10, "policy": "README section 5 reopened 2026-09-21: OPEN",
         "label": label}, indent=1), encoding="utf-8")

    ss = pd.read_csv(REPO_ROOT / "data" / "sample_submission.csv")
    assert len(sub) == len(ss) and (sub["id"].values == ss["id"].values).all()
    assert sub["Will_Buy_EV"].notna().all() and sub["Will_Buy_EV"].between(0, 1).all()
    assert sub["Will_Buy_EV"].nunique() > 1000, "degenerate predictions"
    print(f"submission validated -> {dest / 'submission.csv'}")

    score = ""
    if args.submit:
        out = subprocess.run(["kaggle", "competitions", "submit", COMPETITION,
                              "-f", str(dest / "submission.csv"), "-m", label],
                             capture_output=True, text=True, cwd=REPO_ROOT)
        print(out.stdout or out.stderr)
        if out.returncode != 0:
            raise SystemExit(f"submit failed: {out.stderr}")
        for _ in range(40):
            q = subprocess.run(["kaggle", "competitions", "submissions", COMPETITION, "--csv"],
                               capture_output=True, text=True, cwd=REPO_ROOT)
            r = list(csv.DictReader(q.stdout.splitlines()))
            if r:
                s = r[0].get("publicScore") or r[0].get("public_score")
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
        # "blend" is a COMBINATION_TAG, so this can never be readmitted as a leg of a later
        # stack -- that would count our champion, and each public leg, twice.
        "kernel_ref": "public_blend", "run_tag": "blend",
        "description": label,
        "final_oof_auc": round(float(auc), 6), "public_lb_score": score,
        "n_folds": 5, "cv_seed": 42,
        "preds_dir": dest.relative_to(REPO_ROOT).as_posix(), "notes": args.notes,
    })
    print(f"Appended public blend {run_id} to experiments/runs.csv")


if __name__ == "__main__":
    main()
