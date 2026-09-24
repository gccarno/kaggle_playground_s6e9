#!/usr/bin/env python3
"""Fixed weighted blend of OUR archived champion with PUBLIC OOF artifacts.

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

WHY THE COMBINER DEFAULTS TO RANK, NOT LOGIT (Phase 24). The legs do not live in the same
space. `6view` and `rmlp` ship PERCENTILE RANKS in (0, 1]; our own artifacts and najiama's
ship calibrated probabilities. Their logit SDs are 1.81 and 3.18 respectively, so under
`logit_mean` a nominal "0.34 ours / 0.66 public" actually weights our side 0.34*3.18 = 1.08
against 0.66*1.81 = 1.20 -- an effective ~47/53. Phase 23's dose-response was therefore never
measured at its nominal doses. Worse, `Sergey_LGBM_oof.csv` drops from AUC 0.945329 to
0.872301 under logit(clip(p, 1e-6, 1-1e-6)) alone, because its shipped values are not
calibrated probabilities and the clip silently ties a large block of rows -- a 0.073 AUC loss
no assert here would have caught. `rank_mean` is scale-free and immune to whatever probability
geometry a public file happens to arrive with, so it is the default; `logit_mean` remains
available for reproducing Phase 23's points and carries a guard against the Sergey signature.
On a blend whose legs are ALL in probability space the two agree to 1e-6, so the choice only
ever matters when rank-space legs are present.

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
from scipy.stats import rankdata
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
    # Phase 24: the rest of najiama's 5-fold V-series, registered so the saturation curve in
    # README Phase 24 finding 2 is reproducible from the repo. They are near-twins of each
    # other and of lgbV5, and adding any of them to `xgb5f + lgbV5` LOWERS the blend's OOF
    # monotonically (0.946305 -> .946284 -> .946270 -> .946268). Registered, not recommended.
    "lgbV3": ("s6e9-oof/Pure LGBM_V3_oof.csv", None,
              "s6e9-oof/Pure LGBM_V3_test.csv", None,
              5, "najiama/s6e9-oof"),
    "lgbV6": ("s6e9-oof/Pure LGBM_V6_oof.csv", None,
              "s6e9-oof/Pure LGBM_V6_test.csv", None,
              5, "najiama/s6e9-oof"),
    "lgbV1": ("s6e9-oof/Pure LGBM_V1_oof.csv", None,
              "s6e9-oof/Pure LGBM_V1_test.csv", None,
              5, "najiama/s6e9-oof"),
    # Sergey_LGBM is deliberately NOT registered: its shipped values are not calibrated
    # probabilities and it loses 0.073 AUC under logit clipping (README Phase 24 finding 4).
    # Phase 25: najiama's 10-fold XGBoost twin, and megayak's six views taken APART. Phase 24
    # only ever used the pre-made `ensemble` column, which cannot express "D and F add even
    # though they are weaker" -- the one claim the library's own card makes. Enumerating all
    # of them (scripts/enum_public.py, 3,282 blends) moved the ceiling +0.000016 and closed
    # the axis; they are registered so that result is reproducible, not because any ships.
    "xgb10f": ("s6e9-oof/XGBoost_Triple_TE_10folds_oof.csv", None,
               "s6e9-oof/XGBoost_Triple_TE_10folds_test.csv", None,
               10, "najiama/s6e9-oof"),
}
_SIXVIEW = {
    "sixA": "A_lgbm_triple_te_digits_3seed", "sixB": "B_xgb_on_A_features",
    "sixC": "C_no_digits_windows_lift_sm2_30_300", "sixD": "D_no_exact_key_ladder_windows",
    "sixE": "E_ladder25_250_2500_lift_sm5_50_500", "sixF": "F_exact_rate_as_init_score",
}
for _k, _c in _SIXVIEW.items():
    REGISTRY[_k] = ("s6e9-six-feature-views-oof-library/oof_six_views.csv", _c,
                    "s6e9-six-feature-views-oof-library/test_six_views.csv", _c,
                    10, f"megayak/s6e9-six-feature-views-oof-library (view {_k[-1]})")


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
    # Raw, untransformed. The combiner is chosen in main(), not here -- see the docstring.
    return (np.asarray(_col(o, oof_c), dtype=np.float64),
            np.asarray(_col(t, test_c), dtype=np.float64), nfold, src)


def to_rank(v):
    """Percentile rank in (0, 1). Scale-free, so legs in different spaces combine honestly."""
    return rankdata(v) / (len(v) + 1.0)


def to_logit(v, name):
    """Logit with the Sergey guard: clipping must not change the leg's own ranking."""
    z = logit(np.clip(v, EPS, 1 - EPS))
    lost = len(np.unique(v)) - len(np.unique(z))
    assert lost <= 0 or lost / len(v) < 1e-9, (
        f"{name}: logit clipping collapsed {lost} distinct values -- this leg is not in "
        f"probability space (the Sergey signature). Use --mode rank_mean.")
    return z


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ours", default=OURS_DEFAULT, help="archived run_id of our own side")
    ap.add_argument("--ours-weight", type=float, required=True,
                    help="weight on our side; the public legs share (1 - w) equally. "
                         "0.0 is legal and means a PUBLIC-ONLY blend, which is a probe: it "
                         "contains nothing we trained and is never selectable as a final.")
    ap.add_argument("--mode", choices=("rank_mean", "logit_mean"), default="rank_mean",
                    help="combiner. rank_mean (default) is scale-free; logit_mean reproduces "
                         "Phase 23 and is only safe when every leg is in probability space.")
    ap.add_argument("--public", nargs="+", required=True, choices=sorted(REGISTRY))
    ap.add_argument("--submit", action="store_true")
    ap.add_argument("--notes", default="")
    ap.add_argument("--description", default="")
    args = ap.parse_args()
    assert 0.0 <= args.ours_weight <= 1.0, "--ours-weight must be in [0, 1]"
    tx = ((lambda v, n=None: to_rank(v)) if args.mode == "rank_mean"
          else (lambda v, n=None: to_logit(v, n)))

    truth = pd.read_csv(REPO_ROOT / "data" / "train.csv", usecols=["id", "Will_Buy_EV"])
    y = (truth["Will_Buy_EV"] == "Yes").astype(int).to_numpy()
    test_ids = np.sort(pd.read_csv(REPO_ROOT / "data" / "test.csv", usecols=["id"])["id"].values)

    w = args.ours_weight
    public_only = (w == 0.0)
    if public_only:
        # A probe, never a final: README section 5's hedge is that Final A contains only legs
        # we trained, and this contains none of them. Labelled so the audit can see it.
        od, ours_oof, ours_test, a_ours = None, None, None, float("nan")
    else:
        ours_dir = PREDS / args.ours
        oof_f = sorted(ours_dir.glob("oof_proba_*.csv"))[0]
        test_f = sorted(ours_dir.glob("test_proba_*.csv"))[0]
        od = pd.read_csv(oof_f).sort_values("id")
        td = pd.read_csv(test_f).sort_values("id")
        assert (od["id"].values == truth["id"].values).all(), "our OOF ids misaligned"
        assert (td["id"].values == test_ids).all(), "our test ids misaligned"
        ours_oof = tx(od["proba"].values, f"OURS:{args.ours}")
        ours_test = tx(td["proba"].values, f"OURS:{args.ours}")
        a_ours = roc_auc_score(y, ours_oof)
    pub_oof, pub_test, rows = [], [], []
    for n in args.public:
        lo, lt, nfold, src = load_public(n, truth["id"].values, test_ids)
        # Rank/logit is applied per LEG, before averaging -- that is what makes the weights
        # mean what they say when the legs arrive in different spaces.
        pub_oof.append(tx(lo, n))
        pub_test.append(tx(lt, n))
        rows.append(dict(leg=n, solo_oof=roc_auc_score(y, lo), leg_folds=nfold,
                         weight=(1 - w) / len(args.public), source=src))
    blend_oof = (1 - w) * np.mean(pub_oof, axis=0)
    blend_test = (1 - w) * np.mean(pub_test, axis=0)
    if not public_only:
        blend_oof = blend_oof + w * ours_oof
        blend_test = blend_test + w * ours_test
    auc = roc_auc_score(y, blend_oof)

    head = ([] if public_only else
            [dict(leg=f"OURS:{args.ours}", solo_oof=a_ours, leg_folds=5,
                  weight=w, source="this repo")])
    print(pd.DataFrame(head + rows).to_string(index=False, float_format="%.6f"))
    n10 = sum(r["leg_folds"] == 10 for r in rows)
    print(f"\n[{args.mode}] blend OOF {auc:.6f}", end="")
    if public_only:
        print("   PUBLIC-ONLY probe: contains nothing we trained, NOT final-selectable")
    else:
        print(f"   vs our side alone {auc - a_ours:+.6f}   (shipping gate +0.0000949)")
    if n10:
        print(f"CAUTION: {n10} of {len(rows)} public legs are 10-fold OOF, which reads "
              f"~0.00018 optimistic against this LB (run e1495238). Discount before reading "
              f"this gain as a result; the LB is the arbiter on this axis, not our OOF.")

    run_id = uuid.uuid4().hex[:8]
    dest = PREDS / run_id
    dest.mkdir(parents=True, exist_ok=True)
    # In rank_mean the blend is ALREADY a value in (0, 1); expit-ing it would squash every
    # prediction into (0.5, 0.73). AUC would not notice -- it is rank-invariant -- but the
    # archived artifact would be unusable as a leg later and would read as nonsense on disk.
    unit = (lambda v: v) if args.mode == "rank_mean" else expit
    fold = od["fold"] if (od is not None and "fold" in od) else pd.Series(-1, index=truth.index)
    pd.DataFrame({"id": truth["id"], "fold": fold.values,
                  "proba": unit(blend_oof)}).to_csv(dest / "oof_proba_blend.csv", index=False)
    pd.DataFrame({"id": test_ids, "proba": unit(blend_test)}).to_csv(
        dest / "test_proba_blend.csv", index=False)
    sub = pd.DataFrame({"id": test_ids, "Will_Buy_EV": unit(blend_test)})
    sub.to_csv(dest / "submission.csv", index=False)

    label = args.description or (
        f"public-only {args.mode}: {'+'.join(args.public)} (PROBE, not final-selectable)"
        if public_only else
        f"public-mix {args.mode}: OURS({args.ours})*{w:.2f} + "
        f"{'+'.join(args.public)}*{(1 - w):.2f}")
    # The combiner is part of the recipe, so a manifest without it is incomplete provenance
    # under README section 5 -- which is the condition for being selectable as a final.
    (dest / "manifest.json").write_text(json.dumps(
        {"mode": f"public_weighted_{args.mode}", "combiner": args.mode,
         "ours_run": None if public_only else args.ours, "ours_weight": w,
         "ours_solo_oof": None if public_only else a_ours,
         "public_only": public_only, "final_selectable": not public_only,
         "public_legs": rows, "oof_auc": auc,
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
        "kernel_ref": "public_only" if public_only else "public_blend",
        "run_tag": "blend",
        "description": label,
        "final_oof_auc": round(float(auc), 6), "public_lb_score": score,
        "n_folds": 5, "cv_seed": 42,
        "preds_dir": dest.relative_to(REPO_ROOT).as_posix(), "notes": args.notes,
    })
    print(f"Appended public blend {run_id} to experiments/runs.csv")


if __name__ == "__main__":
    main()
