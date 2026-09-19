#!/usr/bin/env python3
"""Refit the OOF->LB predictor, and quantify how far the frozen fit is extrapolating.

README.md section 4 freezes the shipping GATE at +0.0000949 OOF and says it does not move
again -- that stands, permanently, and this script does not touch it. What it DOES compute
is the PREDICTOR the gate was derived from (slope 1.0832, intercept -0.0784, residual sigma
0.000103), which was fit in Phase 3 (2026-09-04) on 10 points spanning OOF 0.9417-0.9457.
Every run since Phase 14 sits at OOF 0.9460-0.9461 -- outside that calibration range -- and
Phases 16-18 have been chasing a "~-0.00007 unexplained LB residual" that may just be this
fit extrapolated 0.004 past where it was measured.

This script re-fits three ways over experiments/runs.csv:
  1. the frozen 10 points (Phase 3's own selection, for a sanity check against README section 4)
  2. all paired GBDT-family points on the frozen 5-fold split
  3. the rich (>=100 feat) rs-free (fe_recipe_score off) subset alone -- the champion's
     own neighbourhood, which is the fit that actually matters for reading WQ2 onward

Then prints every point's residual under fit 2 and fit 3, so the Phase 16-18 residual
tables can be re-read against a locally-calibrated line instead of an extrapolated one.

Usage:
    python scripts/refit_gap.py
"""
import csv
import json
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
RUNS_CSV = REPO_ROOT / "experiments" / "runs.csv"

# Phase 3's own 10-point series (README.md section 8, "Phase 8" table -- these are the
# runs the frozen slope/intercept/residual-sigma were fit on), keyed by run_tag. Kept
# literal rather than re-derived, since re-deriving the frozen selection defeats the
# sanity-check purpose of fit 1.
FROZEN_TAGS = {"A0", "B5", "H2", "C2", "D4", "I1", "E1", "F2", "E4"}
# The 10th point in Phase 8's table is the 3-leg fixed rank-mean blend, which has no
# single run_tag in runs.csv (learner is blank, several "blend"/"logit_stack" tags exist
# across phases) -- identified instead by its exact archived OOF value.
FROZEN_BLEND_OOF = 0.945743

# cc73801d is an unsubmitted duplicate of WQnoRS; f5c9efb0 carries the real LB score
# (Phase 18's closing note). 92f79e80 is the same class of bug for FQ (Phase 19): an
# improperly-backgrounded collect_run.py call archived it once on its own before a
# second, correctly-tracked call polled the same already-complete kernel and archived
# it again as 5794995c. Exclude both duplicates by run_id so a de-duplicated re-run of
# this script doesn't need runs.csv hand-edited first.
EXCLUDE_RUN_IDS = {"cc73801d", "92f79e80"}
# 10-fold run, not comparable to the frozen 5-fold split -- excluded from the OOF->LB
# regression by design (README.md Phase 15, X4).
EXCLUDE_TAGS = {"X4_10fold"}


def load_points():
    rows = list(csv.DictReader(open(RUNS_CSV, encoding="utf-8")))
    seen = set()
    pts = []
    for r in rows:
        if not r["public_lb_score"] or not r["final_oof_auc"]:
            continue
        if r["run_id"] in EXCLUDE_RUN_IDS or r["run_tag"] in EXCLUDE_TAGS:
            continue
        # GBDT-family only (README section 4: "valid within the GBDT family ONLY" --
        # G1's embedding MLP blew the residual sigma 0.000103 -> 0.000340 when included).
        if r["learner"] not in ("lgb", "xgb", "cat", ""):
            continue
        key = (r["run_tag"], r["public_lb_score"])
        if key in seen:      # duplicate submission of the same tag at the same score
            continue
        seen.add(key)
        pts.append({
            "run_id": r["run_id"], "run_tag": r["run_tag"],
            "oof": float(r["final_oof_auc"]), "lb": float(r["public_lb_score"]),
            "n_feat": int(float(r["n_features"] or 0)),
        })
    return pts


def fit(pts):
    o = np.array([p["oof"] for p in pts])
    l = np.array([p["lb"] for p in pts])
    if len(pts) < 3:
        return None
    s, i = np.polyfit(o, l, 1)
    pred = s * o + i
    resid = l - pred
    # ddof=2: two fitted parameters (slope, intercept), matching README section 4's own
    # convention for "residual sigma" over a least-squares fit.
    sigma = float(resid.std(ddof=2)) if len(pts) > 2 else float("nan")
    return {"slope": float(s), "intercept": float(i), "sigma": sigma, "n": len(pts),
             "resid": dict(zip((p["run_id"] for p in pts), resid))}


def main():
    pts = load_points()
    print(f"{len(pts)} de-duplicated GBDT-family paired points in runs.csv "
          f"(after excluding {EXCLUDE_RUN_IDS} and {EXCLUDE_TAGS})\n")

    frozen_pts = [p for p in pts if p["run_tag"] in FROZEN_TAGS]
    blend = [p for p in pts if abs(p["oof"] - FROZEN_BLEND_OOF) < 1e-6]
    frozen_pts = frozen_pts + blend
    print(f"fit 1 -- frozen Phase-3 selection ({len(frozen_pts)}/10 tags matched):")
    f1 = fit(frozen_pts)
    if f1:
        print(f"  slope={f1['slope']:.4f}  intercept={f1['intercept']:.5f}  "
              f"resid_sigma={f1['sigma']:.6f}")
        print(f"  (README section 4 frozen values: slope=1.0832  resid_sigma=0.000103 "
              f"-- compare, do not overwrite)")
    print()

    print(f"fit 2 -- all {len(pts)} de-duplicated GBDT-family points:")
    f2 = fit(pts)
    print(f"  slope={f2['slope']:.4f}  intercept={f2['intercept']:.5f}  "
          f"resid_sigma={f2['sigma']:.6f}")
    print()

    rich_rsfree = [p for p in pts if p["oof"] >= 0.9459]
    print(f"fit 3 -- rich (OOF>=0.9459) subset alone, the champion's own neighbourhood "
          f"({len(rich_rsfree)} points):")
    f3 = fit(rich_rsfree) if len(rich_rsfree) >= 3 else None
    if f3:
        print(f"  slope={f3['slope']:.4f}  intercept={f3['intercept']:.5f}  "
              f"resid_sigma={f3['sigma']:.6f}")
    else:
        print(f"  fewer than 3 points -- cannot fit a slope from a cluster; this IS the "
              f"finding (see plan Track A/B: build the ladder).")
    print()

    print("per-run residual, all fits (sorted by OOF descending):")
    print(f"{'run_tag':<14}{'oof':>10}{'lb':>9}{'feat':>6}"
          f"{'resid_f1':>10}{'resid_f2':>10}{'resid_f3':>10}")
    for p in sorted(pts, key=lambda p: -p["oof"]):
        r1 = f1["resid"].get(p["run_id"]) if f1 and p["run_id"] in f1["resid"] else None
        r2 = f2["resid"].get(p["run_id"])
        r3 = f3["resid"].get(p["run_id"]) if f3 else None
        def fmt(x):
            return f"{x:+.5f}" if x is not None else "     -"
        print(f"{p['run_tag']:<14}{p['oof']:>10.6f}{p['lb']:>9.5f}{p['n_feat']:>6}"
              f"{fmt(r1):>10}{fmt(r2):>10}{fmt(r3):>10}")


if __name__ == "__main__":
    main()
