#!/usr/bin/env python3
"""Submit an ALREADY-ARCHIVED run's submission.csv, poll for its public score, and write
that score back into experiments/runs.csv.

Why this exists: KAGGLE_PLAYBOOK.md section 1 records S6E8's clearest operational miss --
18 submissions used out of ~290 available. Kaggle gives 5 slots a day and they do not roll
over, so an unspent slot is a paired OOF<->LB point thrown away, and the paired points are
the entire instrument that sections 4 and 5 depend on. Without this script a run can only
be submitted at the moment it is trained, which is exactly what makes slots go unused.

Usage:
    python scripts/submit_run.py B5                    # by run_tag
    python scripts/submit_run.py a9cc34ed              # or by run_id
    python scripts/submit_run.py B5 --no-wait          # fire and forget; fill the score in later
    python scripts/submit_run.py --fill-missing        # poll Kaggle for any run still lacking a score
"""
import argparse, csv, subprocess, sys, time
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
RUNS_CSV = REPO_ROOT / "experiments" / "runs.csv"
COMPETITION = "playground-series-s6e9"


def kaggle(*args):
    return subprocess.run(["kaggle", *args], capture_output=True, text=True, cwd=REPO_ROOT)


def submissions_table():
    out = kaggle("competitions", "submissions", COMPETITION, "--csv")
    return list(csv.DictReader(out.stdout.splitlines()))


def poll_score(description, timeout_min=20, poll_interval=20):
    """Wait for the submission whose description matches, so a concurrent submission
    cannot have its score attributed to this run."""
    deadline = time.time() + timeout_min * 60
    while time.time() < deadline:
        for r in submissions_table():
            if r.get("description", "").strip() != description.strip():
                continue
            s = r.get("publicScore") or r.get("public_score")
            if s not in (None, "", "None", "pending"):
                return s
        time.sleep(poll_interval)
    print("  WARNING: timed out waiting for a public score", file=sys.stderr)
    return ""


def find_run(runs, key):
    hit = runs[(runs["run_id"] == key) | (runs["run_tag"].astype(str) == key)]
    if hit.empty:
        raise SystemExit(f"no run in runs.csv with run_id or run_tag == {key!r}")
    if len(hit) > 1:
        raise SystemExit(f"{key!r} matches {len(hit)} runs: {list(hit['run_id'])}")
    return hit.iloc[0]


def write_score(run_id, score):
    runs = pd.read_csv(RUNS_CSV, dtype=str)
    runs.loc[runs["run_id"] == run_id, "public_lb_score"] = score
    runs.to_csv(RUNS_CSV, index=False)
    print(f"  wrote public_lb_score={score} onto run {run_id} in runs.csv")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run", nargs="?", help="run_id or run_tag of an archived run")
    ap.add_argument("--message", default=None, help="submission description (defaults to the run's)")
    ap.add_argument("--no-wait", action="store_true", help="submit without polling for the score")
    ap.add_argument("--fill-missing", action="store_true",
                    help="do not submit; just poll Kaggle and backfill any missing scores")
    args = ap.parse_args()

    runs = pd.read_csv(RUNS_CSV, dtype=str)

    if args.fill_missing:
        table = {r.get("description", "").strip():
                 (r.get("publicScore") or r.get("public_score")) for r in submissions_table()}
        n = 0
        for _, row in runs.iterrows():
            if pd.notna(row.get("public_lb_score")) and str(row.get("public_lb_score")).strip():
                continue
            s = table.get(str(row["description"]).strip())
            if s and s not in ("None", "pending"):
                write_score(row["run_id"], s)
                n += 1
        print(f"backfilled {n} score(s)")
        return

    if not args.run:
        raise SystemExit("give a run_id/run_tag, or use --fill-missing")

    row = find_run(runs, args.run)
    sub = REPO_ROOT / str(row["preds_dir"]) / "submission.csv"
    if not sub.exists():
        raise SystemExit(f"{sub} is missing -- the artifact was deleted, so this run "
                         "cannot be submitted without retraining")
    msg = args.message or str(row["description"])
    print(f"submitting run {row['run_id']} (tag {row['run_tag']}, OOF {row['final_oof_auc']})\n"
          f"  file: {sub}\n  message: {msg}")

    out = kaggle("competitions", "submit", COMPETITION, "-f", str(sub), "-m", msg)
    if out.returncode != 0:
        raise SystemExit(f"submit failed: {out.stderr}")
    print(out.stdout.strip().splitlines()[-1] if out.stdout.strip() else "submitted")

    if args.no_wait:
        print("  not waiting; run with --fill-missing later to backfill the score")
        return
    score = poll_score(msg)
    if score:
        print(f"  public LB = {score}")
        write_score(row["run_id"], score)


if __name__ == "__main__":
    main()
