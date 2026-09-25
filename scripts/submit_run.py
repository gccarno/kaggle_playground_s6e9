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
        hits = [r for r in submissions_table()
                if r.get("description", "").strip() == description.strip()]
        if len(hits) > 1:
            # Same ambiguity as --fill-missing: a non-unique description means the first
            # match is a guess, not an identification. Refuse rather than mis-attribute.
            print(f"  WARNING: {len(hits)} submissions share this description; refusing to "
                  f"attribute a score. Resolve by submission time.", file=sys.stderr)
            return ""
        for r in hits:
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
        # Matching is by submission DESCRIPTION, which is only safe when the description is
        # unique on BOTH sides. It is not always: stack_logit.py used to emit a generic
        # "rank_mean blend, 4 legs, equal weights" for every blend of that shape, and on
        # 2026-09-25 this loop sprayed one score onto four rows sharing it -- two of which had
        # never been submitted at all, inventing two false paired OOF<->LB points. A false
        # paired point is worse than a missing one: refit_gap.py and every calibration claim
        # in README section 4 are built on exactly these pairs. So ambiguity is now a REFUSAL,
        # never a guess, and it names what to resolve by hand.
        subs = submissions_table()
        by_desc = {}
        for r in subs:
            by_desc.setdefault(r.get("description", "").strip(), []).append(
                r.get("publicScore") or r.get("public_score"))
        unscored = [row for _, row in runs.iterrows()
                    if not (pd.notna(row.get("public_lb_score"))
                            and str(row.get("public_lb_score")).strip())]
        pending = {}
        for row in unscored:
            pending.setdefault(str(row["description"]).strip(), []).append(row["run_id"])
        n = skipped = 0
        for desc, run_ids in pending.items():
            scores = [x for x in by_desc.get(desc, []) if x and x not in ("None", "pending")]
            if not scores:
                continue
            if len(scores) > 1 or len(run_ids) > 1:
                skipped += len(run_ids)
                print(f"  REFUSING to fill an ambiguous match: {len(scores)} submission(s) and "
                      f"{len(run_ids)} unscored run(s) share the description\n"
                      f"    description: {desc!r}\n"
                      f"    runs:        {', '.join(run_ids)}\n"
                      f"    scores:      {', '.join(scores)}\n"
                      f"    resolve by submission TIME with `kaggle competitions submissions "
                      f"-c {COMPETITION}` and write each score with --message, or by hand.",
                      file=sys.stderr)
                continue
            write_score(run_ids[0], scores[0])
            n += 1
        print(f"backfilled {n} score(s)" + (f"; {skipped} skipped as ambiguous" if skipped else ""))
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
