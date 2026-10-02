#!/usr/bin/env python3
"""Post-deadline: write each submitted run's PRIVATE score into experiments/runs.csv.

Matching follows submit_run.py --fill-missing's rule -- by submission description, and ambiguity
is a REFUSAL, never a guess -- with one extra guard: the matched submission's PUBLIC score must
equal the public_lb_score already on the row. That cross-check is what catches a run whose
description was reused (a generic blend label) and a public score that was written down wrong.

A description shared by several submissions is resolved by (description, public score) when that
pair is unique, else by the submission nearest in time to the run's timestamp (within 60 min).

Usage:
    python scripts/backfill_private.py experiments/final/submissions.csv          # dry run
    python scripts/backfill_private.py experiments/final/submissions.csv --write
"""
import argparse, re, sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
RUNS_CSV = REPO_ROOT / "experiments" / "runs.csv"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("submissions_csv", help="output of `kaggle competitions submissions --csv`")
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()

    subs = pd.read_csv(args.submissions_csv).dropna(subset=["privateScore"])
    subs["description"] = subs.description.astype(str).str.strip()
    subs["t"] = pd.to_datetime(subs.date, utc=True, format="mixed")
    runs = pd.read_csv(RUNS_CSV, dtype=str, keep_default_na=False)

    filled = mismatched = refused = 0
    for i, row in runs.iterrows():
        if not row.public_lb_score.strip():
            continue
        pub = float(row.public_lb_score)
        t0 = pd.Timestamp(row.timestamp_utc)
        same_pub = (subs.publicScore - pub).abs() < 5e-6
        cand = subs[subs.description == row.description.strip()]
        if len(cand) > 1:
            if (same_pub & subs.index.isin(cand.index)).sum() == 1:
                cand = cand[same_pub.loc[cand.index]]
            else:
                dt = (cand.t - t0).abs()
                cand = cand[dt == dt.min()] if dt.min() <= pd.Timedelta("60min") else cand.iloc[:0]
        if len(cand) == 1 and abs(cand.iloc[0].publicScore - pub) > 5e-6:
            # A unique description match whose public score disagrees is either a generic blend
            # label that matched a DIFFERENT submission (b5a318cf, "logit stack, 8 legs") or a
            # score written down wrong. The two cannot be told apart here, so it is not filled:
            # drop to the fallbacks, and print it for a human to resolve.
            mismatched += 1
            print(f"  PUBLIC MISMATCH {row.run_id} {row.run_tag!r}: runs.csv {row.public_lb_score}"
                  f", description-matched submission {cand.iloc[0].publicScore:.5f} -- not used",
                  file=sys.stderr)
            cand = cand.iloc[:0]
        if len(cand) == 0:
            # Fallbacks for runs submitted later under a different message ("Phase 21 slot 5:
            # TEXC, ..."), each requiring an EQUAL public score and a submission after the run:
            # (a) the run_id named in the message, at any delay; (b) the run_tag as a whole word,
            # inside 7 days. Unique, or it is refused.
            after = same_pub & (subs.t >= t0 - pd.Timedelta("10min"))
            cand = subs[after & subs.description.str.contains(row.run_id, regex=False)]
            if len(cand) == 0 and row.run_tag.strip():
                tag = re.escape(row.run_tag.strip())
                cand = subs[after & (subs.t <= t0 + pd.Timedelta("7D"))
                            & subs.description.str.contains(rf"(?<![\w-]){tag}(?![\w-])",
                                                            regex=True)]
        if len(cand) != 1:
            refused += 1
            print(f"  REFUSED {row.run_id} {row.run_tag!r}: {len(cand)} candidate submissions",
                  file=sys.stderr)
            continue
        runs.at[i, "private_lb_score"] = f"{cand.iloc[0].privateScore:.5f}"
        filled += 1

    print(f"filled {filled}, public mismatches {mismatched}, refused {refused}")
    if args.write:
        runs.to_csv(RUNS_CSV, index=False)
        print(f"wrote {RUNS_CSV}")


if __name__ == "__main__":
    main()
