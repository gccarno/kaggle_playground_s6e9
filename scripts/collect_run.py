#!/usr/bin/env python3
"""Push the S6E9 kernel to Kaggle, poll until it finishes, and append the run's
metrics (parsed from the notebook's RUN_METRICS_JSON print) to experiments/runs.csv.

Usage:
    python scripts/collect_run.py --description "v1 baseline: raw-feature 5-fold LightGBM"
    python scripts/collect_run.py --description "..." --no-push   # kernel already running
    python scripts/collect_run.py --description "..." --submit    # also submit to the leaderboard

Designed to be run via the harness's background Bash execution -- a full
push -> queue -> run -> complete cycle can take a long time.

Ported from playground_s6e7/scripts/collect_run.py. Two things are generalized,
because S6E7 needed a code edit every time a learner was added:

  * the runs.csv schema EVOLVES -- a metrics key like "xgb_oof_auc" that isn't in
    the header yet causes the CSV to be rewritten with that column appended,
    rather than being silently dropped;
  * prediction artifacts are DISCOVERED by glob (oof_proba_*.csv / test_proba_*.csv)
    instead of being listed learner by learner.
"""
import argparse, csv, json, os, re, shutil, subprocess, sys, time, uuid
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
KERNEL_METADATA = REPO_ROOT / "kernel-metadata.json"
RUNS_CSV = REPO_ROOT / "experiments" / "runs.csv"

COMPETITION = "playground-series-s6e9"

# Baseline schema. Per-learner "<name>_oof_auc" columns are appended automatically
# as new learners appear in RUN_METRICS_JSON (see reconcile_columns).
# `notes` stays free-text and stays LAST-but-one -- it was the single most valuable
# column in S6E7, so give it room and never truncate it.
RUN_COLUMNS = [
    "run_id", "timestamp_utc", "git_commit", "git_dirty",
    "kernel_ref", "kernel_version", "description",
    "notebook_runtime_sec", "n_features", "n_folds", "cv_seed",
    "fold_auc_mean", "fold_auc_std",
    "final_oof_auc", "public_lb_score", "private_lb_score",
    "preds_dir", "notes",
]

# Metrics keys copied straight through to same-named columns when present.
# best_iters and fold_aucs are list-valued and get joined into a string: they are the
# columns that exposed B1's early-stopping confound (fold 0 quitting at 1055 vs 3000
# while the other folds were flat), and that diagnostic was only visible in scrollback
# at the time. Anything capable of overturning a probe's verdict belongs in the log.
PASSTHROUGH = ["notebook_runtime_sec", "n_features", "n_folds", "cv_seed",
               "fold_auc_mean", "fold_auc_std", "final_oof_auc",
               "best_iters", "fold_aucs", "model_seed", "te_cols", "engineered", "run_tag",
               # S6E9: which learner produced the row. Recoverable from whichever
               # <learner>_oof_auc column is populated, but only by inference -- and a
               # multi-learner pool is about to make that inference ambiguous.
               "learner"]

# Artifacts to archive per run, in addition to anything matching PRED_GLOBS.
PRED_FILES = ["submission.csv"]
PRED_GLOBS = ["oof_proba_*.csv", "test_proba_*.csv"]


def run(cmd):
    print(f"$ {' '.join(cmd)}")
    # PYTHONIOENCODING -- the kaggle-cli SUBPROCESS is itself a Python program, and on a
    # non-UTF8 Windows console it crashes with UnicodeEncodeError trying to print a
    # filename/status line containing a character its own stdout codepage can't
    # represent -- after the files it was reporting had already downloaded successfully.
    # This forces UTF-8 in that child process regardless of the console's codepage.
    # encoding="utf-8" on our OWN read of its output is a second, independent guard.
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
    return subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                          errors="replace", cwd=REPO_ROOT, env=env)


def git_commit():
    out = run(["git", "rev-parse", "HEAD"])
    return out.stdout.strip() if out.returncode == 0 else ""


def git_dirty():
    out = run(["git", "status", "--porcelain"])
    return bool(out.stdout.strip())


def kernel_ref(metadata=None):
    meta = json.loads((metadata or KERNEL_METADATA).read_text(encoding="utf-8"))
    return meta["id"]


def stage_push_dir(metadata):
    """`kaggle kernels push -p DIR` only ever reads DIR/kernel-metadata.json, so a repo
    holding two kernels cannot push the second one in place. Stage a throwaway directory
    containing just that kernel's metadata (under the required name) plus its notebook,
    rather than mutating the repo's own kernel-metadata.json and risking leaving it
    swapped if the push fails."""
    meta = json.loads(metadata.read_text(encoding="utf-8"))
    staged = REPO_ROOT / ".kaggle_output" / f"_push_{meta['id'].split('/')[-1]}"
    staged.mkdir(parents=True, exist_ok=True)
    (staged / "kernel-metadata.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    shutil.copy2(REPO_ROOT / meta["code_file"], staged / meta["code_file"])
    print(f"  staged {meta['id']} ({meta['code_file']}) -> {staged}")
    return staged


def push_kernel(metadata=None, accelerator=None):
    push_dir = stage_push_dir(metadata) if metadata else REPO_ROOT
    cmd = ["kaggle", "kernels", "push", "-p", str(push_dir)]
    # `enable_gpu: true` in kernel-metadata.json allocates a Tesla P100 (compute 6.0), and
    # the Kaggle image ships torch 2.10.0+cu128, whose builds contain no Pascal kernels --
    # so the default GPU cannot run PyTorch on Kaggle's own image and every fold dies with
    # "no kernel image is available for execution on the device". The accelerator choice
    # exists ONLY as a push flag; there is no kernel-metadata.json equivalent. See the
    # P100 section of README.md.
    if accelerator:
        cmd += ["--accelerator", accelerator]
    out = run(cmd)
    print(out.stdout); print(out.stderr, file=sys.stderr)
    if out.returncode != 0:
        raise RuntimeError(f"kaggle kernels push failed: {out.stderr}")
    # kaggle-cli EXITS 0 ON A FAILED PUSH and reports the failure only on stdout, e.g.
    # "Kernel push error: Maximum batch GPU session count of 2 reached." (Kaggle allows
    # exactly two concurrent GPU sessions.) Trusting the exit code cost two fabricated
    # rows in runs.csv: the push was rejected, the kernel of that name was still sitting
    # COMPLETE from its PREVIOUS run, so polling returned immediately, the OLD artifacts
    # were downloaded, and the OLD RUN_METRICS_JSON was archived under a NEW run_id --
    # a row that looks like an experiment and is a copy of one from days earlier. Fail
    # here instead: a rejected push must never reach the poller.
    blob = out.stdout + out.stderr
    if re.search(r'(?i)\bpush error\b|\berror:', blob):
        raise RuntimeError(
            'kaggle kernels push was REJECTED (the cli still exited 0): '
            + ' '.join(blob.split())
            + '  --  nothing was pushed, so polling would have collected the '
            + 'PREVIOUS run output and archived it under a new run_id.')
    # Best-effort: kaggle-cli's exact push-success wording isn't guaranteed across
    # versions, so this may return None -- that's fine, only --submit needs a
    # version and it falls back to submitting the file directly.
    m = re.search(r"[Vv]ersion\s+(\d+)", out.stdout + out.stderr)
    return m.group(1) if m else None


def poll_status(ref, poll_interval, timeout_min):
    deadline = time.time() + timeout_min * 60
    while time.time() < deadline:
        out = run(["kaggle", "kernels", "status", ref])
        text = (out.stdout + out.stderr).strip()
        m = re.search(r"KernelWorkerStatus\.(\w+)", text)
        status = m.group(1) if m else text
        print(f"  status: {status}")
        if status in ("COMPLETE", "ERROR", "CANCELLED"):
            return status
        time.sleep(poll_interval)
    raise TimeoutError(f"Kernel did not finish within {timeout_min} minutes")


def _extract_stdout_text(raw_bytes):
    """Kaggle's kernel .log file is a JSON array of {stream_name, time, data}
    entries, not plain text -- reassemble the stdout stream into one string.
    Falls back to the raw decoded text if the file isn't in that format."""
    text = raw_bytes.decode("utf-8", errors="replace")
    try:
        entries = json.loads(text)
    except json.JSONDecodeError:
        return text
    if not isinstance(entries, list):
        return text
    return "".join(e.get("data", "") for e in entries
                   if isinstance(e, dict) and e.get("stream_name") in ("stdout", None))


def pull_run_metrics(ref, workdir):
    workdir.mkdir(parents=True, exist_ok=True)
    out = run(["kaggle", "kernels", "output", ref, "-p", str(workdir), "--force"])
    print(out.stdout); print(out.stderr, file=sys.stderr)
    if out.returncode != 0:
        raise RuntimeError(f"kaggle kernels output failed: {out.stderr}")
    pattern = re.compile(r"RUN_METRICS_JSON:(\{.*\})")
    for p in workdir.rglob("*"):
        if not p.is_file():
            continue
        try:
            raw = p.read_bytes()
        except Exception:
            continue
        m = pattern.search(_extract_stdout_text(raw))
        if m:
            return json.loads(m.group(1))
    raise RuntimeError(f"RUN_METRICS_JSON not found in any file under {workdir}")


def submit_and_get_score(competition, ref, version, message, poll_interval, timeout_min,
                         file_fallback=None):
    # Prefer submitting the pulled submission.csv directly: in S6E7 kernel-based
    # submission failed both without a version ("version required") and with one
    # (400 on CreateCodeSubmission), while file submission worked -- playground
    # competitions accept plain files.
    if file_fallback is not None and Path(file_fallback).is_file():
        print(f"  Submitting file directly: {file_fallback}")
        cmd = ["kaggle", "competitions", "submit", competition, "-f", str(file_fallback), "-m", message]
    elif version:
        cmd = ["kaggle", "competitions", "submit", competition, "-k", ref, "-v", version, "-m", message]
    else:
        cmd = ["kaggle", "competitions", "submit", competition, "-k", ref, "-m", message]
    out = run(cmd)
    print(out.stdout); print(out.stderr, file=sys.stderr)
    if out.returncode != 0:
        raise RuntimeError(f"kaggle competitions submit failed: {out.stderr}")

    deadline = time.time() + timeout_min * 60
    while time.time() < deadline:
        out = run(["kaggle", "competitions", "submissions", competition, "--csv"])
        rows = list(csv.DictReader(out.stdout.splitlines()))
        if rows:
            latest = rows[0]  # assumed most-recent-first, matches the web UI ordering
            score = latest.get("publicScore") or latest.get("public_score")
            if score not in (None, "", "None", "pending"):
                return score
        time.sleep(poll_interval)
    print("  WARNING: timed out waiting for a public score; leaving public_lb_score blank")
    return ""


def archive_preds(workdir, run_id):
    """Copy the run's prediction artifacts into experiments/preds/<run_id>/ so they
    survive later kernel versions (Kaggle only serves the latest output). These
    artifacts are what make any past model retro-blendable with any future model
    without retraining -- KAGGLE_PLAYBOOK.md section 1.

    Returns the repo-relative dir, or '' if nothing was found to archive."""
    dest = REPO_ROOT / "experiments" / "preds" / run_id
    wanted = list(PRED_FILES)
    for pattern in PRED_GLOBS:
        wanted += sorted({p.name for p in workdir.rglob(pattern) if p.is_file()})

    copied = []
    for name in dict.fromkeys(wanted):  # de-dupe, preserve order
        matches = sorted(p for p in workdir.rglob(name) if p.is_file())
        if matches:
            dest.mkdir(parents=True, exist_ok=True)
            shutil.copy2(matches[0], dest / name)
            copied.append(name)
    if not copied:
        print(f"  WARNING: no prediction artifacts found under {workdir}")
        return ""
    print(f"  Archived {len(copied)} file(s) -> {dest}\n    " + "\n    ".join(copied))
    return dest.relative_to(REPO_ROOT).as_posix()


def reconcile_columns(row):
    """Return the column list to write, growing the on-disk header if this run
    introduced a column (e.g. a new learner's <name>_oof_auc) that isn't there yet.

    Without this, S6E7's fixed header silently dropped any metric from a learner
    added after the script was written. Growing the schema is cheap and the log is
    the most valuable artifact in the repo -- never lose a column from it."""
    if RUNS_CSV.exists():
        with RUNS_CSV.open(newline="", encoding="utf-8") as f:
            existing = next(csv.reader(f), []) or list(RUN_COLUMNS)
    else:
        existing = list(RUN_COLUMNS)

    new = [c for c in row if c not in existing]
    if not new:
        return existing

    # Keep notes last so the free-text column stays easy to eyeball in a diff.
    cols = [c for c in existing if c != "notes"] + new + (["notes"] if "notes" in existing else [])
    print(f"  runs.csv schema grew by {new} -> rewriting header")
    if RUNS_CSV.exists():
        with RUNS_CSV.open(newline="", encoding="utf-8") as f:
            old_rows = list(csv.DictReader(f))
        with RUNS_CSV.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            for r in old_rows:
                w.writerow({c: r.get(c, "") for c in cols})
    return cols


def append_run_row(row):
    RUNS_CSV.parent.mkdir(parents=True, exist_ok=True)
    cols = reconcile_columns(row)
    is_new = not RUNS_CSV.exists()
    with RUNS_CSV.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        if is_new:
            w.writeheader()
        w.writerow({c: row.get(c, "") for c in cols})


def build_row(metrics, **fixed):
    row = dict(fixed)
    for k in PASSTHROUGH:
        if k in metrics:
            v = metrics[k]
            row[k] = " ".join(str(x) for x in v) if isinstance(v, list) else v
    # Per-learner OOF scores. The notebook emits "<learner>_oof_auc" keys directly;
    # any learner name is accepted and gets its own column.
    for k, v in metrics.items():
        if k.endswith("_oof_auc"):
            row[k] = v
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--description", required=True, help="Short note on what changed this run")
    ap.add_argument("--no-push", action="store_true", help="Skip pushing; poll an already-running kernel")
    ap.add_argument("--submit", action="store_true", help="Also submit submission.csv to the leaderboard")
    ap.add_argument("--competition", default=COMPETITION)
    ap.add_argument("--metadata", default=None,
                    help="kernel metadata json (default kernel-metadata.json); "
                         "use kernel-metadata-model.json for the modelling kernel")
    ap.add_argument("--accelerator", default=None,
                    choices=["NvidiaTeslaT4", "NvidiaTeslaP100", "Tpu1VmV38"],
                    help="GPU type for this push. REQUIRED for any torch kernel: "
                         "enable_gpu alone gets a P100, which cannot run the image's torch. "
                         "Use NvidiaTeslaT4.")
    ap.add_argument("--poll-interval", type=int, default=60, help="Seconds between status checks")
    ap.add_argument("--timeout-min", type=int, default=180, help="Max minutes to wait for completion")
    ap.add_argument("--notes", default="", help="Freeform notes column -- hypothesis, mechanism, gate")
    args = ap.parse_args()

    metadata = (REPO_ROOT / args.metadata) if args.metadata else None
    ref = kernel_ref(metadata)
    commit = git_commit()
    dirty = git_dirty()
    if dirty:
        print("WARNING: working tree has uncommitted changes; the pushed kernel may not match HEAD.")

    version = None
    if not args.no_push:
        version = push_kernel(metadata, args.accelerator)
        print(f"Pushed. Detected version: {version!r} (best-effort parse; may be None)")

    print(f"Polling {ref} for completion...")
    status = poll_status(ref, args.poll_interval, args.timeout_min)
    if status != "COMPLETE":
        raise RuntimeError(f"Kernel finished with status {status}, not COMPLETE -- not logging a run row.")

    workdir = REPO_ROOT / ".kaggle_output" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    metrics = pull_run_metrics(ref, workdir)
    print("Parsed RUN_METRICS_JSON:", json.dumps(metrics, indent=2))

    run_id = uuid.uuid4().hex[:8]
    preds_dir = archive_preds(workdir, run_id)

    score = ""
    if args.submit:
        sub_file = next(iter(sorted(p for p in workdir.rglob("submission.csv") if p.is_file())), None)
        score = submit_and_get_score(args.competition, ref, version, args.description,
                                     args.poll_interval, args.timeout_min, file_fallback=sub_file)
        print(f"Public LB score: {score}")

    row = build_row(
        metrics,
        run_id=run_id,
        timestamp_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        git_commit=commit,
        git_dirty=dirty,
        kernel_ref=ref,
        kernel_version=version or "",
        description=args.description,
        public_lb_score=score,
        private_lb_score="",
        preds_dir=preds_dir,
        notes=args.notes,
    )
    append_run_row(row)
    print(f"\nAppended run {run_id} to {RUNS_CSV}")
    print(f"  final_oof_auc = {row.get('final_oof_auc', '')}   public_lb = {score or '(pending)'}")


if __name__ == "__main__":
    main()
