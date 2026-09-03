#!/usr/bin/env python3
"""Run s6e9-model.ipynb LOCALLY with a config override, archive its artifacts,
optionally submit the resulting file, and append a row to experiments/runs.csv.

The local loop is ~5 minutes against ~15 for a kernel push/queue/run/collect cycle,
and a locally-produced submission.csv can be submitted directly -- so probes are
screened here and only champions get pushed to Kaggle for the reproducible record.

Usage:
    python scripts/run_local.py --cfg '{"run_tag":"B1","fe_interaction":true}' \
        --description "B1: interaction ratios" --notes "HYPOTHESIS... GATE..." --submit

    # strict-twin check: refuse to run unless exactly one CFG field differs
    python scripts/run_local.py --cfg '{...}' --diff-vs '{"run_tag":"champion"}' ...

Rows land in the same experiments/runs.csv as kernel runs, with kernel_ref="local".
"""
import argparse, importlib.util, json, os, shutil, subprocess, sys, uuid
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK = REPO_ROOT / "s6e9-model.ipynb"

# Reuse collect_run.py's log plumbing rather than reimplementing the schema logic.
_spec = importlib.util.spec_from_file_location("collect_run", Path(__file__).with_name("collect_run.py"))
cr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cr)

def _load_defaults():
    """Read DEFAULTS straight out of the notebook instead of mirroring it here.

    A hand-maintained copy drifted twice in one session, and the second time it made
    --diff-vs compare against a baseline missing the very key the probe was changing,
    so a genuine one-field twin was reported as "0 fields differ". Parsing the single
    source of truth removes that whole class of bug."""
    import ast, re
    src = "".join(
        "".join(c["source"]) for c in json.loads(NOTEBOOK.read_text(encoding="utf-8"))["cells"]
        if c["cell_type"] == "code")
    m = re.search(r"^DEFAULTS = (\{.*?^\})", src, re.S | re.M)
    if not m:
        raise RuntimeError("could not find a DEFAULTS literal in the notebook")
    return ast.literal_eval(m.group(1))


DEFAULTS = _load_defaults()


def git(*args):
    out = subprocess.run(["git", *args], capture_output=True, text=True, cwd=REPO_ROOT)
    return out.stdout.strip() if out.returncode == 0 else ""


def check_strict_twin(cfg, baseline):
    """KAGGLE_PLAYBOOK.md section 3: a probe changes exactly ONE thing. A probe that
    moved two knobs cannot attribute its delta to either, so it is not evidence --
    refuse to run it rather than produce an uninterpretable row."""
    a, b = {**DEFAULTS, **baseline}, {**DEFAULTS, **cfg}
    # Iterate the UNION: iterating only `a` made a probe that introduces a brand-new
    # key invisible, which is exactly the case a strict-twin check exists to catch.
    diff = [k for k in (set(a) | set(b))
            if a.get(k, "<absent>") != b.get(k, "<absent>") and k != "run_tag"]
    if len(diff) != 1:
        raise SystemExit(
            f"NOT A STRICT TWIN: {len(diff)} fields differ from the baseline: {diff}\n"
            f"  baseline: { {k: a[k] for k in diff} }\n"
            f"  probe   : { {k: b[k] for k in diff} }\n"
            "Change exactly one thing, or drop --diff-vs if this is deliberate.")
    print(f"strict-twin OK: only {diff[0]!r} differs "
          f"({a[diff[0]]!r} -> {b[diff[0]]!r})")


def execute_notebook(cfg, out_dir):
    """Run the notebook's code cells as a script in a child process, streaming output live.

    Deliberately NOT nbclient: it buffers all cell output until execute() returns, so a
    run that is interrupted (or merely slow) shows nothing at all -- three runs were lost
    that way with zero diagnostic trace. The notebook stays the single source of truth;
    only its code cells are extracted, in order, and run in one namespace, which is what
    a notebook is anyway."""
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    parts = ["import matplotlib\nmatplotlib.use('Agg')\n"]
    for i, c in enumerate(nb["cells"]):
        if c["cell_type"] == "code":
            parts.append(f"\n# ---- cell {i} ----\n" + "".join(c["source"]) + "\n")

    script = REPO_ROOT / ".kaggle_output" / f"_run_{out_dir.name}.py"
    script.parent.mkdir(parents=True, exist_ok=True)
    script.write_text("".join(parts), encoding="utf-8")

    env = dict(os.environ, S6E9_CFG=json.dumps(cfg), S6E9_OUT=str(out_dir),
               PYTHONUNBUFFERED="1")
    proc = subprocess.Popen([sys.executable, "-u", str(script)], cwd=REPO_ROOT, env=env,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                            bufsize=1)
    captured = []
    for line in proc.stdout:          # live: every line appears as it is produced
        print(line, end="", flush=True)
        captured.append(line)
    if proc.wait() != 0:
        raise RuntimeError(f"notebook script failed with exit code {proc.returncode}")

    marker = "RUN_METRICS_JSON:"
    line = next((l for l in captured if l.startswith(marker)), None)
    if line is None:
        raise RuntimeError("RUN_METRICS_JSON not found in notebook output")
    script.unlink(missing_ok=True)
    return json.loads(line[len(marker):])


def submit(sub_file, message, competition="playground-series-s6e9"):
    out = subprocess.run(["kaggle", "competitions", "submit", competition,
                          "-f", str(sub_file), "-m", message],
                         capture_output=True, text=True, cwd=REPO_ROOT)
    print(out.stdout, out.stderr, file=sys.stderr if out.returncode else sys.stdout)
    if out.returncode != 0:
        raise RuntimeError(f"submit failed: {out.stderr}")


def poll_score(competition="playground-series-s6e9", timeout_min=20, poll_interval=30):
    import csv, time
    deadline = time.time() + timeout_min * 60
    while time.time() < deadline:
        out = subprocess.run(["kaggle", "competitions", "submissions", competition, "--csv"],
                             capture_output=True, text=True, cwd=REPO_ROOT)
        rows = list(csv.DictReader(out.stdout.splitlines()))
        if rows:
            s = rows[0].get("publicScore") or rows[0].get("public_score")
            if s not in (None, "", "None", "pending"):
                return s
        time.sleep(poll_interval)
    print("  WARNING: timed out waiting for a public score")
    return ""


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cfg", default="{}", help="JSON CFG override for the notebook")
    ap.add_argument("--description", required=True)
    ap.add_argument("--notes", default="", help="hypothesis / gate / mechanism -- write the gate BEFORE the result")
    ap.add_argument("--diff-vs", default=None, help="JSON baseline CFG; enforces a strict-twin (exactly one field differs)")
    ap.add_argument("--submit", action="store_true")
    ap.add_argument("--no-log", action="store_true", help="run without appending to runs.csv")
    args = ap.parse_args()

    cfg = {**DEFAULTS, **json.loads(args.cfg)}
    if args.diff_vs is not None:
        check_strict_twin(cfg, json.loads(args.diff_vs))

    run_id = uuid.uuid4().hex[:8]
    out_dir = REPO_ROOT / "experiments" / "preds" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"run_id={run_id}  cfg={json.dumps(cfg)}\n")

    metrics = execute_notebook(cfg, out_dir)
    print(f"\n{'='*70}\nOOF AUC = {metrics['final_oof_auc']}   ({metrics['n_features']} features)")

    score = ""
    if args.submit:
        submit(out_dir / "submission.csv", args.description)
        score = poll_score()
        print(f"Public LB = {score}")

    if not args.no_log:
        row = cr.build_row(
            metrics,
            run_id=run_id,
            timestamp_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            git_commit=git("rev-parse", "HEAD"),
            git_dirty=bool(git("status", "--porcelain")),
            kernel_ref="local",
            kernel_version="",
            description=args.description,
            public_lb_score=score,
            private_lb_score="",
            preds_dir=out_dir.relative_to(REPO_ROOT).as_posix(),
            notes=args.notes,
        )
        row["run_tag"] = metrics.get("run_tag", "")
        row["engineered"] = ",".join(metrics.get("engineered", []))
        cr.append_run_row(row)
        print(f"Appended run {run_id} to experiments/runs.csv")


if __name__ == "__main__":
    main()
