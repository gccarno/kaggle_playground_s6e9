#!/usr/bin/env python3
"""Generate a one-off Kaggle CPU kernel that runs src/pipeline.py under a config override.

Why this exists: Phase 16 hit a hard local memory ceiling -- a 5-fold run on a
160+-feature representation peaks above what a 16GB laptop has free, and probes were
OOM-killed mid-fold four times in a row. Kaggle CPU kernels have ~30GB and allow 5
concurrent sessions, so the fan-out that was previously a convenience (Phase 15's HPO
shards) is now the only way to run a probe at this feature count at all.

The kernel clones this repo from GitHub, so THE CONFIG'S CODE MUST BE PUSHED FIRST --
a kernel built from an unpushed pipeline.py silently runs the previous revision.

    python scripts/make_probe_kernel.py <tag> <cfg.json>     # write .ipynb + metadata
    python scripts/collect_run.py --metadata kernel-metadata-probe-<tag>.json \
        --description "..." --notes "..."                    # push, poll, archive, log
"""
import argparse
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
REPO_URL = "https://github.com/gccarno/kaggle_playground_s6e9.git"

SETUP = '''import subprocess, sys
def run(cmd, **kw):
    r = subprocess.run(cmd, text=True, **kw)
    if r.returncode != 0:
        raise RuntimeError(f"command failed: {cmd}")
    return r

subprocess.run(["rm", "-rf", "kaggle_playground_s6e9"])
run(["git", "clone", "-q", "%s"])
run(["git", "-C", "kaggle_playground_s6e9", "log", "--oneline", "-1"])
''' % REPO_URL

BODY = '''import subprocess, sys, os, json
cfg = json.loads(%s)
env = dict(os.environ, S6E9_CFG=json.dumps(cfg), PYTHONUNBUFFERED="1")
proc = subprocess.run([sys.executable, "-u", "src/pipeline.py"],
                       cwd="kaggle_playground_s6e9", env=env)
if proc.returncode != 0:
    raise RuntimeError("pipeline.py failed")
'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tag", help="short probe tag, e.g. Q1 (used in the kernel slug)")
    ap.add_argument("cfg", help="path to a JSON config override file")
    args = ap.parse_args()

    cfg = json.loads(Path(args.cfg).read_text())
    slug = args.tag.lower().replace("_", "-")
    nb_name = f"probe_{args.tag}.ipynb"

    cells = [SETUP, BODY % json.dumps(json.dumps(cfg))]
    nb = {
        "cells": [{"cell_type": "code", "execution_count": None, "metadata": {},
                   "outputs": [], "source": c.splitlines(keepends=True)} for c in cells],
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                    "name": "python3"},
                     "language_info": {"name": "python"}},
        "nbformat": 4, "nbformat_minor": 5,
    }
    (REPO_ROOT / nb_name).write_text(json.dumps(nb, indent=1))

    meta = {
        "id": f"gcarno/s6e9-probe-{slug}", "title": f"s6e9-probe-{slug}",
        "code_file": nb_name, "language": "python", "kernel_type": "notebook",
        "is_private": True, "enable_gpu": False, "enable_tpu": False,
        "enable_internet": True, "keywords": [], "dataset_sources": [],
        "kernel_sources": [], "competition_sources": ["playground-series-s6e9"],
        "model_sources": [], "machine_shape": "None",
    }
    meta_name = f"kernel-metadata-probe-{slug}.json"
    (REPO_ROOT / meta_name).write_text(json.dumps(meta, indent=1))
    print(f"{nb_name}  +  {meta_name}   (run_tag={cfg.get('run_tag')})")


if __name__ == "__main__":
    main()
