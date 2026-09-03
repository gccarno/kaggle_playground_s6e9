# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository purpose

Kaggle **Playground Series S6E9** ("Predicting Electric Vehicle Purchases"). Binary classification of
`Will_Buy_EV` from 13 features about a household's income, commute, charging access and stated
environmental concern, scored on **ROC AUC**. 668,665 train rows, 286,571 test rows, positive rate
0.174645. **Deadline 2026-09-30.**

**`README.md` is the contract** — the frozen CV split, the metric, the leakage rule, the three
measured numbers and the strategic decisions live there and are not up for renegotiation mid-
competition. Read it before touching a model.

**`KAGGLE_PLAYBOOK.md`** is the method document, distilled from two prior runs — S6E7 (rank 120,
+298 places in the shakeup) and S6E8 (private 0.97030, rank 675/3,532, +6 places). Its thesis:
build the measuring instrument first, and trust it all the way to its conclusion — including when it
says the signal is exhausted, and when it says the teams above you are genuinely ahead. Do not
re-litigate its conclusions; test them here as probes if they need testing.

## What is different about S6E9

Three things reshape the workflow relative to S6E8, and each has a consequence:

1. **The dataset is small and fast.** A full 5-fold LightGBM runs in **32 seconds** on local CPU
   (S6E8: ~5 minutes local, ~15 minutes per kernel push-poll cycle). **Iterate locally.** Kaggle
   kernels are for GPU/neural legs and for the champion's reproducible record — not for screening.
   A probe that would have cost a kernel cycle in S6E8 costs less than a minute here, so the
   binding constraint is thinking, not compute. Run more probes.
2. **The generator is additive in log-odds.** Measured, not assumed: a plain additive GLM reaches
   OOF 0.938402, and adding all 28 pairwise interactions among the 8 strong drivers moves it to
   0.938435 — **+0.000033, i.e. nothing**. LightGBM's +0.0032 over the GLM is per-feature *shape*,
   not interaction. **Do not spend time on interaction search.** Spend it on response shapes:
   per-value target encoding, splines, monotone constraints, calibrated ordinal codings.
3. **There is real headroom.** The raw baseline is 0.94164 against a public top of 0.94644 — a
   0.0048 gap, roughly forty times S6E8's residual σ. Unlike S6E8, the early state of this
   competition is *not* "the signal is exhausted"; assume there is modelling to do until measurement
   says otherwise.

Also inherited unchanged: **zero missingness, zero duplicate rows, no train↔test drift.** Missing-
indicator features, imputation strategies and adversarial-validation drift work are all off the
table — there is nothing there.

## Repo layout

- `src/pipeline.py` — the single source of truth for the model. A `DEFAULTS` config dict at the top;
  every probe is a **config override**, never an edit to the pipeline body. Emits `submission.csv`,
  `oof_proba_<learner>.csv`, `test_proba_<learner>.csv` and a final `RUN_METRICS_JSON:{...}` line.
- `s6e9-model.ipynb` + `kernel-metadata.json` — the Kaggle kernel wrapper. CPU only
  (`enable_gpu: false`); flip it only when a neural leg actually arrives.
- `scripts/run_local.py` — **the primary loop.** Run the pipeline locally with a config override,
  archive artifacts, optionally submit, append a row to `experiments/runs.csv`.
- `scripts/collect_run.py` — kernel equivalent: push → poll → parse `RUN_METRICS_JSON` → archive →
  submit → log.
- `scripts/stack_logit.py`, `leg_probe.py`, `leg_swap.py`, `leg_diversity.py`, `subset_ceiling.py`,
  `compare_oof.py`, `public_gap.py` — ensemble/analysis toolkit ported from S6E8.
- `experiments/runs.csv` — **tracked in git**, append-only, one row per run. The single most
  valuable asset in the repo; both prior competitions' best analyses were pure re-analysis of it.
- `experiments/preds/<run_id>/` — gitignored per-run probability artifacts. **Never delete these.**
- `data/` — gitignored local copy of the competition CSVs.

## Running / iterating

```bash
# re-download the data
kaggle competitions download -c playground-series-s6e9 -p data --unzip

# the primary loop: local run, archived, logged, optionally submitted
python scripts/run_local.py --cfg '{"run_tag":"B1","fe_target_encode":true}' \
  --description "B1: per-value target encoding of the 3 strong numerics" \
  --notes "HYPOTHESIS ... MECHANISM ... GATE ..." --submit

# strict-twin guard: refuse to run unless exactly one config field differs
python scripts/run_local.py --cfg '{...}' --diff-vs '{"run_tag":"champion"}' ...

# kernel path (GPU legs / reproducible champion record)
python scripts/collect_run.py --submit --description "..." --notes "..."
kaggle kernels status gcarno/s6e9-model
```

## Conventions to preserve when editing

- **Leakage discipline is the load-bearing constraint.** Target encoders, bin edges, scalers and
  category vocabularies are fit on the **train fold only**, re-fit at every usage site (feature
  selection, HPO, final stack), never once globally before the CV loop. Unseen categories map to a
  reserved "unknown" level rather than raising. An unsupervised label mapping over train ∪ test is
  *not* a leak; a supervised (target/count) encoding is — say which one it is in a comment when it
  isn't obvious.
- **`roc_auc_score` is the objective everywhere** — CV scoring, any Optuna objective, any neural
  early-stopping. Never accuracy, never logloss as a silent proxy.
- **Submit probabilities, never hard labels.** AUC is rank-based: there is no threshold to tune and
  no calibration that can move the leaderboard — only ranking can. Cross off every knob that touches
  only calibration.
- **Never change the CV split.** `StratifiedKFold(5, shuffle=True, random_state=42)`. Every archived
  OOF matrix is aligned to it; changing it invalidates all cross-run blending retroactively.
- **Always write the per-learner OOF + test probability artifacts**, for every run, even throwaway
  ones. This is the highest-ROI habit in the playbook.
- **Fold-score spread is evaluation-fold difficulty, not test-prediction variance.** Do not respond
  to a wide spread by re-running at a new split seed; S6E7 spent 2.3 GPU-hours proving that buys
  nothing.
- **Every probe needs a written hypothesis and a pre-registered gate before it runs**, and its
  result is logged with the *mechanism*, not just the number. A well-written negative result is
  worth more than a +0.0001 blend.
- **Judge ensemble legs by ADD and SWAP contribution at constant pool size, never by solo score.**
  In S6E8's final round, 10 of 17 probes cleared the solo gate and 0 cleared the stack gate.
- **The gate stays unset in `README.md` §4 until the OOF↔LB residual σ has been measured** over
  ~10 paired runs. Do not invent one earlier, and do not re-derive it later from marginal LB deltas.
- **Burn the daily submission slots.** 5/day, no rollover. Each spent slot is a paired OOF↔LB point,
  and the paired points are what make every offline decision trustworthy. S6E8 used 18 of a possible
  ~290 and paid for it by over-reading four of them.
