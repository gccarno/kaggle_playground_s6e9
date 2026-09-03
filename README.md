# Playground Series S6E9 — Predicting Electric Vehicle Purchases

**This file is the contract.** The frozen CV split, the metric, the leakage rule and the strategic
decisions live here. They are not up for renegotiation mid-competition. Read this before touching a
model. `KAGGLE_PLAYBOOK.md` is the method document carried over from S6E7 and S6E8;
`CLAUDE.md` is the orientation for a fresh agent session.

---

## 1. The competition

| | |
|---|---|
| Competition | `playground-series-s6e9` — "Predicting Electric Vehicle Purchases" |
| Task | **binary classification**, target `Will_Buy_EV` (`Yes`/`No`) |
| Metric | **ROC AUC** (confirmed: sample_submission holds probabilities; baseline OOF 0.94164 lands in the LB's 0.9464 range) |
| Train | 668,665 rows × 13 features |
| Test | 286,571 rows |
| Positive rate | 0.174645 (= the constant in `sample_submission.csv`) |
| Deadline | **2026-09-30 23:59 UTC** |
| Teams at start of work | 460 (2026-09-02); public top 0.94644 |
| Submissions | 5/day, do not roll over |

**Submit probabilities, never hard labels.** AUC is rank-based. What the metric is invariant to,
and therefore what is off the table entirely:

- any monotone transform of the output (calibration, Platt scaling, isotonic — all worth 0.00000)
- any threshold or decision rule
- class weights chosen to fix a decision boundary (they can still change *ranking* via the fit, so
  they are a legitimate model knob — but never a post-hoc correction)

Only **ranking** can move the leaderboard.

---

## 2. The frozen CV split — never change it

```python
StratifiedKFold(n_splits=5, shuffle=True, random_state=42)   # stratified on y
SEED = 42
```

Every archived OOF matrix is aligned to this split. Changing it retroactively invalidates all
cross-run blending. If a probe seems to need a different split, the probe is wrong.

`ID = "id"`, `TARGET = "Will_Buy_EV"`, encoded as `(df[TARGET] == "Yes").astype(int)`.

---

## 3. Leakage rule

Target encoders, quantile bin edges, scalers, imputers and category vocabularies are fit on the
**training fold only**, re-fit at **every usage site** — feature selection, HPO, and the final stack
included. Never fit once globally before the CV loop.

- Unseen categories at inference map to a reserved "unknown" level; never raise.
- An unsupervised label mapping over train ∪ test is **not** a leak. A supervised (target/count)
  encoding **is**. Say which one it is in a comment when it is not obvious.
- `roc_auc_score` is the objective everywhere: CV scoring, any Optuna objective, any neural
  early-stopping criterion. Never accuracy, never logloss as a silent proxy.

This rule is what buys the right to make decisions offline (playbook §4). S6E8's payoff receipt was
Spearman(OOF, private LB) = 0.9974 over 18 runs.

---

## 4. The three numbers on the wall

Set in Phase 0, then used for the rest of the competition. **Unset until measured — do not invent
values earlier.**

| number | value | how it was measured |
|---|---|---|
| OOF→LB slope | *unset* | regress LB on OOF over ≥10 paired runs |
| OOF→LB residual σ | *unset* | same fit; **the gate is ~1σ converted to OOF units** |
| Paired-bootstrap SD, public split | *unset* | paired bootstrap over our own OOF rows at the public split's row count |
| Paired-bootstrap SD, private split | *unset* | same, at the private row count — this is the shakeup's room |
| Seed-noise floor | *unset* | re-run one recipe at a new model seed; nothing below this is a result |

**Shipping gate: *unset* until the residual σ exists.** Until then, probes are logged but nothing is
promoted to the champion recipe on a delta alone.

---

## 5. The strategic decisions, made on day one

**Artifact-sharing policy.** *Pending — to be written before the public leaderboard is tempting.*
The choice is between "best model we can build" (every leg is one we trained) and "best score we can
obtain" (public blend artifacts are legal and are what most of the field does). S6E8 chose the
former, priced it at ~180 places, and the private split charged it in full. Whichever is chosen, it
goes here in writing, with the price attached, before it matters.

**Local-first iteration.** A full 5-fold LightGBM on this data runs in **32 seconds** on local CPU.
This is the single biggest operational difference from S6E8 (~5 min local / ~15 min per kernel
cycle). Kaggle kernels are reserved for GPU/neural legs and for the reproducible record of the
champion. Everything else is screened locally.

---

## 6. What the data is

Thirteen features, **zero missing values**, zero duplicate feature rows, and zero test rows whose
feature tuple appears in train. Distributions are near-identical between train and test (no drift
worth modelling).

| feature | type | nuniq | single-feature AUC | note |
|---|---|---|---|---|
| `Environmental_Concern_Level` | ordinal 1–5 | 5 | **0.84352** | dominant driver; rate 0.006 → 0.518 |
| `Annual_Income_USD` | numeric | 13,214 | **0.74276** | **clipped at 30000** (9.2% of rows) |
| `Subsidy_Available` | binary | 2 | **0.71794** | near-gate: rate 0.0058 (No) vs 0.2747 (Yes) |
| `Daily_Commute_km` | numeric, 1dp | 805 | 0.56024 | **clipped at 5.0** (21.6% of rows) |
| `Home_Charging_Possible` | binary | 2 | 0.55082 | |
| `Age` | int 25–69 | 45 | 0.54788 | non-monotone |
| `Range_Anxiety_Level` | ordinal | 3 | 0.54510 | High ⊂ Home_Charging=No (2194/2194 in train) |
| `Charging_Stations_Near_Home` | int 0–14 | 15 | 0.52407 | ~flat in logit — likely noise |
| `City_Type` | cat | 3 | 0.52345 | |
| `Charging_Stations_Near_Work` | int 0–19 | 20 | 0.51272 | ~flat in logit — likely noise |
| `Current_Car_Type` | cat | 4 | 0.51037 | |
| `Number_of_Cars_Owned` | int 1–4 | 4 | 0.50625 | ~flat in logit — likely noise |
| `Gender` | cat | 3 | 0.50461 | ~flat — likely noise |

### The structural finding

**The generator is close to additive in log-odds.** Measured in Phase 0:

| model | OOF AUC |
|---|---|
| Logistic regression, additive (log-income, splines-free, one-hots) | 0.938402 |
| The same GLM **+ all 28 pairwise interactions among the 8 strong drivers** | 0.938435 |
| LightGBM, raw features, default-ish | **0.941642** |

Pairwise interactions bought **+0.000033** — nothing. So the GBDT's +0.0032 over the GLM is
**per-feature nonlinear shape**, not interaction. `Environmental_Concern_Level` in logit units is
`[-5.17, -3.82, -2.08, -1.11, +0.07]` — strongly non-linear steps; the income logit curve is
non-monotone at its bottom because the 30000 clip mixes two populations.

**Consequence for the plan:** the lever is *recovering per-feature response shapes* (target/rate
encoding, splines, monotone-constrained boosting), not interaction search. Cross this off before
spending a week on it.

---

## 7. Repo layout

- `src/pipeline.py` — the single source of truth for the model. A `DEFAULTS` config dict at the top;
  every probe is a config override, never an edit.
- `scripts/run_local.py` — run the pipeline locally with a config override, archive artifacts,
  optionally submit, append a row to `experiments/runs.csv`. **The primary loop.**
- `scripts/collect_run.py` — the Kaggle-kernel equivalent: push → poll → parse `RUN_METRICS_JSON` →
  archive → submit → log. For GPU legs and the champion's reproducible record.
- `scripts/stack_logit.py`, `leg_probe.py`, `leg_swap.py`, `leg_diversity.py`, `subset_ceiling.py`,
  `compare_oof.py`, `public_gap.py` — the ensemble/analysis toolkit, ported from S6E8.
- `experiments/runs.csv` — **tracked in git**, append-only, one row per run. The most valuable asset
  in the repo.
- `experiments/preds/<run_id>/` — gitignored per-run `oof_proba_<learner>.csv` /
  `test_proba_<learner>.csv`. **Never delete these.**
- `data/` — gitignored local copy of the competition CSVs.

---

## 8. Phase log

*(Each phase gets its hypothesis, gate, result and mechanism appended here as it completes.)*

### Phase 0 — orientation (2026-09-02)

Data downloaded and profiled; LightGBM raw baseline at OOF **0.941642**; GLM structural probe
established that the generator is additive-in-logit (§6). Repo scaffolding ported from S6E8. No
submission yet.
