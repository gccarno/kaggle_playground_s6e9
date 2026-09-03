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

## 4. The numbers on the wall

Measured in Phase 0 (2026-09-02). These set what counts as a result for the rest of the
competition. **Do not re-derive them later from marginal LB deltas** — playbook §5 is the
post-mortem of exactly that mistake.

| number | value | how it was measured |
|---|---|---|
| **Seed-noise floor** | **0.000038** OOF | sample SD of the A0 recipe's OOF AUC over 4 model seeds (42, 1337, 7, 2024): 0.941660 / 0.941656 / 0.941580 / 0.941610. **Nothing below this is a result.** |
| **Paired ΔAUC SD, public split** | **0.000116** | paired bootstrap over our own OOF rows at 57,314 rows |
| **Paired ΔAUC SD, private split** | **0.000055** | same, at 229,257 rows — **this is how much room the shakeup has** |
| **Paired ΔAUC SD, full OOF** | **0.000034** | same, at 668,665 rows; agrees with the seed floor, as it should |
| Single-score SD, public split | 0.001078 | bootstrap of one model's absolute AUC at 57,314 rows |
| Single-score SD, private split | 0.000548 | same, at 229,257 rows |
| OOF→LB slope | *unset* | needs ≥10 paired runs |
| OOF→LB residual σ | *unset* | same fit; **the shipping gate is ~1σ converted to OOF units** |

Reproduce with `python scripts/split_resolution.py <runA> <runB>`. The public split is
**20%** of the 286,571 test rows — 57,314 public / 229,257 private. Confirmed 2026-09-02.

**The single/paired distinction is the trap.** A single public score's own noise is 0.001078 — nine
times the paired SD. So "we are 0.0005 behind rank 40" is *inside one score's noise* as an absolute
statement and simultaneously a **4σ paired difference** against that specific opponent's
predictions. Both are true. Always compare against a named opponent's predictions, never against a
rank (playbook §8).

**Interim shipping gate: +0.0001 OOF** — about 2.6× the seed-noise floor, and comfortably below the
public split's paired resolution so that a passing probe is at least *potentially* visible. This is a
placeholder derived from the noise floor alone; it is replaced by ~1σ of the OOF→LB residual as soon
as ~10 paired runs exist.

## 5. The strategic decisions, made on day one

**Artifact-sharing policy — decided 2026-09-02: ours-only until week 3, then reassess.**
Every leg shipped before 2026-09-21 is one we trained. Public notebooks may be *read* for ideas and
attached as *diagnostics*, but no public submission or blend artifact enters a shipped model. On or
about 2026-09-21 — once the OOF→LB instrument exists and the leaderboard's shape is known (§5,
"read the shape") — the decision is reopened **once**, deliberately, with the price in places written
down at that moment.

*Risk accepted, on the record:* playbook §8 warns that drifting between "best model we can build"
and "best score we can obtain" gets you the worst of both. The mitigation is that this is a single
scheduled reassessment with a written trigger, not an open option to drift. If week 3 arrives and the
decision is not made deliberately, the default is that the ours-only rule stands.

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

### The structural finding — corrected in Phase 1

Phase 0 measured that the generator is **additive in log-odds**, and that part stands:

| model | OOF AUC |
|---|---|
| Logistic regression, additive (log-income, one-hots) | 0.938402 |
| The same GLM **+ all 28 pairwise interactions among the 8 strong drivers** | 0.938435 |
| LightGBM, raw features | 0.941642 |

Pairwise interactions bought **+0.000033** — nothing. **Interaction search is crossed off.**

Phase 0 then drew the wrong conclusion from that, namely that the remaining lever was smooth
per-feature *shape*. Phase 1 refuted it. **`Annual_Income_USD` is a value→target lookup table, not
a magnitude:**

- After fitting the **best possible monotone function** of income (isotonic, weighted, on the 2,213
  values with n≥100 covering 451,901 rows), the per-value residual rate SD is **0.0748** against a
  binomial expectation of **0.0284**. Excess real per-value structure: **0.0685**.
- Within the narrow 80k–90k window alone, 739 distinct values have rate SD 0.0858 vs 0.0321 expected.
  `86095` (n=1082) sits at 0.199 while `96749` (n=1073) sits at 0.115.
- Spearman(income value, per-value rate) is only **0.68** — a real monotone trend exists *underneath*
  the lookup, which is why the backoff for rare values should be the local neighborhood, not the
  global prior.
- 13,214 distinct income values across 668,665 rows (≈50 rows/value); **86.2%** of test income values
  appear in train, so the lookup transfers.

The three probes that establish it, all strict twins of A0 (seed-noise floor 0.000038):

| probe | change | OOF | Δ vs A0 |
|---|---|---|---|
| **B4** | per-value TE of `Annual_Income_USD` | 0.944677 | **+0.003017** |
| **B6** | monotone-increasing constraint on income | 0.940188 | **−0.001472** |
| B2 | `log(income)` added | 0.941660 | +0.000000 |

B6 is the clincher: **forbidding non-monotonicity costs 39σ.** A constraint can only be that
expensive if the thing it forbids is real signal. And B2 shows that a monotone *re-expression* of
income buys exactly zero. So the signal lives in the value identity, and only a per-value encoding
can see it.

**This is S6E8's mechanism recurring** (playbook §7): *any model that reads an integer code as a
magnitude cannot see a non-monotonic lookup, however much capacity it is given.* The Phase 0 GLM
probe was blind to it for precisely that reason — it read income as a number. **The lever is
per-value encoding, not shape and not interaction.**

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
