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

Data downloaded and profiled; LightGBM raw baseline at OOF **0.941660** → public **0.94149**;
repo scaffolding ported from S6E8. The numbers on the wall measured (§4). The GLM probe established
that the generator is additive in log-odds — and produced one wrong conclusion, corrected below.

### Phase 1 — the lookup table (2026-09-02)

Eight strict twins of A0. **Champion moved to B5, OOF 0.944738 → public 0.94511.** The mechanism
and the correction to Phase 0 are in §6. Two large results (B4 +0.003017, B6 −0.001472) and four
clean nulls (B1, B2, B3 all inside the noise floor; B7 −0.000894).

The three negatives cohere rather than being three unrelated nulls: interactions are worth
+0.000033, so raising capacity to 255 leaves **costs** 0.000894 by fitting interactions that are not
there. The generator is additive with a lookup inside it.

### Phase 1b/1c — squeezing the encoder (2026-09-02)

Twins of B5, then of C2. Champion: **D2, OOF 0.945402** (TE of the 3 numerics, `te_smooth=5`,
neighborhood backoff). C2 submitted: OOF 0.945225 → public **0.94549**.

| probe | change | Δ vs its baseline |
|---|---|---|
| C1 | neighborhood backoff (on B5) | +0.000428 |
| C2 | `te_smooth` 20 → 5 (on B5) | +0.000487 |
| C3 | `te_smooth` 20 → 100 (on B5) | −0.000667 |
| C4 | value-count feature (on B5) | +0.000456 |
| C5 / C6 | 15 / 7 leaves (on B5) | +0.000212 / +0.000268 |
| C7 | TE on all 7 numerics (on B5) | −0.000075 |
| D1 | `te_smooth` 5 → 2 (on C2) | +0.000123 |
| D2 | neighborhood backoff (on C2) | +0.000177 |
| D3 | value-count feature (on C2) | +0.000052 |

**The redundancy was predicted and confirmed.** C1, C2 and C4 are three routes to the same problem
— an unreliable rare-value estimate — so stacking them shrinks each one's delta: the backoff was
worth +0.000428 on B5 and +0.000177 on C2; the count feature +0.000456 on B5 and +0.000052 on C2,
which drops it below the gate. Only one of the three ships.

### The two structural negatives that close the encoder axis

Both computed on C2's OOF residuals, with a random-key control landing at 0.99 as it should.

**1. The lookup is fully extracted.** Per-key residual variance, as a ratio to what a correct model
would produce: `Annual_Income_USD` 0.846, `Daily_Commute_km` 0.876, `Age` 0.680 — all *below* 1.0.
(Below rather than at 1.0 because an OOF prediction for value *v* is built from other folds' rows of
*v*, which anticorrelates the error with the realized rate.) There is no per-value signal left.

**2. There is no cross-feature lookup.** Nine joint keys tested; eight land between 0.85 and 0.95.
The one exception, `Annual_Income_USD × Environmental_Concern_Level` at 1.074, is ≈2σ on 300 keys
and is the largest of nine tests — that is selection, not signal, and playbook §5 exists to stop it
being written up as a finding.

**3. The rare-value tail is not where the gap lives.** Only **0.58%** of test rows carry an income
value unseen in train, and only 2.3% land on a value seen fewer than 10 times. This is why the
encoder fixes are worth ~0.0002 rather than ~0.002 — they only touch a few percent of rows. It also
kills the original dataset as a coverage lever: `itzzomkar/ev-adoption-behavior-and-range-anxiety`
(10,000 rows, exact schema match, the source of the synthetic data) could newly cover **82 test
rows**.

### Phase 1d — closing the single-model recipe (2026-09-02)

Strict twins of D4. **Champion: E1, OOF 0.945642 → public 0.94586.**

| probe | change | Δ vs D4 | verdict |
|---|---|---|---|
| **E1** | + neighborhood backoff | **+0.000205** | ships |
| E2 | `te_smooth` 5 → 2 | +0.000117 | clears, but see below |
| **E3** | TE averaged over 4 inner splits | **−0.000019** | **null** |

**The encoder and capacity axes are orthogonal, as hypothesised.** The backoff was worth +0.000177
on C2 and the capacity drop +0.000212; combined in E1 they give +0.000417 against C2, versus
+0.000389 predicted from the parts. Unlike the three encoder fixes, these stack.

**E3 is the important negative, because it was the headline prediction.** The reframe below said the
competition had become a variance problem, and the most direct variance lever — cancelling the
arbitrary choice of which 4/5 of a fold builds each training row's encoding — bought **nothing**.
Mechanism: at ~50 rows per income value, the encoding is already stable across inner draws, so
there was little variance there to cancel. One prediction of the reframe has now failed; the
remaining variance levers (in-fold seed bagging, multi-learner ensembling) are untested, so the
reframe is **weakened but not refuted**, and the next probes are the ones that settle it.

### Phase 2 — the variance reframe is REFUTED (2026-09-03)

After Phase 1c I concluded that signal discovery was finished and the remaining gap to the frontier
was estimation variance. That was **wrong**, and all three of its levers have now been tested:

| lever | probe | result |
|---|---|---|
| encoding variance | E3 — TE averaged over 4 inner splits | −0.000019 **null** |
| model-fitting variance | F2 — 3-seed bagging inside each fold | +0.000034 **null** |
| combiner / correlated error | 6-leg fitted logit stack | +0.000030 over the best single **null** |

All three sit at or below the 0.000038 seed-noise floor. **The reframe is refuted.**

**Why the stack bought nothing, measured rather than assumed.** Pairwise correlation of the six
legs' OOF logits runs **0.9922 to 0.9995**, median 0.9973 — including across the LightGBM/XGBoost
boundary, which is only 0.9980. The fitted combiner put **0 of 6 weights negative**, against 10 of 23
in S6E8. This is playbook §6's discriminator firing exactly as written: *a fitted stack earns its
keep when it can subtract correlated error; if your pool is a set of near-twins, average them.* Six
models on one representation are one model.

**What that leaves.** The remaining ≈0.0006 to the public top is **signal we have not found**, not
noise we have failed to average away. And the failed token probe says something about where it is
*not*:

| probe | change | Δ vs E1 |
|---|---|---|
| F1 | income also as a native LightGBM categorical (13,214 levels) | **−0.000999** |

F1 was the cheap test of the "tokens" idea visible in the public frontier's most-voted notebooks
("EV Adoption Tokens: XGBoost + TinyTokenTransformer", "94.6+ Transformer and GBDT Ensemble"). It
failed hard, and `best_iter` collapsing to ~182 from E1's ~1000 says why: LightGBM's categorical
split is a **free-form, unregularized regrouping** of 13,214 levels at every node, so it memorizes
immediately. That is not evidence against the token idea — it is evidence against the *unregularized*
form of it. A learned embedding is the opposite: a low-rank, shared, continuously-regularized
representation of value identity. **F1 is the wrong proxy, and testing the right one is the next
move.**

### Phase 2b — the token/embedding representation (2026-09-03)

| probe | change | solo OOF | max corr vs pool | stack ADD |
|---|---|---|---|---|
| G1 | token-embedding MLP (emb_dim 16) | 0.943401 | **0.9508** | −0.000002 |
| G2 | G1 retuned (lr 3e-4, emb_dim 8, min_count 20) | 0.943562 | — | not run |

**The representation is genuinely different and still worthless.** G1's max correlation against the
pool is **0.9508**, against a pack whose internal median is 0.9973 — this is by far the most
decorrelated model in the repo, and its ADD contribution is **−0.000002**. That is playbook §7's
rule firing verbatim: *high disagreement is usually weakness, not diversity; the test is whether
disagreement comes WITH competitive solo strength.* It does not — G1/G2 sit ~0.0021 below the
0.945402 pool floor, and playbook §6 says a leg below the floor contributes nothing however
decorrelated. Measured, not assumed.

G2 tested and rejected the obvious excuse for G1 (it peaked at epoch 3 of a 25-epoch OneCycle
schedule, i.e. during LR warmup, which is a schedule bug rather than a capacity verdict). Fixing the
schedule moved it +0.00016 — still 0.0021 short.

### Reading the leaderboard's shape (playbook §8, 2026-09-03)

533 teams. **We are rank 115 at 0.94588.**

| | |
|---|---|
| top score | 0.94647 |
| teams ≥ 0.9460 | **101** |
| top 25 span | 0.94647 → 0.94624 = **0.00023** |
| exact-score spikes | **13 teams at 0.94607**, 9 at 0.94621, 8 at 0.94624 |

**The cluster above us is substantially a shared file.** Thirteen teams do not independently land on
the same five-decimal score. This is the S6E8 pattern — "private ranks 50 through 200 span 0.00001
AUC, which is one CSV, not 150 solutions" — arriving on day 3 rather than at the deadline.

What that changes and what it does not:
- It does **not** mean the score is fake. ~0.9460 is genuinely achievable, and the gap from our
  0.94588 to it is ≈0.0002, which is real but small (1.7σ of the public paired SD).
- It **does** mean rank is not measuring modelling in that band, so rank is not a target to steer by.
- The artifact-sharing policy (§5) anticipated exactly this and **stands until 2026-09-21.** The
  finding is logged now so the week-3 reassessment is made on evidence rather than at the deadline.

The most-voted public notebooks were read for ideas (permitted; no artifact used). Their approach
differs from ours in one testable way: target encoding over **quantile-binned and PAIRED keys with
very heavy smoothing** (`TARGET_SMOOTHING=500`, `PAIR_TARGET_SMOOTHING=1500`) rather than our raw
per-value keys at `te_smooth=5`. Our C3 probe found smoothing 100 *worse* than 5 — but that was on
raw per-value keys, and heavy smoothing over coarse binned keys is a different regime, not a
contradiction. It is the one concrete untested idea the frontier offers, and it is cheap.

### The OOF↔LB instrument, 5 paired points

| run | OOF | public LB | offset |
|---|---|---|---|
| A0 | 0.941660 | 0.94149 | −0.000170 |
| B5 | 0.944738 | 0.94511 | +0.000372 |
| C2 | 0.945225 | 0.94549 | +0.000265 |
| D4 | 0.945437 | 0.94566 | +0.000223 |
| E1 | 0.945642 | 0.94586 | +0.000218 |

**Spearman = 1.0000, slope 1.12, residual σ 0.000122** — and that σ is essentially the public
split's own paired bootstrap SD (0.000116), which is the ideal result: the residual is split noise,
so the OOF is an unbiased ranker. Still provisional at 5 points; the gate stays at the interim
+0.0001 until ~10. **The offset is not a trend** — its range across these five is 0.000542 against a
0.000116 paired SD, and playbook §5 is the post-mortem of reading exactly this kind of series.
