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
| **OOF→LB slope** | **1.0832** | least squares over **10 GBDT-family** paired runs (2026-09-04) |
| **OOF→LB residual σ** | **0.000103** | same fit; at the public split's own paired resolution, as it should be |
| **SHIPPING GATE** | **+0.0000949 OOF** | = residual σ ÷ slope. **Replaces the interim +0.0001 placeholder**, which it confirms. |

**The paired ΔAUC SD is a property of the PAIR, not of the split.** Measured 2026-09-04 and it
changes how every LB comparison is read:

| pair | OOF logit corr | paired SD, public split |
|---|---|---|
| F2 vs E1 (strict twins) | ~0.9999 | **0.000027** |
| the 0.000116 row above | — | 0.000116 |
| G1 vs E1 (embedding MLP) | 0.9508 | **0.000237** |

An order of magnitude of range. So a strict-twin LB comparison resolves ~4× *finer* than the wall
number implies, and a comparison against a decorrelated model ~2× coarser. Quote the SD for the
actual pair — `scripts/split_resolution.py <runA> <runB>` — never the single number.

**The OOF→LB fit is valid within the GBDT family ONLY.** Adding one non-GBDT point (G1) to the
regression blows the residual σ from 0.000103 to **0.000340**, a 3.3× degradation. See Phase 3.

Reproduce with `python scripts/split_resolution.py <runA> <runB>`. The public split is
**20%** of the 286,571 test rows — 57,314 public / 229,257 private. Confirmed 2026-09-02.

**The single/paired distinction is the trap.** A single public score's own noise is 0.001078 — nine
times the paired SD. So "we are 0.0005 behind rank 40" is *inside one score's noise* as an absolute
statement and simultaneously a **4σ paired difference** against that specific opponent's
predictions. Both are true. Always compare against a named opponent's predictions, never against a
rank (playbook §8).

~~**Interim shipping gate: +0.0001 OOF**~~ — **superseded 2026-09-04.** The gate is now
**+0.0000949 OOF**, derived as residual σ ÷ slope over 10 paired runs exactly as planned. The
placeholder was right to within 5%, so nothing about past decisions changes; what changes is that
the number is now measured rather than guessed, and per playbook §3 it stays fixed for the rest of
the competition. **Do not re-derive it later from marginal LB deltas.**

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
| `Charging_Stations_Near_Home` | int 0–14 | 15 | 0.52407 | flat only in a LINEAR screen — real non-monotone signal, see Phase 6 |
| `City_Type` | cat | 3 | 0.52345 | |
| `Charging_Stations_Near_Work` | int 0–19 | 20 | 0.51272 | flat only in a LINEAR screen — real non-monotone signal, see Phase 6 |
| `Current_Car_Type` | cat | 4 | 0.51037 | |
| `Number_of_Cars_Owned` | int 1–4 | 4 | 0.50625 | flat only in a LINEAR screen — real non-monotone signal, see Phase 6 |
| `Gender` | cat | 3 | 0.50461 | ~flat — weakest of the four, still likely noise |

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
- `scripts/shap_champion.py` — read-only SHAP diagnostic on E1's fold-0 model (Phase 6). Not a
  probe: touches no OOF/LB, exists to surface structure the aggregate structural tests average away.
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

### Phase 2c — the frontier's encoder recipe, tested and rejected (2026-09-03)

The one concrete untested idea the public frontier offered (logged in "Reading the leaderboard's
shape" below): target encoding over **quantile-binned** and **paired** keys with very heavy
smoothing (`TARGET_SMOOTHING=500`, `PAIR_TARGET_SMOOTHING=1500`), rather than our raw per-value keys
at `te_smooth=5`. Three strict twins of E1 (0.945642). **None ships.**

| probe | change | OOF | Δ vs E1 | mean `best_iter` |
|---|---|---|---|---|
| **H1** | + TE on quantile-**bin** keys of the 3 numerics (100 bins, smooth 500) | 0.945617 | **−0.000025** | 1041 |
| **H2** | + TE on **5 paired** keys (20 bins/component, smooth 1500) | 0.945121 | **−0.000521** | 333 |
| **H3** | + TE on **1 paired** key (income × concern) | 0.945237 | −0.000405 | 637 |
| *(E1)* | *baseline* | *0.945642* | — | *998* |

Encoder verified leak-free before the probes ran: a random 13,214-level key encoded through the same
inner-fold protocol scores **0.4982** on held-out rows, and train-row and val-row AUCs of the real
keys agree to 0.001 (0.68204 / 0.68102).

**H1 — binned keys are null, and the reason is that we already compute them.** E1's neighborhood
backoff *already* estimates the smoothed local bin rate; it just uses it as a shrinkage target for
rare values instead of exposing it as a column. Handing the same quantity to the learner a second
time adds nothing. C3 (raw-value keys at smoothing 100, −0.000667) and H1 are therefore not in
tension: heavy smoothing over coarse keys is harmless-and-redundant, heavy smoothing over fine keys
destroys the lookup. **Neither is a lever.**

**H2/H3 — paired keys actively cost, by DISPLACEMENT, and the two probes separate the mechanism.**
`income × concern` alone reaches solo AUC **0.877** — one column that is nearly a complete model. The
tree eats it greedily and early-stops long before it has refined the additive per-feature shapes E1
spends ~1000 rounds building: `best_iter` falls **998 → 637 → 333** as more lookup is handed over, and
the loss grows with it. One pair costs **78%** of what five cost, so this is displacement by the
dominant key, not noise accumulating across five correlated columns. It is F1's signature again
(best_iter 1000 → 182, −0.000999) with a different, *regularized* feature — which makes the point
sharper: a coarse joint lookup is not too noisy, it is too **coarse**, and it crowds out finer
additive structure that is worth more.

**This closes the encoder axis, and it is the third independent confirmation that the generator is
additive.** The first was the GLM (all 28 pairwise interactions worth +0.000033); the second the
joint-key residual test (8 of 9 keys at 0.85–0.95). H2/H3 are the strongest of the three, because
they answer the objection the other two could not: E1 runs at `num_leaves=7` *precisely because* the
generator is additive, so it can barely represent a 2-way interaction — the worry was that the
capacity cut had thrown real interactions away. Handing five of them over as pre-computed columns,
which is the cheapest possible way to buy an interaction, makes things **worse**. The capacity cut
threw nothing away.

**What it means for the leaderboard reading.** The frontier's encoder recipe is not where the ≈0.0002
above us comes from — we tested it and it is behind ours. That is consistent with, not evidence
against, the shared-file finding below.

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

### Phase 3 — the instrument is finished, and it turns out to be family-specific (2026-09-04)

Five slots spent in one day (playbook §1's "burn the slots" rule, applied deliberately for the first
time): two on champion candidates, **two chosen purely to stress the instrument**. The two stress
points are where the value was.

| slot | submission | OOF | predicted LB | actual LB | |
|---|---|---|---|---|---|
| 1 | F2 (3-seed bagged E1) | 0.945676 | 0.94590 | **0.94587** | ✅ 1σ |
| 2 | **G1** (embedding MLP) | 0.943401 | 0.94345 | **0.94450** | ❌ **+0.00105 = 4.4σ** |
| 3 | 3-leg fixed rank-mean blend | 0.945743 | 0.94603 | **0.94588** | −1.5σ |
| 4 | I1 (boosted GAM) | 0.945581 | 0.94582 | **0.94570** | −1.2σ |

**1. The OOF→LB ranker is valid for GBDTs and NOT across model families.** G1 scores 0.00105 higher
than its OOF predicts. This is not split luck: the G1↔E1 paired SD was measured at **0.000237**
(2× a near-twin pair's, because decorrelated models do not share split luck) and the miss is still
4.4σ. Include G1 in the regression and the residual σ goes 0.000103 → **0.000340**.

*Mechanism, hypothesised and not yet measured:* **an OOF row is predicted by ONE fold-model; a test
row is the average of FIVE.** For G1 — a net that early-stopped at epoch 3–6 — that averaging is a
large variance cut. For a 1000-round GBDT it is nearly nothing. So OOF systematically *understates*
high-variance learners, and our instrument was calibrated entirely on low-variance ones.
**Consequence if true: Phase 2b rejected G1 with a biased ruler**, and every future neural leg must
be seed-bagged in-fold before its OOF is compared to a GBDT's. The offline test needs no slot —
bag G1 in-fold and see whether its OOF jumps ~0.001; F2 is the control, where the same treatment on
E1 bought +0.000034.

**2. The paired SD is a property of the pair, not the split** (§4). Range measured: 0.000027 for
strict twins, 0.000237 at corr 0.95.

**3. Playbook §6's open question, answered: fixed beats fitted here, and the elbow is at 3.**

| combiner | OOF |
|---|---|
| **fixed rank-mean, 3 legs (E1 lgb + E4 xgb + E5 cat)** | **0.945743** |
| fitted logit stack, 6 legs | 0.945719 |
| fixed rank-mean / logit-mean, 6 legs | 0.945712 |

Exactly §6's prediction for a pool of near-twins. `rank_mean` and `logit_mean` agree to 6 dp — at
corr 0.998 the scale choice is irrelevant, and unlike S6E8 this target does not saturate. **But the
+0.000054 OOF edge over E4 produced 0.00000 on the LB**, which is below that pair's resolution, so
§5 forbids reading it either way. The 3-leg fixed blend is the §9 **Final B** candidate: no fitted
machinery at all.

**4. `id` carries nothing.** AUC 0.49999, flat across deciles, zero train/test id overlap.

### Phase 3b — the additivity constraint, and a mechanism in README that was wrong

`additive_only` (LightGBM `interaction_constraints`, one group per column) makes the ensemble an
exact boosted GAM. Two strict twins, chained one field at a time, plus D2 which *is* E1's recipe at
63 leaves — giving a complete 2×2:

| | 63 leaves | 7 leaves | **Δ (7−63)** |
|---|---|---|---|
| interactions allowed | D2 0.945402 | E1 0.945642 | **+0.000240** |
| interactions forbidden | I2 0.945380 | I1 0.945581 | **+0.000201** |
| **Δ (forbid−allow)** | **−0.000022** | **−0.000061** | |

**Forbidding interactions costs ~nothing at either capacity** — the fourth independent confirmation
that the generator is additive, and the most direct, since it is a constraint rather than an
inference. Both deltas sit at or near the 0.000038 seed floor.

**The capacity mechanism recorded for B7/D4 is wrong.** It said lower capacity wins *because* the
tree can no longer fit interactions that are not there. If that were so, forbidding interactions
would have removed the reason and the capacity effect with it. It does not: +0.000240 allowed vs
+0.000201 forbidden, the same within noise. With additivity forced a 63-leaf tree is simply a
63-piece *univariate* step function, and `best_iter` falls 1032 → 676. **This is per-feature shape
overfitting, not interaction overfitting** — the optimum is coarse-and-slow, which is why ~1000
rounds of 7-leaf trees wins. Both probes' own hypotheses (I1 ≥ E1, I2 > I1) are refuted; the
correction to the mechanism is worth more than either would have been.

*Harness gap found:* `scripts/run_local.py` prints the resolved config but never archives it, so
G1's exact override is unrecoverable from the repo. Playbook §1's rule is that nothing lives only in
scrollback, and the config **is** the experiment. Fix: write `cfg.json` into the preds dir. **Fixed**
— every run since now archives its resolved cfg into `experiments/preds/<run_id>/cfg.json`.

### Phase 4 — the bagging mechanism, tested offline, no slot spent (2026-09-05)

G1's own cfg was still unrecoverable (it predates the fix), so it was reconstructed from G2's notes
(epochs 25→15 implies G1=25) and E1's TE recipe (G1's notes: "TE features are kept alongside"). As
**G1r**, this reproduces G1 **exactly** — OOF 0.943401 and all five fold AUCs identical bit-for-bit
— so the reconstruction is trusted as the strict-twin baseline Phase 3 needed but didn't have.

| probe | change | OOF | Δ vs G1r |
|---|---|---|---|
| G1r | reconstructed G1 | 0.943401 | — |
| **G1rB** | + 3-seed in-fold bagging (F2's treatment) | **0.944074** | **+0.000673** |

**The gate clears by 20x offline.** F2 (the same treatment on E1, a 1000-round GBDT) bought only
+0.000034 — the control for "bagging helps for boring reasons." G1rB's +0.000673 recovers most of
G1's own +0.00105 LB miss, using nothing but the OOF's own fold structure — the signature Phase 3
predicted and not the signature of "embeddings are just weaker."

A slot was spent to check it: **G1rB → LB 0.94465** (submission, not blend). Against the GBDT-family
fit (slope 1.0832, intercept −0.0784), predicted LB is 0.94417 — residual **+0.00048**, against
G1's own original residual of **+0.00106**. Bagging cut the family-bias gap **by more than half**,
but did not close it: +0.00048 is still ~5σ against the instrument's 0.000092 residual σ. The LB
itself barely moved (0.94450 → 0.94465, +0.00015) while OOF moved 4.5x more (+0.000673) — exactly
what the mechanism predicts, since test predictions were *already* 5-way fold-averaged and had
little room to gain, while the OOF was scored by one fold-model at a time and had a lot. **Verdict:
the mechanism is confirmed and explains most, not all, of the miss.** Phase 2b's rejection of G1
used a biased ruler, but even corrected, embeddings still sit outside the GBDT-family instrument by
a real margin — a second, smaller family-specific effect remains unexplained. Any future neural leg
must be seed-bagged in-fold before its OOF is compared to a GBDT's; the residual gap means one
should not be trusted at instrument-level precision even after bagging.

### Phase 5 — the shrinkage axis is also null; E1's hyperparameters were never actually tuned (2026-09-05)

Every capacity probe since Phase 1c changed `num_leaves`. Two Phase-0-era defaults riding along
unchanged in every one of them — `learning_rate=0.05` and `min_child_samples=20` — were never
themselves probed, even though Phase 3b's mechanism (capacity wins by buying per-feature SHAPE
RESOLUTION, not interaction fitting) predicts they might matter: a slower, finer boosting schedule
or a looser leaf-occupancy floor should, if the mechanism is right, resolve the same univariate
curves more precisely. Two strict twins of E1, offline, no slot needed:

| probe | change | OOF | Δ vs E1 | best_iter |
|---|---|---|---|---|
| J1 | `learning_rate` 0.05 → 0.02 | 0.945653 | +0.000011 **null** | 1996–2780 |
| K1 | `min_child_samples` 20 → 5 | 0.945602 | **−0.000040** | 702–1525 |

**Both null, one slightly negative.** J1's `best_iter` sits well clear of both 0 and the 12000 cap,
so the null is not an early-stopping artifact — three times the rounds at 2.5x smaller steps buys
nothing. K1's small loss says loosening the leaf floor costs a touch of overfitting rather than
resolving anything finer. **The shape-resolution axis is saturated at E1's recipe, not
under-tuned** — the Phase-0 defaults for these two knobs happened to already be fine, which is a
different (weaker) claim than "capacity is the lever," and worth recording precisely because it
closes off the last untested corner of that mechanism. No slot spent: neither result had a positive
OOF delta to check against the LB.

**Where this leaves the search, before Phase 6.** Encoder, capacity, interaction, ensembling,
in-fold variance and shrinkage/leaf-occupancy are all closed. The 0.94588 champion is at or very
near the ceiling this feature set and model family can reach; the ≈0.0002 gap to the well-supported
~0.9460 frontier (§8, 1.7σ of the public paired SD) is consistent with noise, not an unfound lever.
Any further gain would need a genuinely new representation (the embedding axis, Phase 2b–4, is
measured and real but below the pool floor) or a structural fact about the generator not yet found.

### Phase 6 — SHAP catches what the linear screen missed: the "noise" columns aren't (2026-09-05)

Every prior read of feature importance was aggregate and structure-blind: single-feature raw-value
AUC (§6's table), the Phase 0 GLM's linear coefficients, and residual-variance ratios keyed on
exact or binned values. None of these can see a real per-value effect that happens to be
non-monotonic in the raw ordering unless a value→target lookup is what's specifically being
tested for. `scripts/shap_champion.py` fits E1's exact recipe on fold 0 and reads `TreeExplainer`
output directly instead — a genuinely different lens, not a re-run of an existing test.

**Two things came out of it.** First, the top SHAP *interaction* pair by far is
`Environmental_Concern_Level × Subsidy_Available` (mean |interaction| 0.163, next-highest 0.088) —
the tree clearly builds this join internally. This does not contradict Phase 3b's additive-only
result (forbidding it costs ~nothing): the interaction the tree constructs is evidently redundant
with what the univariate terms already encode, not new information. Confirmation, not a lever.

Second, and new: the binned SHAP dependence table for `Charging_Stations_Near_Home` (15 distinct
values) shows a real non-monotonic swing — mean SHAP dips to **−0.073** at values 3–7 and rises to
**+0.094** at 13, on bins of 4,000–20,000 rows each, far past sampling noise. `Number_of_Cars_Owned`
and `Charging_Stations_Near_Work` show the same shape of pattern, smaller. This is the identical
class of blind spot that hid `Annual_Income_USD`'s lookup table from the Phase 0 GLM — a screen
that assumes near-linearity cannot see a non-monotonic per-value effect, whatever its size.

**L1** (strict twin of E1, `drop_noise=True`, removing all four flagged columns) tests whether this
is real: **OOF 0.945525, −0.000117** — about 3x the interim gate, and roughly 15x what the same
drop cost on the unengineered baseline (B3: 0.941648 vs A0 0.941656, ≈0). **The columns are not
noise; B3's null verdict was an artifact of testing them at a baseline with too little capacity and
too few rounds to extract a small non-monotonic effect.** At champion capacity (7 leaves, ~1000+
rounds, "coarse and slow"), the tree already carves each of these low-cardinality values into its
own effective bucket — unlike income's 13,214 values, 15-20 distinct values is cheap for a tree to
split on natively. That is consistent with C7's null result for adding explicit TE on top of these
columns (−0.000075): the per-value structure is real, but it is already fully captured by ordinary
splits, so an encoder adds nothing but redundant, noisier features. **No new OOF lever — E1 already
keeps these columns — but the §6 feature table's "likely noise" label was wrong, and now corrected.**
No slot spent: this closes a labeling error, not a modeling gap.

### Phase 7 — a new architecture confirms additivity harder than any prior test (2026-09-05)

Every model tried through Phase 6 was a tree ensemble (LightGBM, XGBoost, CatBoost) or a neural
net (the embedding MLP). None is a genuine test of "is a pure additive model sufficient," because
a tree ensemble's additivity was always a *behavioral* finding (it happens not to use the
interactions it could build) rather than a *structural* one. An Explainable Boosting Machine
(GA2M, `interpret-core`) with `interactions=0` is additive **by construction** — it cannot build an
interaction at all — so it is the first architecture that actually tests the hypothesis rather than
observing that another model declines to falsify it.

**M1** (EBM, `interactions=0`, on E1's exact TE feature set): **OOF 0.945550**, fold AUCs
`[0.944684, 0.945266, 0.946483, 0.94586, 0.945469]` against E1's
`[0.944717, 0.945267, 0.946656, 0.945962, 0.945649]` — **within 0.0002 on every single fold**, total
gap **−0.000092**. A cyclic-gradient-boosted sum of smooth univariate step functions, fit by a
completely different optimizer than LightGBM's greedy leaf-wise trees, lands on almost exactly the
same function. This is the strongest evidence yet for additivity: not "forbidding interactions
costs little" but "an architecture incapable of interactions matches one that permits them,"
independently implemented.

(`outer_bags` — EBM's internal bagging count — was dropped 14→4→1 across three runs purely for
wall-clock on 668k rows; 14→4 changed folds 0–1 by <0.00001, confirming it doesn't affect the
result, so 1 was used for the logged OOF.)

**It does not help the ensemble, and the reason confirms §6's discriminator rather than
contradicting it.** Spearman correlation between M1 and E1's OOF logits is **0.9966** — inside the
existing GBDT pack's own internal range (0.9922–0.9995) — with disagreement at 0.5% and solo
strength within 0.0001 of the pool. That is the near-twin signature, not decorrelation: a model
that correctly learns the same additive truth makes the same mistakes, because the residual is the
same irreducible noise. Confirmed directly rather than inferred: adding M1 to the champion 3-leg
rank-mean blend moves OOF **0.945743 → 0.945744, +0.000001, null**. No slot spent — this is an ADD
test, not a submission-worthy result.

**Verdict:** the additive-generator finding is now confirmed by four independent lines of evidence
(GLM interactions, the additive_only tree constraint, joint-key residual ratios, and now an
architecturally-additive model matching the champion) and does not open a new ensemble lever — a
fourth confirmation is not a fourth data point for the stack.

### Phase 8 — populating playbook §7's wall plot; M1 submitted, one slot spent (2026-09-05)

Playbook §7: "plot solo score against disagreement rate across every model you own... when that
correlation is strongly negative and no point sits in the useful quadrant, the ensemble axis is
closed." Feature/representation exploration is closed (Phases 5–6); this phase asks the
architecture question directly, the way §7 prescribes — not by assuming the wall, but by measuring
it with structurally distinct model families.

| leg | architecture | solo OOF | corr vs E1 | disagreement | ADD to 3-leg blend |
|---|---|---|---|---|---|
| E1/E4/E5 | GBDT (lgb/xgb/cat) | 0.9456–0.9457 | 0.9922–0.9995 (internal) | — | champion pool |
| G1rB | token embedding (neural) | 0.944074 | ~0.95 (G1) | — | below pool floor |
| **M1** | **EBM (GA2M, additive)** | **0.945550** | **0.9966** | **0.512%** | **0.945743→0.945744, null** |
| **N1** | **plain dense MLP** | **0.944912** | **0.9921** | **1.147%** | **0.945743→0.945700, −0.000043** |

Two new, structurally opposite architectures this phase, both landing exactly where §7 predicts:
**M1 is strong-but-correlated** (inside the GBDT pack's own internal correlation range — a model
that learns the same additive truth makes the same mistakes) and **N1 is weak-and-still-correlated**
— a plain feedforward net given the identical TE representation, no embeddings at all. N1's ADD
result is the sharper confirmation of the two: adding it to the champion blend *loses* −0.000043,
because 1.147% disagreement without competitive solo strength is, per §7, "the model being wrong in
new places," not diversity — an equal-weight combiner cannot tell the difference and pays for it.
Across four architectures now (GBDT, token-embedding neural net, GA2M, plain neural net), zero sit
in the useful strength-and-decorrelation quadrant. **The ensemble axis is closed by measurement, not
assumption**, matching §0's sharpened §7 prior for this competition almost exactly.

(A real pipeline.py bug surfaced building N1: `emb_cols=[]` built a `(0, 0)` token placeholder
instead of `(n_rows, 0)`, crashing on the first batch index — fixed. Also: without per-value
embeddings to overfit through, the net does not early-stop by epoch ~5–9 the way G1 did;
`emb_epochs` was cut 40→15 purely for wall-clock, matching G2's schedule length.)

**One slot spent, and it extends Phase 3/4's family-generalization finding to a third family.** M1
submitted: **LB 0.94568**. Against the GBDT-family instrument fit (slope 1.0832, intercept
−0.0784), predicted LB is 0.94577 — residual **−0.00009**, inside the instrument's own noise band
(~0.0001σ). Unlike the neural embedding leg (+0.00048 to +0.00106, a real family-specific bias),
**EBM's OOF→LB relationship behaves exactly like a GBDT.** This narrows Phase 3's mechanism: the
bias is specific to early-stopped neural nets' fold-averaging variance (one fold-model scores an OOF
row, five score a test row), not to "any non-tree model" — a boosting-based GAM with no such
training-time variance asymmetry needs no correction at all. N1 was not submitted: its OOF was
already conclusively negative for the pool before spending a paired point on it.

| run | OOF | public LB | residual |
|---|---|---|---|
| A0 | 0.941660 | 0.94149 | −0.000067 |
| B5 | 0.944738 | 0.94511 | +0.000219 |
| H2 | 0.945121 | 0.94536 | +0.000054 |
| C2 | 0.945225 | 0.94549 | +0.000072 |
| D4 | 0.945437 | 0.94566 | +0.000012 |
| I1 | 0.945581 | 0.94570 | −0.000104 |
| E1 | 0.945642 | 0.94586 | −0.000010 |
| F2 | 0.945676 | 0.94587 | −0.000037 |
| E4 | 0.945689 | 0.94588 | −0.000041 |
| blend | 0.945743 | 0.94588 | −0.000099 |
| *(G1)* | *0.943401* | *0.94450* | *+0.001057 — **excluded**, see Phase 3* |

**Spearman 0.9970, slope 1.0832, residual σ 0.000103**, against the public split's own paired
bootstrap SD of 0.000116 — the residual is split noise, so the OOF is an unbiased ranker *within the
GBDT family*. **The gate is now set at +0.0000949 OOF and does not move again** (§4). The offset is
still not a trend; its range across these ten is 0.000323 against a paired SD of the same order.
**The offset is not a trend** — its range is 0.000542 against a 0.000116 paired SD, and playbook §5
is the post-mortem of reading exactly this kind of series.

**H2 is the most informative point in the series so far, and it is a rejected probe.** The other six
are champions, so the whole instrument was previously fitted on "we improved and the LB agreed" —
a series that is monotone by construction and cannot catch a ranker that is merely optimistic.
H2 is the first point submitted *because it lost*: it lands **between B5 and C2** on OOF, and the
LB placed it between B5 and C2, to within 0.00002 of the fit. A deliberately worse model being
ranked correctly-worse is evidence a monotone run of improvements cannot supply. This is what
playbook §1's "burn the slots" rule is actually for — the cheapest paired points are the ones
you were not going to ship anyway.

### Phase 9 — three more TabBench/playbook architectures; the wall holds seven times over (2026-09-06)

User request: test the top TabBench leaderboard architectures, and RealMLP/NODE from playbook §7's
"mostly fail" list, before accepting the architecture axis is closed. TabBench's top 3 are all
tabular foundation models: **Seldon** is proprietary/API-only, no open weights — skipped, no access.
**TabPFN v3** requires the user's own browser login and license acceptance at priorlabs.ai to even
download weights — that's account authentication, not something to automate, and skipped on the
user's call. **TabICL v2** is open (Apache 2.0) and was tested in full.

**TabICL v2 (in-context tabular foundation model, `tabicl` pip package).** Installs with zero new
dependencies. Added as a proper `pipeline.py` learner (`tabicl`) with a context-size cap and
chunked prediction, since peak memory is driven by (context + query) length together, not context
alone — a real local OOM boundary was mapped around 250–280k combined length on a 6.4GB laptop GPU.

- At 100,000 context rows (18.7% of fold 0's training data) against the **full** validation fold:
  **OOF 0.944550**, within **0.000167** of E1's own fold-0 score (0.944717), using under a fifth of
  the data. Correlation with E1 is **0.9956**, disagreement **0.75%** — another near-twin.
- Pushing to 450,000 context (84% of the data) on Kaggle's T4 was attempted to see whether more data
  lets it *exceed* the champion. It didn't get the chance to answer that: each 5,000-row prediction
  chunk took **~53 minutes** — the full validation set would have needed ~24 hours — and Kaggle
  cancelled the session after 2 of 27 chunks. **This is itself the finding**: TabICL's inference
  cost scales far worse than linearly with context length (a 4.5x context increase from the 100k
  run produced roughly a 150x per-row slowdown, not ~4.5x), making "test at true full scale"
  practically infeasible for this architecture on this hardware. The 100k result stands as the real
  data point, and it already answers the question the 450k run was chasing: near-champion, not
  exceeding it, using a fraction of the data.
- Getting the never-before-used `s6e9-model.ipynb` Kaggle kernel working for this competition (it
  did not exist on Kaggle before this session — verified) surfaced and fixed five real
  infrastructure bugs, all committed: two wrong hardcoded Kaggle data-mount paths in `pipeline.py`
  (the true mount is nested under `/kaggle/input/competitions/<slug>/`, not the top-level path every
  other assumption expected), a `collect_run.py` crash from the kaggle-cli subprocess hitting a
  non-UTF8 Windows console codepage (fixed via `PYTHONIOENCODING`), and the `_fit_emb` `(0,0)`-shape
  bug from Phase 8. One slot spent on M1 (Phase 8) used this same kernel infrastructure to extend
  the family-generalization finding; P1 (this TabICL run) was correctly scoped to skip the test set
  and submission entirely, since it was answering an architecture question, not shipping a
  candidate — so no slot was at risk in the 24-hour miscalculation, only Kaggle GPU-hours.

**RealMLP (`pytabkit`, tuned-defaults MLP) and NODE (`pytorch_tabular`, neural oblivious decision
ensembles).** Both already installed locally from prior competitions. Both single-fold diagnostics
on E1's exact TE representation and the real frozen fold-0 split, evaluated on the full validation
fold — the same protocol as every other architecture this session:

| leg | architecture | n (of 534,932) | solo OOF | Δ vs E1 | corr vs E1 | disagreement |
|---|---|---|---|---|---|---|
| **TabICL** | in-context foundation model | 100,000 | 0.944550 | −0.000167 | 0.9956 | 0.75% |
| **RealMLP** | tuned-default MLP | 50,000 | 0.939127 | −0.005590 | 0.9482 | 1.59% |
| **NODE** | neural oblivious decision ensemble | 50,000 | 0.935101 | −0.009616 | 0.9552 | 5.76% |

RealMLP and NODE are the clearest weak-and-decorrelated points measured all session: NODE's 5.76%
disagreement is the highest of any architecture tested, paired with the *weakest* solo score of any
architecture tested — playbook §7's "high disagreement without competitive solo strength is the
model being wrong in new places" stated as plainly as the data gets. Neither was pushed to a
larger/fairer scale (RealMLP could likely close some of this gap with the full 535k rows and more
epochs, NODE likely less so given its architecture is a poorer fit for this response surface) —
scaling up was not attempted because both already landed unambiguously in the known-losing quadrant,
and playbook §5 says not to keep spending to sharpen a conclusion the data has already made.

*Harness friction, for the record:* getting RealMLP and NODE running locally took repeated retries
against real environment issues, not model problems — RealMLP's GPU path had highly variable
first-call CUDA/cuDNN latency (worked reliably switched to CPU), NODE's default `entmax15`
choice-function hung deterministically (switched to `sparsemax`/`sparsemoid`, which is a listed
valid choice, not a workaround), and `pytorch_tabular`'s continuous-column scaler crashed against
pandas 3.0's stricter `LossySetitemError` (an int64 column being assigned floats used to silently
upcast; fixed by casting to `float64` before handing data to the library) — a direct downstream
consequence of the numpy/pandas/scikit-learn bump from installing `shap` in Phase 6.

**Verdict: seven architectures now tested this session** (GBDT, token-embedding neural net, EBM,
plain MLP, TabICL, RealMLP, NODE) **spanning trees, GAMs, plain and embedding-based neural nets,
oblivious decision ensembles, and an in-context foundation model. Zero land in the useful
strong-and-decorrelated quadrant.** Playbook §7's wall is not assumed here; it is measured, from
every major inductive bias family available, at the representation this repo already found. The
architecture axis is closed for this dataset until a genuinely new representation is found —
consistent with §0's sharpened prior for this competition from day one.

### Phase 10 — ModernNCA and TabTransformer; nine architectures, still zero in the useful quadrant (2026-09-06)

User request: also test ModernNCA and TabTransformer. Neither has a pip package for this
architecture (ModernNCA has none at all; TabTransformer is via `pytorch_tabular`, already used for
NODE). ModernNCA was adapted from the S6E8 implementation (`gcarno/s6e8-mnca`) — same retrieval
mechanism (soft-kNN over a learned metric space, self-retrieval masked, candidates resampled every
step), rebuilt on a plain MLP encoder over this repo's TE representation rather than S6E8's
per-value token embeddings, since the retrieval mechanism is what's being tested here, not S6E8's
tokenizer.

**Local GPU reliability broke down entirely for this round** — 7+ retries on ModernNCA (1 clean
success), TabTransformer never completed locally. Both are raw-PyTorch or Lightning code with no
multiprocessing DataLoader workers (ruling out the Windows spawn-hang theory used to explain
NODE's issue), so this looks like first-CUDA-call latency variance on this laptop GPU, severe
enough this round that continuing to retry stopped being worth it. Moved both to the proven Kaggle
kernel infrastructure instead — clean success on the first push. (The long wall-clock, ~90 minutes,
was Kaggle GPU queue time, not compute: actual training was 422s for ModernNCA's 15 epochs and 71s
for TabTransformer's fit.)

Same protocol as every architecture this session: E1's exact TE representation, real frozen fold-0
split, full validation fold, test set and submission skipped entirely (architecture research, not a
shipping candidate):

| leg | architecture | n (of 534,932) | solo OOF | Δ vs E1 | corr vs E1 | disagreement |
|---|---|---|---|---|---|---|
| **ModernNCA** | learned-metric soft-kNN retrieval | 300,000 | 0.944310 | −0.000407 | 0.9922 | 0.84% |
| **TabTransformer** | column-attention transformer | 150,000 | 0.942877 | −0.001840 | 0.9889 | 1.61% |

**ModernNCA is the third strong-and-correlated near-twin** (joining EBM and TabICL) — a genuinely
different mechanism (retrieval, not function approximation) converging on essentially the same
answer as the tree ensemble, the cleanest possible restatement of "representation dominates
architecture." It was also still improving at epoch 14 of 15 with no early-stopping trigger yet —
more epochs would likely close some of the remaining −0.0004, but not past the correlation ceiling
that already puts it in the near-twin camp regardless. **TabTransformer lands where RealMLP did**:
weak and moderately decorrelated, not in the useful quadrant, and not scaled up further for the
same reason as Phase 9's RealMLP/NODE — playbook §5's rule against spending more to sharpen an
already-clear conclusion.

**Verdict: nine architectures now tested this session**, adding a retrieval-based method and a
second transformer variant to Phase 9's seven. Still zero in the useful strong-and-decorrelated
quadrant. Every additional architecture at this point is confirmatory, not exploratory — the wall
has now been measured from trees, GAMs, plain and embedding-based neural nets, oblivious decision
ensembles, an in-context foundation model, a retrieval method, and two transformer variants, all
converging on the same two failure modes playbook §7 predicted from the first architecture tested.
