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

**Phase 19 correction to how this predictor is read, not to the gate.** The slope/intercept
above were fit on 10 points spanning OOF 0.9417–0.9457. Every champion since Phase 14 sits
at 0.9460–0.9461 — outside that calibration range — so using this line to predict an LB
score up there is an extrapolation, not an interpolation. Phase 19 refit a second line on a
matched OOF band and found the champion family's residual against it is inside 1.1σ; the
≈−0.00007 residual Phases 16–18 chased was two separate, smaller mechanisms (a known
`fe_recipe_score` cost and a newly-found TE-weakening cost), not a property of this fit being
wrong. **The SHIPPING GATE itself does not change**: it is residual σ ÷ slope from this exact
fit, frozen by design, and stays frozen. See Phase 19 (§8) for the full account.

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

### The reassessment was held on 2026-09-21, on schedule, and the policy is now OPEN

**The price, written down at the moment of the decision as §5 required.** Our ours-only champion
stood at public **0.94638 = rank 448 / 2,648**. Public artifact-pooling was reaching ~0.94656 =
rank ~90. **The ours-only rule was costing roughly 350 places of public rank.** The 0.9465–0.94675
pack was confirmed by direct inspection to be largely *stacks over shared public OOF libraries*
(Den Pugovkin's 15-teacher stack over `najiama/s6e9-oof` and megayak's six-view library), not
better modelling — which is exactly the S6E7 pattern playbook §8 warns evaporates in a shakeup.

**Decision: OPEN, with the hedge carried in the two final slots rather than in the rule.**
Public artifacts may now enter a shipped model. The exposure is bounded structurally instead:

- **Final A = the best ours-only model.** No public artifact in it, at any weight. This is the
  entry that survives if the public-pooled pack evaporates on the private split, as it did in S6E7.
- **Final B = the best public-mixed model.** This is the entry that wins if the pack's gains are
  real.

This replaces playbook §9's usual A/B split (honest champion vs variance-reduced twin) **for this
competition only**, and it is a deliberate substitution, not a drift: the hedge still costs nothing
(S6E8 measured Final A and Final B tying exactly at 0.97030 private), and it now hedges the one
risk that is actually live here.

**The load-bearing consequence: provenance must be auditable off disk.** Every public-mixed file is
written by `scripts/public_blend.py`, which records the source dataset, per-leg weight and each
leg's own fold count in `manifest.json`. A candidate whose public content cannot be read back off
disk is **not selectable as a final** — that is what makes the A/B hedge enforceable on 2026-09-30
rather than a thing we remember approximately.

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

**Corrected in Phase 14: this verdict was wrong.** The "genuinely new representation" this paragraph
called for turned out to exist and be reachable — a multi-scale neighbour-pooled income encoder,
worth +0.000347 OOF once found. The gap was real signal on an axis (encoder granularity) this phase
had not yet tried, not noise at this instrument's resolution. See Phase 14 for the mechanism and the
measurement that overturned it.

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

### Phase 11 — three instrument-stress submissions, no new champion, one genuine surprise (2026-09-06)

Three slots spent, deliberately, on artifacts that already existed with zero incremental compute
(`submission.csv` sitting unused in three preds dirs) — playbook §1's "burn the slots" rule applied
to points that sharpen the OOF→LB instrument rather than to new champion candidates. Fit used: the
10-point GBDT-family regression refit here from the logged pairs, slope **1.0832**, intercept
**−0.0784**, residual σ **0.000092** — matches §4's numbers to within rounding.

| probe | OOF | predicted LB | actual LB | residual |
|---|---|---|---|---|
| **logit_stack** | 0.945719 | 0.94595 | **0.94588** | −0.00007 |
| **L1** | 0.945525 | 0.94574 | **0.94577** | +0.00003 |
| **N1** | 0.944912 | 0.94508 | **0.94497** | **−0.00011** |

**logit_stack confirms "fixed beats fitted" harder than OOF alone could.** The fixed 3-leg rank-mean
blend (OOF 0.945743, champion) and this fitted 6-leg logit stack (OOF 0.945719) differ by +0.000024
offline — under the near-twin paired resolution (0.000027), a wash. On the LB they land on the
**exact same score, 0.94588**, to five decimals. Playbook §6's elbow-at-3 call is now confirmed on
the actual leaderboard, not just inferred from an OOF gap too small to trust.

**L1 confirms Phase 6's SHAP finding on the LB, cleanly.** Dropping the four SHAP-flagged
"not-actually-noise" columns cost −0.000117 OOF against E1; on the LB it cost E1 0.94586 → L1
0.94577, **−0.00009**, and lands at residual +0.00003 — inside the GBDT-family band, exactly as
expected for a LightGBM run that changes only which columns it sees. The deliberately-worse-model
check (H2's trick, applied to a different axis: feature retention, not encoding) holds.

**N1 is the interesting one, and it argues against the naive form of the neural-bias mechanism.**
G1 (embedding MLP, early-stopped epoch 3–6) sits at residual **+0.00105**; G1rB (same net, in-fold
bagged) at **+0.00048** — both *positive*, read as "OOF, scored by one fold-model, understates a
high-variance learner that the 5-way test-time average smooths out." N1 is a plain dense MLP with
no per-value embeddings, trained through most of its epoch budget rather than early-stopping at
epoch 3–9 the way G1 did (Phase 8's note). If the positive bias is caused by *premature stopping*
specifically, a more fully-trained net should show little or none of it. N1 shows **−0.00011** —
not just smaller, but the **opposite sign**, ~1.2σ of the GBDT band and in a direction the "fold-
averaging rescues an undertrained model" story does not predict. This is a real data point against
reading G1/G1rB's bias as "neural nets get an LB bonus" in general: it looks specific to nets that
are *still changing fast* when each fold stops, not to gradient-descent training as such. Consistent
with, not yet conclusive of, the mechanism — one point, and N1's own solo OOF (0.944912) is far
enough below the pool floor that this is read as an instrument point, not grounds for revisiting
the closed ensemble axis.

No champion moved. `experiments/runs.csv` rows `718b2ed3`, `04d11559`, `83b7b495` now carry their
`public_lb_score`.

### Phase 12 — five leads from the public frontier, tested offline, zero clear the gate (2026-09-09)

The public leaderboard was re-read (top now 0.94672, a dense cluster at 0.9464–0.9465, up from
Phase 8's 0.94647) and four notebooks pulled in full: `cdeotte/fable-5-1-eda-original-data-insights`,
`georgymamarin/s6e9-starter-how-to-tell-a-real-gain-from-noise`,
`dariushafshar/s6e9-0-exact-twins-31-discrete-zero-shift`, and its companion
`s6e9-0-94585-1-of-4-ingredients-carries-it`. Read for ideas only, per the artifact-sharing policy
(§5) — no code or blend adopted. One genuine structural finding came out of it: Chris Deotte
recovered the ORIGIN dataset's exact NumPy seed (101) and reproduced all 10,000 rows bit-for-bit,
reading off the literal generator rule (`buy_score = 1.2*(income/1e5) + 0.6*concern + 2*subsidy -
1*(anxiety=Medium) - 3*(anxiety=High)`, threshold 5.5) — and Georgy Mamarin's companion piece
establishes that ORIGIN itself is synthetic (a "custom Python script", per its own dataset
description) and that Kaggle's own generator is a *second, different machine* fitted to those 10,000
rows, which is the most plausible mechanism yet for why the competition's income column carries
per-value lookup structure the literal linear recipe does not predict.

Five strict twins of E1 (0.945642), all local, all offline, zero submission slots spent:

| probe | change | OOF | Δ vs E1 | best_iter | verdict |
|---|---|---|---|---|---|
| P1 | TE the 6 native categoricals too (not just the 3 numerics) | 0.945634 | −0.000008 | 966–1347 | null |
| P2 | `max_bin` 255 → 1023 | 0.945644 | +0.000002 | 772–1619 | dead null |
| P3 | + Deotte's exact `buy_score`/`worry_score` as two engineered columns | 0.945498 | **−0.000144** | 369–444 | negative |
| P4 | unsupervised frequency encoding, all 13 raw columns | 0.945694 | +0.000052 | 953–1158 | positive, sub-gate |
| P5 | 10,000 ORIGIN rows appended to every fold's TRAIN partition only | 0.945413 | **−0.000229** | 560–1474 | negative |

**P1/P2 close the two concrete external levers this round offered, definitively, against our own
recipe rather than a weaker one.** dariushafshar reports TE-everything as their single largest gain
(+0.0011) and georgymamarin reports the bin cap as theirs — both on baselines without per-value TE.
On E1 both are null: P1 matches Mamarin's own control on his own baseline ("declaring a low-
cardinality column categorical... bought nothing, both encodings land in the same place") — LightGBM's
native categorical split already sorts levels by gradient/hessian, and with thousands of rows per
level on 2–4-level columns there is no sparse-value shrinkage problem for TE to solve, unlike income's
13,214 values at ~50 rows/value. P2 directly confirms Phase 1c's "the lookup is fully extracted"
finding by measurement rather than inference: once TE already hands the tree exact per-value
resolution on a separate column, raising the raw column's own bin cap has nothing left to add.

**P3 is F1/H2/H3's displacement signature a fourth time, and the sharpest version of it yet.**
`best_iter` collapses to 369–444 from E1's 777–1098 — the tree leans on the two composite columns and
early-stops before building the fine per-feature shapes E1 spends ~1000 rounds on, exactly the
mechanism that broke F1 (native categorical income, −0.000999), H2 (5 paired keys, −0.000521) and H3
(1 paired key, −0.000405). The difference this time is the input: not a guessed proxy but the
generator's own recovered coefficients, fit on zero competition data. That the true formula still
displaces resolution is the strongest evidence yet that E1's per-feature shapes already resolve finer
than any fixed-coefficient linear combination of the same columns can — Georgy Mamarin independently
ran the identical test on his own baseline and got the same null ("worth nothing"), a second dataset,
second implementation, same result.

**P4 is a genuine maybe, not a clean null, and is the one candidate worth a second seed rather than a
slot.** +0.000052 is 1.4x the 0.000038 seed-noise floor but well under half the +0.0000949 gate —
too small to trust on one draw, too large to file as pure noise without checking. `best_iter` shows no
collapse (953–1158, in E1's own range), so if it is real the mechanism is additive rather than
displacing: unlike P1's high-count categoricals, `freq_cols` also covers `Charging_Stations_Near_Home`
/`_Work` and `Number_of_Cars_Owned`, whose real non-monotonic structure (Phase 6) is cheap for a tree
to split on natively but whose *rarity* a frequency column exposes directly. Re-run at `model_seed`
1337 before deciding either way; do not ship on this one draw regardless of sign.

One slot spent to check it anyway (playbook S1's burn-the-slots rule: this was a cheap paired point,
not a champion bet): **P4 → LB 0.94588**, against a GBDT-family-fit prediction of 0.94598 (slope
1.0832, intercept −0.0784) — residual **−0.00010, 1.1σ of the instrument's own 0.000092**, unremarkable
and inside the GBDT band. Against E1's own 0.94586 that is **+0.00002**, indistinguishable at the
near-twin paired resolution (0.000027). The LB agrees with the second-seed-before-deciding read: on
this one point there is nothing here yet, consistent with the OOF delta being sub-gate rather than a
false negative the LB would have caught.

**P5 closes the "should we augment with ORIGIN" question, and closes it hard.** −0.000229 is ~6x the
seed-noise floor, the largest-magnitude result of the round. Mechanism, per Mamarin's §16: ORIGIN and
the competition data are drawn from two *different* generators (the literal script vs. Kaggle's own
fitted model), so 10,000 rows from the wrong conditional distribution P(y|x) measurably pull the
fit away from the population E1 is actually scored against — a genuine distribution-mismatch cost,
not overfitting noise. This was flagged as explicitly untested by the community ("someone should put
a number on it") and now has one: naive unweighted concatenation into the training fold is a net
negative for this competition. Downweighting ORIGIN rows or excluding them entirely from the TE fit
(raw features only) might change the sign, but that is a new probe, not a re-read of this one.

**Verdict: no champion moved, and every axis this round tested closes.** Combined with Phases 0–11,
the encoder, capacity, interaction, ensembling, in-fold variance, shrinkage, architecture (nine
families), and now categorical encoding, bin resolution, the literal generator formula, blanket
frequency encoding, and ORIGIN augmentation are all measured closed. The ≈0.0002 gap to the ~0.9465
frontier stands as it did after Phase 5: consistent with noise at this instrument's resolution, not an
unfound lever. `experiments/runs.csv` rows `fb2aefef`, `18cf4e13`, `c6addbeb`, `13692fb1`, `3e9e7b34`.

**Corrected in Phase 14: this verdict was also wrong, on the same axis Phase 5's was.** "Encoder" here
meant raw-value TE, binned TE, categorical TE and bin resolution — it never tried a neighbour-pooled,
multi-scale encoding, which is precisely what this round's own read of the public frontier's two
concrete ideas (P1/P2) screened out without noticing the recipe's THIRD, untested ingredient (its
income shape encoder) sitting one layer beneath the ones that were checked. See Phase 14.

### Phase 13 — three meta-models over existing artifacts, zero incremental training (2026-09-09)

Playbook §1's "burn the slots" rule applied to the ensemble axis rather than to a new leg: every
architecture family accumulated since Phase 8–12 (`O1` CatBoost 0.945150, `M1`/`O2` EBM
0.945550/0.945015, `N1` plain-MLP 0.944912, `P4` freq-encoded LightGBM 0.945694) already has complete
OOF+test artifacts, so combining them costs a `stack_logit.py` call, not a training run. `leg_probe.py`
first checked each individually against the curated 6-leg pool (`E1/E4/E4d/F2/E2/D2`, floor 0.945402) —
**all five miss gate 2 solo**, contribution +0.000001 to +0.000039 against the +0.0002 bar, `P4` best
of the five. Three combinations followed:

| probe | pool | mode | OOF | Δ vs champion | LB | mechanism |
|---|---|---|---|---|---|---|
| **A** | all 11 legs (curated 6 + 5 new) | fitted logit stack | 0.945774 | +0.000031 | **0.94590** | highest OOF on record; fitted combiner down-weights the weaker new legs instead of diluting into them |
| **D** | 6 legs: curated pool with `D2`+`E4d` (its two weakest) swapped for `P4`+`M1` | fitted logit stack | 0.945768 | +0.000025 | **0.94590** | the literal ADD/SWAP-at-constant-pool-size test this file's conventions call for; best single swap found |
| **B** | same 11 legs as A | fixed rank-mean blend | 0.945730 | −0.000013 | 0.94584 | fixed combiner, no down-weighting available |

**None clears the +0.0002 gate — this is not a champion change.** A and D both miss on OOF (+0.000031,
+0.000025) but both land LB 0.94590, +0.00002 over the prior best 0.94588 — a delta far under the
GBDT-family residual σ (0.000092, README §4), i.e. noise-indistinguishable from the shipped champion.
`pool.json` and the champion designation are unchanged.

**B reverses Phase 11's "fixed beats fitted" finding, and the reversal replicates on the LB.** At 6
near-twin legs (corr 0.9922–0.9995) a fitted combiner had nothing to subtract and the fixed blend won
(champion 0.945743 vs `logit_stack` 0.945719). Add five weaker, architecturally different legs (cat,
2×EBM, plain-MLP, all soloing under the 6-leg pool floor) and the ordering flips: fixed blend B
(0.945730) loses −0.000044 OOF to fitted stack A (0.945774), because averaging dilutes toward the weak
members while a fitted combiner shrinks their weight instead (A's weight table: the three weakest legs
get 0.14–0.40 vs the strongest LightGBM's 1.27). The gap survives on the leaderboard too — B is 0.94584
against A/D's 0.94590, same ordering, −0.00006. Confirms `stack_logit.py`'s own note in the least
ambiguous way available: which combiner wins is a property of the **pool's composition**, not a fixed
rule — re-test at every pool-size change rather than reusing Phase 11's verdict.

**A separately-suggestive, unconfirmed point: mixing in non-tree architectures may cost a small,
consistent LB offset.** All three probes here land 0.00016–0.00018 *below* the GBDT-family regression's
predicted LB (slope 1.0832, intercept −0.0784) — roughly 1.8σ of its own 0.000092 residual, and in the
same direction for all three combiners regardless of fitted-vs-fixed. The existing 6-leg `logit_stack`
(pure LightGBM/XGBoost) sat inside the band at −0.00007. One consistent-direction observation across
three paired points is a lead, not a finding — filed here rather than acted on; would need dedicated
GBDT-only vs. mixed-architecture paired stacks to separate "mixing architectures" from "this specific
pool" as the cause.

No slots wasted on redundant confirmation: a fourth combination (E1 + all four non-LightGBM/XGBoost
legs, OOF 0.945701, run `d7beea36`) scored below every alternative above and was logged but not
submitted — it only re-confirms `leg_diversity.py`'s standing conclusion that decorrelation pays
nothing without solo strength to decorrelate. `experiments/runs.csv` rows `631bbe8f`, `a536d5e7`,
`2aa6afb9`, `d7beea36`.

### Phase 14 — the frontier recipe was a representation gap, not noise; new champion (2026-09-10)

**Phase 5 and Phase 12's "the ≈0.0002 gap to the frontier is consistent with noise, not an unfound
lever" is corrected here, the way Phase 1 corrected Phase 0 and Phase 3b corrected B7/D4.** It was
wrong, and direct measurement is what found the error: the leaderboard had grown from 533 teams
(Phase 8) to **1451**, the public top from 0.94647 to 0.94672, and the exact-score cluster the
competition was already known to have (Phase 8's "13 teams at 0.94607") turned out to trace to an
identifiable **model recipe**, not a blend artifact — `jazivxt/single-model-zoom-zoom`, forked as
`najiama/pure-lgbm-model-cv-0-94606-lb-0-94637` (55 votes) and re-forked across the whole
0.9463–0.94644 band. Its published OOF file was downloaded as a **diagnostic** (permitted by §5:
"may be read for ideas and attached as diagnostics") and scored against our own labels on our own
frozen split: **OOF 0.946064**, +0.000422 over E1 (4.4× the gate), at logit-correlation **0.993**
to our own legs (a near-twin, not a decorrelated model) — meaning the gap was a better
**representation**, not headroom the ensemble axis could reach. No artifact from their submission
or code entered anything shipped; every ingredient below was reimplemented from scratch under this
repo's own leakage protocol, per the ours-only policy (§5).

**The recipe's five ingredients, read from source, and the mechanism identified as most likely
before any run:** digit decomposition (`floor(v/10^p) % 10`), a multi-scale neighbour-pooled income
encoder (equal-width bins at two widths, each emitting position/central/gaussian-smoothed/
left/right/slope/curvature/log-count channels), dual TE smoothing (`smooth=10` and an
empirical-Bayes `"auto"` side by side), blanket frequency encoding, and their own hyperparameters
(`lr 0.02, num_leaves 31, max_depth 5, colsample 0.303, reg_alpha 0.07, reg_lambda 2.03,
max_bin 1024`). The prime suspect: our own per-value TE keys `Annual_Income_USD` at ~50 rows/value,
so the rate estimate's own SE (~0.053) sits close to the real per-value SD (0.0748, §6) —
signal-to-noise near 1.4:1. Pooling adjacent bins cuts that SE while explicit slope/curvature
channels hand the tree the response's local derivative — a materially finer regime than H1's own
quantile-bin test (100 bins, smoothing 500, ~132 income values/bin).

**A read-only check first, to keep from double-counting a source of signal already priced in.**
Income's ones-digit alone looked dramatic (rate SD 0.021 at 13.8× its own SE), but restricted to
unclipped rows (excluding the 9.21%-of-rows floor at 30000, rate 0.0443 vs 0.1879 elsewhere — which
E1's per-value TE already captures exactly) the ratio falls to 4.9×: real, but modest. This
correctly predicted the digit block's small role below.

`src/pipeline.py` gained five new config axes, every one defaulting off/empty so every archived run
stays bit-identical: `fe_digit_cols`/`fe_digit_powers`/`freq_digit_cols` (unsupervised digit
features), `te_shape_cols`/`te_shape_bins`/`te_shape_smooth` (`apply_shape_target_encoding`, the
neighbour-pooled encoder — same fold-only-fit, inner-K-fold-protected protocol as
`apply_target_encoding`), and `te_multi_smooth` (one TE column per smoothing value, including an
`"auto"` empirical-Bayes rule in the spirit of, not a bit-exact port of, sklearn's). Scope decisions
made during implementation, logged for auditability: `fe_digit_cols` covers the 7 native numeric
columns only, not the 6 categoricals their notebook first integer-codes (digit-decomposing an
already-compact 0–3 code is near-degenerate); `te_multi_smooth` applies only to E1's existing 3
`te_cols`, not to all 61 raw-plus-digit keys their notebook TEs (a direct test of "does a second
smoothing view help," not a full 320-feature port). R0 therefore does not fully close the gap to
their 0.946064 diagnostic ceiling — expected, and consistent with the scoping.

**R0 (the full reimplementation, strict twin of nothing — a new representation) reproduces the
mechanism cleanly:**

| run | change vs E1 | OOF | Δ vs E1 | features |
|---|---|---|---|---|
| E1 | *(old champion)* | 0.945642 | — | 16 |
| **S1** | + shape encoder ONLY, E1's own params | **0.945787** | **+0.000145** | 32 |
| **R2** | + everything except the shape encoder | **0.945877** | **+0.000235** | 138 |
| **R1** | + everything except the digit block | **0.945939** | **+0.000297** | 50 |
| **R0** | + all five ingredients, their hyperparameters | **0.945989** | **+0.000347** | 154 |

`compare_oof.py` against E1 confirms it is not split luck: paired bootstrap 95% CI
**[+0.000286, +0.000409]**, paired SE 0.000031 — **11× the SE**, identical fold assignment. The
ablation ladder attributes the gain cleanly: the digit block alone is worth +0.00005 (R0 − R1,
matching the read-only prediction above), the shape encoder alone is worth +0.000112 within the
bundle (R0 − R2) and +0.000145 solo (S1 − E1) — **the single largest contributor** — and the
remaining ~0.00018 (freq encoding + dual TE smoothing + their hyperparameters, jointly) is real but
not attributable to one ingredient without a further probe. `leg_probe.py` against the standing
6-leg pool: R0 **clears gate 2** on its own, ADD **+0.000308** to the pool (6-leg 0.945719 → 7-leg
0.946027) — the first leg this competition to clear that gate solo since the pool was curated.

**All five of today's slots spent, and every paired point landed as predicted or better:**

| submission | OOF | predicted LB | actual LB | residual |
|---|---|---|---|---|
| R0 | 0.945989 | 0.94630 | **0.94614** | −0.00016 |
| R1 | 0.945939 | 0.94624 | **0.94614** | −0.00010 |
| **7-leg stack (R0 + old pool)** | **0.946027** | 0.94634 | **0.94616** | −0.00018 |
| R2 *(deliberately worse — H2's trick)* | 0.945877 | 0.94617 | **0.94600** | −0.00017 |
| S1 *(shape encoder only)* | 0.945787 | 0.94608 | **0.94608** | **+0.00000** |

R2 landed correctly below R0/R1 on the LB, the same deliberately-worse-model confirmation H2 supplied
in Phase 8 and L1 in Phase 11 — the ranker is not merely monotone-by-construction here. **Champion
moved from the 6-leg stack (OOF 0.945719 → LB 0.94588, Phase 13) to the 7-leg stack (OOF 0.946027 →
LB 0.94616), +0.00028 LB, +115 ranks on a same-day leaderboard snapshot (rank ~420 → ~305 of 1451).**
`pool.json` now carries R0 as a 7th leg and `champion_stack_oof=0.946027`.

**The new finding: a small, consistent, feature-bulk-proportional negative LB offset, not a
family-specific one.** All four Phase-14-recipe submissions land **0.00010–0.00018 below** what the
GBDT-family fit (slope 1.0832, intercept −0.0784, §4) predicts — 1–2× the instrument's own residual
σ (0.000092), same sign, same rough size, across a solo model, an ablation, and a stack. **S1 is the
control that localizes it**: same architecture, same family, but only the ONE ingredient (the shape
encoder) that matters most — and its residual is **+0.00000**, dead on the fit. The offset therefore
tracks the *amount of new feature engineering* (digit block + freq block + dual-smoothing TE +
their hyperparameters, stacked together), not the architecture and not the shape encoder itself.
This is a different mechanism from Phase 3/8's neural-fold-averaging bias — it is measured here
entirely within the GBDT family — and it is a lead, not yet a probed finding: the next move, if
pursued, is isolating which of freq/dual-TE/hyperparameters (or their combination) carries it, the
same way S1 isolated the shape encoder from the rest.

**What stays closed, what reopens.** The architecture axis (Phase 9–10, nine families) and the
ensembling axis (Phase 8/13, zero legs in the useful quadrant except now R0 itself) are untouched by
this phase — R0 is a GBDT on an enriched representation, not a new architecture. What reopens is the
**encoder axis** specifically: Phase 1c/6/12 closed it against raw-value TE, binned TE, categorical
TE and bin-resolution increases, but never tested neighbour-pooled multi-scale encoding, which is
now the single largest lever found since Phase 1's original per-value TE discovery. The natural next
probe — extending `te_shape_cols` to `Daily_Commute_km` and `Age`, the other two `te_cols` members —
is untested and cheap; it was not run today because the day's slots were better spent locking in
R0's confirmed gain than making a novel offline hypothesis without a submission to check it against.
`experiments/runs.csv` rows `bdc92691`, `90e00e54`, `e87bed07`, `ea2531a7`, `0f742d1d`.

### Phase 15 — five axes closed in one day; the feature-bulk offset sharpened, not the gap (2026-09-11)

Leaderboard re-read: 1,542 teams (up from 1,451), top **0.94672**, rank 50 at **0.94644**. We stood
at rank 342, public **0.94616**. Phase 14 left one named-and-cheap probe (extend `te_shape_cols` to
`Daily_Commute_km`/`Age`) and reopened the encoder axis generally; this phase tested that probe plus
four more axes the day's compute made newly affordable — Kaggle CPU kernels run in parallel (capped
at **5 concurrent sessions**, not the 2-GPU cap this repo had previously measured) rather than the
32-second local loop, since a full 5-fold run on R0's 154-feature representation now costs
~5-7 minutes locally, not 32 seconds.

**X1 — shape-encoder granularity, closed.** A local measurement (per-bin excess-signal/expected-SE
ratio, trend removed) predicted the ratio peaks at 1024-2048 bins and is worst at R0's own
8192/16384 — independently corroborated by the public frontier's much coarser `//100`/`//1000`
"smooth keys". The prediction was wrong in its practical conclusion:

| bins | OOF | Δ vs R0 (0.945989) |
|---|---|---|
| `[1024,4096]` | 0.945974 | −0.000015 |
| `[2048,8192]` | 0.946009 | **+0.000020** (best, still under the seed floor) |
| `[512,2048,8192]` | 0.946008 | +0.000019 |
| `[1024,4096,16384]` | 0.945989 | 0.000000 |
| `[256,1024]` | 0.945917 | **−0.000072** |

Every grid from 1024 to 16384 bins lands within the 0.000038 seed-noise floor of R0 — the axis is
flat, not peaked, across that whole range. Only pushing to genuinely coarse (256/1024) costs real
signal. **The local per-bin SNR calculation was correct about local rate-estimate precision and
still didn't predict the model's behaviour**: R0's neighborhood backoff on the raw per-value
`te_cols` already supplies a smoothed local estimate, so the shape encoder's job is adding the
slope/curvature *derivative* information on top of that, and this is apparently insensitive to
exactly how many bins carry it, provided there are enough to keep the derivative meaningful. Mirrors
H1's "we already compute this" finding (Phase 2c) one level down.

**X2 — five ingredient twins of R0, all closed.**

| probe | change | OOF | Δ vs R0 |
|---|---|---|---|
| Ia | `fe_recipe_score: false` | 0.945995 | +0.000006 |
| Ib | `te_multi_smooth: [10,"auto",100]` | 0.946008 | +0.000019 |
| Ic | `te_inner_folds: 5→10` | 0.945969 | −0.000020 |
| Id | `te_shape_cols` + `Daily_Commute_km` | 0.945964 | −0.000025 |
| Ie | `te_shape_cols` + commute + `Age` | 0.945956 | −0.000033 |

**Ia is the interesting null.** P3 (Phase 12) measured `fe_recipe_score` (the ORIGIN generator's
fixed-coefficient `buy_score`/`worry_score`) at **−0.000144** on E1's 16-feature representation —
displacement, the F1/H2/H3/O1 signature. On R0's 154-feature representation the same feature is a
dead null (+0.000006). Read together with R0's `best_iter` (777-1098, nowhere near F1/H2/H3's
collapse to 300-450): the richer representation's per-feature shapes are no longer being fully
resolved before the tree runs out of budget the way E1's leaner one was, so a coarse composite
column that used to compete for early rounds no longer measurably displaces anything. **Id/Ie land
exactly as pre-registered** — commute (830 rows/value) and age (14,859 rows/value) have almost no
per-value SE left to cut by pooling, so extending the shape encoder to them is closer to redundant
columns than to new signal, and the small negative deltas are consistent with that reading, not with
a real cost.

**X3 — Optuna HPO on R0's representation, closed, and instructively so.** No hyperparameter search
existed in this repo; `scripts/optuna_probe.py` is new, single-fold (fold 0), reuses
`pipeline.py`'s own feature/encoder/`fit_predict` functions, and — for wall-clock only, after TE
fitting — subsamples the fold-0 **training** rows to 40%. Four independent Optuna studies (distinct
sampler seeds, 80 trials each, 320 trials total) ran as four parallel Kaggle CPU kernels and
converged tightly on one region, sharply different from R0's borrowed (public-notebook) params:

| | R0 (borrowed) | HPO consensus (4/4 shards) |
|---|---|---|
| `max_depth` | 5 | **4**, all four shards |
| `learning_rate` | 0.02 | **0.011-0.014**, all four shards |
| `min_child_samples` | 10 | **35-96**, all four shards |
| `colsample_bytree` | 0.303 | 0.22-0.32 |

The three best-looking trials (highest single-fold AUC on the 40%-subsample screen), confirmed on
the **full** 535k-row, 5-fold data:

| probe | OOF | Δ vs R0 |
|---|---|---|
| best of shard 2 (AUC 0.944848 on screen) | 0.945996 | +0.000007 |
| best of shard 3 (AUC 0.944827 on screen) | 0.945984 | −0.000005 |
| consensus (median across all 4 shards) | 0.945989 | **0.000000** |

**All three collapse to noise on full data.** The mechanism is a clean, well-understood screening
artifact, not a bug: HPO on a 40%-subsampled single fold rationally favours *more* regularization
(shallower trees, slower learning, higher leaf-occupancy floor) because less data has more variance
to guard against — and that extra regularization buys nothing once the model sees the full 535k
rows. **R0's borrowed hyperparameters, despite being tuned for a different feature count and fold
count, are already at or near the full-data optimum for this representation.** A harness bug was
found and fixed along the way (below); this HPO result is not an artifact of that fix, since the
full-data confirmations ran after the fix on the corrected code path.

**X4 — does 10-fold explain the frontier's residual gap? Refuted, cleanly, with a slot.** The
public frontier notebook's own published OOF scores 0.946064 on our split — only +0.000075 over R0
— yet its LB (0.94637) beats R0's (0.94614) by +0.00023, a gap the OOF comparison alone can't
explain. Every frontier notebook read this competition trains on 10 folds, not 5; a test row is then
averaged over 10 fold-models instead of 5, the same mechanism Phase 3/4 measured as a real,
quantified bias for early-stopped neural nets (G1: +0.00105 raw, +0.00048 after in-fold bagging).
Pre-registered: **ΔLB ≥ +0.00015 supports fold-averaging and reopens fold count; below +0.00008
closes it.** R0's exact recipe at `n_folds=10` (OOF 0.946122, *not comparable* to any 5-fold run,
excluded from `pool.json` and the OOF→LB regression by design) scored **LB 0.94617** — **+0.00001**
over R0's 5-fold 0.94614. **Refuted, not merely null: the frontier's 0.00023 residual gap is not a
fold-count artifact.** The mechanism that explains neural nets' fold-averaging bias does not
transfer to a 1000+-round GBDT the way Phase 3 itself already predicted (test predictions are
"already 5-way fold-averaged and had little room to gain" — Phase 4's own words, now confirmed for
10-vs-5 as well as bagged-vs-not). What remains unexplained about the frontier's extra 0.00023 is
still open; X1-X3 above rule out the shape-encoder granularity, four cheap ingredient corners, and
LightGBM hyperparameters as the source, so it most plausibly sits in ingredients this repo
deliberately scoped narrower in Phase 14 (dual vs. triple-smoothing TE covering only 3 keys vs.
their ~61, or the "smooth keys" string-categorical representation, which is structurally different
from a numeric per-value key) — a new probe, not a re-read of today's.

**X5 — cross-family legs on the new representation, closed by measurement.** The pool was stranded:
6 of 7 legs on the old 16-feature E1 representation, only R0 itself on the new one. Trained the
first two cross-family legs on R0's exact feature config:

| leg | solo OOF | max corr vs pool | ADD to 7-leg stack |
|---|---|---|---|
| `xgb` (E4r0) | 0.945941 | 0.9986 vs R0 | 0.946027→0.946035, **+0.000009**, miss |
| `cat` | 0.945886 | 0.9915 vs F2 | 0.946027→0.946051, **+0.000025**, miss |
| both together (9-leg stack) | — | — | 0.946027→**0.946055**, **+0.000028**, miss |

Both miss gate 2 solo, and combined they still miss it — the highest OOF on record this competition
(0.946055) still isn't a 0.0002 ADD. **Mechanism: near-twin, again.** `xgb`/`cat` on the *identical*
feature representation as `R0` correlate 0.99+ with it, playbook §7's "same truth, same mistakes"
signature repeating on the new representation exactly as it did on the old one (E1/E4/E5 in Phase 3)
— changing the learner without changing the representation is not a diversity lever here either.

**A harness bug, found and fixed, not a probe.** X3's first HPO push failed on trial 0:
`LGBMClassifier.fit() got an unexpected keyword argument 'eval_X'`. `pipeline.py`'s `lgb` branch used
the `eval_X=`/`eval_y=` keyword form, present only in LightGBM 4.7+ (this repo's local pinned
version) and absent from Kaggle's older pinned LightGBM. **No `lgb`-learner run had ever previously
been pushed to a Kaggle kernel** — every `lgb` champion since Phase 0 was screened locally — so this
portability gap was latent for the whole competition until today's HPO kernels were the first `lgb`
code path ever executed on Kaggle. Fixed to the universal `eval_set=[(Xva, yva)]` form, verified
bit-identical to `eval_X=`/`eval_y=` on a synthetic fit (same `best_iteration_`, max abs prediction
diff 0.0) before trusting any result built on it — every archived run's result is unaffected.

**Five slots spent, all on instrument points, since nothing cleared the shipping gate.** Per
playbook §1's rule, unused slots are wasted paired points, not saved ones:

| submission | OOF | predicted LB (GBDT fit) | actual LB | residual |
|---|---|---|---|---|
| X4 (R0 @ 10-fold) | *(n/a, 10-fold)* | — | 0.94617 | — |
| 9-leg stack | 0.946055 | 0.94637 | 0.94614 | **−0.00023** |
| G2 (`[2048,8192]`) | 0.946009 | 0.94632 | 0.94612 | −0.00020 |
| `xgb` solo (E4r0) | 0.945941 | 0.94624 | 0.94610 | −0.00014 |
| G5 (`[256,1024]`, *deliberately worse*) | 0.945917 | 0.94622 | 0.94610 | −0.00012 |

*(for reference: R0 solo residual −0.00016, 7-leg champion stack residual −0.00018, both from Phase 14)*

**Two real findings inside what looks like a flat day of nulls.** First, **G5 is the fourth
deliberately-worse-model check to pass** (after H2 in Phase 8, L1 in Phase 11, R2 in Phase 14): its
LB (0.94610) is correctly the lowest of the five, confirming the ranker still discriminates
correctly at this richer, 154-feature representation. Second, and new: **the feature-bulk-
proportional negative offset Phase 14 found (S1 at +0.00000, the four richer-recipe points at
−0.00010 to −0.00018) now holds across a different learner** (`xgb` solo: −0.00014, essentially
R0's own −0.00016) **and gets WORSE when legs are stacked, monotonically with stack size** (solo
~−0.00012 to −0.00016, 7-leg −0.00018, 9-leg −0.00023). This is exactly why the 9-leg stack's higher
OOF (0.946055 vs the 7-leg champion's 0.946027) produced a *lower* LB (0.94614 vs 0.94616) — a real
inversion, and precisely the case the ADD gate exists to catch: **the gate correctly refused to
promote the 9-leg stack**, and the LB confirms that refusal was right, not merely conservative.

**Verdict: every axis this round tested closes, and today extends Phase 14's two most important
findings rather than reopening either.** Encoder granularity, four ingredient corners, LightGBM
hyperparameters, and cross-family legs on the new representation are now measured closed on top of
Phase 0-13's list. The champion is unchanged: the 7-leg stack, OOF 0.946027 → LB 0.94616. The
frontier's remaining ~0.00023 residual gap is narrowed (fold count is ruled out) but not closed;
the next candidates are the triple- vs. dual-smoothing TE coverage gap and the string-categorical
"smooth keys" representation, both flagged, neither run. `experiments/runs.csv` rows include
`77c52611`, `47838d89`, `2a552362`, `023f3e27`, `11872e0c` (X2), `f602a9f5`, `f5b957d4`, `9d3ff376`,
`4fb4705f`, `a1e0ca8e` (X1), `f8881980`, `2463e900`, `a16f0116`, `d3622bdc` (X3 shards),
`6fcd493b`, `d68bf43c`, `6ebe2430` (X3 confirm), `e1495238` (X4), `7c151fb7`, `d300b5ff`,
`826f0978` (X5).

### Phase 16 — the window encoder replaces the shape encoder; new champion, +0.00007 LB (2026-09-12)

Leaderboard re-read: **1,632 teams**, top 0.94672, rank 100 at 0.94644, rank 200 at 0.94637.
We stood at rank 369 on 0.94617. The 49-team spike at 0.94644 is **not a model** — it traces to
`talhatursun/s6e9-daily-rank-average-ensemble`, a rank-average of public submission files, which
§5 puts out of bounds until the 2026-09-21 reassessment and which is not a modelling lever anyway.

**The day's lead came from `megayak/s6e9-0-94645-four-feature-views-beat-the-blend`** (read for
ideas only, per §5), which reports on its own 10-fold split that a different *learner* on identical
features adds nothing (corr 0.9995 — our own X5 result, independently reproduced) while two
different *feature views* do add at corr 0.9979–0.9989. Its two untested ingredients: **centred-
window target rates** (the smoothed buy rate over income `v±2…±200`) and a **quantisation ladder**
(`//10, //50, //500, //5000`) used instead of the exact-value key.

**Step 0 killed the ensemble half of that before a line of model code was written.** Their published
four-member OOF was scored against our labels as a *diagnostic* (the Phase 14 move, §5-permitted):

| added to our 7-leg stack (0.946024 refit) | ADD |
|---|---|
| A (their best) | +0.000070 |
| C (the windows view) | +0.000064 |
| D (the ladder view) | +0.000064 |
| **all four together** | **+0.000065** |
| C, *given A is already in* | **+0.000000** |
| D, given A already in | −0.000002 |

All four correlate with `R0` at Spearman 0.9952–0.9953 — indistinguishably, and no further from R0
than R0 is from our own E1 (0.99569). **Their four views carry exactly one model's worth of
information relative to our pool.** The "+0.00003 per view" is their own pool's redundancy
structure, not transferable signal. The view-ensemble build was abandoned on this measurement; the
*ingredients* were then tested as single-model levers, which is where the value turned out to be.

**Two new config axes**, both default-off so every archived run stays bit-identical (R0's cfg
re-run after the edits reproduces OOF 0.945989 and all five fold AUCs exactly):
`apply_window_target_encoding` / `te_window_cols` / `te_window_radii`, and `fe_quant_cols` /
`fe_quant_divisors`. Leak control passed before any probe was trusted: a random 13,214-value key
scores **0.4959–0.5019** through the same inner-fold path, and real income's train-row AUC sits
*below* its val-row AUC (0.709 vs 0.714) — the correct sign, since the inner-fold encoding is
noisier than the full-fold one.

| probe | change | feats | OOF | Δ vs R0 | LB | residual |
|---|---|---|---|---|---|---|
| **WQ** | + windows + ladder | 172 | **0.946086** | **+0.000097** | **0.94624** | −0.00016 |
| Q1 | + ladder (added to the exact key) | 166 | 0.946067 | +0.000078 | — | — |
| **WS** | windows **replacing** the shape encoder | 144 | 0.946051 | +0.000062 | 0.94614 | −0.00022 |
| LWQ | lean base (no digits) + windows + ladder | 52 | 0.946046 | +0.000057 | — | — |
| W1 | + windows (shape encoder kept) | 160 | 0.946035 | +0.000046 | 0.94617 | −0.00018 |
| LW | lean base + windows replacing shape | 40 | 0.946002 | +0.000013 | 0.94615 | −0.00016 |
| LWQE | LWQ on E1's hyperparameters | 52 | 0.945896 | −0.000093 | — | — |
| W2 | S1 (lean, E1 params) + windows | 38 | 0.945881 | −0.000108 | 0.94617 | **−0.00001** |
| Q2 | ladder **replacing** the exact-value key | 164 | 0.945849 | −0.000140 | — | — |
| LWE | LW on E1's hyperparameters | 40 | 0.945803 | −0.000186 | — | — |

**1. The centred window is the shape encoder, better parameterised — and keeping both is worse than
keeping one.** WS (windows only, 144 features) beats both R0 (shape only, 154) by +0.000062 and W1
(both, 160) by +0.000016. The fixed equal-width grid pools a value asymmetrically when it falls near
a bin edge, and its radius is whatever the grid width happens to be; a centred window puts the value
at the centre of its own neighbourhood at six scales at once. This explains Phase 15's X1 null
(the encoder was *flat* from 1024 to 16384 bins): grid **width** was never the binding limitation,
the arbitrary **origin** was, and X1 could not see that because every setting it tried shared the
defect. **`te_shape_cols` is now superseded by `te_window_cols`** and should not appear in new
recipes.

**2. The ladder adds as a supplement and fails as a replacement, which localizes what it is.**
Q1 (+0.000078, added alongside the exact key) against Q2 (−0.000140, replacing it). Our key
resolutions previously jumped straight from the exact value (13,214 levels, rate SE ~0.053) to the
backoff neighbourhood (200 quantile bins); `//50` (3,160 levels) and `//500` (316) are real
intermediate resolutions. But they cannot carry the per-value lookup itself — Phase 1's B4/B6 result
stands, and Q2's `best_iter` (1076–1386, no collapse) says the loss is information, not displacement.
WQ combines them at **78% of the additive prediction** (+0.000097 against +0.000124) — mild
redundancy, not Phase 1c's strong C1/C2/C4 collapse.

**3. A false trail, recorded because the experiment that refuted it is the interesting part.**
Mid-session a strong-looking result appeared: LB residual regressed on feature count gave
r = −0.727, p = 0.0004, with lean (≤40 feats) runs at mean −0.00004 and rich (≥130) at −0.00016,
Welch p < 0.0001. It predicted a lean model would land on the raw fit and a rich one 0.00016 below.
**W2 (38 feats) landed at −0.00001 and LW (40 feats) at −0.00016** — near-identical feature counts,
opposite bands. The variable is the **hyperparameter family**, not feature bulk: E1's params
(7 leaves, lr 0.05) average residual −0.00004 (n=5, sd 0.00004) against R0's borrowed params
(31 leaves, depth 5, lr 0.02, max_bin 1024) at −0.00015 (n=6, sd 0.00004), Welch **p = 0.0014** —
and feature count explains nothing *within* either family (p = 0.147, p = 0.987). Every rich run had
inherited R0's params in Phase 14 and every lean run kept E1's, so the two were perfectly confounded
until today's W2/LW pair broke them apart. **This corrects Phase 15's X3**, which concluded R0's
borrowed hyperparameters were "already at or near the full-data optimum" — true on OOF, and on the
board they cost ~0.00011.

**4. The offset is real and NOT harvestable, which is the practically important half.** LWE and LWQE
put today's ingredients on E1's hyperparameters and lost 0.000186 / 0.000093 of OOF, with `best_iter`
collapsing to 358–536 and 420–728 against LW's 1075–1355 — plain under-capacity: 7 leaves at lr 0.05
cannot carry 40+ features with the freq block and dual TE. The params that avoid the penalty cannot
fit the representation that earns the score. **The open lead is whether something between 7 and 31
leaves gets the OOF without the offset** — untested, and the first thing to probe next.

**5. A plateau reading I made and the board refuted within the hour.** R0 (0.94614), W1 (0.94617)
and WS (0.94614) span 0.000062 in OOF and land within 0.00003 on the board, and I read that as "OOF
gains no longer reach the leaderboard on this representation." WQ then took the largest OOF gain of
the day (+0.000097) and produced **+0.00010 of LB over R0**. The three earlier points were simply
closer together than the near-twin paired resolution (0.000027) can resolve; the instrument was
working the whole time. Playbook §5 is the post-mortem of exactly this kind of over-read, and it
applied here to a conclusion drawn from three points in a single afternoon.

**Five slots spent, every one a paired point:**

| submission | OOF | raw predicted LB | actual LB | residual |
|---|---|---|---|---|
| **WQ** | 0.946086 | 0.94640 | **0.94624** | −0.00016 |
| W1 | 0.946035 | 0.94635 | 0.94617 | −0.00018 |
| W2 *(lean control)* | 0.945881 | 0.94618 | 0.94617 | **−0.00001** |
| LW *(the refutation)* | 0.946002 | 0.94631 | 0.94615 | −0.00016 |
| WS | 0.946051 | 0.94636 | 0.94614 | −0.00022 |

**Champion moves from the 7-leg stack (OOF 0.946027 → LB 0.94616) to `WQ` solo (OOF 0.946086 →
LB 0.94624), +0.00007 LB** — and WQ beats the old champion on OOF and LB simultaneously. WQ's ADD to
the 7-leg pool is +0.000100, missing the +0.0002 leg gate, so the 8-leg *stack* (OOF 0.946127, the
highest recorded here) was **not** shipped — Phase 15's inversion is the precedent, and there was no
slot left to check it. `pool.json` now carries WQ as an 8th leg with `champion_stack_oof=0.946127`
as the baseline for future ADD tests; **verifying that stack is the first slot tomorrow.**

*Harness notes.* Local runs hit a hard memory ceiling — a 5-fold fit on a 160+-feature
representation peaks above what a 16GB machine has free, and four probes were OOM-killed mid-fold.
Two fixes: the fold loop now releases its encoded frames before the next iteration allocates its own
(peak was two folds' worth of a 535k × n_features float64 matrix), and `scripts/make_probe_kernel.py`
generates a one-off Kaggle CPU kernel per probe, which restored the 5-way parallelism Phase 15 used.
Kernels clone from GitHub, so **a probe kernel built against an unpushed `pipeline.py` silently runs
the previous revision** — push first. One real mistake: racing a `--no-push` re-collection against a
still-running background collector archived Q1 and WQ **twice each**, under different run_ids
(`d6ca7bd5`/`e9dcd2ff`, `7d456f26`/`529fbde2`) with identical OOF. The rows are left in place —
`runs.csv` is append-only — but **de-duplicate on `run_tag` before any re-analysis of this phase.**

### Phase 16b — the 8-leg stack, checked; the ADD gate's refusal holds (2026-09-13)

Phase 16 ended with one candidate unchecked: WQ's ADD to the 7-leg pool was +0.000100, missing the
+0.0002 leg gate, so the 8-leg stack (OOF **0.946127**, the highest recorded in this competition)
was not shipped and no slot remained to test whether the gate was right to refuse it.

Pre-registered before submitting: raw GBDT-fit prediction **0.94644**; the stack-family residual band
(−0.00012 to −0.00023 across the 6-, 7-, 9- and 11-leg stacks) puts the expectation at
**0.94621–0.94626**, straddling WQ solo's actual 0.94624. ≥0.94627 promotes the stack; ≤0.94621
confirms Phase 15's inversion reading.

**Result: LB 0.94625, residual −0.00019** — mid-band, clearing neither gate.

| | OOF | raw pred | LB | residual |
|---|---|---|---|---|
| WQ solo *(champion)* | 0.946086 | 0.94640 | 0.94624 | −0.00016 |
| **8-leg stack** | **0.946127** | 0.94644 | **0.94625** | **−0.00019** |

**The stack's +0.000041 OOF over WQ solo produced +0.00001 on the board** — an order of magnitude
inside the near-twin paired resolution (0.000027), i.e. a wash. **The champion does not move: WQ
solo stays.** The ADD gate refused this stack at +0.000100, and the leaderboard neither confirms nor
contradicts that refusal so much as it makes the question moot — the two artifacts are the same
score. Between two indistinguishable candidates the repo takes the simpler one: a single model
carries no meta-fit optimism, and Phase 15's 9-leg inversion is the standing precedent for stack
size buying OOF and paying it back in offset. `pool.json` keeps `champion_stack_oof=0.946127`, now a
*verified* number rather than a projected one.

**One genuinely new structural fact: the first negative stack weight of this competition.**

| leg | solo | weight |
|---|---|---|
| **WQ** | 0.9461 | **1.9877** |
| R0 | 0.9460 | 0.6485 |
| E4 | 0.9457 | 0.5234 |
| **D2** | 0.9454 | **−0.3787** |
| E1 / F2 / E4d / E2 | 0.9455–0.9456 | 0.08–0.15 |

Phase 2 recorded "**0 of 6 weights negative**" against S6E8's 10 of 23, and read it as the signature
of a pool of near-twins: playbook §6 says a fitted combiner earns its keep by SUBTRACTING correlated
error, and there was nothing to subtract. WQ's arrival changes that — it takes weight 1.99, three
times R0's, and the combiner now *subtracts* D2, the pool's weakest and oldest leg. So §6's
discriminator has finally fired here. **It fired without paying**: +0.000041 OOF, +0.00001 LB. The
mechanism being available is not the same as there being correlated error worth removing, and one
negative weight on an 8-leg pool of 0.995+ correlates is not the S6E8 regime.

**Leaderboard context, for the §5 week-3 reassessment now eight days out.** The board grew
**1,632 → 1,820 teams overnight** and compressed: rank 100 now needs 0.94645, rank 200 0.94641,
rank 300 0.94635. We gained +0.00008 since yesterday and moved **369 → 395**, because the ~0.94644
public-blend cluster keeps absorbing new entrants above us. This is §8's standing finding sharpening,
not changing: rank is not measuring modelling in that band, and the gap to the *modelling* frontier
(their best single model, ~0.94633) is ~0.00008, not the ~0.0002 the rank implies.

**What stays open**, unchanged from Phase 16: whether a capacity between E1's 7 leaves and R0's 31
gets the OOF without the ~0.00011 hyperparameter-family LB penalty. That is the one lead with a
measured mechanism behind it and no probe run yet.

### Phase 17 — the offset was never hyperparameters; it is `fe_recipe_score` (2026-09-14)

Phase 16 left one open lead: whether a capacity between E1's 7 leaves and R0's 31 gets R0's OOF
without the ~0.00011-0.00019 "R0-family" LB offset first measured that day. Five slots spent
today, all instrument points, isolating one variable at a time — and the lead was wrong in its
premise. **The offset was never about hyperparameters. It is `fe_recipe_score`.**

**Capacity, closed.** A clean sweep on the LW base (R1's lean 40-feature recipe with windows
replacing the shape encoder), varying ONLY `num_leaves` and holding every other R0-family knob
fixed:

| leaves | OOF | Δ vs LW(31) |
|---|---|---|
| 7 | 0.946022 | +0.000020 |
| 13 | 0.946018 | +0.000016 |
| 19 | 0.946000 | −0.000002 |
| 25 | 0.946006 | +0.000004 |
| 31 *(LW)* | 0.946002 | — |

Flat across the whole range — under the 0.000038 seed floor. `best_iter` compensates (7 leaves
needs 2132-3426 rounds; 31 needs 930-1355) but OOF does not move. **LWcap7 submitted anyway**, as
the sharpest available test of whether the offset itself tracked leaves even though OOF didn't:
LB 0.94617, residual **−0.00016** — squarely the R0-family band. Capacity is not the driver.

**max_bin and learning_rate, closed the same way.** `LWb255` (max_bin 1024→255): OOF 0.945988,
flat (−0.000014), no `best_iter` change. `LWlr05` (lr 0.02→0.05): OOF 0.945958, and its
`best_iter` collapses to 392-536 — an almost exact match to `LWE`'s full-five-knob collapse
(358-536) from Phase 16. That looked like the smoking gun. Submitted: LB 0.94612, residual
**−0.00014** — still the R0-family band, despite the mechanistic match. Three single-knob
hyperparameter swaps, three failures to move the residual.

**The pivot, and the actual answer.** Every "bad-band" run all along shared one thing every
"good-band" run lacked: the `freq_cols`/`fe_recipe_score`/`te_multi_smooth` feature bundle,
independent of which hyperparameters process it. Testing that directly, holding hyperparameters
fixed at `W2`'s own E1 values (its own residual was the best measured all competition, −0.00001):

| probe | added to W2 | OOF | Δ vs W2 | best_iter | LB | residual |
|---|---|---|---|---|---|---|
| W2 *(baseline)* | — | 0.945881 | — | normal | 0.94617 | **−0.00001** |
| **W2TM** | `te_multi_smooth` only | 0.945902 | +0.000021 | normal (775-1151) | 0.94620 | **+0.00000** |
| W2FB | freq_cols + recipe_score + multi-smooth | 0.945799 | −0.000082 | collapse (340-660) | 0.94595 | −0.00014 |
| **W2RS** | `fe_recipe_score` **only** | 0.945721 | **−0.000160** | **collapse (288-555)** | **0.94587** | **−0.00013** |

`te_multi_smooth` is innocent (W2TM lands dead on the good-band prediction). `fe_recipe_score`
**alone** — two fixed-coefficient columns (`buy_score`/`worry_score`, the recovered ORIGIN
generator's linear formula) — reproduces nearly all of W2FB's OOF loss and LB penalty by itself,
with the sharpest `best_iter` collapse measured all day. This directly extends Phase 12's P3
finding (displacement on E1's 16-feature baseline: −0.000144 OOF, `best_iter` 369-444) to this
representation, and the collapse is *worse*, not better, after four days of encoder work:
**the fixed composite score is still the cheapest thing in the feature set for a tree to grab
early, and grabbing it costs resolving the fine per-value structure everything else this
competition has built exists to supply.**

**What this means for the champion, and why the premise was wrong from the start.** WQ (the
current champion) *does* carry `fe_recipe_score=True`. Phase 15's own `Ia` probe already measured
dropping it from R0 as a dead null on OOF (+0.000006) and explained why: R0's 154-feature
representation has enough competing structure that `best_iter` (777-1098) was nowhere near
collapse, so two extra composite columns don't measurably displace anything further there. That
reasoning holds up: the mechanism is representation-size-dependent, not fixed. Whether it still
holds for WQ specifically (172 features, windows+ladder added since R0) is checked directly below.

**WQnoRS (offline, no slot spent): OOF 0.946084**, vs WQ's 0.946086 — a dead null (−0.000002),
`best_iter` normal (983-1210, no collapse). Confirms Ia's finding holds at WQ's own scale: on the
172-feature representation, `fe_recipe_score` displaces nothing, so today's mechanism does not
touch the champion. **No free improvement is available, and none should be sought here** — the
mechanism just closed explicitly requires displacement, which is not happening on this
representation, so a future slot testing WQ-minus-recipe-score on the LB would almost certainly
reproduce WQ's own −0.00016 residual with statistically the same OOF. Not worth spending on.

**What stays open.** WQ's own −0.00016 LB residual is real (measured twice now: WQ itself and the
8-leg stack both landed there) and still unexplained — today closed capacity, max_bin, learning
rate, `te_multi_smooth`, and `fe_recipe_score` as candidates, all on representations where they
*could* plausibly matter, and none of them touch it at WQ's actual feature count. The lead for a
future session: isolate `freq_cols` and the window/ladder encoders themselves (untested alone,
still confounded together in every rich-representation run to date) as the remaining candidates,
or accept that the offset is a property of representation richness in some diffuse sense that a
single-ingredient swap cannot isolate.

**Five slots spent, four of them decisive isolations, in order:**

| submission | probe | raw pred | actual LB | residual | verdict |
|---|---|---|---|---|---|
| 1 | LWcap7 (leaves→7) | 0.94633 | 0.94617 | −0.00016 | capacity: not the driver |
| 2 | LWlr05 (lr→0.05) | 0.94626 | 0.94612 | −0.00014 | learning rate: not the driver |
| 3 | W2FB (full bundle added) | 0.94609 | 0.94595 | −0.00014 | confirms the feature bundle carries it |
| 4 | W2TM (multi-smooth only) | 0.94620 | 0.94620 | **+0.00000** | cleared |
| 5 | W2RS (recipe_score only) | 0.94600 | 0.94587 | −0.00013 | **confirmed carrier** |

No champion move — none of today's probes were built to beat WQ, and none accidentally did (best:
LWcap7 at 0.94617, still below WQ's 0.94624). `experiments/runs.csv` carries all nine probe rows
plus the offline WQnoRS check; `pool.json` is unchanged. Board re-read: 1,877 teams (up from 1,820
yesterday), rank 409 at our unchanged 0.94625.

### Phase 18 — the window encoder is saturated; feature bulk is dead; `fe_recipe_score` is a real LB cost (2026-09-16)

Five slots, all spent, plus eight offline probes. **No OOF lever found — all seven new probes came
back negative — and the champion still moved +0.00007 LB**, because the night's gains came from
*removing* things rather than adding them. Board re-read: 2,036 teams (up from 1,877), top 0.94674,
rank 100 at 0.94649, rank 300 at 0.94638. We stood at rank 445 on 0.94625.

*(Zero submissions were made on 2026-09-15 UTC — five paired points thrown away, the exact miss
playbook §1 exists to prevent. Recorded here because the log should show it.)*

**1. `fe_recipe_score` costs real leaderboard score, and Phase 17's exemption of the champion was
wrong.** Phase 17 identified the two fixed-coefficient composite columns as the carrier of the
R0-family LB offset, then exempted `WQ` on the grounds that the displacement mechanism cannot operate
at 172 features (`WQnoRS` was a dead null on OOF, +0.000002, `best_iter` uncollapsed) and declared a
slot on it "not worth spending." That reasoning was about OOF; the question was about the residual.
`WQnoRS` had been sitting archived and unsubmitted since Phase 17.

| | feats | OOF | LB | residual |
|---|---|---|---|---|
| WQ *(rs on)* | 172 | 0.946086 | 0.94624 | −0.00016 |
| **WQnoRS** *(rs off, strict twin)* | 170 | 0.946084 | **0.94629** | **−0.00011** |

**+0.00005 LB at a −0.000002 OOF change** — 1.9σ of the near-twin paired resolution. Restricting to
rich runs only (≥100 features), so feature count cannot explain the split: rs=True n=8 mean
**−0.00018**, rs=False n=4 mean **−0.00007**, Welch **p = 0.016**. The paired single-field estimate
(+0.00005) is the conservative one and the group difference (+0.00011) the upper bound; both say the
same thing. **`fe_recipe_score` is struck from every recipe.** It was never worth anything on OOF at
any feature count, and it costs on the board at every one.

But the offset is **not fully explained by it** — rs removal recovers about a third of the −0.00016,
leaving ~−0.00007 standing after six isolation attempts across Phases 16–18.

**2. Feature bulk is refuted, on the cleanest test this competition has produced.** Phase 16 §3
raised a feature-count effect (r = −0.727, p = 0.0004) and then abandoned it as a "false trail" on
the W2-vs-LW pair — which was confounded by hyperparameters *and* `fe_recipe_score` simultaneously,
so it never actually settled anything. Tonight's five submissions break both confounds: identical R0
hyperparameters, `fe_recipe_score` off on every one, spanning 50 to 194 features.

| run | feats | OOF | LB | residual |
|---|---|---|---|---|
| LWQnoRS | 50 | 0.946048 | 0.94630 | −0.00006 |
| **WQ2** | **154** | 0.946080 | **0.94631** | −0.00008 |
| WQnoRS | 170 | 0.946084 | 0.94629 | −0.00011 |
| WR | 175 | 0.946080 | **0.94633** | −0.00006 |
| WD | 194 | 0.946049 | **0.94633** | −0.00003 |

**corr(n_features, residual) = +0.001** across a 144-feature span. Flat, and the sign of every
sub-comparison is wrong for bulk. The sharpest single pair is **WQ2 vs WR: OOF matched to all six
decimals (0.946080) at 154 vs 175 features** — the extra 21 features scored **+0.00002 higher**,
against a bulk prediction of −0.00003. `LWQnoRS` landed at 0.94630 against a pre-registered 0.94629
(refutation) vs 0.94636 (bulk). **The axis is closed.** What Phase 16 measured as a feature-count
trend was `fe_recipe_score` riding along with feature count, exactly as Phase 17 suspected but could
not confirm without a rich rs-free point.

**3. The derivative hypothesis — the night's headline probe — is refuted, three times over.**
Reading `apply_window_target_encoding` against the shape encoder it replaced: the shape encoder
emitted eight channels per bin width *including* left, right, slope and curvature, and Phase 14 named
those explicit derivative channels as a main mechanism ("hand the tree the response's local
derivative"). Phase 16 promoted centred windows on the strength of fixing the grid's arbitrary
*origin* — and silently dropped the derivative, since a symmetric rate per radius carries no
asymmetry and **a tree cannot difference two columns**. That looked like the reason `WQ` still
carried *both* encoders despite Phase 16 §1 declaring `te_shape_cols` superseded.

New default-off axis `te_window_sides` (left/right/slope/curvature; one-sided windows exclude the
query value, so they are strictly *more* leak-resistant than the centred rate; verified bit-identical
on W2's archived config before use). Result, on three structurally different bases:

| probe | base | feats | Δ OOF |
|---|---|---|---|
| W2D | lean, E1 params | 62 | −0.000043 |
| WD | rich, R0 params | 194 | −0.000035 |
| WDS | rich, shape dropped | 178 | −0.000041 |

One answer, three bases, `best_iter` normal throughout (988–1494), so this is not displacement in the
F1/H2/H3/P3 sense. **Phase 14's attribution of the shape encoder's gain to its slope/curvature
channels does not survive re-parameterisation**: what the window encoder kept (correct centring) was
the part that mattered, and what it dropped was not.

**4. The window encoder is saturated on every parameter it has.** Seven probes, all negative, none
near the +0.0000949 gate — against `WQnoRS` 0.946084:

| probe | change | Δ OOF | feats |
|---|---|---|---|
| WQ2 | `te_shape_cols` dropped | −0.000004 | 154 |
| WR | radii 6 → 11, [1…250] | −0.000004 | 175 |
| WSM | `te_window_smooth` 10 → 2 | −0.000011 | 170 |
| WC | `te_window_count` (occupancy) | −0.000017 | 176 |
| WD | `te_window_sides` | −0.000035 | 194 |
| WDS | sides + shape dropped | −0.000041 | 178 |

**WSM is the informative null of the small ones.** Its exact-value analogue is the second-largest
single-knob gain of this competition (C2, `te_smooth` 20 → 5, +0.000487), and it does not transfer: a
window at radius 25+ already pools thousands of rows, so smoothing strength is near-irrelevant there,
and the fine structure that made the per-value key sensitive is exactly what pooling has already
averaged away.

**5. A Phase 16 explanation, corrected.** Phase 15's X1 found the shape encoder flat from 1024 to
16384 bins; Phase 16 explained that null away as the arbitrary-origin defect being shared by every
setting tried, implying a correctly-centred encoder *would* show real sensitivity to its pooling
scale. `WR` nearly doubles the radius ladder and widens both ends for **−0.000004**. A correctly
centred encoder is flat in its radii too. **Neighbour-pooling granularity is flat in general — X1's
null was a property of the pooling, not of the broken grid**, and Phase 16's account of it was wrong.

**Champion moves: `WQ` (OOF 0.946086 → LB 0.94624) → `WQ2` (OOF 0.946080 → LB 0.94631), +0.00007 LB,
rank ~445 → ~399.** `WQ2` is `WQ` with `fe_recipe_score` off and the redundant fixed-grid shape
encoder dropped: **154 features against 172, at an OOF change of −0.000006 — one sixth of the seed
floor.** Both removals, nothing added.

**On not promoting WD or WR, which both scored 0.94633.** Tonight's five submissions span 0.00004 of
LB, **1.5σ of the near-twin paired resolution — they are one score.** `WD`'s OOF is 0.000031 *below*
`WQ2`'s, and `WR`'s matches it exactly while carrying 21 more features. The OOF is the trusted ranker
(§4) and the repo's standing tiebreak between indistinguishable candidates is the simpler model
(Phase 16b). Promoting `WD` would be reading a 1σ LB difference against a measured OOF deficit —
precisely the over-read Phase 16 §5 is the post-mortem of. **`WQ2` ships.**

**What stays open.** The residual ~−0.00007 that survives removing `fe_recipe_score`, now with
capacity, `max_bin`, learning rate, `te_multi_smooth`, `fe_recipe_score` and feature bulk all measured
closed as candidates. The encoder axis that produced every gain since Phase 14 is saturated in its
parameterisation. `pool.json` still references `WQ` as its 8th leg and its `champion_stack_oof`
(0.946127) predates tonight — the stack was not re-measured and that number is stale until it is. The
§5 artifact-sharing reassessment is due **2026-09-21**, five days out.

`experiments/runs.csv` rows `221f2964` (WQ2), `874fe751` (WR), `351fb556` (WD), `2f08d839` (LWQnoRS),
`435587bd` (WSM), `5c2a97ca` (WC), `d7c3e9da` (WDS), `a51f88b2` (W2D); `f5c9efb0` carries WQnoRS's
score — **`cc73801d` is its un-submitted duplicate, so de-duplicate on `run_tag` before any
re-analysis.**

### Phase 19 -- the residual decomposes into three mechanisms, one of them new (2026-09-16/17)

Phase 18 closed with one open item: a ~-0.00007 LB residual surviving fe_recipe_score
removal, unexplained after six isolation attempts across Phases 16-18. Before touching code,
re-reading README section 4 itself found the more basic problem: the frozen OOF->LB fit (slope
1.0832, residual sigma 0.000103) was estimated in Phase 3 on 10 points spanning OOF 0.9417-0.9457.
Every run since Phase 14 sits at 0.9460-0.9461, outside that calibration range -- and 14 of the
most recent submissions sit within 0.0001 of each other on OOF, so no regression fit to them can
estimate a local slope; it can only extrapolate. Playbook section 5 exists for exactly this
shape of mistake: a series of LB points read as a trend when its range is smaller than the
split's own resolution.

So today bought what the champion's own neighbourhood never had: OOF spread. Five slots, spent
on a deliberately-degraded ladder built on WQ2's own 154-feature config, submitted in descending
OOF order, each pre-registered under both the frozen fit and a same-day refit before its LB score
was known (experiments/runs.csv notes carry the numbers verbatim, written before submission).

**Zero-slot work first, all local/offline:**

- scripts/refit_gap.py (new): refits the OOF->LB predictor three ways from the archived run
  log -- the frozen 10-point selection (reproduces slope 1.0832, sigma 0.000103 exactly, confirming
  README section 4's own arithmetic), all 45 then-archived points (slope 1.0496, sigma 0.000080),
  and the rich cluster alone (slope 0.8354 off only 20 points spanning 0.0002 OOF -- noise, not a
  measurement, which is the plan's whole thesis stated in code).
- scripts/public_gap.py, run for the first time in S6E9. Its named library
  (szymonkapiski/s6e9-oof-library-47-models) now 403s; substituted
  dariushafshar/s6e9-golem-oof-library (19 members, OOF 0.9381-0.9445, all below our pool floor --
  a diversity library, not a state-of-the-art one). It ships folds_seed42.npy, which the script
  now verifies bit-identical to our frozen split (668,665/668,665 rows) before trusting anything --
  a real check the original szymonkapiski plan could never offer. Result: our 8-leg pool (WQ
  swapped for its rs-free successor WQ2; pool.json updated) contributes +0.001424 to a
  union stack over this library, and the niche we're missing from it is +0.000023 -- under the
  seed floor. This library adds nothing; it says nothing about the actual ~0.9467 frontier, which
  it does not contain.
- pool.json refreshed: 8th leg WQ -> WQ2, champion_stack_oof recomputed 0.946136 (was
  0.946127, stale since before Phase 18).
- Six ladder rungs screened on Kaggle CPU kernels (5-concurrent cap), zero LB spent. A real
  harness bug caught here, not on a submission: the first TS100/TS1000 attempts (te_smooth
  5->100/1000) landed bit-identical to WQ2 -- pipeline.py's own comment says te_smooth is
  ignored for te_cols whenever te_multi_smooth is non-empty, and WQ2 carries
  te_multi_smooth=[10,"auto"]. Both were silent no-ops. Corrected (TS100b/TS1000b: override
  te_multi_smooth itself) before either reached a submission.

**The ladder, OOF only (screened, no LB yet at this point):**

| rung | change from WQ2 | OOF | delta vs WQ2 (0.946080) |
|---|---|---|---|
| FQ | freq_cols/freq_digit_cols dropped | 0.946101 | +0.000021 -- near-twin, closes Phase 18's last open lead |
| TS100b | te_multi_smooth -> [100] | 0.945951 | -0.000129 |
| TS1000b | te_multi_smooth -> [1000] | 0.945909 | -0.000171 |
| NOTE | te_cols emptied (per-value TE dropped) | 0.945859 | -0.000221 |
| NE300 | n_estimators 3500->300, early stop off | 0.945406 | -0.000674 |
| BARE | te_cols/te_window_cols/fe_quant_cols all emptied | 0.944209 | -0.001871 |
| NE120 | n_estimators 3500->120, early stop off | 0.941051 | -0.005029 |

**The five submitted (descending OOF): TS100b -> NOTE -> NE300 -> BARE -> NE120.**

| rung | OOF | frozen pred | refit pred | actual LB | resid vs frozen | resid vs refit |
|---|---|---|---|---|---|---|
| TS100b | 0.945951 | 0.94623 | 0.94613 | 0.94605 | -0.00018 | -0.00008 |
| NOTE | 0.945859 | 0.94613 | 0.94603 | 0.94591 | -0.00022 | -0.00012 |
| NE300 | 0.945406 | 0.94564 | 0.94556 | 0.94561 | -0.00003 | +0.00005 |
| BARE | 0.944209 | 0.94435 | 0.94430 | 0.94430 | -0.00005 | 0.00000 |
| NE120 | 0.941051 | 0.94093 | 0.94099 | 0.94127 | +0.00034 | +0.00028 |

**Neither pre-registered outcome won outright.** The refit (1.0496) tracks better at the low end
(BARE dead on, NE300 within 1 sigma) but NE120 misses both fits by 4x the fit's own residual sigma, in
the opposite direction from TS100b/NOTE. A single local slope cannot be the answer when the
residual's sign flips depending on which rung produced it.

**Fitting the two historical eras separately resolves this.** The Phase 0-3 lean cluster (<=21
features, n=15, the frozen fit's own era) refits to slope 1.0898, sigma 0.000083 -- matching the
frozen fit closely, as it should. Reading every rung and every WQ2-family anchor against that
line, grouped by mechanism, rather than feature count:

| group | tags | resid vs lean line | paired sigma (this pair) |
|---|---|---|---|
| champion family (full TE, rs off) | WQ2 -0.00006, WR -0.00004, WD -0.00001, WQnoRS -0.00009, LWQnoRS -0.00004 | all <=1.1 sigma | -- |
| fe_recipe_score on (known cost, Phase 18) | R0 -0.00013, WQ -0.00014, W1 -0.00015, LW -0.00014, LWcap7 -0.00014 | ~1.6-1.8 sigma | -- |
| TE weakened, not removed (new) | TS100b -0.00018, NOTE -0.00022 | 2.2 sigma, 2.7 sigma | WQ2 vs NOTE: 0.000097 (public split) |
| TE fully removed / fit mildly truncated | BARE -0.00003, NE300 -0.00003 | ~0.4 sigma | -- |
| fit severely truncated (new) | NE120 +0.00038 | 4.6 sigma, opposite sign | WQ2 vs NE120: 0.000363 (public split) |

**Three mechanisms, not one, and Phase 16-18 were hunting a single flat offset that was never
there:**

1. **fe_recipe_score costs LB** -- already found (Phase 18), independently reproduced here by a
   completely different route (residual against a historical calibration line rather than a
   paired ablation). Confirms Phase 18 rather than adding to it.
2. **New: a per-value TE that is weakened but still present costs MORE than either the full-
   strength version or no TE at all.** BARE (TE gone entirely) sits on the lean line; the
   champion family (TE at full strength) sits on the lean line; NOTE/TS100b (TE present but
   degraded) sit 2-3 sigma below it -- the worst residuals of any non-fe_recipe_score group. This
   is a non-monotonic effect in encoder strength, structurally the same shape as the F1/H2/H3/P3
   displacement signature (README section 6, Phase 2c): a partially-resolved lookup is something
   the tree leans on with more confidence than it should, in a way a fully-resolved or fully-absent
   lookup does not produce.
3. **New: severely truncating boosting rounds inflates the LB relative to OOF, not the reverse.**
   NE120 (120 rounds, no early stopping) scores +0.00038 above what its own OOF predicts -- the
   Phase 3/4 fold-averaging mechanism (an OOF row is scored by one fold-model; a test row is the
   5-fold average), previously established only for early-stopped neural nets (G1: +0.00105 raw,
   +0.00048 after in-fold bagging), now demonstrated for the first time in a GBDT whose capacity
   was cut hard enough to make it genuinely high-variance. NE300 (300 rounds -- less severe) shows
   only a trace of the same effect, consistent with the mechanism being magnitude-dependent: the
   variance cut from 5-fold test-averaging only matters once the underlying fold-model is
   high-variance to begin with. The champion's own ~1000-1400-round early-stopped fit is far from
   this regime -- this finding does not suggest the champion's OOF is currently mismeasured, only
   that the ladder rungs built to probe the low end crossed into a regime where a different,
   previously-neural-net-only bias applies.

**What this means for the champion, stated plainly.** WQ2 and its rs-free siblings show a
residual against the historical calibration line of -0.00001 to -0.00009 -- all within 1.1 sigma of
the lean-era fit's own 0.000083 residual sigma. Phase 18's "~-0.00007 unexplained residual" is, once
fe_recipe_score's already-known cost and the newly-found TE-weakening effect are separated out,
not present for the actual champion. It was Phase 16-18 conflating the champion's own
near-zero residual with the larger residuals of related-but-different configs (R0, WQ, LW,
LWcap7) that still carried fe_recipe_score or the pre-WQ2 recipe -- plus, at the session's
start, a fit extrapolated 0.004 past its own calibration range. The residual hunt that ran
through three phases is retired: there was no single thing left to find, because there was never
one mechanism producing it.

**The shipping gate does not move.** README section 4 freezes it at +0.0000949 OOF, derived once
and not re-derived from marginal deltas -- that stands exactly as written. What changes is the
predictor (the OOF->LB relationship itself), which was always a diagnostic reading, never the
gate's basis. The two are different objects; conflating them is the mistake this phase exists to
correct.

**No champion move.** All five slots went to instrument work by design (this phase's brief);
WQ2 (OOF 0.946080 -> LB 0.94631) still ships. Two things are now queued for the next session,
neither actioned tonight: the parked 5-leg rs-free blend (WQ2+WR+WD+WQnoRS+LWQnoRS,
logit-mean OOF 0.946136, computed via stack_logit.py, near-equal fitted weights 0.199-0.203
confirming the near-twin/fixed-combiner read) -- its predicted LB now has a properly local
calibration to check against; and whether the TE-weakening effect (mechanism 2) has a real,
untested optimum distinct from the OOF optimum, which the ladder was not built to resolve.

Board re-read before the first slot: 2,132 teams (up from 2,036), top 0.94674, we stood at
rank 410 on our best-of 0.94633. experiments/runs.csv rows: 5794995c (FQ), e882223f
(TS100, broken/no-op -- superseded by 1b01b99a TS100b), `d52cd631` (TS1000, same defect --
superseded by TS1000b), ad825508 (NOTE), 260eae35 (NE300), cb4fd044 (BARE), 1b01b99a (TS100b),
`31e2d10a` (TS1000b), `adb8cfde` (NE120), de07aba9 (8-leg stack re-fit). pool.json updated in this
phase (WQ -> WQ2, champion_stack_oof 0.946136). 92f79e80 is an unsubmitted duplicate of FQ (5794995c carries the real record) -- the same class of bug as Phase 18's `cc73801d`/WQnoRS pair, from an improperly-backgrounded push that archived once on its own before a correctly-tracked retry archived the same kernel a second time. De-duplicate on run_tag before any re-analysis; scripts/refit_gap.py excludes it by run_id already.

### Phase 20 -- FQ and a never-run TE-coverage lead both beat WQ2; new champion, +0.00004 LB (2026-09-17/18)

Board re-read before the first slot: 2,252 teams (up from 2,132), top **0.94675**, we stood at rank
439 on our best-of 0.94633. Phase 19 closed with two artifacts parked, unsubmitted, at higher OOF
than the champion (`FQ` 0.946101, the 5-leg rs-free blend 0.946136 fitted / 0.946121 as an honest
equal-weight mean) and two Phase 15 X4 leads never run (TE key coverage; smooth keys as native
categoricals). Tonight closed all four with five slots, all spent.

**Zero-slot work first.** `scripts/refit_gap.py` reproduced the lean-era line exactly (slope
1.0898, intercept -0.08470, sigma 0.000083 -- matches Phase 19's own fit to the last digit) and
also fit the full 45-point de-duplicated set (slope 1.0248, intercept -0.02326, sigma 0.000085),
now spanning OOF 0.9410-0.9461 thanks to Phase 19's ladder -- an interpolating fit for the first
time since the champion moved past the frozen fit's calibration range. `scripts/stack_logit.py
--mode logit_mean` recomputed Phase 19's parked blend as an honest equal-weight mean (not the
fitted combiner): B5 (the parked 5 legs) 0.946121, not 0.946136 -- the 0.000015 gap is exactly the
meta-model optimism the script's own docstring names, confirmed again by the fitted weights coming
back at 0.199-0.203 (Phase 2's discriminator: a pool of near-twins gets averaged, not fitted).
Adding `FQ` as a sixth leg (B6) raised it to 0.946130, the best blend found. One contained infra
addition: `fe_quant_divisors_by_col`, a per-column override of the quantisation-ladder divisors
(default `{}`), needed because the global `[10,50,500,5000]` ladder degenerates on
`Daily_Commute_km` (5.0-98.7) to 10/2/1/1 levels, the last two constant. **Verified bit-identical**
by re-running `FQ`'s exact archived config against the patched pipeline before trusting anything
built on it: OOF 0.946101 and all five fold AUCs/best_iters reproduced exactly -- the check that
would have caught Phase 19's `te_smooth` no-op before it reached a submission.

**Five kernels fanned out on Kaggle CPU (the 5-concurrent cap), all built on `FQ`'s 93-feature
base, screened free before any slot was spent:**

| probe | change | feats | OOF | Delta vs FQ (0.946101) |
|---|---|---|---|---|
| **TEX** | commute quantisation ladder (`[1,2,5,10]`) added to `te_cols` | 105 | **0.946122** | **+0.000021** |
| SKa | `q_Annual_Income_USD_5000` (32 lvl) as native categorical | 94 | 0.946097 | -0.000004 |
| TS2 | `te_multi_smooth` strengthened to `[2,10,"auto"]` | 100 | 0.946063 | -0.000038 |
| SKb | `q_Annual_Income_USD_500` (284 lvl) as native categorical | 94 | 0.946023 | -0.000078 |
| SKc | `q_Annual_Income_USD_50` (2,026 lvl) as native categorical | 94 | 0.945759 | -0.000342 |

**The SK ladder closes the categorical-identity axis cleanly.** A smooth monotone decline with
cardinality (32 lvl null, 284 lvl -0.000078, 2,026 lvl -0.000342), not a cliff, and `best_iter`
normal throughout (SKc: 823/1154/988/...) -- this is information loss, not the F1/H2/H3/P3
displacement signature. Consistent with Phase 1c's structural finding that the per-value lookup is
already fully extracted (residual variance ratios 0.68-0.88): handing the same value identity to a
second, coarser mechanism only degrades the split search, it recovers nothing the TE missed.

**Five slots, submitted in this order, each pre-registered in `experiments/runs.csv` before
submission:**

| slot | run | OOF | predicted LB | actual LB | read |
|---|---|---|---|---|---|
| 1 | `FQ` (`5794995c`) | 0.946101 | 0.94635 | **0.94627** | misses prediction by 0.00008; does NOT clear the fewer-features tiebreak against WQ2 (0.94631) |
| 2 | `B6` blend (`26ed552b`) | 0.946130 | 0.94638 (no penalty) / ~0.94626 (7-leg rate) | **0.94633** | between the two, 0.00001 short of the promotion gate; beats FQ but not TEX |
| 3 | `TEX` (`43db301d`) | 0.946122 | 0.94637 | **0.94635** | clears WQ2 (0.94631) outright, at 105 feats vs 154 |
| 4 | `SKc` (`ec6afd42`) | 0.945759 | 0.94600 | **0.94598** | within 0.00002 of prediction -- confirms the axis-closing read |
| 5 | `TS2` (`67c48e45`) | 0.946063 | 0.94631-0.94632 | **0.94628** | ~0.00003-0.00004 below prediction -- a soft miss, not the 2-3 sigma NOTE/TS100b signature |

**New champion: `TEX`, OOF 0.946122 -> LB 0.94635, at 105 features against `WQ2`'s 154.** This
promotion does not rely on a tiebreak or a prediction: TEX beats WQ2 on OOF (+0.000042), on feature
count (105 vs 154), and on the actual, submitted LB score (0.94635 vs 0.94631) -- three independent
readings agreeing, the strongest promotion basis this repo has produced since Phase 14. Phase 15
X4's "TE key coverage" lead, named and left untested for five phases, is real: the frontier's wider
key coverage was never about the encoder's parameterisation (Phase 16-18 exhausted that), it was
about which columns get a per-value key at all. Rank 439 -> **405** at 0.94635 (2,252 teams).

**`FQ` itself was a false lead, and the mechanism is now visible in hindsight.** Its higher OOF
than WQ2 (+0.000021) came entirely from *removing* `freq_cols`/`freq_digit_cols`, exactly the class
of move that worked twice in Phase 18 (`fe_recipe_score`, the shape encoder) -- but FQ's actual LB
residual against the all-points line is -0.00008, a real miss, not a null. Removing the frequency
blocks costs something on the board that the OOF ladder cannot see, distinct from and larger than
any mechanism identified so far. **Not chased further tonight** -- TEX (built on the same FQ base,
plus the commute ladder) inherits whatever FQ's residual is and still cleared WQ2 comfortably, so
the missing frequency blocks are not a blocking cost, but the FQ-alone result is flagged as an open
question for a future session rather than folded into an existing mechanism it may not belong to.

**The blend result is a genuine wash, read against Phase 14/16b's stack penalty.** B6's equal-
weight logit mean scored 0.94633, almost exactly WQ2's own 0.94631 and a hair below TEX's 0.94635,
despite carrying the highest OOF of the night (0.946130). This does not distinguish the two
pre-registered outcomes cleanly -- it sits between "no penalty" (0.94638) and "7-leg fitted rate"
(~0.94626) -- but it leans toward *some* dilution surviving even a fixed-weight mean over legs from
one representation, since the winning single leg (TEX, discovered only after B6 was already
computed and submitted) beats the six-leg average of legs that do not include it. The honest
reading: a mean over near-twins is not free, it is merely cheaper than a fitted stack, matching
Phase 2's discriminator directionally but not proving the penalty is zero.

**`SKc`'s LB landing within 0.00002 of its predicted 0.94600 is the cleanest confirmation of an
axis-closing read this competition has produced** -- a -0.000342 OOF cost with normal `best_iter`
predicted to land exactly on the calibration line if it were pure information loss, and it did.
Categorical identity keys are closed for good: the per-value TE already extracts what they carry,
and coarsening the key only removes information the finer encoding kept.

**`TS2`'s soft miss (~0.00003-0.00004 below its 0.94631-0.94632 prediction) does not cleanly
confirm mechanism 2's directional claim, but does not refute it either.** It is well inside the
near-twin/family residual band (~0.00003-0.00004), nowhere near NOTE/TS100b's 2.2-2.7 sigma misses
on the weakening side. **Read together with FQ's own -0.00008 residual on the same all-points
line, a pattern worth naming for a future session: three of tonight's five slots (FQ, B6, TS2) all
landed 0.00001-0.00008 below their predictions, while only TEX and SKc landed on or above.** This
could be the all-points line's own residual sigma (0.000085) doing what residual sigma does, or a
real small negative bias in tonight's specific ladder -- one night is not enough data to tell them
apart, and the next session should read new points against this line before assuming either.

**A duplicate-archive artifact, same class as Phase 18's `cc73801d`/`WQnoRS` and Phase 19's own
`92f79e80`/`FQ`.** An early attempt to background the TEX kernel poll used a shell-level `&`/
`disown` that fought the harness's own background-process tracking; the orphaned process finished
and archived on its own (`dad8253f`) before a correctly-tracked `--no-push` retry archived the
identical kernel a second time (`43db301d`, the run actually passed to `submit_run.py`). Both rows
carry identical OOF/LB (0.946122 / 0.94635) since they are the same kernel output. **De-duplicate
on `run_tag=TEX`, keeping `43db301d`, before any re-analysis.**

**Pool and champion updated.** `pool.json`: 8th leg `WQ2` -> `TEX` (`43db301d`); the 8-leg curated
stack refit (fit-only, `b5a318cf`, no slot spent) to honest OOF **0.946178**, +0.000282 over the
new equal-weight champion, with TEX taking the largest fitted weight (1.919) of any leg in the
pool's history -- a strong single-leg signal, not stack-inflation, since a lone dominant weight is
the opposite of what a redundant-legs stack produces. `champion_stack_oof` updated to 0.946178.
This stack has **not** been submitted -- it is a fit-only measurement, exactly like Phase 19's
`de07aba9` re-fit, and stays that way until a slot is deliberately spent on testing whether TEX's
fitted-stack ADD survives contact with the LB the way its solo promotion just did.

**What stays open for next session.** (1) FQ's -0.00008 residual, unexplained and not yet folded
into an existing mechanism. (2) Whether the fixed-weight blend penalty is real or a one-night
artifact (B6's inconclusive wash). (3) Whether extending TEX's commute-ladder idea to `Age` (a TEX2
probe, never run, config-only) recovers more of the frontier's ~61-key coverage advantage --
plausible given TEX's strength was real and the axis was only tested on one of three strong-driver
columns. (4) The fitted 8-leg TEX stack (0.946178 OOF) is unsubmitted and untested against the ADD
gate. `experiments/runs.csv` rows: `5794995c` (FQ), `26ed552b` (B6 blend), `43db301d`/`dad8253f`
(TEX, duplicate pair), `d7121f80` (SKa), `d38d5c54` (SKb), `ec6afd42` (SKc), `67c48e45` (TS2),
`9bbe0c46` (B5 mean), `a03e61c6` (B5f), `b5a318cf` (8-leg TEX stack re-fit, unsubmitted).

### Phase 21 -- coverage generalization closes, the stack washes a fourth time, and the calibration line was the noisy instrument (2026-09-19)

Board re-read before the first slot: 2,336 teams (up from 2,252), top **0.94675**, we stood at
rank 437 on our best-of 0.94635 (TEX). Five wave-1 probes screened free on Kaggle CPU tested
whether TEX's win -- widening TE key COVERAGE, specifically giving `Daily_Commute_km` a
quantisation ladder -- generalizes to other columns and encoders. **It does not.** All five slots
were spent regardless, per the standing "burn the slots" rule, both to close the open leads
decisively and to buy paired points for the calibration-line question Phase 20 flagged.

**Zero-slot work first.** `scripts/refit_gap.py` crashed on a legacy row storing `n_features` as
`"13.0"` -- fixed (`int(float(...))`), committed, harmless to every prior result since it only
affects parsing. Refit before tonight's points (55 archived pairs): all-points fit slope 1.0227,
intercept -0.02131, sigma 0.000100; rich-subset fit (OOF>=0.9459, n=24) slope 0.9974, intercept
0.00262, sigma 0.000061.

**Wave 1, all five strict twins of TEX (105 feats, OOF 0.946122), screened on Kaggle CPU before
any slot was spent:**

| probe | change | feats | OOF | delta vs TEX | best_iter |
|---|---|---|---|---|---|
| TEX2 | Age quantisation ladder ([2,5,10]) added to te_cols | 114 | 0.946109 | -0.000013 | normal (1119-1445) |
| TXWC | window encoder extended income -> income+commute | 111 | 0.946111 | -0.000011 | normal (1022-1472) |
| TXWA | window encoder extended income -> income+age | 111 | 0.946099 | -0.000023 | normal (1098-1505) |
| TEXC | per-value TE added for the 4 columns with no key at all | 113 | 0.946101 | -0.000021 | normal (844-1600) |
| TEXF | WQ2's freq_cols/freq_digit_cols blocks restored | 166 | 0.946125 | +0.000003 | normal (1037-1506) |

**The coverage-generalization hypothesis is refuted, cleanly.** TEX's own commute-ladder win
(Phase 20, +0.000021 OOF) does not repeat for Age via either the ladder (TEX2) or the window
encoder (TXWA), does not repeat for commute via the window encoder (TXWC, redundant with the
ladder that already works there), and does not repeat for the four still-uncovered low-cardinality
columns (TEXC, -0.000021 -- the same magnitude as Phase 1c's C7 on the 16-feature E1 base,
-0.000075, now independently reproduced on a representation four generations removed from it).
Every one of the four sits inside or barely outside the 0.000038 seed floor, `best_iter` normal
throughout -- these are information nulls, not the F1/H2/H3/P3 displacement signature. **The
commute ladder was a real, specific finding about that one column, not a template.** TEXF landed
flat on OOF as its own mechanism predicts (freq_cols are unsupervised value-count features,
historically near-null on OOF alone).

**Five slots, submitted in this order, each pre-registered in `experiments/runs.csv` before
submission:**

| slot | run | OOF | LB | vs TEX (0.94635) | read |
|---|---|---|---|---|---|
| 1 | 8-leg fitted stack (`b5a318cf`, OOF 0.946178) | 0.946178 | **0.94634** | -0.00001 | washes -- 4th confirmation of the stack-penalty pattern |
| 2 | TEXF | 0.946125 | **0.94637** | +0.00002 | inside near-twin resolution (0.000027); see below |
| 3 | TEX2 | 0.946109 | 0.94635 | +0.00000 | ties, despite -0.000013 OOF -- closes the Age-ladder lead |
| 4 | TXWA | 0.946099 | 0.94636 | +0.00001 | ties, despite -0.000023 OOF (the WORST OOF of the five, a top-2 LB) |
| 5 | TEXC | 0.946101 | 0.94635 | +0.00000 | ties -- closes the low-cardinality coverage lead |

**Tonight's five LB scores span 0.00003 -- inside the near-twin paired resolution (0.000027).
They are one score, not five, exactly as playbook section 5 warns.** The clearest illustration:
TXWA has the LOWEST OOF of the batch (0.946099, -0.000023 vs TEX) and a top-2 LB (0.94636); the
8-leg stack has the HIGHEST OOF (0.946178, the best recorded all competition) and the LOWEST LB
of the six (0.94634). **No ranking can be read across this batch** -- the OOF differences being
probed (all within ~0.00003 of each other and of TEX) are simply below this instrument's
resolution floor, and the correct reading is "one score", not "TXWA's window encoder beats
TEXC's TE" or any other pairwise story.

**The stack's fourth wash is the cleanest result of the night.** 0.946178 OOF -> 0.94634 LB,
essentially tied with TEX solo (0.94635) and clearly below the raw-fit prediction (~0.94635).
This is the same pattern as Phase 15's 9-leg inversion, Phase 16b's 8-leg WQ-pool wash, and now a
third pool generation's 8-leg TEX-anchored stack -- three different champion families, the same
verdict. **Phase 20's open lead (4) closes: the fitted stack does not beat its own best leg on the
board**, however high its OOF climbs. `champion_stack_oof` (0.946178) is now a verified, submitted
number rather than a projected one, and it stays unshipped.

**The calibration-line question Phase 20 raised is resolved, and the answer is that the "bias" was
mostly the fit's own noise.** Phase 20 flagged three of its five slots (FQ, the B6 blend, TS2)
landing 0.00001-0.00008 BELOW their predicted LB and deferred the question of whether that was a
real negative bias or the line's own sigma. Re-reading those same five points against tonight's
fully-refit 60-point line (slope 1.0241, intercept -0.02263, sigma 0.000097) instead of the
smaller, staler fit available at the time:

| group | mean residual (refit line) | sign count (of 5) |
|---|---|---|
| Phase 20's 5 points, re-read | **+0.00004** | 1 negative / 5 |
| Phase 21's 5 points | **+0.00006** | 1 negative / 5 |

**Both nights land close to on-line, mildly positive if anything -- the opposite of what the
smaller fit made Phase 20's points look like.** FQ's own residual moved from the -0.00008 Phase 20
read to -0.00000 against the refreshed fit. This is in-sample shrinkage doing exactly what it
should: a regression fit on few, clustered, recent points reads its own noise as a trend, and
adding more data erases most of it. **There is no persistent negative bias.** This also means
Phase 20's headline finding that fe_recipe_score-free freq removal (FQ) cost real LB is weaker
than it looked -- against the better-calibrated line the cost is close to zero, not a confirmed
-0.00008.

**TEXF is a genuinely ambiguous result and is flagged rather than resolved.** It produced our
**best public score of the competition, 0.94637**, moving rank 437 -> **393** (2,338 teams) -- a
real leaderboard gain. But it does not clear the internal promotion bar: OOF is flat vs TEX
(+0.000003, inside the seed floor) and it carries 61 MORE features (166 vs 105), and its LB edge
(+0.00002) sits inside the near-twin paired resolution. Per the standing tiebreak (Phase 16
section 5, Phase 18: OOF is the trusted ranker, and between indistinguishable candidates the repo
takes the simpler model), **TEX remains the champion for future twin ablations.** But TEXF's
direction (restoring freq blocks helps, however slightly) is at least consistent with, not
contradicted by, Phase 20's original (now-weakened) freq-removal finding, and is the one open
thread worth an independent confirmatory twin next session rather than one more coin-flip-sized
read tonight.

**What stays open.** (1) Whether TEXF's freq-block LB edge is real -- needs a second, independent
paired point before it can move the champion designation, regardless of what the public
leaderboard currently shows. (2) The fixed-weight blend penalty question (Phase 20 lead 2) --
untouched tonight, deliberately deprioritized behind the coverage and stack questions. (3)
Coverage as a lever is now closed on every column tested (income, commute, age, the four
remaining low-cardinality columns) -- any future OOF gain needs a genuinely different mechanism,
not more key coverage. `experiments/runs.csv` rows: `c924bc0a` (TEX2), `ad44e7d4` (TXWC, screened
not submitted), `98f5bf60` (TXWA), `9daece89` (TEXC), `1b80818a` (TEXF), `b5a318cf` (8-leg stack,
submitted tonight). `pool.json` unchanged -- no leg swap, no promotion.

### Phase 22 -- the 2-leg blend is a real promotion, not a wash; CatBoost carries its own family offset (2026-09-20/21)

Five slots, all spent, closing both items Phase 21 left open plus a fresh, previously-untested
axis. Four probes screened free on Kaggle CPU (5-concurrent), one computed free locally from
already-archived artifacts, all pre-registered before submission.

**Zero-slot work first.** `scripts/refit_gap.py` refit against 60 de-duplicated points: all-points
fit slope 1.0241, intercept -0.02263, sigma 0.000097 (fit 2); rich-subset fit (OOF>=0.9459, n=29)
slope 1.1225, intercept -0.11568, sigma 0.000060 (fit 3) -- the champion's own local calibration,
now with 5 more points than Phase 21 had.

**The five, screened OOF then submitted in descending order:**

| probe | change | OOF | delta vs base | LB | fit3 pred | fit3 resid |
|---|---|---|---|---|---|---|
| **blend** | fixed logit-mean, TEX + TEXF (2 legs) | **0.946151** | +0.000029 vs TEX | **0.94638** | 0.94637 | +0.00001 |
| TEXbag | TEX + `seed_bag=3` (in-fold model-seed bagging) | 0.946137 | +0.000015 vs TEX | 0.94636 | 0.94636 | +0.00000 |
| TEX2F | TEX2 + freq_cols/freq_digit_cols restored | 0.946124 | +0.000015 vs TEX2 | 0.94636 | 0.94634 | +0.00002 |
| CatTEX | TEX's rep, learner -> CatBoost, no native cats | 0.946114 | -0.000008 vs TEX | 0.94625 | 0.94633 | -0.00008 |
| CatIncome | CatTEX + `Annual_Income_USD` as native CatBoost categorical | 0.945985 | -0.000129 vs CatTEX | 0.94613 | 0.94619 | -0.00006 |

**1. The 2-leg fixed blend is a genuine promotion, and it answers Phase 20's open "fixed-weight
blend penalty" question.** `stack_logit.py --mode logit_mean` on TEX (0.946122/0.94635) + TEXF
(0.946125/0.94637) -- corr high, both champion-family, one freq-block field apart -- computed free
from already-archived OOF/test artifacts, no kernel needed. OOF 0.946151 clears the reopening gate
over TEX solo (+0.000029 >= 0.0000949) and the LB agrees: **0.94638, the best public score of the
competition**, beating both TEX solo (0.94635) and TEXF solo (0.94637). This is the strongest kind
of promotion this repo makes (Phase 20's own standard) -- OOF and LB agree, and it isn't a
tiebreak between indistinguishable candidates, since +0.000029 clears the near-twin resolution
(0.000027). **`champion` moves from `TEX` solo to the fixed 2-leg blend
`logit_mean(TEX=43db301d, TEXF=1b80818a)`, archived as run `61fb5598`.**

The four prior fixed-blend results at 6-8 legs all washed (Phase 15's 9-leg inversion, Phase 16b's
8-leg WQ-pool wash, Phase 20's 6-leg B6 wash, Phase 21's 8-leg TEX-anchored stack wash). Tonight's
2-leg blend does not. **The wash is leg-count-dependent, not an inherent property of averaging
near-twins.** The mechanism this suggests: at 2 legs sharing ~166-105=61 features' worth of
difference (freq blocks), there is still a little uncorrelated fold noise left to cancel; by 6-8
legs drawn from the same representation family, the legs have converged enough that averaging only
dilutes. This reopens small (2-3 leg) fixed blends as a live, cheap lever for future champion
candidates -- but the champion is now itself a blend, so the next "strict twin" probe against it
means re-running both TEX and TEXF's diff and re-blending, not a single retrain.

**2. TEXbag banks the Final-B variance-reduced twin, cleanly.** OOF +0.000015 over TEX solo (F2's
own bagging of E1 bought +0.000034, same order), LB ties TEX solo within near-twin noise
(0.94636 vs 0.94635). Per playbook section 9, Final A = the honest best-OOF champion, Final B =
the same idea with less fitted machinery / lower variance -- `seed_bag=3` is exactly that knob,
and it costs nothing on this pair. Banked for the deadline-week final-slot decision, not submitted
as a promotion attempt.

**3. TEX2F is a second directionally-consistent but still sub-gate confirmation of the freq-block
lead.** Built on TEX2 (Age ladder) rather than TEX, so this is a genuinely independent prediction
vector, not a re-read of TEXF's own residual. Result: OOF flat (+0.000015, matching TEXF's own
+0.000003 -- freq blocks are unsupervised value-count features, historically null on OOF alone),
LB edge over TEX2 solo (0.94636 vs TEX2's known 0.94635) is **+0.00001** -- inside the near-twin
paired resolution (0.000027), same as TEXF's own +0.00002 edge over TEX. Two independent bases,
two small positive edges, neither individually crossing the pre-registered confirmation gate.
**Read together, not singly:** two same-signed near-twin-sized results is weak supportive evidence
for a real small effect, not proof of one -- the honest reading is that if freq blocks carry a real
LB-only edge, it is smaller than this instrument can confirm in one paired point, and a third
independent base would be needed to move past "consistent but unconfirmed."

**4. CatBoost carries its own family-specific LB offset, distinct from LightGBM's, even at matched
OOF -- the first solo CatBoost submission of the competition.** Every prior CatBoost result (E5,
E5r0) was a stack leg only, never submitted solo, so there was no direct CatBoost point on the
OOF->LB line before tonight. CatTEX's OOF (0.946114) is a near-twin of TEX's own (0.946122,
-0.000008) -- by the OOF instrument these are indistinguishable. Their LB scores are not:
**0.94625 vs TEX's 0.94635, a real -0.00010 gap at matched OOF.** Against the rich-subset
calibration line this is -0.00008, -1.4 sigma (against the broader all-points line, -0.00004,
-0.4 sigma -- the two fits disagree on how surprising this is, which is itself informative: the
tight rich-subset line was fit almost entirely on LightGBM-family points, so its sigma may
understate CatBoost's true scatter). **This extends the family-generalization caution (Phase 3's
G1, Phase 19's NE120 -- both early-stopped/high-variance learners) to a normally-trained,
non-early-stopping-pathological GBDT of a different implementation.** The OOF->LB fit's own
disclaimer -- "valid within the GBDT family ONLY" -- was calibrated on LightGBM/XGBoost points
almost exclusively; today's evidence says CatBoost may need its own local offset even though it
is unambiguously still a GBDT. Not chased further tonight (one point is not a slope), but a
second CatBoost point (e.g. a future retrain at a different feature count) would settle whether
this is a fixed offset or noise.

**5. CatIncome closes the native-categorical-income axis for CatBoost too, and the isolation is
clean.** Read against CatTEX (its own same-day, same-family baseline) rather than the generic
calibration line, the LB delta (-0.00012) matches the OOF delta (-0.000129) almost exactly --
confirming this is ordinary information loss, not displacement, extending Phase 20's SK-ladder
finding (LightGBM's native categorical costs smoothly with cardinality, -0.000004/-0.000078/
-0.000342) to a structurally different boosting implementation. **Phase 1c's structural claim --
the per-value lookup is already fully extracted by the target encoder, and no downstream
mechanism recovers more from the raw column -- now holds for both boosting families tested.**

**What stays open.** (1) The freq-block lead is still short of confirmed after two independent
near-twin-sized positive reads (TEXF, TEX2F) -- a third base or a larger-effect isolation would be
needed to settle it either way. (2) CatBoost's own -0.00010 LB gap at matched OOF is one point, not
a slope -- worth a second CatBoost submission before treating it as a fixed family offset. (3) The
champion is now a 2-model fixed blend rather than a single trainable config, which changes what a
"strict twin" of it means going forward -- future probes need to diff one side of the pair (TEX or
TEXF's own recipe) and re-blend, or the comparison is confounded. `experiments/runs.csv` rows:
`61fb5598` (blend, new champion), `50d84171` (TEXbag), `f0ba7251` (TEX2F), `3da3ad60` (CatTEX),
`74c16e6e` (CatIncome). `pool.json` unchanged -- the curated 8-leg stack's own leg list is a
separate question from the solo/blend champion tracked here.

**Another duplicate-archive artifact, same class as Phase 18/19/20's.** An early attempt to
background the TEX2F kernel poll raced the harness's own process tracking and got killed
mid-flight, but not before it had already archived independently: `f59fab26` carries TEX2F's same
OOF (0.946124) with no LB score, three seconds after the correctly-tracked retry archived the
identical kernel as `f0ba7251` -- the run actually passed to `submit_run.py`. **De-duplicate on
`run_tag=TEX2F`, keeping `f0ba7251`.**

### Phase 23 -- the artifact policy opens, and the public axis is real but three times smaller than OOF says (2026-09-21/22)

Board re-read before the first slot: **2,648 teams**, and a new #1 at **0.94827**, +0.00152 clear
of an unchanged #2 (0.94675). One team, one jump, nobody following -- the shape of a leak or of
public-LB probing, not of a mechanism. Noted, not chased. We stood at rank **448** on 0.94638.

**Zero-slot work first, and it decided the whole night.**

*1. The public frontier was inspected directly and has nothing mechanistic for us.* Four of the
day's kernels were pulled and read. Marc Maldonado's "the generator remembers the original rows"
is a genuinely good notebook -- and its best model is **OOF 0.94596, below our 0.946151**. Its
headline lever, `max_bin` 255 -> 4095 (+0.00178 on a raw 13-column model), is the one we closed as
a dead null in Phase 12 (P2, +0.000002) and again in Phase 18 (LWb255), for the reason its own
ablation shows: on top of target encoding the same knob is worth +0.00020, and our TE is richer
than theirs. Its `% 1000` digit family: +0.00001. The "breaking ties wins" guide is closed in one
line -- **our champion submission has 286,571 unique values in 286,571 rows. Zero ties.** Worth
exactly 0.00000, as AUC rank-invariance says it must be.

*2. The community froze the same CV split we did.* `dariushafshar/s6e9-golem-oof-library` ships
`folds_seed42.npy`, and it is **bit-identical** to this repo's `StratifiedKFold(5, shuffle=True,
random_state=42)` over `train.csv` in original row order. Public OOF arrays therefore stack with
ours row-for-row. Alignment is not what makes a *fixed* blend honest -- nothing is fitted, and every
leg's OOF prediction for a row comes from a model that never saw it -- but it means a fitted stack
over these legs would be fold-honest too, which is why the top of the board is stacking them.

*3. The ours-only blend axis was enumerated exhaustively and is SATURATED.* 6,461 fixed logit-mean
blends over 14 distinct champion-family recipes, k = 2..6:

| legs | best OOF | vs champion 0.946151 |
|---|---|---|
| k=2 | 0.946189 | +0.000038 |
| k=3 | 0.946194 | +0.000043 |
| k=4 | 0.946196 | +0.000045 |
| k=5 | 0.946196 | +0.000045 |
| k=6 | 0.946196 | +0.000045 |

**It is a plateau, not a hump, and its ceiling is half the shipping gate.** Phase 22's reading --
"2 legs gain, 6-8 legs wash" -- does not survive full enumeration: there is no wash by leg count at
all. The four prior washes were all *fitted* stacks; **the wash is a property of FITTING, not of
leg count**, and Phase 22's leg-count mechanism is hereby corrected. `CatTEX` appears in all ten
top blends -- the CatBoost leg is the only real diversity our own pool contains. This enumeration
cost zero slots and saved two.

**The five slots.**

| slot | model | OOF | vs champ | LB | read |
|---|---|---|---|---|---|
| 1 | ours-only `TEX+TEX2F+CatTEX+E4r0` | 0.946196 | +0.000045 | **0.94634** | CatBoost offset propagates |
| 2 | `OURS*0.50 + 6view+rmlp` | 0.946329 | +0.000178 | **0.94644** | public axis is real |
| 3 | `OURS*0.34 + 6view+rmlp` | 0.946374 | +0.000223 | **0.94646** | best of the night |
| 4 | `OURS*0.25 + 6view+rmlp` | 0.946393 | +0.000243 | **0.94645** | turns over |
| 5 | `OURS*0.50 + xgb5f` **(control)** | 0.946255 | +0.000105 | **0.94644** | ties slot 2 |

**1. The public axis is real, and it is the first LB gain since Phase 22.** All four public-mixed
blends beat the ours-only champion (0.94638). Best: slot 3 at **0.94646**, rank **448 -> 261**,
**+187 places**. `champion` moves to the public-mixed blend for **Final B** purposes; Final A stays
ours-only per the §5 hedge.

**2. But OOF overstates this axis by about 3x, and the control is what proves it.** OOF promised
+0.000178 to +0.000243; the LB delivered +0.00006 to +0.00008. The decisive point is slot 5:
**matched to slot 2 on our own weight (0.50) but with no 10-fold leg anywhere in it**, it scores
OOF 0.946255 against slot 2's 0.946329 -- a **+0.000073 OOF deficit that bought exactly 0.00000 LB**
(both 0.94644). The pre-registered fork said that outcome means fold-count inflation explains the
extra OOF, and it does. **The ~0.00018 10-fold discount measured on our own `e1495238` (OOF
0.946122 -> LB 0.94617, where TEX at the IDENTICAL OOF scored 0.94635) reproduces on public
artifacts.** Any future public leg must be read at its own fold count, and a 5-fold-honest public
leg is worth as much as a 10-fold one that looks 0.00007 better on OOF.

**3. The dose-response is flat, and must be read as one score, not three.** Slots 2/3/4 span
0.94644-0.94646 -- **0.00002, inside the near-twin paired resolution (0.000027)**. Per playbook §5
and Phase 21's own lesson, the honest reading is that public share anywhere in 0.50-0.75 buys the
same thing; the apparent interior optimum at w=0.34 is not resolvable and **no weight tuning should
be built on it**. What *is* resolved is the step from 0.00 to 0.50 public share: +0.00006, roughly
2x the near-twin resolution.

**4. CatBoost's family offset propagates through a fixed blend.** Slot 1 put `CatTEX` in at 1/4
weight, gained +0.000045 OOF, and **lost 0.00004 LB** (0.94634 vs the champion's 0.94638) -- landing
in the pre-registered "offset propagates" branch. Phase 22's -0.00010 solo offset is therefore a
property of the predictions, not of the solo submission, and it survives averaging at roughly its
weighted share. **CatTEX is not a free diversity leg**, whatever the ours-only enumeration's OOF
says about it, and this is now the second measured case (with the 10-fold discount above) of our
OOF ranking a leg that the LB then refuses.

**What stays open.** (1) Final B's exact recipe -- tonight shows public share 0.50-0.75 is a
plateau, so the choice should be made on the *5-fold-honest* legs it contains, not on OOF-max.
(2) A 5-fold-only multi-leg public blend is untested and is the natural next probe, since slot 5
showed a single 5-fold leg already matches two 10-fold ones. (3) Final A is now capped: the
ours-only axis is enumerated and saturated at +0.000045 OOF, below gate, so Final A stays
`blend(TEX, TEXF)` (0.946151 / 0.94638) unless a genuinely new ours-only mechanism appears.
(4) The #1 at 0.94827 is unexplained. `experiments/runs.csv` rows: `7844a725` (slot 1),
`413d78a3` (slot 2), `3096f75f` (slot 3), `e685c35f` (slot 4), `4b46b9dd` (slot 5, the control).

### Phase 24 -- the fold-honesty question, and five public legs that were never read (2026-09-22/23)

Phase 23 left one lead marked as the natural next probe: *"a 5-fold-only multi-leg public blend is
untested."* Its own control slot is why -- a blend carrying one **5-fold** public leg (`xgb5f`)
tied a blend carrying two **10-fold** legs (both 0.94644) while sitting **-0.000073 on OOF**, which
reproduced on public artifacts the ~0.00018 10-fold discount already measured on our own
`e1495238`. That made the discount a *description*. Tonight asks whether it is a **predictor**:
does a fold-honest public blend that OOF ranks BELOW the 10-fold one actually beat it on the board?
The answer decides Final B's recipe, which is the last open modelling decision here -- Final A is
fixed and the ours-only axis is enumerated and saturated (Phase 23).

**Five zero-slot findings, measured before any slot was spent. Two of them change what gets
submitted.**

**1. Five 5-fold-honest public legs were sitting in `data/public/` unregistered and unread.**
`public_blend.py`'s REGISTRY carried only `xgb5f` and `lgbV5` out of `najiama/s6e9-oof`. Solo OOF
against `train.csv` labels on our frozen split:

| leg | solo OOF | folds | corr vs OURS |
|---|---|---|---|
| `lgbV5` (`Pure LGBM_V5`) | **0.946170** | 5 | 0.99624 |
| `xgb5f` (`XGBoost_Triple_TE_5folds`) | 0.946142 | 5 | 0.99720 |
| `lgbV6` | 0.946068 | 5 | 0.99641 |
| `lgbV3` | 0.946064 | 5 | 0.99634 |
| `lgbV1` | 0.945866 | 5 | 0.99543 |
| *(OURS `61fb5598`)* | *0.946151* | *5* | -- |

`lgbV5` alone **outscores our own champion at matched fold count**, and nothing in the repo had
ever read it.

**2. The fold-honest public side saturates at TWO legs.** Equal-weight public side, ours-weight
0.34: `xgb5f+lgbV5` **0.946305**; `+lgbV3` 0.946284; `+lgbV6` 0.946270; `+lgbV1` 0.946268.
Monotone decreasing -- the `Pure LGBM` V-series are near-twins of each other. Settled without a
slot; two legs is the recipe.

**3. The public legs are percentile RANKS and ours are probabilities, so Phase 23's nominal weights
were never its effective weights.** Logit SDs: OURS **3.179**, `6view` **1.814**, `rmlp` **1.806**,
`xgb5f` **3.150**. Under `logit_mean`, "0.34 ours / 0.66 public" weighted our side 0.34x3.18 = 1.08
against 0.66x1.81 = 1.20 -- an **effective ~47/53, not 34/66**; at w=0.50 it was ~64/36. Phase 23's
"flat plateau over public share 0.50-0.75" was therefore measured across an effective range of only
~36-64%, which is a smaller span than it looked and partly explains why it read as flat.

Switching to `rank_mean` (scale-free) is worth **+0.000039 / +0.000021 / +0.000008** OOF at
w = 0.50 / 0.34 / 0.25 -- real, correctly signed (largest where the mismatch is worst), and **below
the +0.0000949 shipping gate**. This is a parameterization correction, not a free win. For a blend
whose legs are all in probability space it is a **no-op** (0.946305 logit vs 0.946304 rank), which
is exactly what leaves tonight's slot 1 unconfounded.

**4. `Sergey_LGBM_oof.csv` is structurally unsafe under a logit combiner, and this is the
load-bearing reason the default flips.** Raw AUC 0.945329; after `logit(clip(p, 1e-6, 1-1e-6))` it
collapses to **0.872301**. Its shipped values are not calibrated probabilities, and the clip
silently ties a large block of rows -- a 0.073 AUC loss that no assert in the repo would have
caught. A rank combiner is immune to whatever probability geometry a public file happens to arrive
with. `public_blend.py` now defaults to `rank_mean` and, in `logit_mean` mode only, refuses any leg
whose logit-clipped AUC differs from its raw AUC by more than 1e-4.

**5. `dariushafshar/s6e9-golem-oof-library` is 19 members on our exact frozen 5-fold split, and is
below the pool floor.** Best member `h` = 0.944507, 0.0017 under our champion; its three strongest
(`a`, `h`, `i`) self-disclose early stopping on the held-out validation fold, so their OOF is
optimistic on top of being weak. Correlation vs OURS 0.984 -- genuinely looser than the 0.996-0.997
public pack, so it is the most decorrelated public material available -- but playbook section 6 says
a leg below the pool floor contributes nothing however decorrelated (G1, Phase 2b). Offline ADD
test only; no slot.

**6. The rank combiner introduces ties, and they cost exactly nothing -- checked rather than
assumed, because the tie density differs per slot and would otherwise have been a differential
bias across the very comparisons tonight is built to make.** `logit_mean` on distinct
probabilities is tie-free; `rank_mean` collides by construction, because a weighted average of
percentile ranks lands on a lattice. Unique OOF values out of 668,665, and the gain from breaking
every tie with an infinitesimal logit-blend perturbation:

| blend | unique | OOF | tie-broken | gain |
|---|---|---|---|---|
| s1 `0.34o + 0.66[x5,v5]` | 666,638 | 0.946304 | 0.946304 | **-0.000000** |
| s2 `0.34o + 0.66[6v,rm]` | 667,608 | 0.946395 | 0.946395 | **+0.000000** |
| s3 `0.25o + 0.75[4 legs]` | 663,148 | 0.946388 | 0.946388 | **-0.000000** |
| s4 `0.50o + 0.50[x5,v5]` | 619,774 | 0.946297 | 0.946297 | **+0.000000** |
| s5 public only `[x5,v5]` | 554,468 | 0.946249 | 0.946249 | **+0.000000** |

Zero at six decimal places even at slot 5, where **17% of rows are tied**. *Mechanism:* a tie only
forms between rows the blend already scores near-identically, so resolving it is a coin flip and
AUC's own tie handling (counting a tied pair as 0.5) is already the correct expected value.
Breaking them buys the variance of a coin flip, not a gain. This is Phase 23's "breaking ties wins
is worth 0.00000" verdict re-confirmed in the regime where ties actually exist -- there it was
vacuous, because our champion had none. **No tiebreak is applied and none is needed**, and the
five slots are not differentially biased by their tie densities.

**The five slots, submitted in order, every one pre-registered in `experiments/runs.csv` before
submission (commit `23a29c4`), and every OOF verified against the plan to 6 dp before any slot was
spent.**

| slot | recipe (all `rank_mean`) | OOF | LB | pre-registered gate | verdict |
|---|---|---|---|---|---|
| 1 | `0.34 OURS + 0.66[xgb5f, lgbV5]` (5-fold) | 0.946304 | **0.94644** | >=0.94646 confirms; <=0.94643 falsifies | **NO RESOLUTION** |
| 2 | `0.34 OURS + 0.66[6view, rmlp]` (combiner control) | 0.946395 | **0.94645** | tie expected | **tie -- closed** |
| 3 | `0.25 OURS + 0.75[6view, rmlp, xgb5f, lgbV5]` | 0.946388 | **0.94646** | >0.94646 => new Final B | **ties, does not exceed** |
| 4 | `0.50 OURS + 0.50[xgb5f, lgbV5]` (5-fold) | 0.946297 | **0.94644** | >=0.94644 | **clears** |
| 5 | `[xgb5f, lgbV5]`, **no ours** (the anchor) | 0.946249 | **0.94639** | -- | see below |

**1. H1 is NOT promoted, and the pre-registration is what makes that call rather than hindsight.**
Slot 1 landed at 0.94644 against Phase 23 slot 3's 0.94646 -- a -0.00002 difference, squarely inside
the pre-registered **no-resolution band** (the 0.000027 near-twin paired SD). The gate demanded
>=0.94646 to confirm and <=0.94643 to falsify; it delivered neither. Slot 4 cleared its own gate
(0.94644, tying **both** of its paired Phase-23 points exactly). The joint condition was *both*, so
per the pre-registration **the 10-fold discount stays a description and is not promoted to a
predictor.** It would have been easy to read slot 1 as a win -- it carries a **-0.000070 OOF
deficit** and gave up only 0.00002 on the board -- and playbook section 5 exists precisely to stop
that reading.

**2. What tonight actually establishes is stronger than what it was testing: the public axis is ONE
BIT, and we already have it.** Four blends at public shares 0.50 / 0.66 / 0.75, with 5-fold legs,
10-fold legs and mixtures of both, under two different combiners, span **0.94644-0.94646**. Their
OOFs span 0.946297-0.946395, a range **four times larger**. Phase 23 found the *dose* flat; tonight
extends that to **leg identity, fold count and combiner**. Everything with public content in it
scores 0.94644-0.94646; our best ours-only model scores 0.94638. **The public axis is a single step
of +0.00006-0.00008 and it saturates immediately** -- there is no composition of these artifacts
left to search, and no further slot should be spent searching it.

**3. The anchor is the most informative point of the night, and it reframes the mechanism.** Slot 5
is two public legs we did not train, with nothing of ours in it: **OOF 0.946249 -> LB 0.94639**,
against our own ours-only champion's **OOF 0.946151 -> LB 0.94638**. At **matched fold count** the
public side is +0.000098 on OOF and **+0.00001 on the board** -- inside near-twin resolution. The
rich-subset calibration line predicted +0.00011. So the OOF advantage evaporates even with fold
count held fixed, which means **the effect was never specifically 10-fold inflation.**

The residual table makes it unmissable. Against both refit lines, **every public-content blend sits
below the line and every one of our own runs sits above it:**

| run | OOF | LB | resid (fit 2) |
|---|---|---|---|
| blend (P23 s4, 10-fold) | 0.946393 | 0.94645 | **-0.00014** |
| blend (P23 s3, 10-fold) | 0.946374 | 0.94646 | **-0.00011** |
| blend (P23 s2, 10-fold) | 0.946329 | 0.94644 | **-0.00009** |
| blend (P24 s5, **5-fold, public-only**) | 0.946249 | 0.94639 | **-0.00006** |
| **OURS champion `61fb5598`** | 0.946151 | 0.94638 | **+0.00003** |
| TEXbag / TEXF / TEX2F / TEX / TEX2 / TEXC / TXWA | ~0.9461 | 0.9463-0.9464 | **+0.00003 to +0.00009** |

A clean sign split with no overlap, and the negative residual grows with how much public content the
blend carries. **The OOF->LB line is a property of the SOURCE, not only of the model family.** Our
own artifacts sit on one line; outside artifacts sit on another roughly 0.0001 below it at matched
OOF, whatever their fold count. This is the same class of finding as Phase 3's G1 (family), Phase 22's
CatBoost offset (implementation) and Phase 23's fold-count discount -- four instances now, and the
general rule they share is: **an OOF produced by a pipeline we did not calibrate does not enter our
instrument at face value.**

**4. The combiner question is closed, and it closed the way the instrument said it would.** Slot 2
is Phase 23 slot 3's strict twin in one field (`logit_mean` -> `rank_mean`): OOF +0.000021, **below
the +0.0000949 shipping gate**; LB 0.94645 vs 0.94646, **-0.00001, below the near-twin resolution**.
A sub-gate OOF move produced a sub-resolution LB move -- the instrument behaving exactly as
specified. `rank_mean` is kept as the default because it is the *correct parameterization* (it is
scale-free, so the weights mean what they say, and it cannot be silently destroyed by a leg that
ships something other than calibrated probabilities -- finding 4), **not because it buys anything.
It buys 0.00000.**

**5. Final B does not move.** Slot 3 tied the competition's best public score at 0.94646 but the gate
required strictly greater, so **Final B stays Phase 23 slot 3 (`3096f75f`)** and **Final A stays
`61fb5598`** (ours-only `blend(TEX, TEXF)`, 0.94638). One thing is worth carrying to the deadline
rather than deciding tonight: slot 3 (`1184af71`) reaches the *same* 0.94646 with **half its public
side fold-honest and four legs instead of two**, which is a genuine playbook section 9 Final-B
argument -- same score, less concentrated provenance -- and it is the only such argument on record.
It is flagged, not promoted.

**6. The calibration line's local slope is not stable and must not be extrapolated.** `refit_gap.py`
over 70 de-duplicated points: fit 2 (all) slope 1.0175, intercept -0.01632, sigma 0.000094; fit 3
(rich, OOF>=0.9459, n=39) slope **0.8524**, intercept 0.13986, sigma 0.000060. The rich-subset slope
has now read 0.9974 (n=24, Phase 21), 1.1225 (n=29, Phase 22) and 0.8524 (n=39, tonight) across
three successive refits. **That is not a moving truth, it is a noise-dominated estimate** -- Phase 21
already found this fit reads its own noise as a trend on few clustered points, and adding 10 more
points has now swung it the other way. The frozen Phase-3 values in section 4 (slope 1.0832, sigma
0.000103) and the shipping gate derived from them stay exactly as they are, per section 4's standing
rule. **Do not use the local slope to predict an LB score.**

**7. The golem library is closed by ADD test, zero slots** (finding 5's prediction confirmed).
Against slot 1's recipe as the pool, at a 1/3 public share and at a light 10% dose:

| member | solo OOF | ADD (1/3 of public share) | ADD (10% dose) |
|---|---|---|---|
| `h` | 0.944507 | -0.000045 | +0.000004 |
| `i` | 0.944407 | -0.000054 | +0.000001 |
| `a` | 0.943742 | -0.000100 | -0.000012 |
| `d_catboost` | 0.941672 | -0.000230 | -0.000050 |
| `h_spline_gam` | 0.939913 | -0.000296 | -0.000061 |
| `g_mlp` | 0.938375 | -0.000373 | -0.000080 |
| `j_logreg` | 0.938094 | -0.000379 | -0.000080 |

Null to negative everywhere, and **monotone in solo strength** -- the contribution tracks how good
the leg is, not how decorrelated it is, even though this is the most decorrelated public material
available (corr 0.984 against the pack's 0.996-0.997). The best case, `h` at a 10% dose, is
**+0.000004 = one tenth of the seed-noise floor.** This is playbook section 7's rule firing again:
*high disagreement is weakness, not diversity, unless it comes with competitive solo strength.*
**19 members, zero usable, no slot spent.**

**What stays open.** (1) Nothing on the public axis -- it is one bit, measured at four shares, two
fold counts, two combiners and 21 candidate legs, and it is saturated at 0.94646. Our best public
score is unchanged from Phase 23. (2) Final A is capped by Phase 23's ours-only enumeration; only a
genuinely new ours-only mechanism would move it, and six phases of search have found none. (3) The
source-offset finding in point 3 is the one genuinely new mechanism tonight produced, and it is the
reason no further public artifact should be admitted on the strength of its OOF alone. (4) The #1 at
0.94945 (up from 0.94827) remains unexplained and un-chased. `experiments/runs.csv` rows:
`e6db88da` (s1), `b0eaa810` (s2), `1184af71` (s3), `7c4cdc76` (s4), `44e81167` (s5, the anchor).

### Phase 25 -- the public axis survives an exhaustive leg-level test; the slots turn to variance (2026-09-24)

Phase 24 closed the public axis as "one bit" on the strength of 4 shares x 2 fold counts x 2
combiners. That claim had a real hole in it: **every one of those blends used megayak's pre-made
`ensemble` column as a single unit.** The six individual views A-F were never enumerated, najiama's
V-series was never mixed with the 10-fold legs, and the 10-fold XGBoost twin was never loaded at
all. megayak's own card says D and F add *despite being weaker* ("they see the data differently",
F the loosest at 0.99761), which is exactly the case a library-level blend cannot express. So the
axis was closed by extrapolation from five points, not by measurement.

**It is now closed by measurement. 3,282 fixed `rank_mean` blends, 14 public legs, k = 1..4, three
weights, zero slots.**

| public leg | folds | solo OOF | | public leg | folds | solo OOF |
|---|---|---|---|---|---|---|
| `ENS` (megayak ensemble) | 10 | 0.946345 | | `V5` | **5** | 0.946170 |
| `A` | 10 | 0.946281 | | `xgb5f` | **5** | 0.946142 |
| `B` | 10 | 0.946258 | | `F` | 10 | 0.946134 |
| `xgb10f` | 10 | 0.946243 | | `D` | 10 | 0.946077 |
| `E` | 10 | 0.946230 | | `V6` | **5** | 0.946068 |
| `C` | 10 | 0.946223 | | `V3` | **5** | 0.946064 |
| `rmlp` | 10 | 0.946182 | | `V1` | **5** | 0.945866 |

**The ceiling over all 3,282 is 0.946404** (`0.25 OURS + 0.75[B, D, rmlp, xgb10f]`). The blend we
already shipped as Phase 24 slot 3 scores 0.946388. **Decomposing the library into its six views,
adding a 10-fold XGBoost and four more LightGBM legs, and searching every subset up to size four
buys +0.000016 OOF** -- one sixth of the shipping gate and less than half the 0.000038 seed-noise
floor. Phase 24's "one bit" verdict survives the test that could have refuted it.

**And OOF's preference is diagnostic rather than useful.** Every one of the top 25 blends is at
`w = 0.25`, the heaviest public dose searched, and **24 of the top 25 contain no fold-honest leg at
all.** Sorted by fold-honesty of the public side, the OOF ceiling falls monotonically:

| public side | best OOF | vs champion |
|---|---|---|
| 0% fold-honest (`B+D+rmlp+xgb10f`) | 0.946404 | +0.000254 |
| 50% (`ENS+rmlp+xgb5f+V5`) | 0.946388 | +0.000237 |
| 75% (`rmlp+xgb5f+V5+V6`) | 0.946344 | +0.000194 |
| 100% (`xgb5f+V5`) | 0.946304 | +0.000153 |

That is the source offset of Phase 24 point 3 drawn as a curve: **our OOF pays for 10-fold content
at about +0.00005 per quarter of the public side, and the board does not.** Two of tonight's slots
are a matched pair on exactly this, at identical weight.

**The ours-only side, measured free from the archive.** Playbook section 9's Final B is the
*variance-reduced* twin -- the same idea with fitted machinery removed -- and it is chosen for that
property, not its score. `TEXbag` (Phase 22, `seed_bag=3`) is that knob and has never been blended:

| ours-only blend | OOF | vs Final A |
|---|---|---|
| `blend(TEX, TEXF)` = `61fb5598`, **today's Final A** | 0.946150 | -- |
| `blend(TEXbag, TEXF)` | 0.946154 | +0.000004 |
| `blend(TEX, TEXF, TEX2F)` | 0.946156 | +0.000006 |
| **`blend(TEXbag, TEXF, TEX2F)`** | **0.946158** | **+0.000008** |

All four are one score -- every gap is inside the seed floor, as three near-twins of the same recipe
must be. The reason to prefer the last one is not the +0.000008: it is **three legs instead of two,
one of them already averaged over three model seeds**, which is strictly less prediction variance
for an unseen split at no measured cost. Per playbook section 9 you can only *select* what you
*submitted*, and this has never been submitted -- which is what makes it worth a slot rather than a
line in this file.

**The five slots, each pre-registered in `experiments/runs.csv` before submission (commit
`ea0cdc2`), every OOF verified against the enumeration before any slot was spent.**

| slot | model | OOF | LB | pre-registered gate | verdict |
|---|---|---|---|---|---|
| 1 | ours-only `rank_mean(TEXbag, TEXF, TEX2F)` | 0.946158 | **0.94638** | >=0.94635 => adopt | **CLEARS -- new Final A** |
| 2 | `0.34 OURS3 + 0.66[6view, rmlp, xgb5f, lgbV5]` | 0.946378 | **0.94646** | >=0.94645 => adopt | **CLEARS -- new Final B** |
| 3 | pure public **10-fold** `[6view, rmlp]`, no ours | 0.946383 | **0.94641** | (a) <=0.94641 / (b) >=0.94648 / (c) between | **(a) fires** |
| 4 | `0.25 OURS + 0.75[sixB, sixD, rmlp, xgb10f]` -- OOF-max of 3,282 | 0.946404 | **0.94646** | >0.94646 => new best | **ties, no new best** |
| 5 | `0.25 OURS + 0.75[rmlp, xgb5f, lgbV5, lgbV6]` -- 75% fold-honest | 0.946344 | **0.94646** | read only as s4 - s5 | **s4 - s5 = 0.00000** |

**1. Slot 3 settles the fold question, and it is the first UNDILUTED measurement of it in this
competition.** Its matched arm is Phase 24 slot 5. Both are pure public, both `rank_mean`, both two
legs, and **neither contains a single prediction we trained** -- so the only difference between them
is the fold count of the legs:

| arm | legs | folds | OOF | LB |
|---|---|---|---|---|
| Phase 24 slot 5 (`44e81167`) | `xgb5f + lgbV5` | **5** | 0.946249 | 0.94639 |
| **Phase 25 slot 3 (`99d6eaf9`)** | `6view + rmlp` | **10** | 0.946383 | **0.94641** |
| | | | **+0.000134** | **+0.00002** |

**A +0.000134 OOF advantage bought +0.00002 on the board: a transfer ratio of about 15%.** Branch
(a) of the pre-registered fork fires -- the gain does not transfer -- and the implied inflation for
a 5 -> 10 fold step is **~0.000114**, the same order as the ~0.00018 measured on our own
`e1495238` and now measured a second time by a completely independent route. Every previous read of
this effect was on a blend diluted with our own predictions; this one is not, which is why it is
worth more than the four before it.

**2. Slots 4 and 5 are the matched pair, and they confirm the source offset at identical weight.**
Same ours run, same ours-weight (0.25), same number of public legs (4), differing **only** in fold
composition -- 0% fold-honest against 75%:

| | public side | fold-honest | OOF | LB |
|---|---|---|---|---|
| slot 4 | `sixB + sixD + rmlp + xgb10f` | 0/4 | 0.946404 | 0.94646 |
| slot 5 | `rmlp + xgb5f + lgbV5 + lgbV6` | 3/4 | 0.946344 | 0.94646 |
| | | | **+0.000060** | **0.00000** |

The pre-registration said `s4 - s5 <= 0` confirms the offset and `~+0.00006` refutes it.
**It is exactly 0.00000.** Our OOF pays about +0.00005 per quarter of the public side converted to
10-fold content and **the board pays nothing at all.** Together with slot 3 this is two independent,
separately pre-registered tests landing on the same verdict in one night, on top of Phase 23 slot 5
and Phase 24's residual sign-split. **The effect is established; it should not be probed again.**

**3. Slot 4 closes the public axis on the board, not just offline.** It is the OOF maximum over all
**3,282** enumerated blends -- every subset of 14 public legs up to size four, at three weights,
with megayak's library taken apart into its six views. It scores **0.94646**: identical to Phase 23
slot 3, Phase 24 slot 3, and both of tonight's other mixed blends. **Exhaustive OOF search over the
entire public axis buys 0.00000.**

And the ceiling is hard. Tonight's three mixed blends span OOF **0.946344-0.946404** across three
structurally different compositions and **all three score 0.94646** -- as did two earlier blends at
yet other compositions. Five different recipes, five different OOFs, one leaderboard score.
**0.94646 is a wall, and it is where this axis ends.**

**4. Both finals are upgraded, at zero cost in public score and strictly lower variance.** This is
playbook section 9's pattern doing exactly what S6E8 measured it doing -- the hedge is free.

- **Final A moves to `d8ef7c11`**: ours-only `rank_mean(TEXbag, TEXF, TEX2F)`, OOF 0.946158,
  **LB 0.94638 -- tying the outgoing `61fb5598` exactly.** Three legs instead of two, one of them
  already averaged over three model seeds. Adopted on the variance property, not on the +0.000008
  OOF, which is inside the seed floor and is not the argument.
- **Final B moves to `2e2c756d`**: `0.34 OURS3 + 0.66[6view, rmlp, xgb5f, lgbV5]`, OOF 0.946378,
  **LB 0.94646 -- tying the outgoing `3096f75f` exactly.** Four public legs instead of two with
  **half of them fold-honest**, over the three-leg seed-bagged ours side, under the correct
  scale-free combiner. Same score, more legs on both sides, less concentrated provenance. This is
  the candidate Phase 24 point 5 flagged and declined to promote; it now has its own paired point.

The README section 5 hedge is unchanged in kind -- Final A contains no public artifact at any
weight, Final B carries the public mix -- and both sides of it are now the lower-variance member of
their own family.

**What stays open.** (1) **Nothing on the public axis.** It is saturated at 0.94646, confirmed by
3,282 offline blends and five board points at five different compositions. No further slot should be
spent on it, and that includes the remaining nights. (2) The fold/source offset is established by
two undiluted pre-registered tests and needs no further probing. (3) Final A remains capped by Phase
23's ours-only enumeration; only a genuinely new ours-only mechanism would move it, and seven phases
have found none. (4) The #1 at 0.94945 is still unexplained and still un-chased. With six nights of
slots left and every axis closed, the honest remaining use of a slot is a paired point, not a
search. `experiments/runs.csv` rows: `d8ef7c11` (s1, **Final A**), `2e2c756d` (s2, **Final B**),
`99d6eaf9` (s3), `39873a34` (s4), `0e2fa556` (s5).

### Phase 26 -- pre-registration: chasing the 0.94945 (2026-09-24, before any probe ran)

Every axis this repo has opened is closed (Phase 25). The leaderboard at 2026-09-24 23:18 UTC:

| rank | team | public | note |
|---|---|---|---|
| 1 | Team Alicia | **0.94945** | submitted 23:08 today |
| 2 | Prior | 0.94676 | |
| 3 | Chris Deotte | 0.94672 | Kaggle GM |
| -- | *our Final B* | *0.94646* | the wall, five recipes deep |

**+0.00269 clear of a Grandmaster, and no public notebook above 0.94656** (`kaggle kernels list
--sort-by scoreDescending`: the entire public frontier is 0.9465x blend-tracking). A gap that size,
that isolated, is an *information* difference and not a tuning difference -- and Phase 25 left it as
the one thing still un-chased with six nights of slots in hand. Tonight chases it. Five probes, all
offline, all pre-registered here before any of them ran.

**Z1 -- the `id` axis.** HYPOTHESIS: the generator emitted rows in an order carrying target
information (batch or curriculum drift), so `id` holds signal no recipe here has ever read -- `id`
has never appeared in a probe. MECHANISM: if generation is sequential over a drifting latent, the
per-id-block target rate departs from the global rate beyond binomial noise. GATE to open the axis:
single-feature AUC of `id` >= 0.505, **or** id-block rate SD > 2x binomial expectation. Otherwise
closed in one line.

**Z2 -- the parent-cluster leak.** HYPOTHESIS: all 955,236 synthetic rows descend from the 10,000
rows of `EV_Adoption_and_Range_Anxiety_Dataset.csv`; if generation is per-row conditional, parent
identity is a latent variable and the parent's own `Will_Buy_EV` is a noisy read of it -- signal that
lives *outside* the 13 features and therefore outside everything measured so far. MECHANISM: Phase 1
closed this dataset on **coverage** (exact feature-tuple match covers 82 test rows) and never tested
**nearest-parent** at all, which is a different claim. GATE: mean champion-OOF residual must differ
between nearest-parent-label=Yes and =No by more than 3 SE, with the SE from the actual group sizes.
Otherwise the original dataset is closed for good, on mechanism and not just on coverage.

**Z3 -- the ceiling.** HYPOTHESIS: 0.94945 is reachable only if the true log-odds field is
materially more dispersed than our model's. MECHANISM: for a perfectly calibrated model, AUC is a
functional of the distribution of true `p` alone. Simulate `y ~ Bernoulli(p_oof)` and measure
`AUC(p_oof, y_sim)`: that is the AUC a *perfect* model reaches on labels whose probability field
looks like ours. Then solve for the log-odds scale `s` at which `s * our_logit` would reach 0.94945
under its own labels. No gate -- this is the diagnostic that says whether the 0.00269 is a modelling
gap or an information gap, and it is the number that decides how the remaining five nights are spent.

**Z4 -- the joint-key lookup, with a permutation null this time.** HYPOTHESIS: Phase 1's
nine-joint-key test ran on C2 (OOF 0.945225) and correctly read its largest-of-nine as selection;
at champion strength, with keys fixed in advance and a permutation null instead of a binomial one,
a real 2-way lookup may survive. MECHANISM: the same per-key residual-variance ratio, but the null
comes from permuting the key labels within the realized model predictions, which absorbs exactly the
anticorrelation that pushed Phase 1's ratios below 1.0 and made them hard to read. GATE: ratio must
exceed the permutation null's 99th percentile on a key named before the number is seen.

**Z5 -- test-augmented (pseudo-label) target encoding.** HYPOTHESIS: the income lookup's precision
is bounded by ~50 train rows per value; pseudo-labelling the 286,571 test rows adds ~43% more
support per value, sharpening the lookup precisely where backoff currently dominates. MECHANISM:
this is the one mechanism visible on the public frontier that this repo has never run
(`crystalbaby/pseudo-labels-without-lying-to-cv-15-teachers`), and unlike a blend it changes the
*representation*, which is the only thing that has ever moved this competition. GATE: the
+0.0000949 shipping gate, on a strict twin of the champion recipe.

Slots are not committed in advance tonight. Phase 25's verdict stands -- the public axis is
saturated and no slot should be spent searching it -- so a slot gets spent only if a probe above
produces something that needs a board point.

### Phase 26 -- results: four nulls, the lookup finally EXPLAINED, and a live mechanism (2026-09-25)

**Z1 -- the `id` axis is null, and it is a clean null.** `AUC(id) = 0.499990`. Per-id-block target
rate SD against binomial expectation, at four block counts: 10 blocks ratio 0.796, 50 blocks 0.964,
200 blocks 0.961, 1000 blocks 1.024. Residual-vs-`id` slope +0.002897 against a bootstrap SD of
0.006074, **z = +0.48**. The generator's emission order carries nothing. Closed; the gate demanded
AUC >= 0.505 or a rate-SD ratio > 2.0 and got neither.

**Z3 -- the ceiling, and it is the most decision-relevant number of the night.** For a perfectly
calibrated model, AUC is a functional of the distribution of true `p` alone, so simulating
`y ~ Bernoulli(p_oof)` and scoring `AUC(p_oof, y_sim)` gives the AUC a *perfect* model reaches on a
probability field shaped like ours. Rate pinned at 0.174645 by bisection so only dispersion varies
(the first pass rescaled about the mean logit, which moved the positive rate to 0.30 and made AUC
rise for two reasons at once -- that pass is not the one reported):

| | logit SD | ceiling AUC |
|---|---|---|
| our field (TEXbag OOF) | 3.1911 | **0.945665** |
| observed against real labels | -- | **0.946137** |

**Observed sits +0.000472 ABOVE the self-consistent ceiling**, so the true probability field is
slightly *sharper* than ours and we are demonstrably not at any ceiling. Then the operative
question -- how much independent missing signal would explain Team Alicia's 0.94945:

| independent missing signal, log-odds SD | ceiling AUC | vs ours |
|---|---|---|
| 0.0 | 0.945665 | -- |
| 0.3 | 0.945721 | +0.000056 |
| 0.5 | 0.946177 | +0.000512 |
| 0.7 | 0.947306 | +0.001641 |
| **1.0** | **0.949144** | **+0.003479** |

**The gap to #1 is worth an independent feature of about 1.0 log-odds SD.** Our entire logit field
has SD 3.19, so that is a feature carrying roughly 9% of the total log-odds variance. The
calibration that makes this land: everything this repo has found in 25 phases, baseline 0.94164 to
0.946137, is **+0.0045 -- which on the same table is a missing-signal SD of about 1.1.** So
**#1's edge is the same size as every mechanism we have discovered put together.** It is not a
refinement, it is not a stack of +0.0001s, and that retroactively explains why seven phases of
careful search found nothing: you do not find a 9%-of-variance feature by tuning. It is either
reachable or it is not.

**Z4 -- no joint-key lookup, and this time the null is properly calibrated.** Phase 1 ran nine joint
keys against a *binomial* null on C2's residuals, which put the ratios at 0.85-1.07 and made them
hard to read (an OOF prediction for value *v* is built from other folds' rows of *v*, which
anticorrelates the error and pushes the ratio below 1). Replacing it with a **permutation null that
shuffles key assignment within probability strata** absorbs exactly that, and the null mean lands
where it should, at ~1.0. Ten keys fixed before any number was read, plus a control:

| key | nkeys | ratio | null p99 | z |
|---|---|---|---|---|
| `nHome x nWork` | 254 | 1.1327 | 1.2048 | +1.57 |
| `ECL x Subsidy x HomeCharge` | 20 | 1.2319 | 1.7559 | -0.17 |
| `income5k x HomeCharge` | 56 | 0.9178 | 1.4102 | -0.38 |
| `income5k x commute5` | 295 | 0.9211 | 1.1288 | -0.59 |
| *(six more, all null)* | | 0.57-0.92 | | -3.37 to -0.89 |
| **CONTROL random 300 keys** | 300 | **1.1397** | 1.1614 | **+1.78** |

**The random-key control's z is higher than every one of the ten real keys.** Nothing here is signal,
and the control is what licenses saying so. Note also the magnitude argument from Z3: a missing
feature of 1.0 log-odds SD would drive a ratio many multiples above 1, not 1.13. Joint-key lookups
cannot host a thing that size. Closed.

**Z2 was answered with the wrong key, and Z6 is the right one -- it EXPLAINS README section 6.**
Z2 matched each train row to its nearest of the 9,466 usable original rows in 13-d standardised
space: median distance 1.22, 9,093 distinct parents used, and a clean null (residual difference
z = **-0.37** on the nearest quartile; adding the parent label *lowered* AUC). That null is real but
it answers the wrong question, because nearest-neighbour in 13-d is not how the generator indexes
its source. **`Annual_Income_USD` is a near-unique key into the original dataset**, and an exact
join is a completely different instrument:

- 8,915 distinct income values in the original; **8,571 of them map to exactly one original row**
  (335 to two, 8 to three, and income 30000 -- the clip point -- to 557).
- **75.8% of train rows and 75.8% of test rows** carry an income value with a unique parent.
- Among the 5,524 income values with >= 20 train rows and exactly one parent, the **train per-value
  target rate is 0.2744 when that parent bought and 0.1804 when it did not -- a +0.0940 gap at
  z = +16.84**, against a global rate of 0.1746.

**That is the lookup table.** README section 6 has described `Annual_Income_USD` as a
"value -> target lookup, not a magnitude" since Phase 1, with Spearman(value, rate) = 0.68 and a
per-value residual rate SD of 0.0748 -- correct, load-bearing, and until tonight unexplained. The
mechanism is that **income identifies a source row, and the source row's own label leaks through the
generator into every synthetic child of it.** Phase 1's B6 result -- forbidding non-monotonicity in
income costs 39 sigma -- is that leak being forbidden. The lever this repo has been pulling since
day one now has a name.

**And Z6b closes it in the same breath: it is already ~95% extracted, and no feature can get the
rest.** Against TEXbag's own OOF, row level, on the 506,588 train rows with a unique parent: mean
residual +0.003416 for parent-bought against -0.001047 for parent-did-not, a difference of
**+0.004463 at z = +4.32** -- real, but 5% of the +0.0940 the raw per-value rate shows, because the
per-value TE has already absorbed the rest. Adding it as an additive bump *lowers* AUC at every dose
tried (0.945116 -> 0.945072 -> 0.944812 -> 0.943964 at eps = 0.003 / 0.01 / 0.03). And it is **flat
across train support** rather than concentrated where the encoder is weakest, which is where a
feature would have had room:

| train support of the income value | rows | residual diff | z |
|---|---|---|---|
| 0-5 | 903 | +0.011139 | +0.55 |
| 5-10 | 1,871 | +0.004967 | +0.46 |
| 10-25 | 13,705 | +0.010174 | +2.06 |
| 25-50 | 55,136 | +0.003884 | +1.22 |
| 50-100 | 127,770 | +0.004605 | +2.31 |
| 100+ | 307,203 | +0.004093 | +3.01 |

No trend. And the pocket where the parent could beat the encoder is tiny: of test rows with train
support < 10 (2.29% of test) only 22.05% have a unique parent, and of the 0.58% on an income value
unseen in train only **4.95%** do. Upper bound if the parent label were used *perfectly* on the
low-support pocket: **0.000022 / 0.000033 / 0.000265** of AUC at support < 5 / 10 / 25 -- and the
last of those assumes perfect exploitation of a z ~ 2 effect. Below the gate at best. **The original
dataset is now closed on mechanism, not just on Phase 1's coverage argument -- and it is closed
having finally explained the biggest single finding in this repo.**

**Z5 -- the one live mechanism, and the first version of it was wrong because my own diagnostic
leaked.** The pre-registered question was whether pseudo-labelling the 286,571 test rows sharpens
the income encoder. The first pass used the archived `test_proba_lgb.csv` as the soft labels and
reported the encoder predicting held-out per-value rates better by wRMSE -0.003190 / standalone AUC
+0.005782, uniformly across all five folds. **That is not the mechanism, it is a leak:**
`src/pipeline.py` builds that file as `test_proba += pt / n_folds`, the 5-fold average, so it carries
information from every train row's label including the held-out fold's. Rebuilt with per-fold
pseudo-labels from a model trained on folds != f only, the same measurement gives wRMSE **-0.001039**
/ AUC **+0.001963** -- a third of the size. The uniform sign across folds was the tell, and it is
recorded here because the leaked number would have justified a much bigger claim.

Then end to end, five folds, frozen split, lean recipe (Z5c/Z5d):

| arm | OOF | vs train-only |
|---|---|---|
| A train-only TE | 0.944001 | -- |
| **B pseudo-augmented TE** | **0.944431** | **+0.000430** |
| D augmented with the CONSTANT PRIOR | 0.944088 | +0.000087 |
| E augmented with SHUFFLED soft labels | 0.943765 | **-0.000237** |

**D and E are what make B believable.** The augmented encoder sees 42.9% more rows per income value,
so at fixed `te_smooth` it shrinks less -- a different encoder even if the soft labels say nothing.
D holds the support constant and sets the information to zero: it explains only **+0.000087**, below
the shipping gate, so smoothing is a fifth of the effect. E keeps the support *and* the soft-label
marginal distribution and destroys only the row-to-value association: it scores **-0.000237**,
actively worse than baseline. An encoder that is corrupted by scrambling which value a soft label
belongs to is an encoder that was using that association. **B - D = +0.000343, 3.6x the shipping
gate, and it is information.**

**Z5e -- and it survives the champion's representation.** The obvious objection is redundancy: the
champion carries eleven TE columns, eight of them multi-scale quantile-bucket encoders
(`q_income_10/50/500/5000`, `q_commute_1/2/5/10`) that are themselves pooled estimates of the same
per-value rate at coarser support, and README Phase 1b/1c measured three separate times that several
routes to one problem shrink each other. Same harness, champion-shaped feature set:

| | OOF | pseudo-label gain |
|---|---|---|
| 3-column TE, train-only / pseudo | 0.944057 / 0.944520 | **+0.000463** |
| 11-column TE, train-only / pseudo | 0.944370 / 0.944704 | **+0.000334** |
| multi-scale pooling alone (11col - 3col, train-only) | +0.000313 | -- |

The redundancy is real but partial -- it takes 28% of the gain, not all of it -- and **+0.000334 is
still 3.5x the shipping gate**, with all five folds positive in both arms. This is the first new
ours-only mechanism since Phase 20.

**`te_pseudo` is implemented in `src/pipeline.py` (commit `c5e4771`), default off.** Each fold runs
two passes: pass 1 fits a lean model on the training fold only and predicts the test rows; pass 2
refits every `te_cols` encoder with those rows folded in as soft-labelled support, overwriting the
same column names so the feature set and its order are unchanged and the probe stays a strict twin.
`_te_stats` is a groupby sum/count, so a soft label of 0.3 contributes 0.3 of a buyer and 1 of a row
-- a fractional observation, already well defined.

**The one way to get this wrong, and why pass 1 has its own fitter.** `fit_predict()` always passes
`eval_set=[(Xva, yva)]` with early stopping, so routing pass 1 through it would choose the pass-1
model's number of rounds -- and therefore its test predictions -- **using the validation fold's
labels**, which then build the encoder that the same validation fold is scored against. No assert in
this repo would have caught it. `_pseudo_label_test()` is a separate function with fixed rounds and
no `eval_set` for exactly that reason. In-pipeline strict twin at a lean smoke config:
**0.945552 -> 0.945656, +0.000104, all five folds positive.**

**Z7 -- the champion-recipe test is on Kaggle now**, three kernels on commit `c5e4771`: `PS0`
(champion TEXbag recipe, `te_pseudo` off -- the strict-twin control fitted by the same code), `PS`
(pass 1 at 400 rounds / lr 0.05, matching what Z5c-Z5e measured) and `PSR` (pass 1 at 1500 rounds /
lr 0.02, to ask whether the gain is bounded by soft-label quality, since the pass-1 model in every
measurement so far scores ~0.943 against the champion's 0.9461). **Gate: +0.0000949 OOF over PS0.**
No slot is spent until that gate is cleared.
