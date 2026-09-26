# S6E9 — what we tried, what we learned, and how to pick the final two

**Audience: you, coming back to this cold.** This is the orientation layer. It summarises 27 phases
of work, says which axes are open and which are closed *and why*, and gives the decision procedure
for the two final submissions.

It is deliberately not the source of truth. Three documents sit under it:

| document | role | when to read it |
|---|---|---|
| **`README.md`** | **the contract.** The frozen CV split, the metric, the leakage rule, the measured constants, and the full phase-by-phase log with every number. | Before touching a model. It is long; use the index in §8 below to jump. |
| **`KAGGLE_PLAYBOOK.md`** | the method, distilled from S6E7 and S6E8. Why we build the instrument first. | Before designing a probe. |
| **`experiments/runs.csv`** | 181 runs, 95 of them with a paired public-LB score. Append-only, in git. | Any time you want to re-analyse. Both prior competitions' best insights were pure re-analysis of this file. |

Deadline **2026-09-30 23:59 UTC**. 5 submissions/day, no rollover.

---

## 1. The problem, in one page

Predict `Will_Buy_EV` (binary) from 13 features about a household. **Scored on ROC AUC.**

| | |
|---|---|
| Train / test | 668,665 / 286,571 rows |
| Positive rate | 0.174645 |
| Public / private split | **20% / 80%** — 57,314 public rows, 229,257 private |
| Missing values, duplicate rows, train↔test drift | **all zero.** Imputation, missing-indicators and adversarial validation are off the table — there is nothing there. |
| CV | `StratifiedKFold(5, shuffle=True, random_state=42)`, **never changed** |

Two consequences of the metric that cross whole categories off the list:

- **AUC is rank-based.** Any monotone transform of the output is worth exactly 0.00000 — calibration,
  Platt scaling, isotonic, threshold tuning. Only *ranking* can move the board. Submit probabilities.
- **A single public score's own noise is 0.001078**, but a *paired* comparison of two of our models on
  the same public rows resolves to **0.000116**, and for strict twins to **0.000027**. So "we are
  0.0005 behind rank 40" and "that is a 4σ difference against that specific opponent" are both true.
  Always compare against a named opponent's predictions, never against a rank.

### What the data actually is

One feature dominates: `Environmental_Concern_Level` (ordinal 1–5) has single-feature AUC **0.84352**.
Then `Annual_Income_USD` 0.74276 and `Subsidy_Available` 0.71794. The rest are 0.50–0.56.

**The generator is additive in log-odds.** An additive GLM reaches OOF 0.938402; adding *all 28
pairwise interactions* among the 8 strong drivers moves it to 0.938435 — **+0.000033, nothing.**
Interaction search is crossed off, and nine different architectures later confirmed it (§4.3).

**But `Annual_Income_USD` is a value→target lookup table, not a magnitude.** This is the single most
important fact in the competition and it took Phase 1 to find and Phase 26 to explain:

- After fitting the *best possible monotone function* of income, per-value residual rate SD is
  **0.0748** against a binomial expectation of 0.0284.
- Per-value target encoding of that one column: **+0.003017 OOF.** Forbidding non-monotonicity with a
  monotone constraint: **−0.001472** — a constraint can only cost that much if what it forbids is real.
  Adding `log(income)`: **+0.000000**, because a monotone re-expression cannot see a lookup.
- **Why it exists (Phase 26):** income is a near-unique key into the 10,000-row source dataset.
  **8,571 income values map to exactly one original row**, covering 75.8% of train *and* test, and
  where that source row bought, the train per-value rate is 0.2744 against 0.1804 where it did not —
  a +0.094 gap at z=+16.8. The generative model memorised its source, and the source row's label
  leaks into every synthetic child. *The lookup is a memorisation artifact.*

The transferable version: **a model that reads an integer code as a magnitude cannot see a
non-monotonic lookup, however much capacity you give it.** Phase 0's GLM probe was blind to this and
produced a confident wrong conclusion because of it.

---

## 2. The measuring instrument — read this before any number

Everything else in this repo is downstream of these constants. They were measured in Phase 0–3 and
are **frozen by design** (`README.md` §4). Do not re-derive them from marginal LB deltas later; that
was S6E8's clearest mistake.

| constant | value | meaning |
|---|---|---|
| **Seed-noise floor** | **0.000038** | SD of the same recipe's OOF over 4 model seeds. **Nothing below this is a result.** |
| **SHIPPING GATE** | **+0.0000949 OOF** | = OOF→LB residual σ ÷ slope. The bar a probe must clear to ship. |
| OOF→LB slope | 1.0832 | fit over 10 paired GBDT runs |
| OOF→LB residual σ | 0.000103 | same fit |
| Paired ΔAUC SD, public | 0.000116 | two of our models, same public rows |
| Paired ΔAUC SD, **private** | **0.000055** | **this is how much room the shakeup has** |
| Paired ΔAUC SD, strict twins | 0.000027 | one-field twins resolve ~4× finer |
| Single-score SD, public | 0.001078 | one model's absolute public score |

Two limits on the instrument that cost real effort to learn:

- **It is valid within the GBDT family only.** Adding one non-GBDT point blows the residual σ from
  0.000103 to **0.000340**, a 3.3× degradation (Phase 3).
- **Its local slope is noise-dominated and must not be extrapolated.** Refit on the rich band it has
  read 0.9974 (n=24), 1.1225 (n=29) and 0.8524 (n=39) across three successive refits. That is not a
  moving truth, it is an estimate reading its own noise (Phase 21, Phase 24 point 6).

---

## 3. The scoreboard

| model | OOF | public LB |
|---|---|---|
| Phase 0 raw-feature LightGBM baseline | 0.941660 | 0.94149 |
| Best single model (`PS`, `7eb59fd6`) | 0.946139 | **0.94643** |
| **Final A** — ours-only (`b6177d6b`) | 0.946205 | **0.94642** |
| **Final B** — public-mixed (`03d73b32`) | 0.946397 | **0.94646** |
| #2 on the board | — | 0.94676 |
| **#1 on the board (Team Alicia)** | — | **0.94945** |

3,016 teams. The board is brutally compressed: **358 teams sit at or above 0.94646** and 523 at or
above 0.94640, so 0.00004 of score is worth roughly 100 places. This is exactly why rank is a
terrible unit and paired comparisons are the only honest ones.

---

## 4. What was tried, by axis

Each verdict below is a measurement, not an opinion. The phase pointer is where the numbers live.

### 4.1 Feature engineering & representation — the only axis that ever paid

| what | verdict | where |
|---|---|---|
| **Per-value target encoding of `Annual_Income_USD`** | **+0.003017. The single biggest win in the competition.** | Phase 1 |
| Encoder tuning: smoothing 20→5, neighbourhood backoff, inner-fold protection | +0.0005 to +0.0009 combined, with heavy mutual redundancy | Phase 1b/1c |
| **Multi-scale quantile-bucket TE** (`q_income_10/50/500/5000`) | real; part of the champion | Phase 14 |
| **Window encoder** — each value centred in its own neighbourhood at six radii | real; replaced the shape encoder, new champion | Phase 16 |
| Two smoothings per column (`te_multi_smooth: [10, "auto"]`) | real; part of the champion | Phase 14 |
| Digit decomposition (`floor(v/10^p) % 10`) | real; part of the champion | Phase 14 |
| Wider TE key coverage (commute quantisation ladder) | real for commute, **does not generalise** to other columns | Phase 20, 21 |
| Pairwise interactions (all 28 among strong drivers) | **+0.000033 — crossed off** | Phase 0 |
| Monotone constraint on income | **−0.001472** | Phase 1 |
| `log(income)` and other monotone re-expressions | **0.000000** | Phase 1 |
| `fe_recipe_score` (fixed-coefficient composite columns) | **struck from every recipe — worth 0 on OOF and a real LB cost** | Phase 17, 18 |
| Feature bulk generally (more columns for their own sake) | **dead** | Phase 18 |
| Frontier's heavy-smoothing binned/paired TE recipe | rejected, three twins | Phase 2c |
| Cross-feature / joint-key lookups | **null**, 9 keys then 10 more with a proper permutation null | Phase 1, Phase 26 Z4 |

### 4.2 The source dataset & generator fingerprints — closed, having explained the lookup

| what | verdict | where |
|---|---|---|
| Original dataset as extra training rows | covers 82 test rows; worthless on coverage grounds | Phase 1 |
| Nearest-parent match in 13-d space | **null (z=−0.37)** — and it was the *wrong key* | Phase 26 Z2 |
| **Exact income→unique-parent join** | **explains the income lookup** (z=+16.8 on per-value rate) | Phase 26 Z6 |
| Parent label as an additive signal | ~95% already absorbed by the TE; flat across support; ceiling 0.00027 | Phase 26 Z6b |
| Parent label modulated by feature agreement | real and rising (+0.0235 at match≥8) but only z=+2.57; gate not met | Phase 27 G2 |
| Parent block as model features | **−0.000085, negative in all 5 folds** | Phase 27 G3 |
| Does the generator copy columns? | **No** — match ratios 1.00–1.36 vs chance; the dominant driver sits at 1.02 | Phase 27 G1 |
| De-clipping income/commute via the parent | **impossible — the clipping is in the SOURCE data**, original min 30000.0 / 5.0, zero rows below | Phase 27 G1 |
| `id` / row-emission order | **null** — AUC(id)=0.499990, block-rate ratios ~1.0 | Phase 26 Z1 |

### 4.3 Model families & architectures — nine tried, zero usable

The playbook's test is a wall plot: solo score against disagreement rate. **No architecture landed in
the useful quadrant** (competitive solo *and* decorrelated). LightGBM, XGBoost, CatBoost, an
embedding MLP, a plain MLP, EBM/GA2M, NODE, TabICL, ModernNCA, TabTransformer.

The decisive one: the **embedding MLP** achieved max correlation **0.9508** against a pool whose
internal median is 0.9973 — by far the most genuinely different representation we own — and its stack
ADD was **−0.000002**. And an **EBM with `interactions=0`** is additive *by construction*, so it is
the first architecture that actually tests additivity rather than declining to falsify it. It
confirmed it. Phases 7–10.

> **Playbook §7, earned the hard way: high disagreement is weakness, not diversity, unless it comes
> with competitive solo strength.**

### 4.4 Hyperparameters, capacity, variance — all null or tiny

| what | verdict | where |
|---|---|---|
| Capacity (`num_leaves` sweeps) | 7 leaves beats 63 by +0.000240 — because capacity buys per-feature *shape resolution*, and the interactions it would otherwise fit are not there | Phase 3b |
| Learning rate, `min_child_samples`, `max_bin` | null | Phase 5 |
| Optuna HPO (4 sharded kernels) | no lever | Phase 15 |
| Encoding variance (TE averaged over inner splits) | −0.000019, null | Phase 2 |
| Seed bagging inside a fold | +0.000034 — inside the noise floor, but kept as a *variance* device | Phase 2, 4 |
| Re-running at a new outer split seed | **never do this.** S6E7 spent 2.3 GPU-hours proving it buys nothing. Fold-score spread is evaluation-fold difficulty, not prediction variance. | playbook |

### 4.5 Ensembling — washes almost every time

Fitted logistic stacks, rank-mean and logit-mean blends, 8-leg pools, meta-models over the
architecture zoo. **The stack washed four separate times** (Phases 13, 16b, 21). In S6E8's final
round, 10 of 17 legs cleared the solo gate and **0** cleared the stack gate.

> **Judge a leg by its ADD and SWAP contribution at constant pool size, never by its solo score.**

The one thing that *did* work, and only in Phase 27: **pool diversity that isn't family diversity.**
Final A had been three lgb near-twins at rank-correlation 0.9997–0.9998, ranking **305th of 336**
pools over an exhaustive k=3..5 enumeration. Adding the one leg with a genuinely different
*representation* was worth +0.00004 on the board.

### 4.6 Public artifacts — real, one bit wide, and saturated

Policy opened on schedule on 2026-09-21 (`README.md` §5). The price was written down at the moment of
the decision: ours-only was costing ~350 places of public rank.

**The axis is one bit and we already have it.** Measured across 4 public shares × 2 fold counts × 2
combiners, then by an exhaustive **3,282-blend enumeration** over 14 individually-registered public
legs at k=1..4 — decomposing megayak's library into its six views and adding a 10-fold XGBoost.
The entire exhaustive search buys **+0.000016 OOF over what we had already shipped**, and
**six different compositions spanning OOF 0.946344–0.946404 all score exactly 0.94646.**

> **0.94646 is a wall. Do not spend another slot searching this axis.** Phases 23, 24, 25, 27.

A 19-member public OOF library was also tested and closed by ADD test with zero slots: null to
negative everywhere, and **monotone in solo strength rather than in decorrelation** — §4.3's rule
again.

### 4.7 Transductive / pseudo-labelling — the subtle one

Folding the 286,571 test rows into the target encoders as *soft-labelled support* (42.9% more rows
per income value). Controls mattered enormously here:

| arm | OOF (lean recipe) |
|---|---|
| train-only encoders | 0.944001 |
| **pseudo-augmented** | **0.944431 (+0.000430)** |
| augmented with the **constant prior** (same support, zero information) | 0.944088 (+0.000087) |
| augmented with **shuffled** soft labels | 0.943765 (**−0.000237**) |

So the gain is *information*, not smoothing — and scrambling which value a soft label belongs to
corrupts the encoder below baseline, which is what genuine per-value signal looks like. It iterates
once (+0.000427 at two rounds) then saturates.

**But on the full champion recipe it is worth +0.000002.** The champion already buys a low-variance
per-value rate four separate ways. See §6 for the two lessons this produced — they are the most
transferable things in the document.

### 4.8 The gap to #1 — priced, not closed

Team Alicia sits at **0.94945**, +0.00269 clear of a Grandmaster at 0.94672, with no public notebook
above 0.94656. Phase 26 Z3 priced it rather than chasing it:

Simulate `y ~ Bernoulli(p_oof)` with the positive rate pinned, and score `AUC(p_oof, y_sim)` — that
is the AUC a *perfect* model reaches on a probability field shaped like ours. It comes out at
**0.945665** against our observed 0.946137, so **we are not at any ceiling**; the true field is
slightly sharper than ours. Then ask what would close the gap:

| independent missing signal, log-odds SD | ceiling AUC |
|---|---|
| 0.0 | 0.945665 |
| 0.5 | 0.946177 |
| **1.0** | **0.949144** |

**Reaching 0.94945 needs an independent feature of ~1.0 log-odds SD.** Our whole logit field has SD
3.19, so that is ~9% of total log-odds variance. And everything this repo discovered in 25 phases —
baseline 0.94164 → 0.946137 — is **+0.0045, which is ~1.1 on the same scale.**

> **#1's edge is the size of every mechanism we found, combined.** It is not a refinement and it is
> not a stack of +0.0001s. That retroactively explains why seven phases of careful search found
> nothing: you do not find a 9%-of-variance feature by tuning. Every channel outside the 13 columns
> that we can reach — `id` order, the source dataset, joint keys — is closed by measurement. So it is
> either information we have no access to, or it is not generalisable signal. **Do not spend the
> remaining slots searching for it.**

---

## 5. The five offsets — the repo's most transferable finding

An OOF gain does not convert to board score at a constant rate. It depends on *where the OOF came
from*. Five instances, each measured independently:

| offset | size | meaning | where |
|---|---|---|---|
| **Family** | cat ≈ **−0.00010**, xgb ≈ −0.00004 at matched OOF | non-LightGBM legs' OOF overstates their board value | Phase 22, confirmed Phase 27 |
| **Source** | ≈ **−0.0001** | an OOF produced by a pipeline we did not calibrate (someone else's public artifact) sits on a *lower* line than ours | Phase 24 |
| **Fold count** | ≈ **−0.000114** for a 5→10 fold step | a 10-fold OOF is optimistic against a 5-fold-calibrated line | Phase 25 |
| `fe_recipe_score` | ≈ −0.00018 | a specific feature block that cost board score at zero OOF cost | Phase 17, 18 |
| **Representation (`te_pseudo`)** | **+0.00007** at matched OOF | **the first POSITIVE one** — encoders fitted with test-row support are built partly *for the test distribution*, which OOF has no way to reward | Phase 27 |

The general rule: **an OOF produced by a pipeline we did not calibrate does not enter our instrument
at face value.** Phase 27's cleanest demonstration — two arms 0.000010 apart on OOF landed 0.00007
apart on the board, in the pre-registered direction:

| arm | non-lgb share | LB vs old Final A |
|---|---|---|
| + representation leg only | 0% | **+0.00004** |
| + cat and xgb legs only | 40% | **−0.00003** |
| both | 50% | **+0.00001** (= the sum) |

---

## 6. Two methodological lessons that cost the most to learn

**1. A lean stand-in can only UPPER-BOUND a representation gain, never estimate it.** The
pseudo-label mechanism measured +0.000430 on a lean recipe and +0.000334 on a deliberately
"champion-shaped" replica — which turned out to be a **28% shrink where the real champion produced
99.5%**. Screening on a partial replica is legitimate for deciding *whether to build* something and
worthless for predicting *what it will be worth*. The ADD test has to run on the actual champion
config. (This cuts both ways and makes the lean screen a good cheap **kill** test: a negative upper
bound cannot be positive on the champion — that is how Phase 27 G3 closed in one run.)

**2. Check whether your diagnostic is the thing that's leaking.** The first pseudo-label measurement
used the archived `test_proba_*.csv` as soft labels and reported an effect **three times** the true
size. `pipeline.py` builds that file as `test_proba += pt / n_folds` — the 5-fold average — so it
carries information from every train row's label, including the held-out fold's. *The tell was that
the effect was uniformly positive across all five folds.* Relatedly, `fit_predict()` always
early-stops on `(Xva, yva)`, so any two-pass scheme must not route its first pass through it; that is
why `_pseudo_label_test()` exists as a separate function with fixed rounds and no `eval_set`.

**Two operational traps, both now guarded in code:**

- **Duplicate archive rows.** A poller plus a retry can archive the same kernel twice (Phases 18, 19,
  20, 22, 26). **De-duplicate on `run_tag` before any re-analysis.**
- **Fabricated paired points.** `submit_run.py --fill-missing` matched submissions to runs *by
  description*, and `stack_logit.py` emitted a generic one, so one score got written onto four rows —
  two of which had never been submitted at all. A fabricated paired point is worse than a missing one,
  because §2's whole instrument is built from these pairs. Both sides are fixed: the backfill now
  **refuses** ambiguous matches and blend descriptions are self-identifying. Phase 27.

---

## 7. How to choose the final two submissions

### 7.1 The mechanics — do not skip this

Kaggle scores **two** selected submissions on the private split. **If you do not choose explicitly,
Kaggle defaults to your best *public* score**, which is precisely the choice the hedge exists to
avoid. Select both, by hand, on the competition's Submissions page, **before 2026-09-30 23:59 UTC.**

### 7.2 The strategy, decided in advance and not to be improvised now

`README.md` §5 replaced the playbook's usual A/B split with one tuned to this competition's live risk.
The 0.9465–0.94675 pack was confirmed by direct inspection to be **largely stacks over shared public
OOF libraries, not better modelling** — which is exactly the S6E7 pattern that evaporated in a
shakeup. So:

- **Final A = the best ours-only model.** No public artifact in it, at any weight. This is the entry
  that survives if the public-pooled pack collapses on the private split.
- **Final B = the best public-mixed model.** This is the entry that wins if the pack's gains are real.

**The load-bearing constraint: provenance must be auditable off disk.** Every public-mixed candidate
is written by `scripts/public_blend.py`, which records the source dataset, per-leg weight and each
leg's fold count in `manifest.json`. **A candidate whose public content cannot be read back off disk
is not selectable.** That is what makes the hedge enforceable rather than remembered approximately.

### 7.3 The current two

| | run_id | recipe | OOF | public LB |
|---|---|---|---|---|
| **Final A** | **`b6177d6b`** | ours-only `rank_mean(TEXbag, TEXF, TEX2F, PS)` | 0.946205 | **0.94642** |
| **Final B** | **`03d73b32`** | `rank_mean(A_NEW × 0.34 + [6view, rmlp, xgb5f, lgbV5] × 0.66)` | 0.946397 | **0.94646** |

Why Final A is this and not the single best public score (`PS` solo at 0.94643): the two are **inside
near-twin resolution (0.000027)**, and a four-leg blend carries strictly less prediction variance on
an unseen split than a single model. That is playbook §9's rule, and choosing the lower-variance
member when scores tie is the same reasoning that picked the previous Final A.

### 7.4 The decision rule

**The hedge is close to free, and that is measured, not assumed.** In S6E8 Final A and Final B tied
*exactly* on private (0.97030). Here the A–B gap is 0.00004 on public, against a **private paired SD
of 0.000055** — well inside noise. So you are not sacrificing expected score to buy the hedge.

Apply in order:

1. **Keep one of each kind.** A must contain no public artifact at any weight; B carries the public
   mix. Do not let both finals end up on the same side of the shakeup risk — that throws the hedge
   away for nothing.
2. **Within each kind, take the best public LB score.** Offline OOF has done its job; for choosing
   between candidates that already have board points, the board point is better evidence.
3. **Break ties on variance, not on OOF.** More legs beats fewer; seed-bagged beats single-seed; a
   fixed equal-weight blend beats a fitted stack (no meta-fit optimism). Never break a tie on a
   difference smaller than the 0.000027 near-twin resolution — that is reading noise.
4. **Discount non-LightGBM legs before counting their OOF** (§5). Cross-family pooling is closed on
   the board: it *lost* 0.00003 while its OOF rose.
5. **You can only select what you submitted.** An unsubmitted candidate, however good its OOF, is not
   selectable. This is why slots get burned rather than hoarded.

### 7.5 What to do with the remaining slots (~25, five nights)

In priority order. Every axis is closed, so these are paired points and candidate promotions, not
searches.

1. **Rebuild Final B over `A_LGB` instead of `A_NEW`.** `03d73b32`'s ours side carries the family
   offset Phase 27 measured, so this should come in ≥ 0.94646. This is the one candidate the evidence
   directly points at. *Do this first.*
2. **A second paired point on the `te_pseudo` offset.** At 2.6σ it is the only offset with a sign that
   favours us, and it is the one thing measured that OOF is structurally unable to price. If it
   confirms, `PS`-carrying pools are systematically undervalued by our own instrument.
3. **Burn the rest on paired points in the rich OOF band.** 5/day, no rollover; an unspent slot is a
   paired point thrown away. S6E8 used 18 of ~290 available and paid for it by over-reading four of
   them.
4. **Do not** spend a slot on the public axis (§4.6, a wall at 0.94646 across six compositions), on
   cross-family pooling (§5), on the generator/parent axis (§4.2, closed three ways), or on chasing #1
   (§4.8, priced at more signal than we have ever found).

### 7.6 Pre-deadline checklist

- [ ] Re-read `README.md` §5 so the hedge is applied as written, not as remembered.
- [ ] Confirm Final A's `manifest.json` contains **no** public source, at any weight.
- [ ] Confirm Final B's `manifest.json` lists every public leg, its weight and its fold count.
- [ ] Confirm both candidates were actually **submitted** and carry a `public_lb_score` in
      `experiments/runs.csv`.
- [ ] De-duplicate on `run_tag` before any final re-analysis (§6).
- [ ] **Explicitly select both** on Kaggle. Do not let the default take over.
- [ ] Write the predicted private outcome down *before* the reveal, so the post-mortem is honest.

---

## 8. Where to dive deep

`README.md` §8 is the full log. Jump by topic:

| you want | go to |
|---|---|
| The frozen constants and how they were measured | `README.md` §4 |
| The A/B hedge and the artifact policy, with the price written down | `README.md` §5 |
| What the 13 features are, and the income-lookup evidence | `README.md` §6 |
| **The income lookup discovered** | Phase 1, 1b, 1c, 1d |
| A confident wrong conclusion, and how measurement caught it | Phase 0 → corrected by Phase 1; Phase 2 → Phase 3b; Phase 12 → Phase 14 |
| The instrument built, and found to be family-specific | Phase 3 |
| Additivity proved structurally (EBM), not just behaviourally | Phase 7 |
| The architecture wall, nine models | Phases 8, 9, 10 |
| The champion's representation (quantile buckets, digits, multi-smooth) | Phase 14 |
| The window encoder | Phase 16, saturated in Phase 18 |
| `fe_recipe_score` found to be a real LB cost | Phases 17, 18 |
| The residual decomposed into three mechanisms | Phase 19 |
| The CatBoost family offset | Phase 22 |
| The public artifact policy opening, and the price | Phase 23 |
| The source offset, and a structurally unsafe public file | Phase 24 |
| The exhaustive 3,282-blend public enumeration | Phase 25 |
| **The income lookup EXPLAINED (memorisation of the source)** | Phase 26 Z6 |
| **The gap to #1 priced** | Phase 26 Z3 |
| Pseudo-label encoders, with the controls that make them readable | Phase 26 Z5–Z8, Phase 27 |
| **The current finals, and the diversity decomposition** | Phase 27 |

Useful tools:

```bash
python scripts/run_local.py --cfg '{...}' --diff-vs '{"run_tag":"champion"}' ...  # strict-twin guard
python scripts/make_probe_kernel.py <tag> <cfg.json>   # fan out 5-wide on Kaggle CPU
python scripts/split_resolution.py <runA> <runB>       # the paired SD for THIS pair
python scripts/leg_probe.py / leg_swap.py              # ADD and SWAP at constant pool size
python scripts/refit_gap.py                            # refit the OOF->LB line (read §2's caveat)
python scripts/submit_run.py --fill-missing            # backfill LB scores (now refuses ambiguity)
```

---

## 9. If you remember only five things

1. **Build the measuring instrument first, then trust it** — including when it says an axis is
   exhausted, and when it says the teams above you are genuinely ahead.
2. **Nothing below 0.000038 is a result; nothing below +0.0000949 ships.** Write the hypothesis,
   mechanism and gate down *before* the probe runs. A well-written negative result is worth more than
   a +0.0001 blend.
3. **An OOF gain's conversion to board score depends on where the OOF came from** (§5). Five
   independent instances.
4. **Controls are the whole game.** The constant-prior arm, the shuffled-label arm and the random-key
   control are what turned three suggestive effects into one real finding and two closed axes.
5. **Burn the slots.** 5/day, no rollover, and every spent slot is a paired point that makes the next
   offline decision trustworthy.
