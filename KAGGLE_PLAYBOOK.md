# Kaggle Playbook — transferable lessons from Playground S6E7 and S6E8

**Audience:** a Claude Code instance starting a new Kaggle competition (tabular, but most of this
generalizes). This is a *method* document, distilled from two full competition runs. Every claim
below is backed by a number that was actually measured; the numbers are cited so you can recognize
the same pattern rather than take the advice on faith.

**The two runs this method has produced:**

| | **S6E7** | **S6E8** |
|---|---|---|
| Task | 3-class balanced accuracy, 690k rows | binary ROC AUC, 691k rows, 296k test |
| Public → private | 0.95091 → **0.95050** | 0.97060 → **0.97030** |
| Rank | **120**, up **298** places | **675 / 3,532** (top 19.1%), up **6** places |
| Runs logged / submissions | 41 / 38 | 89 / 18 |
| Spearman(OOF, private LB) | 0.87 | **0.9974** |

S6E7's thesis was that refusing to chase the public leaderboard past the point where your own CV
says the signal is exhausted *earns* you places in the shakeup. **S6E8 tested that and came back
with a conditional.** The discipline was right both times — in S6E8 the OOF ordered all 18
submissions in exactly the order the private set did, without a single inversion — but the +298
places were never a law. They were a payout for a specific thing being true: the public leaders were
overfitting a split. When they are not, the shakeup returns nothing, and the honest ceiling is
simply your score.

So the thesis, revised and now the spine of §4, §5 and §8:

> **Build an instrument that tells you the difference between a better model and a luckier one, and
> then believe it — including when it tells you that you are done, and including when it tells you
> that the teams above you are genuinely ahead.**

---

## 0. How S6E9 differs — read this before applying the rest

This document is carried into **Playground S6E9** ("Predicting Electric Vehicle Purchases", binary
ROC AUC, 668,665 train / 286,571 test, deadline 2026-09-30). Everything below still applies, but
three properties of S6E9 change which sections bind hardest. They were measured on day one, not
assumed.

| | S6E8 | **S6E9** |
|---|---|---|
| Local 5-fold GBDT wall-clock | ~5 min (kernel cycle ~15 min) | **32 s** |
| Baseline → public top gap | small; signal near-exhausted early | **0.94164 → 0.94644 = 0.0048** |
| Interaction structure | per-value lookup table, strongly non-additive | **additive in log-odds** (all 28 pairwise interactions among the 8 strong drivers: **+0.000033**) |
| Missingness / drift | present, worth modelling | **none at all** |

What that implies for the sections below:

- **§1 (harness) binds harder, not less.** When a probe costs 32 seconds, the number of probes
  explodes and an un-logged experiment is the only way to lose. Build `runs.csv` and the artifact
  archive before the first real model, exactly as written.
- **§3's cheap-probe budget is enormous.** The negative-result round that S6E8 could only afford at
  the very end (§3, "know what a negative result costs") is affordable continuously here. Run it.
- **§7's "representation over architecture" finding transfers directly and is sharpened.** In S6E8
  the data was a value→target lookup table, so target encoding won. In S6E9 the data is
  *additive-in-logit with nonlinear per-feature shapes*, so the representation lever is **response
  shape** — per-value target encoding, splines, monotone constraints — and the thing to cross off in
  advance is **interaction search**, which is measured to be worth nothing.
- **§1's "burn every submission slot" was S6E8's clearest operational miss** (18 of ~290). With a
  28-day window at 5/day there are ~140 slots. Use them; they are the paired OOF↔LB points that
  make §4 and §5 possible.
- **§8's shakeup arithmetic must be recomputed here.** 286,571 test rows is a *smaller* test set
  than S6E8's, so the private split has more room to move than S6E8's did. Compute the paired
  bootstrap SD at both split sizes before assuming either way.

Do not treat any conclusion below as settled for this competition. They are priors with receipts
attached; the receipts are from other datasets.

---

## 1. Build the experiment harness before the first model

Do this on day one. It costs a few hours and it compounds for the entire competition. Both runs
confirm every item here; nothing in this section changed.

**A one-command run collector.** `scripts/collect_run.py`: push the kernel → poll to completion →
parse a `RUN_METRICS_JSON` blob printed by the notebook → archive artifacts → append one row to
`experiments/runs.csv`. Never run an experiment whose result lands only in a scrollback buffer.

**An append-only experiment log in git.** One row per run: run id, git commit, config knobs,
per-learner OOF scores, final OOF, LB score, and a *long* free-text `notes` field. Commit it after
every run. S6E7 ended with 41 rows, S6E8 with 89, and in both competitions the highest-value analysis
of the whole project was pure re-analysis of that file with no new training. In S6E8 that included
the entire post-mortem: filling in 18 private scores turned a month of arguments into a table.

**Archive OOF and test probabilities for every learner, every run.** Still the highest-ROI decision
in the whole project. `experiments/preds/<run_id>/{oof_proba,test_proba}_<learner>.csv`, gitignored,
keyed by run id. Consequences:

- Any past model can be retro-blended with any future model without retraining.
- You can compute *disagreement* between any two models you have ever built.
- Late in the competition you can sweep the back catalogue for models judged only on OOF. S6E8's
  final 25-leg pool was assembled this way, mechanically, from `runs.csv` plus the preds inventory.

**Freeze the CV split globally.** Both projects used `StratifiedKFold(5, shuffle=True,
random_state=42)`. This is what makes OOF matrices from different runs, weeks apart, directly
blendable. Pick the split once, write it in the README, never change it.

**Automate submission slots.** Kaggle gives 5–10 submissions/day and they do not roll over. A
scheduled queue that burns a priority list before the 00:00 UTC reset turns an unused daily resource
into free OOF↔LB data points, which is what makes §4 possible. **S6E8 did not do this and it was a
real cost:** 18 submissions across a month, against a possible ~290. It was enough to fit the OOF→LB
line, but the four stack points that Phase 5 over-read (§5) were four points *because nobody
submitted more*. Cheap paired points are the defence against fitting noise.

---

## 2. Leakage discipline is the load-bearing constraint

Everything else is tuning; this is correctness. The rule: target encoders, quantile bin edges,
scalers, imputers, and category vocabularies are **fit on a training fold only and applied to the
val/test fold** — never fit once on the full training set before the CV loop.

The same discipline has to be repeated identically in *every* usage site: feature selection, HPO, and
the final stack. It is tempting to fit the encoder once "just for feature selection." Don't — a
leak there silently inflates OOF and destroys the OOF↔LB relationship you will depend on in §4.

**S6E8 is the payoff receipt.** Its single largest feature win was per-value target encoding of nine
numeric columns — the most leak-prone transform there is, refit inside every fold at every usage
site. The result: Spearman(OOF, private LB) = **0.9974** over 18 runs, with the OOF→private residual
σ (0.000119) indistinguishable from the OOF→public one (0.000118). A leak anywhere in that path
would have shown up as OOF running ahead of both leaderboards. It did not. Discipline here is what
buys you the right to make decisions offline in §4.

Corollaries that saved time:
- Unseen categories at inference must map to a reserved "unknown" level, never raise.
- An unsupervised label mapping over train ∪ test is **not** a leak; a supervised (target/count)
  encoding is. Say which one it is in a comment when it is not obvious.
- Optimize the **actual competition metric** everywhere — CV scoring, the Optuna objective, and
  neural early stopping. In S6E7 that was `balanced_accuracy_score`; in S6E8, `roc_auc_score`.
- **Know what the metric does not care about.** AUC is rank-based: there is no threshold to tune, no
  per-class weight to search, and probability calibration cannot move the leaderboard — only ranking
  can. S6E7 spent real effort on class weights and a decision rule because balanced accuracy needed
  it; carrying that habit into S6E8 would have been pure waste. Ask, once, at the start: *what
  transformations of my output leave this metric invariant?* Everything invariant is off the table.

---

## 3. Run experiments as falsifiable probes with a pre-registered gate

The format that worked in both:

> **Hypothesis.** X will help because Y. **Test.** Strict-twin ablation changing only X.
> **Gate.** Ship only if OOF improves by ≥ the pre-registered threshold. **Result.** …
> **Mechanism.** …

Three rules make this actually bite.

**Pre-register the gate, before seeing the result.** Derive it from §4, not from taste. In S6E7 the
gate was +0.0003 OOF ≈ 1σ of the measured OOF↔LB residual; about nine probes after the champion was
found cleared exactly zero of them, and because the gate was pre-registered none got rationalized
into a submission. In S6E8 the gate was +0.0002 OOF, and the private split later confirmed it had
been right all along: 1σ of the OOF→private residual is 0.000119 in LB units, which is 0.000133 in
OOF units, putting the gate at ≈1.5σ. **A gate you set from the residual σ stays valid for the whole
competition.** Do not re-derive it later from marginal LB deltas — see §5 for how badly that goes.

**Use two gates when you ship into an ensemble.** S6E8's most productive refinement. Gate 1 is solo
OOF versus the probe's strict twin; gate 2 is *contribution to the stack*, as both an ADD and a SWAP
at constant pool size. They are almost unrelated, and only gate 2 is a shipping decision. In the
final validation round of 17 probes, **10 cleared gate 1 and 0 cleared gate 2.** Solo deltas spanned
−0.0229 to +0.0106 — 0.033 AUC of range — while every one of those probes' stack contributions fell
between −0.000013 and +0.000046. A probe that improves a model in isolation tells you nothing about
whether it improves your submission.

**Strict-twin ablations.** Change exactly one thing. S6E7's cleanest experiment held features,
backbone, embeddings, LR schedule, epoch count and seed-bagging identical to the champion's neural
leg and changed *only the output head and loss* (3-way softmax → ordinal CORAL): OOF 0.93560 vs
0.95051, **−0.0149**, attributable to the ordinal *assumption* rather than to capacity or tuning.
A non-twin comparison would have been an anecdote.

**Log the mechanism, not just the number.** The CORAL failure decomposed into per-class recall:
`at-risk` 0.936 → 0.878, with true `at-risk` rows bleeding symmetrically to both extremes.
Mechanism: a 1-D latent severity score can only predict the middle rank between two thresholds, and
`at-risk` was not a midpoint on a continuum — it was the diffuse majority *catch-all* class. That
mechanism, once written down, explained three other failures retroactively and pre-emptively killed a
whole family of future ideas. **A well-written negative result is worth more than a +0.0001 blend.**

**Know what a negative result costs to produce, and produce it anyway.** S6E8's closing act was 17
strict-twin probes run specifically to test assumptions that had only ever been *argued*. All 17
missed. That round is the most valuable thing in the repo (§5) because it converted a conclusion
resting on bad arithmetic into a conclusion resting on measurement. Budget for it.

---

## 4. Measure whether your own OOF predicts the LB — then trust the answer

Do not accept folklore like "OOF inverts the LB here." Measure it, on every run that has both.

| | S6E7 (26 pairs) | S6E8 (18 pairs) |
|---|---|---|
| Spearman(OOF, LB) | 0.87 | **0.9974** |
| Residual σ (LB units) | 0.00029 | **0.000119** |
| Slope ΔLB / ΔOOF | — | **0.89** |
| Worst outlier | none beyond ~2σ | no rank inversion at all |

**Report the relationship as a slope plus a residual σ, not as an offset.** This is the sharpest new
lesson of S6E8 and it took the private split to see it. The *offset* (LB − OOF) is a property of the
split, not of your models: S6E8's was 0.00123 on the public 20% and 0.00089 on the private 80% for
the very same submissions, because the public slice was simply easier. Anything you infer from the
offset's level or its drift is inference about which rows landed where. The **slope** is the part
that transferred: 0.907 against public, 0.893 against private — the same number, flat across the
whole range from 0.9633 to 0.9694.

What the measurement buys you:

1. **It sets the gate.** Any delta below ~1σ is noise; chasing it is a coin flip. Convert to OOF
   units by dividing by the slope, and set the gate there or a little above.
2. **It licenses you to stop submitting to decide things.** With a reliable OOF you can kill probes
   offline and spend submissions on genuine finalists. S6E8 decided a month of work on 18 submissions.
3. **It converts an LB gap into a modelling gap you can size.** S6E8's private rank 100 needed OOF
   ≈ 0.97029 against our 0.969434 — a gap of 0.00086 OOF, ≈6.5σ of the residual. That is *real*, and
   knowing it is real is what stops you looking for it in tuning.
4. **It tells you when the public LB is lying — sometimes.** Teams above you on public but not on
   OOF are either overfitting the public split or exploiting something that will not transfer. In
   S6E7 the public toppers sat ~0.0015 above an honest ceiling — five times the residual σ — and
   evaporated. **In S6E8 they sat +0.0005 above us and it was real.** The test in §8 is how to tell
   the two apart *before* the deadline instead of hoping.

**A public score many σ above what your CV can explain is a red flag. A public score a couple of σ
above it is probably just a better solution than yours.**

---

## 5. Do not fit a theory to leaderboard deltas smaller than the split's own resolution

New section, and it exists because S6E8 built an entire phase on a slope that was not there. It is
the most instructive mistake in either competition, because the repo *already contained the number
that refutes it*.

**What happened.** Four submitted stacks of growing size showed a CV→LB offset that fell
monotonically: 0.001304 → 0.001282 → 0.001246 → 0.001166. That was read as a real decay, explained
with a plausible mechanism (a fitted meta-model's OOF grows more optimistic as members are added),
and turned into an operational "transfer ratio" that fell to **33%** by the last step — from which
followed the conclusion that no additional leg could move the leaderboard. The private split:

| stack | OOF | public offset | private offset |
|---|---|---|---|
| 16 legs | 0.967976 | 0.001304 | 0.000884 |
| 19 legs | 0.969228 | 0.001282 | 0.000922 |
| 22 legs | 0.969314 | 0.001246 | 0.000886 |
| 23 legs | 0.969434 | 0.001166 | 0.000866 |
| 25 legs | 0.969425 | 0.001155 | 0.000875 |
| | | range 0.000149 | **range 0.000056, non-monotonic** |

**There was no decay.** 16 legs and 25 legs agree to 0.000009 on the private set. The public series
was one split's noise that happened to arrive in sorted order. And the 33% transfer ratio at the
final step was **83%** on private.

**The tell was available at the time.** The same repo had computed the public split's paired
bootstrap SD at 0.000096. The 22→23 ΔLB it fitted a theory to was 0.000040 — less than half its own
error bar. The rule:

> **Before reading a series of LB points as a trend, compare the series' *range* to the split's
> paired SD. If the range is not several multiples of it, plot the points and refuse to fit them.**

Compute that SD early; it is a paired bootstrap over your own OOF rows at the public split's size and
costs minutes. Get it on the wall next to the residual σ.

**Two corollaries, both about monotonicity as a seductress:**

- Four points arriving in sorted order feel like evidence. With four points, sorted order happens by
  chance about 1 in 12 times under pure noise — and you only notice and write it up in the cases
  where it happened. Selection is doing the work, not the data.
- A mechanism that *would* explain the trend is not evidence that the trend exists. The
  optimism-grows-with-members story was correct physics and still described nothing.

**What saved it.** The conclusion — modelling exhausted — was right, but not for the stated reason.
It survived because someone re-derived it on OOF instead of trusting the LB arithmetic: 17 strict
twins, every stack contribution inside ±0.000013 against a 0.000066 seed-noise floor, no leaderboard
involved. **When a decision rests on LB arithmetic, re-derive it on OOF before acting.** OOF has
691k rows; the public split had 59k.

---

## 6. Ensembling: what actually moves the needle

**Fitted stack or fixed blend — test both, assume neither.** The two competitions disagree, and the
disagreement is the lesson. S6E7's champion was a `bagged FT-PLR : XGBoost` blend at a **fixed 2:1**
ratio; its 5-learner logistic-regression stack scored *worse*, because the meta-learner spread mass
across redundant learners. S6E8's champion was a **23-leg fitted logit stack** that beat the best
equal-weight blend in the repo by **+0.0017 OOF** (0.969434 vs 0.967733) — an enormous margin — with
**10 of its 23 weights negative**. That is
the discriminator: a fitted stack earns its keep when it can *subtract* correlated error, which an
average structurally cannot do. If your pool is a set of near-twins, average them; if it contains
legs whose errors point in different directions, fit weights. Decide it by running both under nested
CV (fit weights on 4 folds, score the held-out 5th), not by carrying a rule over from last time.

**Find the elbow and stop.** S6E8's stack curve, on the private set: 16 → 19 legs bought **+0.00129**;
19 → 25 legs bought **+0.00015 in total**, all of it inside the 0.000119 residual σ. Nineteen legs is
where the competition ended, and the six legs after it are not visible on the private leaderboard.
Plot final score against pool size as you go — the elbow is usually obvious and usually early.

**Solo strength stops predicting stack contribution almost immediately.** S6E8's 17-probe round
measured solo deltas spanning **0.033 AUC** while the corresponding stack contributions spanned
**0.00006**. Once your pool is more than a handful of legs, judge every candidate by its ADD and SWAP
contribution at constant pool size. Never by its solo score, and never by its correlation with the
champion — S6E8's Phase 3 measured directly that max-correlation does not predict stack contribution
either.

**A leg weaker than your pool floor contributes nothing, however decorrelated.** Measured in S6E8 and
worth knowing before you spend a week on an exotic architecture: below the floor, the meta-model
either zeroes it or uses it to fit noise.

**Prefer fixed, round, mechanically-derived membership over fitted selection.** S6E7: a hill-climbed
simplex over 51 models found +0.00014 OOF — 0.5σ, pure selection bias — and one hill-climb collapsed
to weight 1.0 on a single model. S6E8 put this to a direct test by building two finals that differed
in *exactly* the fitted-selection machinery: Final A's 23 legs came through per-probe admission gates
plus a grid search over the regularization constant; Final B applied one solo-OOF threshold to
everything in the repo, pinned the constant, and derived membership mechanically from `runs.csv`.
**They scored identically on the private set — 0.97030 both.** The entire selection apparatus was
worth 0.00000. Build the rule instead; it is cheaper, reproducible, and free of the variance.

**Seed-bagging inside a fold: yes. Re-running at a new split seed: no.** Averaging 3 neural nets per
fold gave a real gain. Averaging the whole S6E7 champion across two outer-CV seeds gave **+0.00009
OOF / −0.00010 LB** — nothing — because stack test predictions are *already* averaged over 5
fold-models, so a new split seed re-randomizes variance that was largely cancelled. The two seeds
flipped 210 of 295,753 predictions. **The spread across per-fold scores is evaluation-fold
difficulty, not test-prediction variance;** do not mistake one for the other and spend 2.3 GPU-hours
on it like we did. Corollary from S6E8: measure your **seed-noise floor** early (re-run one recipe at
a new model seed) and refuse to interpret any contribution below it. That floor was 0.000066, and it
killed several results that looked positive.

**Check the decision rule once, then leave it alone — if the metric has one at all.** For balanced
accuracy, a 2-DOF per-class multiplier search gave in-sample +0.00005 but held-out **−0.00010**;
plain argmax was already optimal, and prior correction was catastrophic. For AUC there is no decision
rule to check (§2). Test these; don't assume them.

---

## 7. Recognizing the wall — the most important skill

Every competition has a point where the signal is exhausted and further work buys zero. Recognizing
it early is worth more than any model.

**The strength ↔ decorrelation tradeoff.** For a blend to improve, you need a model that is both
*strong solo* and *decorrelated* from your champion's legs. Plot solo score against disagreement rate
across every model you own. In S6E7: **Spearman(strength, decorrelation) = −0.84** (p<1e-4) over 38
of our own learners, **−0.90** (p=0.0002) over 11 external models. No probe among 49 was
simultaneously ≥0.950 solo *and* >1.5% decorrelated. When that correlation is strongly negative and
no point sits in the useful quadrant, the ensemble axis is closed. Compute this plot — it turns "I
feel like we're stuck" into a number.

**High disagreement is usually weakness, not diversity.** Repeatedly, a novel architecture disagreed
with the champion on 5% of rows, looked promising, and got *literally zero* blend weight. The
disagreement was the model being wrong in new places. The test: does disagreement come *with*
competitive solo strength? If not, it is noise wearing a costume.

**The representation is the unit of ensemble value, not the model.** S6E8's deepest finding, and it
arrived from three independent directions. Two unrelated architectures given the same feature
representation bought the same thing; six neural architectures given the repo's strongest engineered
family all landed in the same place. The clinching measurement was a tabular foundation model probe:
growing the in-context training set 17× (4k → 69k rows) moved AUC by **+0.007** and plateaued, while
changing *only the representation* — target-encoded instead of raw, same model, same 16k context —
moved it by **+0.019**. When you are choosing what to try next, a new representation dominates a new
architecture and a bigger anything.

**Novel architectures mostly fail, and they fail for a discoverable reason.** Across both runs:
TabICL/TabPFN, ModernNCA, NODE, TabM, FT-Transformer, TabTransformer, RealMLP, a 1-D CNN, an ordinal
CORAL head. Nearly all were *strong-but-correlated* or *decorrelated-but-weak* — the two jaws of the
same wall — and the failures clustered on a single property of the data. In S6E7 it was a diffuse
majority catch-all class that alternative inductive biases could not carve out. In S6E8 it was that
the data is a **value→target lookup table** (per-value rates 0.119–0.986, Pearson(value, rate) =
−0.044, cardinality to 1,459): any model that normalizes an integer code and reads it as a
*magnitude* cannot see a non-monotonic lookup however much capacity or context it is given. **Find
that property and you can predict which architectures will fail without training them.**

**Distinguish supervised from unsupervised levers.** The biggest single feature win in both
competitions was raw per-value target encoding, which is *supervised* (value → label rate). S6E8
spent a probe testing whether 256-bin quantile quantization — the unsupervised analogue — could
recover it. It could not, and could not have: no unsupervised binning recreates a value→label
mapping. Ask "is this lever supervised?" before building the unsupervised imitation of it.

**Check the licence before the benchmark.** TabPFN-3's weights forbid commercial and production use;
TabICL is BSD-3. A no-prize Playground competition is very probably fine under either, but "very
probably fine" is not a licence review and the permissive option cost nothing.

---

## 8. Read the leaderboard's shape before you plan around it

This section is entirely S6E8's, and it is the correction to S6E7's most quotable claim.

**The shakeup dividend is conditional, and the condition is checkable in advance.**

| | S6E7 | S6E8 |
|---|---|---|
| Residual σ of our OOF→LB fit | 0.00029 | 0.000119 |
| Leaders' excess over our honest ceiling | +0.0015 ≈ **5σ** | +0.00146 ≈ **12σ** |
| What the excess was made of | public-split overfitting | a genuinely better shared artifact |
| Spearman(public rank, private rank) | — | **0.9962** |
| Median absolute rank move | — | **13 places** |
| Our shakeup | **+298** | **+6** |

Nothing evaporated in S6E8 because nothing was overfitted. Private #1 was public #1; **19 of the
public top 20 finished in the private top 30.**

**Note what the discriminator was not: size.** The S6E8 leaders were *further* above our ceiling than
the S6E7 leaders were — 12σ against 5σ — and every bit of it held. "Many σ above what your CV can
explain" (§4) flags a score as *unexplained*, which is a question, not a verdict. What separated the
two cases was *what the excess was made of*, and you can usually find that out by reading the top
notebooks — a free afternoon that S6E8 spent on 2026-08-07 and that correctly identified the shared
blend for what it was, three weeks before it mattered.

**Estimate the shakeup's room before you count on it.** It is bounded by the private split's noise,
which you can compute on day one from the test-set row count via a paired bootstrap over your own
OOF. S6E8's private slice was 237k rows with a paired ΔAUC SD of **0.000039**. At that resolution the
ordering is nearly deterministic and there is simply nothing to shake. A plan whose upside is "the
shakeup will sort this out" needs the private split to be small enough to have an opinion; check
that it is.

**When the top of the leaderboard is a shared file, rank stops measuring modelling.** S6E8's private
ranks 50 through 200 span **0.00001 AUC** — 150 teams inside a hundred-thousandth, which is one CSV,
not 150 solutions. 373 teams finished at or above 0.97090. The #1-voted public notebook was four
cells that read a blended `submission.csv` from a private dataset and wrote it out; that one file was
rank 60. Agreement to five decimal places is not convergence.

**Decide the artifact-sharing policy on day one, in writing, with the price attached.** This is the
single highest-leverage strategic decision and it cannot be retrofitted at the deadline. Using public
artifacts is legal under Kaggle rules and is what most of the teams above us were doing. S6E8's rule
was that every model shipped is one we trained; it was priced at ~180 places on 2026-08-12,
reaffirmed with the price known, and **the private split charged it in full and refunded none of
it.** That is a defensible outcome — but only because it was a decision. Write down which game you
are playing (best model you can build / best score you can obtain) before the leaderboard tempts you,
because the two have different optimal moves and drifting between them gets you the worst of each.

**Distinguish paired differences from absolute scores; they live on different scales.** S6E8's
paired bootstrap of our stack against a rival, on the public split: SD of the paired ΔAUC =
**0.000096**, SD of a single absolute score = **0.000631**. So a 0.0003 gap to a specific rival is
3σ and would replicate, while the entire spread from rank 248 to rank 1 was smaller than one score's
own noise. Both statements are true simultaneously, and confusing them is how a leaderboard eats a
month. Compare *against a named opponent's predictions*, never against a rank.

> **Once the shape is established, stop reading the public leaderboard for guidance.** Its top may be
> a shared file, the ordering inside a few σ is split noise, and acting on either is the failure mode.

---

## 9. Final submission selection

Kaggle lets you pick 2 finals, and **only from entries you actually submitted** — so submit anything
you might want to select, even if you don't expect it to top the public LB.

The pattern, unchanged across both runs:

- **Final A** = the honest champion: best OOF, simplest defensible recipe.
- **Final B** = the *variance-reduced* twin of the same recipe. Chosen not because the public LB
  likes it — in S6E7 it scored 0.00010 worse, in S6E8 0.00002 worse — but because it has strictly
  lower prediction variance, which is the right property for an unseen split.

"Variance-reduced" means: remove the steps that were fitted on data. In S6E7 that was averaging over
two independent fold partitions. In S6E8 it was replacing every fitted selection step — the per-probe
admission gates and the grid search over the stacker's regularization constant — with one fixed
threshold and a pinned constant, with membership derived mechanically from the run log.

**S6E8 settled what this hedge costs: nothing.** Final A and Final B scored **0.97030 private,
exactly tied.** The hedge is free, and in the world where the fitted selection *had* overfit, it is
the one that survives. Take it every time.

What we deliberately did **not** select, in either competition: any banked public-notebook submission
scoring above our own ceiling. In S6E7 those were exactly the entries that evaporated. In S6E8 they
would have scored — and that is the §8 decision, made in advance and on the record, not a
final-slot improvisation.

Rule of thumb: **one submission you believe for CV reasons, one that is the same idea with less
fitted machinery. Never spend a final slot on a public score you cannot explain.**

---

## 10. Checklist for the next competition

**Week 1 — infrastructure**
- [ ] Run collector: push → poll → parse metrics → archive artifacts → append log row.
- [ ] `experiments/runs.csv` in git, with a long free-text notes column.
- [ ] Per-learner OOF + test probability artifacts saved for *every* run, keyed by run id.
- [ ] Global frozen CV split, documented, never changed.
- [ ] Automated submission-slot queue so no daily slot is wasted — the paired points are the
      defence against §5.
- [ ] A single baseline end-to-end, submitted, logged. Establish the OOF↔LB pair early.

**Week 1 — the three numbers to put on the wall**
- [ ] **Residual σ** of the OOF→LB fit, and its **slope**. Gate = ~1σ, converted to OOF units.
- [ ] **Paired-bootstrap SD** at the public split's row count, and at the private split's. Nothing
      smaller than this is readable; the private one tells you how much shakeup room exists.
- [ ] **Seed-noise floor**: re-run one recipe at a new model seed. Nothing below this is a result.

**Week 1 — the one strategic decision**
- [ ] Read the top public notebooks. Is the frontier a set of independent solutions or a shared file?
- [ ] Write down the artifact-sharing policy, with its price in places, before it is tempting.
- [ ] Write down what the metric is invariant to, and cross off every knob that touches only that.

**Ongoing — per probe**
- [ ] Written hypothesis + mechanism + pre-registered gate.
- [ ] Strict-twin ablation: exactly one thing changes.
- [ ] Leakage audit of every fitted transform in the new path.
- [ ] **Two gates**: solo OOF, then ADD *and* SWAP contribution at constant pool size. Only the
      second is a shipping decision.
- [ ] Result logged with the *mechanism*, not just the score.

**Midpoint — reliability**
- [ ] Recompute Spearman(OOF, LB), the slope and the residual σ over all runs.
- [ ] Compute Spearman(solo strength, disagreement-vs-champion) over all models. If strongly
      negative with an empty useful quadrant, the ensemble axis is closed.
- [ ] Plot final score against pool size. Where is the elbow? You are probably past it.
- [ ] Sweep the back catalogue: any model judged only on OOF that has never been stack-tested.
- [ ] Before believing any LB trend: is the series' range several multiples of the paired SD?

**Endgame**
- [ ] Re-derive every LB-based conclusion on OOF before acting on it. If a claim rests on deltas
      below the split's resolution, it is not a claim.
- [ ] Run the negative-result round: strict twins for every assumption that was only ever argued.
- [ ] Stop when probes stop clearing gate 2. Extra probes past the wall bought 0.00000 in both runs.
- [ ] Final A = honest champion. Final B = the same recipe with every fitted step removed. The hedge
      is free.
- [ ] Do not select a public score your CV cannot explain, however tempting the rank.

---

## 11. The two-sentence version

Build the logging and artifact infrastructure first, freeze one CV split, keep every OOF matrix you
ever produce, and measure three numbers early — the OOF→LB slope and residual σ, the split's paired
bootstrap SD, and your seed-noise floor — because together they tell you what a real improvement
looks like and, more importantly, what is too small to be anything at all. Then trust that
instrument all the way to its conclusion: judge legs by stack contribution rather than solo strength,
refuse to fit a story to deltas below the resolution you measured, decide on day one whether you are
playing for the best model or the best score, and when the instrument says the signal is exhausted,
stop and submit the model you can defend — sometimes that earns 298 places in the shakeup, and
sometimes it earns six, and the method is the same either way.
