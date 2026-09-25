"""Phase 27 G4 -- the one axis the finals do not use: MODEL FAMILY.

Final A is rank_mean(TEXbag, TEXF, TEX2F) and all three are lgb near-twins -- Phase 25
adopted it on a variance argument worth +0.000008 OOF, explicitly not on score. Family
diversity is the same argument with real decorrelation behind it instead of three seeds of
one recipe, and the archive has never had a non-lgb leg in a shipped ours-only final:

  CatTEX  3da3ad60  cat  OOF 0.946114  LB 0.94625   (0.000023 under TEX)
  E4r0    7c151fb7  xgb  OOF 0.945941  LB 0.94610
  PS      7eb59fd6  lgb  OOF 0.946139  LB --        (te_pseudo: ties solo, different
                                                     REPRESENTATION, so a diversity candidate
                                                     even though Phase 26 rejected it as a
                                                     replacement)

Judged by ADD and SWAP at constant pool size, never by solo score (playbook section 6).
Read-only over archived artifacts. No slot.

NOTE on the family offset: Phase 22 measured CatBoost carrying its own OOF->LB offset, and
the pair here shows it -- TEX 0.946122 -> 0.94635 against CatTEX 0.946114 -> 0.94625, so cat
gives up ~0.00010 of board at matched OOF. Any OOF gain from a cat leg must be discounted by
that before it counts as a board gain. Stated here so it is applied, not rediscovered.
"""
import itertools, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from scipy.stats import rankdata

tr = pd.read_csv("data/train.csv", usecols=["id", "Will_Buy_EV"])
y = (tr["Will_Buy_EV"] == "Yes").astype(int).values
LEGS = {"TEXbag": ("50d84171", "lgb"), "TEXF": ("1b80818a", "lgb"),
        "TEX2F": ("f0ba7251", "lgb"), "TEX": ("43db301d", "lgb"),
        "PS": ("7eb59fd6", "lgb"), "CatTEX": ("3da3ad60", "cat"),
        "CatIncome": ("74c16e6e", "cat"), "E4r0": ("7c151fb7", "xgb"),
        "E4": ("2b471d6b", "xgb")}
P, R = {}, {}
for nm, (rid, lr) in LEGS.items():
    d = pd.read_csv(f"experiments/preds/{rid}/oof_proba_{lr}.csv").sort_values("id")
    P[nm] = d["proba"].values
    R[nm] = rankdata(P[nm]) / len(P[nm])
    print(f"{nm:10s} {lr:4s} solo OOF {roc_auc_score(y, P[nm]):.6f}")

FINAL_A = ["TEXbag", "TEXF", "TEX2F"]
def blend(names):
    return roc_auc_score(y, np.mean([R[n] for n in names], axis=0))
base = blend(FINAL_A)
print(f"\nFinal A = rank_mean{tuple(FINAL_A)}  OOF {base:.6f}  (archived d8ef7c11: 0.946158)")

print("\n--- correlation of OOF RANKS against the Final A pool mean ---")
pool = np.mean([R[n] for n in FINAL_A], axis=0)
for nm in LEGS:
    print(f"  {nm:10s} corr {np.corrcoef(R[nm], pool)[0,1]:.5f}   "
          f"family {LEGS[nm][1]}")

print("\n--- ADD: Final A + one leg (pool grows 3 -> 4) ---")
for nm in LEGS:
    if nm in FINAL_A: continue
    s = blend(FINAL_A + [nm])
    print(f"  + {nm:10s} OOF {s:.6f}   ADD {s-base:+.6f}")

print("\n--- SWAP at CONSTANT pool size 3: replace TEX2F (the weakest of the three) ---")
for nm in LEGS:
    if nm in FINAL_A: continue
    s = blend(["TEXbag", "TEXF", nm])
    print(f"  TEX2F -> {nm:10s} OOF {s:.6f}   SWAP {s-base:+.6f}")

print("\n--- best cross-family pools, exhaustive over the 9 legs, k = 3..5 ---")
res = []
for k in (3, 4, 5):
    for c in itertools.combinations(LEGS, k):
        res.append((blend(list(c)), k, c))
res.sort(reverse=True)
for s, k, c in res[:12]:
    fam = "".join(sorted({LEGS[n][1][0] for n in c}))
    print(f"  {s:.6f}  k={k}  fam={fam:3s}  {'+'.join(c)}")
print(f"\n  ... {len(res)} pools evaluated; Final A sits at rank "
      f"{1+sum(1 for s,_,_ in res if s > base)} of {len(res)}")
print("\ngate +0.0000949   seed floor 0.000038   cat family offset ~-0.00010 LB at matched OOF")
