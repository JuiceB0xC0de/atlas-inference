# Experiment 01B — Contrastive Concept Direction (Gate 2)

**Date:** 2026-06-08 · **Status:** PASSES at L18 (p≈0.02), FAILS at L35 (p≈0.09).
**Model:** `Qwen/Qwen3-8B-Base` · **SAE:** `Qwen/SAE-Res-Qwen3-8B-Base-W64K-L0_50` (resid_post, TopK k=50)
**Run:** `modal run experiments/modal_bridge_01b.py` — `n_shuffles=200`, `gaussian×5`

## Question

Can a contrastive, prompt-derived concept direction in SAE feature space produce
concept-matched vocabulary, beyond what matched controls produce?

```
math_delta = mean_z(math prompts) − mean_z(control prompts)   # 65,536-dim, contrastive
residual   = W_dec @ math_delta                               # NO b_dec (contrast removes common mode)
logits     = final_RMSNorm(residual) @ lm_head.T
classify top-20 tokens → math_specific / numeric / code_syntax / other   (pre-registered scorer)
```

## Result

Permutation test over label shuffles: `p = (null≥observed + 1) / (n_shuffles + 1)`, n=200.

| layer | math_delta (math-specific) | null mean | null max | null ≥ obs | p | verdict |
|------:|:--:|:--:|:--:|:--:|:--:|:--|
| 18 | 0.200 | 0.008 | 0.25 | 3/200 | **0.020** | PASS (significant) |
| 35 | 0.150 | 0.032 | 0.60 | 17/200 | 0.090 | FAIL (null tail) |

All non-shuffle controls ≈ 0.00 math-specific at both layers: `code_delta`, `poetry_delta`,
`random_real_matched`, and `gaussian×5` (math_max = 0.000). The shuffled-label null is the
discriminating control.

## Interpretation

- **L18 — math is a separable semantic concept.** The direction decodes to math *words*
  (`circumference, consecutive, 平方, nonzero`), **zero digits**, significantly above a
  label-shuffle null (p≈0.02). ~1.5% of random 40/40 splits reach 0.20 by chance — the null
  tail is real but small.
- **L35 — math has collapsed into the shared numeric/output axis.** The direction is **50%
  digits — the same numeric fraction as `code_delta`** — and its math-specific notation
  (`½ ∜ π`, 0.15) sits inside the null tail (p≈0.09; one shuffle reached 0.60). At the output
  layer, "math" ≈ "numbers," which code shares.
- This **inverts the prior expectation** that L35 would be the cleaner math layer. L35's clean
  vocabulary is the *non-specific* numeric axis; concept-specificity lives at L18.

## Caveats

- p≈0.02 at L18 is significant but **not airtight** — the null is not empty (3/200). A larger
  prompt pool (e.g. 100 math / 100 control) shrinks the combinatorial null tail toward p<0.01.
  More shuffles will not help (the tail is real, not undersampled).
- Single prompt set, single SAE variant (l0_50), single layer pair.

## Implication for Gate 3

Probe layer-to-layer **dynamics at L18**, where the concept is cleanly separable. Use L35 for
output / "mouth" diagnostics only.

Full per-condition tokens + all 200 null runs: [`artifacts_01b.json`](artifacts_01b.json).
