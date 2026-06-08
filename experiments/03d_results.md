# Experiment 03D — Functional Decode of Composed Chain (Gate 3 payoff)

**Date:** 2026-06-08 · **Status:** MIXED — chain reaches mouth (0.29 end-to-end), concept direction destroyed in transit.
**Model:** `Qwen/Qwen3-8B-Base` · **SAE:** `Qwen/SAE-Res-Qwen3-8B-Base-W64K-L0_50` (resid_post, TopK k=50)
**Run:** `modal run experiments/modal_transition_03d_decode.py` — n_decode=2000

## Question

03C showed R²=0.445 over 17 composed linear hops. Does that statistical fidelity translate to behavioral fidelity at the vocabulary boundary? And does the math concept that Gate 2 isolated at L18 survive transport through the chain to L35?

## (a) Functional decode

Composed z₃₅ decoded through `scatter → W_dec + b_dec → RMSNorm → lm_head → top-10 tokens`. Two reference points:
- **vs actual_z35**: isolates z-prediction quality (same decode path, different z)
- **vs model_real**: end-to-end (folds in SAE-decode loss from Gate 1)

| method | vs actual_z35 | vs model_real | math | numeric | code | other |
|--------|:---:|:---:|:---:|:---:|:---:|:---:|
| actual_z35 | 1.000 | 0.472 | 0.001 | 0.071 | 0.006 | 0.922 |
| **free_chain** | **0.440** | **0.291** | 0.000 | 0.067 | 0.003 | 0.930 |
| direct 18→35 | 0.516 | 0.345 | 0.000 | 0.081 | 0.004 | 0.914 |
| teacher_forced | 0.729 | 0.457 | 0.000 | 0.076 | 0.006 | 0.918 |
| mean_predictor | 0.258 | 0.148 | 0.000 | 0.000 | 0.000 | 1.000 |

**The chain reaches the mouth.** Free chain overlap vs actual_z35 = 0.440, well above mean_predictor (0.258). End-to-end, the composed chain recovers 29.1% of the model's top-10 next-token predictions through 17 composed linear maps and an SAE decode.

**The distribution shape is preserved.** Free chain produces 6.7% numeric tokens vs actual's 7.1%. Mean_predictor produces 0% numeric and 100% function words — the default mode of the residual stream. The chain is not in default mode. It is degraded but alive.

**The SAE-decode ceiling is 0.472.** Even actual z₃₅ only captures 47.2% of the model's top-10 tokens. This means the SAE round-trip itself loses ~53% of token information. The chain's 0.291 end-to-end = 0.440 (z-prediction) × 0.472 (SAE ceiling) ≈ 0.208, plus some alignment benefit. The two losses compound but don't simply multiply.

**Composed < direct.** Direct 18→35 (0.516/0.345) outperforms the chain (0.440/0.291), consistent with 03C's composed < direct R² gap.

## (b) Math direction through the chain — FAILS

```
cos(chain(math_18), math_35) = -0.053
cos(random through chain)       = -0.073
```

Indistinguishable from random. The math concept that was separable at L18 (p≈0.02, Gate 2) is **destroyed** by 17 hops of linear composition.

```
chain_math tokens:  " ", ",", "and", "(", "in", ".", "the", "a", "to", "-"
actual_math35 tokens: "1", "2", "3", "4", "5", "0", "6", ".", "7", "8"
```

Chain-transported math decodes to pure function words — the residual stream's default mode. Actual math at L35 decodes to digits (the shared numeric axis, consistent with Gate 2's L35 finding).

Only 7/20 math features were in the L18 top-2048 basis, 11/20 at L35. The concept was undersampled going in, and 17 linear maps scattered what little signal remained.

## What 03D proves

1. **R²=0.445 converts to a real behavioral signal at the vocabulary boundary.** 29% of the model's top-10 predictions are recoverable end-to-end through 17 composed linear maps and an SAE decode. This is well above the mean-predictor floor (15%) and confirms the chain doesn't collapse.

2. **The distribution shape is preserved.** The chain produces similar numeric/code/other fractions to actual z₃₅. It's not in default mode.

3. **Concept directions do NOT survive composition.** The math-specific semantic content isolated at L18 is destroyed by 17 hops of linear composition. The aggregate statistics compose; individual semantics don't.

## What 03D does NOT prove

- That any concept direction survives transit. Math was the only concept tested through the chain. It's possible that stronger/larger concept directions fare better, but the math direction was significant at p≈0.02 at L18 — it's not a weak signal going in.
- That the chain produces fluent text. 29% top-10 overlap is meaningful but far from useful generation.

## Path forward

The aggregate composition works but concept transport fails. The bottleneck is likely:

1. **Basis width.** Top-2048 captures common features but misses rare ones (only 7/20 math features). A wider basis or importance-weighted basis could retain concept signal.
2. **Linearity.** Ridge regression captures change-of-basis but not the nonlinear MLP computation. Transcoder-style sparse→nonlinear→sparse operators could capture the δ that linear maps scatter.
3. **Both.** Wider basis + nonlinear hops is the most likely path to concept survival.

Full data: [`artifacts_03d_decode.json`](artifacts_03d_decode.json)