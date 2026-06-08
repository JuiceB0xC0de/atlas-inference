# Experiment 03 — Feature-State Dynamics (Gate 3)

**Date:** 2026-06-08 · **Status:** PASSED (one-hop and chain composition).
**Model:** `Qwen/Qwen3-8B-Base` · **SAE:** `Qwen/SAE-Res-Qwen3-8B-Base-W64K-L0_50` (resid_post, TopK k=50)
**Runs:** `modal run experiments/modal_transition_03a.py` (03A), `modal run experiments/modal_transition_03c_chain.py` (03C)

## Question

Can feature states at one layer predict feature states at the next? If the atlas can compute layer-to-layer transitions, you get a white-box surrogate where every intermediate state is a concept you can read.

## 03A — One-hop transition: L18 → L19

N=66,409 tokens (wikitext-2-raw-v1 test), λ=1000.0, basis overlap=211/2048 (~10%).

Operator: linear ridge regression on top-2048 most-active features per layer (~4M params on ~10-15k token pairs).

| condition       | cosine | top-k overlap | R²    |
|-----------------|--------|---------------|-------|
| fitted_T        | 0.880  | 0.742         | 0.917 |
| mean_predictor  | 0.431  | 0.249         | 0.000 |
| shuffle_null    | 0.371  | 0.237         | -0.039|

Math-direction thread:
- math feats in basis: L18=7/20, L19=8/20 (rare math features excluded from top-2048)
- `cos(T(math_in), math_out) = 0.751` vs random ctrl 0.050
- T learns to carry the math concept across the basis change despite near-zero raw cosine

**One-hop proves linear recoverability.** R²=0.917 is high partly because the residual stream is additive (`z₁₉ ≈ z₁₈ + δ`). Most of z₁₉ is z₁₈'s content re-expressed in a new dictionary. A linear map handles change-of-basis even if the layer computed little. This is necessary and encouraging, but does not yet prove the map computes. The load-bearing test is composition.

## 03C — Chain composition: L18 → L19 → ... → L35

17 composed hops, each using a per-layer transition matrix T_L trained independently on top-2048 active features. Cross-basis throughout (different learned dictionaries per layer, ~10% overlap on average).

N=55,378 tokens (wikitext-2-raw-v1 test, after BOS exclusion), splits: n_train=47,071, n_test=8,307.

### Teacher-forced per-hop (real z_L as input)

| layer | cosine | R²    | top-k overlap |
|------:|--------|-------|---------------|
| 19    | 0.8821 | 0.747 | 0.748         |
| 20    | 0.8954 | 0.770 | 0.764         |
| 21    | 0.8846 | 0.756 | 0.759         |
| 22    | 0.8636 | 0.721 | 0.727         |
| 23    | 0.8661 | 0.736 | 0.728         |
| 24    | 0.8810 | 0.768 | 0.745         |
| 25    | 0.8966 | 0.802 | 0.769         |
| 26    | 0.9002 | 0.810 | 0.777         |
| 27    | 0.9006 | 0.814 | 0.780         |
| 28    | 0.9003 | 0.816 | 0.780         |
| 29    | 0.9095 | 0.835 | 0.787         |
| 30    | 0.9165 | 0.852 | 0.790         |
| 31    | 0.9255 | 0.869 | 0.795         |
| 32    | 0.9267 | 0.872 | 0.793         |
| 33    | 0.9264 | 0.869 | 0.791         |
| 34    | 0.9130 | 0.850 | 0.767         |
| 35    | 0.9221 | 0.819 | 0.779         |

Per-hop teacher-forced R² **climbs** in deeper layers: 0.75 → 0.87. The residual stream becomes more linearly predictable late in the network.

### Free-running composition (predicted-into-predicted, no teacher forcing)

| layer | cosine | R²    | top-k overlap |
|------:|--------|-------|---------------|
| 19    | 0.8821 | 0.747 | 0.748         |
| 20    | 0.8608 | 0.696 | 0.703         |
| 21    | 0.8304 | 0.646 | 0.666         |
| 22    | 0.7948 | 0.585 | 0.624         |
| 23    | 0.7698 | 0.555 | 0.591         |
| 24    | 0.7594 | 0.538 | 0.573         |
| 25    | 0.7424 | 0.516 | 0.556         |
| 26    | 0.7295 | 0.493 | 0.542         |
| 27    | 0.7177 | 0.473 | 0.534         |
| 28    | 0.6981 | 0.447 | 0.510         |
| 29    | 0.6962 | 0.442 | 0.504         |
| 30    | 0.6948 | 0.430 | 0.491         |
| 31    | 0.6882 | 0.424 | 0.473         |
| 32    | 0.6817 | 0.418 | 0.460         |
| 33    | 0.6749 | 0.423 | 0.450         |
| 34    | 0.6804 | 0.436 | 0.446         |
| 35    | 0.7606 | 0.445 | 0.511         |

Decay saturates around R²=0.42–0.45 from hop ~12 and **ticks up at L35** — it does not cliff toward the floor.

### Baselines at z₃₅

| condition | cosine | R²    | top-k overlap |
|-----------|--------|-------|---------------|
| composed (17 hops) | 0.7606 | 0.445 | 0.511 |
| direct 18→35 | 0.8094 | 0.561 | 0.591 |
| mean predictor | 0.5310 | -0.000 | 0.329 |

The composed chain (0.445) is well above the mean-predictor floor (-0.000) and below the direct shortcut (0.561). Composition leaks ~0.12 R² versus one big map.

### Basis overlap (shared features between adjacent layer top-2048)

| layer | overlap |
|------:|--------:|
| 18→19 | 203     |
| 19→20 | 207     |
| 22→23 | 213     |
| 24→25 | 223     |
| 25→26 | 262     |
| 29→30 | 186     |
| 33→34 | 175     |
| 34→35 | 113     |

Basis overlap is consistently low (~10%), confirming cross-basis is real throughout the chain.

## What Gate 3 proves

1. **Linear feature-space dynamics compose to a meaningful, bounded degree.** R²=0.445, cosine=0.76, top-k=0.51 at z₃₅ through 17 cross-basis hops. The database can run a linear forward pass.
2. **Per-hop operators are individually excellent.** Teacher-forced R² climbs from 0.75 to 0.87 in deeper layers — the residual stream becomes more linearly predictable late.
3. **Composed < direct.** The 17-hop chain (0.445) leaks ~0.12 R² versus a single direct 18→35 map (0.561). Per-hop operators are composable-but-lossy.
4. **The math concept survives transit.** `cos(T(math_in), math_out) = 0.751` vs random ctrl 0.050 (one-hop). The transition matrix learns to carry semantic content across the basis change.

## What Gate 3 does NOT prove

1. **Top-2048 subspace only.** The transition operates on the top-2048 most-active features per layer. Rare features — including ~12/20 of the 01B math features — are outside this basis. "The map computes" is shown for the common-feature manifold, not the full 65,536-dim state.
2. **Residual carry-over is doing real work.** The direct 18→35 map hitting R²=0.56 means much of z₃₅ is linearly predictable from z₁₈ because the residual stream carries forward. This is linear composability, not cleanly isolated per-layer nonlinear computation.
3. **This is feature-state R², not a functional decode.** I haven't yet asked whether the composed z₃₅ decodes to the right tokens through the unembed. R²=0.445 in feature space doesn't directly tell you whether the atlas produces the right next-token predictions.

## Next experiments

1. **Functional decode of composed z₃₅** — decode predicted vs actual z₃₅ → top-k tokens, overlap. Turns R²=0.445 into "does it actually say the right thing?" The real payoff test.
2. **Math direction through the 17-hop chain** — does `cos(chain(math_z₁₈), math_z₃₅)` survive the full composed transit?
3. **Nonlinear operators** — does a small MLP per hop close the composed-vs-direct gap (0.445 → 0.561), capturing the computation a linear map can't?

Full chain data: [`artifacts_03c_chain.json`](artifacts_03c_chain.json).