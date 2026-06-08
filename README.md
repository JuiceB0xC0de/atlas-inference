# atlas-inference

**Can a static feature atlas of a frozen language model be driven as a runnable, legible surrogate of that model?**

I built a ~570 MB SQLite atlas of `Qwen3-8B-Base` — 65,536 SAE features × 36 layers × 2 sparsity variants (L0_50, L0_100). It combines SAE feature statistics with channel-level activation, compliance, taxonomy, OV, DAS, SNMF, and cross-layer geometry tables. It maps the model's internals at high resolution, and it's fully queryable.

This repo asks the next question: **is that map enough to compute?**

Not "build a better LLM" — the model already exists. The question is whether its forward computation is *recoverable in the SAE feature basis*: whether a sequence of sparse, named, inspectable feature states can reproduce a piece of what the dense network does. If so, you get a white-box surrogate where every intermediate state is a concept you can read.

## Status

Three gates tested. Aggregate composition passes; concept transport fails.

## Gates

I test load-bearing premises in order. Each one must hold for the next to matter:

1. **Bridge** — Can atlas feature directions produce coherent vocabulary through the model's real unembed? ✅
2. **Concept** — Can a contrastive, prompt-derived direction in feature space produce concept-matched vocabulary? ✅ (L18) / ❌ (L35)
3. **Dynamics** — Can feature states at one layer predict feature states at the next? ✅ aggregate / ❌ concept transport

---

## Gate 1: Decoder → Unembed Bridge — PASSED

The bridge works, with caveats. SAE feature directions, decoded through the real SAE decoder and projected through the model's final RMSNorm + unembed, produce semantically coherent vocabulary. Individual atlas-selected features decode into tight bilingual clusters — poetry, humor, ML-infrastructure, UX/design, math — not noise.

### Step 0 — Reconstruction gate

`encode → TopK(50) → decode (x̂)`, measured against the real residual `x`:

| layer | cos (no bdec → bdec) | top-12 overlap (b_dec) | overlap (no BOS) |
|------:|:----------------------:|:-----------------------:|:-----------------:|
| 3     | 0.81 → 0.97            | 0.595                   | 0.595             |
| 18    | 0.72 → 0.90            | 0.452                   | 0.488             |
| 35    | 0.58 → 0.97            | 0.690                   | 0.679             |

Key findings:

- **`b_dec` is mandatory.** Cosine jumps 0.58 → 0.97 at L35 when the decoder bias is included.
- **Token overlap is the honest metric.** Cosine 0.97 does not mean faithful — only 69% of the model's top-12 tokens survive the SAE round-trip at L35. The SAE is genuinely lossy at the vocab boundary.
- **The 0.69 is real, not a BOS artifact.** Excluding BOS changed overlap by <0.01 at L3 and L35; L18 showed a modest 0.036 shift. Loss is spread across ordinary prose tokens. Digits reconstruct perfectly (1.0); prose tokens scatter 0.2–0.5.
- **A k=50 SAE round-trip drops ~31% of the head's top tokens.** This bounds the claim: the map speaks, but it is not an exact surrogate.

### Step 01A — Feature → tokens

Atlas-selected features (ranked by `topic_fstat`) decoded individually to top-k tokens, vs controls:

- **Selective features → crisp, monosemantic, bilingual concept clusters.** Poetry, humor, ML-infrastructure, UX/design, fairness — each feature has a clear semantic identity that emerges from the decode, not from the atlas labels.
- **Gaussian (random residual directions) → junk.** The unembed is not doing the work — arbitrary directions produce garbage.
- **Norm-matched random real columns → mixed.** Some random features produce semi-coherent clusters, but with weaker category focus than selective features. The honest ranking: gaussian < random-real < selective.

### Step 01B — Contrastive concept direction → math-specific tokens

Can a prompt-derived contrastive direction (`mean_z(math prompts) − mean_z(control prompts)`) in SAE feature space produce math-specific vocabulary beyond what matched controls produce?

```
math_delta = mean_z(math prompts) − mean_z(control prompts)
residual   = W_dec @ math_delta                    # no b_dec — contrast subtracts common mode
logits     = final_RMSNorm(residual) @ lm_head.T
```

Scored with a pre-registered 4-bucket classifier (math_specific / numeric / code_syntax / other). Digits are NOT math-specific. Bare `_`, `^`, `$` are NOT math.

**L18 — PASSES (p≈0.02).** math_specific = 0.20, all from math words (circumference, consecutive, 平方, nonzero), zero digits. 3/200 shuffles ≥ observed (null mean 0.008, null max 0.25). Significant and real, but not airtight — ~1.5% of random 40/40 splits reach 0.20 by combinatorics. The concept is cleanly, specifically separable at mid-stack.

**L35 — FAILS (p≈0.09).** math_specific = 0.15, but 17/200 shuffles ≥ observed (null mean 0.032, null max 0.60). At the output layer, the math direction is 50% digits — the shared numeric axis that math and code occupy together. Random label splits also surface this axis. Math-specific notation (½, ∜, π) sits inside the null's tail.

The finding: concepts live semantically at mid-stack (L18), where they're cleanly separable. At late-stack (L35), concepts dissolve into generic output vocabulary. The shuffled-label null earned its keep — without it L35 would have been called a pass.

---

## Gate 2: Concept — PASSED at L18 (p≈0.02), FAILED at L35

n=200 shuffles + 5 gaussians. All non-shuffle controls ≈ 0.00 math-specific. Full results: [`experiments/01b_results.md`](experiments/01b_results.md).

---

## Gate 3: Feature-State Dynamics — PASSED

Can feature states at one layer predict feature states at the next? This is the load-bearing test for the core premise.

### 03A — One-hop transition: L18 → L19

N=66,409 tokens (wikitext-2-raw-v1 test), λ=1000.0, basis overlap=211/2048 (~10%).

| condition       | cosine | top-k overlap | R²    |
|-----------------|--------|---------------|-------|
| fitted_T        | 0.880  | 0.742         | 0.917 |
| mean_predictor  | 0.431  | 0.249         | 0.000 |
| shuffle_null    | 0.371  | 0.237         | -0.039|

Math-direction thread: `cos(T(math_in), math_out) = 0.751` vs random ctrl 0.050. The transition matrix learns to carry the math concept across the basis change despite near-zero raw cosine between in/out math features.

**One-hop proves linear recoverability.** R²=0.917 is high partly because the residual stream is additive (`z₁₉ ≈ z₁₈ + δ`). Most of z₁₉ is z₁₈'s content re-expressed in a new dictionary. A linear map handles change-of-basis even if the layer computed little. This is necessary and encouraging, but does not yet prove the map computes. The load-bearing test is composition.

### 03C — Chain composition: L18 → L19 → ... → L35

17 composed hops, each using a per-layer transition matrix T_L trained independently on top-2048 active features. Cross-basis throughout (different learned dictionaries per layer, ~10% overlap).

**Teacher-forced per-hop** (each hop uses the real z_L as input):

| layer | cosine | R²    | top-k |
|------:|--------|-------|-------|
| 19    | 0.8821 | 0.747 | 0.748 |
| 22    | 0.8608 | 0.721 | 0.724 |
| 26    | 0.9002 | 0.810 | 0.777 |
| 30    | 0.9165 | 0.852 | 0.790 |
| 35    | 0.9267 | 0.872 | 0.793 |

**Free-running composition** (predicted-into-predicted, no teacher forcing):

| layer | cosine | R²    | top-k |
|------:|--------|-------|-------|
| 19    | 0.8821 | 0.747 | 0.748 |
| 22    | 0.7948 | 0.585 | 0.624 |
| 26    | 0.7295 | 0.493 | 0.542 |
| 30    | 0.6948 | 0.430 | 0.491 |
| 35    | 0.7606 | 0.445 | 0.511 |

**Baselines at z₃₅:**

| condition | cosine | R²    | top-k |
|-----------|--------|-------|-------|
| composed (17 hops) | 0.7606 | 0.445 | 0.511 |
| direct 18→35 | 0.8094 | 0.561 | 0.591 |
| mean predictor | 0.531 | -0.000 | 0.329 |

**The chain composes.** R²=0.445 over 17 composed linear hops, well above the mean-predictor floor (R²=0.000). The decay saturates around 0.42–0.45 from hop ~12 and ticks up at z₃₅ — it does not cliff toward the floor. Error accumulates but bounds. A chain of 17 linear operators runs a feature-space forward pass and still predicts the final layer.

### What Gate 3 proves (honestly)

1. **Linear feature-space dynamics compose to a meaningful, bounded degree.** R²=0.445, cosine=0.76, top-k=0.51 at the final layer through 17 hops of cross-basis transitions. The database can run a linear forward pass.
2. **Per-hop operators are individually excellent** (teacher-forced R² 0.72→0.87, climbing in deeper layers). The residual stream becomes more linearly predictable late in the network.
3. **Composed < direct.** The 17-hop chain (0.445) leaks ~0.12 R² versus a single direct 18→35 map (0.561). The per-hop linear operators capture a composable-but-lossy approximation — they don't capture more than a direct linear map.
4. **The math concept survives one hop.** In the 03A one-hop test, `cos(T(math_in), math_out) = 0.751` vs random ctrl 0.050. The transition matrix learns to carry semantic content across a single basis change.
5. **The chain reaches the mouth.** 03D: free chain recovers 44.0% of actual z₃₅ tokens, 29.1% of model's real top-10 predictions end-to-end. Well above mean_predictor (25.8%/14.8%). The distribution shape is preserved.

### 03D — Functional decode of composed z₃₅

The 03C result (R²=0.445) was internal to feature space. 03D asks the behavioral question: does the composed z₃₅ decode to the right tokens?

Composed z₃₅ decoded through `scatter → W_dec + b_dec → RMSNorm → lm_head → top-10`, compared to two references:

| method | vs actual_z35 | vs model_real | numeric | other |
|--------|:---:|:---:|:---:|:---:|
| actual_z35 | 1.000 | 0.472 | 0.071 | 0.922 |
| **free_chain** | **0.440** | **0.291** | 0.067 | 0.930 |
| direct 18→35 | 0.516 | 0.345 | 0.081 | 0.914 |
| teacher_forced | 0.729 | 0.457 | 0.076 | 0.918 |
| mean_predictor | 0.258 | 0.148 | 0.000 | 1.000 |

**The chain reaches the mouth.** 29.1% of the model's top-10 predictions recovered end-to-end through 17 composed linear maps and an SAE decode, well above the 14.8% floor. The distribution shape is preserved (6.7% numeric vs actual's 7.1%).

**But concept directions don't survive transit.** The math concept separable at L18 (p≈0.02) is destroyed by 17 hops: `cos(chain(math_18), math_35) = -0.053` vs random ctrl -0.073. Chain-transported math decodes to function words; actual math_35 decodes to digits.

Full results: [`experiments/03d_results.md`](experiments/03d_results.md)

### What Gate 3 does NOT prove

1. **Top-2048 subspace only.** The transition operates on the top-2048 most-active features per layer. Rare features — including ~12/20 of the 01B math features — are outside this basis. "The map computes" is shown for the common-feature manifold, not the full 65,536-dim state.
2. **Residual carry-over is doing real work.** The direct 18→35 map hitting R²=0.56 means much of z₃₅ is linearly predictable from z₁₈ because the residual stream carries forward. This is linear composability, not cleanly isolated per-layer nonlinear computation.
3. **Concept directions do NOT survive composition.** The math concept separable at L18 (p≈0.02) is destroyed by 17 hops of linear composition: `cos(chain(math_18), math_35) = -0.053`, indistinguishable from random (-0.073). Chain-transported math decodes to function words ("the", "a", "in"). Aggregate statistics compose; individual semantics don't.

---

## What comes next

The chain composes at the aggregate level but loses concept directions in transit. The path forward:

1. **Wider basis** — Top-2048 captures common features but misses rare ones. An importance-weighted or concept-aware basis could retain semantic signal.
2. **Nonlinear transition operators** — Ridge regression captures change-of-basis but not the MLP computation. Transcoder-style sparse→nonlinear→sparse operators could capture the δ that linear maps scatter.
3. **Both** — Wider basis + nonlinear hops is the most likely path to concept survival through composition.

---

## Data

The atlas is available on HuggingFace: [`juiceb0xc0de/qwen3-8b-base-atlas`](https://huggingface.co/datasets/juiceb0xc0de/qwen3-8b-base-atlas). Point `ATLAS_DB` at the downloaded `atlas.sqlite`. The underlying SAEs are public ([`Qwen/SAE-Res-Qwen3-8B-Base-W64K-L0_50`](https://huggingface.co/Qwen/SAE-Res-Qwen3-8B-Base-W64K-L0_50)); `Qwen3-8B-Base` weights are on the Hugging Face Hub.

## Why public

This is an odd-shaped corner of interpretability and surrogate inference, and I'm exploring it in public. If it's interesting to you, you're welcome here. See [`CONTRIBUTING.md`](CONTRIBUTING.md) for ground rules. Negative results count.

## License

MIT. See [`LICENSE`](LICENSE).