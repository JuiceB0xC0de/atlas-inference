# atlas-inference

**Can a static feature atlas of a frozen language model be driven as a runnable, legible surrogate of that model?**

We built a ~570 MB SQLite atlas of `Qwen3-8B-Base` — 65,536 SAE features × 36 layers × 2 sparsity variants (L0_50, L0_100). It combines SAE feature statistics with channel-level activation, compliance, taxonomy, OV, DAS, SNMF, and cross-layer geometry tables. It maps the model's internals at high resolution, and it's fully queryable.

This repo asks the next question: **is that map enough to compute?**

Not "build a better LLM" — the model already exists. The question is whether its forward computation is *recoverable in the SAE feature basis*: whether a sequence of sparse, named, inspectable feature states can reproduce a piece of what the dense network does. If it can, you get a white-box surrogate where every intermediate state is a concept you can read.

## Status

Early and exploratory. We test load-bearing premises one at a time, in public, and report the result either way. Negative results count.

## Gates

We're testing three premises in order. Each one must hold for the next to matter:

1. **Bridge** — Can atlas feature directions produce coherent vocabulary through the model's real unembed? (Does the map have a mouth?)
2. **Concept** — Can a contrastive, prompt-derived direction in feature space produce concept-matched vocabulary? (Can the map speak *intentionally* — math direction → math tokens, not just any tokens?)
3. **Dynamics** — Can feature states at one layer predict feature states at the next? (Can the map *compute*, layer to layer?)

## Gate 1: Decoder → Unembed Bridge — PASSED

The bridge works, with caveats. SAE feature directions, decoded through the real SAE decoder and projected through the model's final RMSNorm + unembed, produce semantically coherent vocabulary. Individual atlas-selected features decode into tight bilingual clusters — poetry, humor, ML-infrastructure, UX/design, math — not noise.

### Step 0 — Reconstruction gate

`encode → TopK(50) → decode (x̂)`, measured against the real residual `x`:

| layer | cos (no b_dec → b_dec) | top-12 overlap (b_dec) | overlap (no BOS) |
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

Can a prompt-derived contrastive direction (`mean_z(math prompts) − mean_z(control prompts)`) in SAE feature space produce math-specific vocabulary beyond what shuffled-label nulls produce?

```
math_delta = mean_z(math prompts) − mean_z(control prompts)
residual   = W_dec @ math_delta                    # no b_dec — contrast subtracts common mode
logits     = final_RMSNorm(residual) @ lm_head.T
```

Scored with a pre-registered 4-bucket classifier (math_specific / numeric / code_syntax / other). Digits are NOT math-specific. Bare `_`, `^`, `$` are NOT math.

**L18 — PASSES (p≈0.02).** math_specific = 0.20, all from math words (circumference, consecutive, 平方, nonzero), zero digits. 3/200 shuffles ≥ observed (null mean 0.008, null max 0.25). Significant and real, but not airtight — ~1.5% of random 40/40 splits reach 0.20 by combinatorics. The concept is cleanly, specifically separable at mid-stack.

**L35 — FAILS (p≈0.09).** math_specific = 0.15, but 17/200 shuffles ≥ observed (null mean 0.032, null max 0.60). At the output layer, the math direction is 50% digits — the shared numeric axis that math and code occupy together. Random label splits also surface this axis. Math-specific notation (½, ∜, π) sits inside the null's tail.

The finding: concepts live semantically at mid-stack (L18), where they're cleanly separable. At late-stack (L35), concepts dissolve into generic output vocabulary. The shuffled-label null earned its keep — without it, L35 would have been called a pass.

## Gate 2: Concept — LOCKED. PASSED at L18 (p≈0.02), FAILED at L35 (p≈0.09)

n=200 shuffles + 5 gaussians. All non-shuffle controls ≈ 0.00 math-specific. Full results: [`experiments/01b_results.md`](experiments/01b_results.md).

## Gate 3: Dynamics — NOT YET

Can feature states at one layer predict feature states at the next? This is the big one. If the atlas can compute layer-to-layer transitions, you get a white-box surrogate where every intermediate state is a concept you can read.

Probe target: L18, where concepts are cleanly separable — not L35, where they dissolve into output vocabulary.

## Data

The atlas itself is **not** in this repo (too large, and it's a research artifact). Point `ATLAS_DB` at your own build. The underlying SAEs are public (released by the Qwen team); `Qwen3-8B-Base` weights are on the Hugging Face Hub.

## Why public

This is an odd-shaped corner of interpretability and surrogate inference, and we're exploring it in public. If it's interesting to you, you're welcome here. See [`CONTRIBUTING.md`](CONTRIBUTING.md).

## License

MIT. See [`LICENSE`](LICENSE).