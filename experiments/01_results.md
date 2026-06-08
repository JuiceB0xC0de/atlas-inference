# Experiment 01 — Decoder → Unembed Bridge (Gate 1)

**Date:** 2026-06-07 · **Status:** PASSED, with caveats. Gate 1 of 3 (bridge → concept → dynamics).
**Model:** `Qwen/Qwen3-8B-Base` · **SAE:** `Qwen/SAE-Res-Qwen3-8B-Base-W64K-L0_50` (resid_post, TopK k=50)
**Run:** `ATLAS_DB=/path/to/atlas.sqlite modal run experiments/modal_bridge_01a.py`

## Claim

SAE feature directions, decoded through the SAE decoder and projected through the model's
*real* unembedding (final RMSNorm → `lm_head`), produce coherent vocabulary. The atlas's
feature space has a "mouth." This does **not** show the atlas computes — only that its
feature directions can *speak*.

## Step 0 — reconstruction gate

`encode → TopK(50) → decode (x̂)`, measured against the real residual `x`:

| layer | cos (no-bdec → bdec) | top-12 overlap (bdec) | overlap (no-BOS) |
|------:|:--------------------:|:---------------------:|:----------------:|
| 3  | 0.81 → 0.97 | 0.595 | 0.595 |
| 18 | 0.72 → 0.90 | 0.452 | 0.488 |
| 35 | 0.58 → 0.97 | 0.690 | 0.679 |

- **`b_dec` is mandatory.** cos jumps 0.58 → 0.97 at L35. Decode convention: `x̂ = f @ W_dec.T + b_dec`.
- **Token overlap is the honest gate** (Codex). At the final layer `unembed(x)` is the model's
  *actual* next-token distribution, yet only **0.69** of its top-12 survive the SAE round-trip.
  cos 0.97 vs overlap 0.69 = the head is highly sensitive to small residual deviations.
- **The 0.69 is real, not a BOS artifact.** Excluding BOS changed it by <0.01 (and went *down*
  slightly at L35). The hypothesis that BOS/function positions dragged the aggregate down is
  **not supported** — loss is spread across ordinary prose tokens. One clean signal: **digits
  reconstruct perfectly** (`'9'`,`'6'`=1.0, `'2'`=0.92); prose tokens scatter 0.2–0.5 regardless
  of function/content. A k=50 round-trip is genuinely lossy at the next-token boundary.

## Step 01A — feature → tokens

Decode individual atlas-selected (top `topic_fstat`) features → top-k tokens, vs controls:

- **Selective features → crisp, monosemantic, bilingual concepts** — poetry, humor, email,
  agile/Scrum, transformer-internals, ML-fairness, simulation, startups, …
- **Gaussian (arbitrary residual directions) → junk.** The load-bearing control: the bridge is
  *not* "any vector → plausible tokens."
- **Norm-matched random real columns → mixed** (some coherent, many fragments).
- Honest contrast: **gaussian (junk) < random-real (mixed) < selective (clean)** — not
  "signal vs noise." Random columns aren't noise; they're all trained features.

## What this does NOT prove

- Concept *selection by the atlas* (gate 2 / 01B). These are individual features; identity
  emerged from the decode. The atlas stores selectivity, not concept identity.
- Any *dynamics* (gate 3). No layer-to-layer computation was tested.

Full token reads + per-position overlap: [`artifacts_01a.json`](artifacts_01a.json).
