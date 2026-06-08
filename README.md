# atlas-inference

**Can a static feature atlas of a frozen language model be driven as a runnable, legible surrogate of that model?**

We built a ~800 MB SQLite atlas of `Qwen3-8B-base` — every SAE feature across 36 layers (two sparsity variants, 65,536 features each), annotated with activation statistics, topic separation, compliance deltas, taxonomy class, per-head OV decompositions, DAS rotations, SNMF membership, and cross-layer coherence. It maps the model's internals at high resolution, and it's fully queryable.

This repo asks the next question: **is that map enough to compute?**

Not "build a better LLM" — the model already exists. The question is whether its forward computation is *recoverable in the SAE feature basis*: whether a sequence of sparse, named, inspectable feature states can reproduce a piece of what the dense network does. If it can, you get a white-box surrogate where every intermediate state is a concept you can read.

## Status

Early and exploratory. We test load-bearing premises one at a time, in public, and report the result either way. Negative results count.

## First experiment: the Decoder → Unembed Bridge

Before any dynamics, the simplest probe — can an atlas concept become vocabulary evidence?

```
concept centroid (SAE feature space)
  → SAE decoder            → residual-stream vector
  → final norm + unembed   → top-k tokens
```

Take a concept the atlas knows — we start with **mathematics** — form its centroid in feature space, decode it to a residual vector, and push that through the model's unembedding (logit-lens style). If the top-k tokens come out math-shaped, the atlas has a "mouth": its concepts can speak in vocabulary. If they don't, we debug basis / layer / scaling / centering / decoder availability — or learn that the atlas entries are metadata-only and the generative weights live elsewhere.

One concept, one decode, one read. A clean yes/no on whether the map can talk.

## Data

The atlas itself is **not** in this repo (too large, and it's a research artifact). Point `ATLAS_DB` at your own build. The underlying SAEs are public (released by the Qwen team); `Qwen3-8B-base` weights are on the Hugging Face Hub.

## Why public

Frontier labs aren't probing this corner — it's odd-shaped, solo-dev territory. If it's interesting to you, you're welcome here. See [`CONTRIBUTING.md`](CONTRIBUTING.md).

## License

MIT. See [`LICENSE`](LICENSE).
