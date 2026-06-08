#!/usr/bin/env python3
"""
Experiment 01 — Decoder → Unembed Bridge
========================================

HYPOTHESIS
    A concept living in SAE feature space can be turned into vocabulary evidence
    by skipping the transformer entirely:

        f (sparse, 65,536-dim)
          → W_dec            → residual-stream vector   [4096]
          → (final RMSNorm)  → normalized residual
          → lm_head.Tᵀ       → logits over vocab
          → top-k            → tokens

    If the top-k tokens are coherent (e.g. math-shaped for a math feature), the
    atlas/SAE has a "mouth": its concepts can speak in tokens without a forward pass.

SUCCESS / FAILURE CRITERION
    Step 1a (plumbing): a single SAE feature's decoder column should top-k into a
        coherent token cluster. If it's noise, the bug is orientation / scale /
        normalization — toggle --no-rmsnorm vs --rmsnorm and --scale to localize it.
    Step 1b (atlas-driven): pull the top-N SAE features by topic_fstat from the
        atlas and decode each. The atlas selects the *meaningful* features; this
        script reveals *what they say*. Concept identity (math/code/…) emerges from
        the decode rather than being assumed — the atlas has no topic labels.

WHY THE ATLAS DOESN'T DRIVE CONCEPT SELECTION DIRECTLY
    The atlas labels concepts on MODEL channels (q/k/v/gate/up/mlp), not on SAE
    features. The only SAE-space signal it stores is topic_fstat (how selective a
    feature is, not *for what*). So we rank by selectivity and read meaning here.

WEIGHTS
    SAE decoder : Qwen/SAE-Res-Qwen3-8B-Base-W64K-L0_50  (layer{N}.sae.pt, resid_post)
    Unembed     : Qwen/Qwen3-8B-Base  (lm_head.weight + model.norm.weight only —
                  pulled lazily from safetensors, the 8B model is never fully loaded)

This is research scaffolding. Run it, read the tokens, report what you saw.
"""

from __future__ import annotations

import argparse
import json
import sqlite3

import torch
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from transformers import AutoTokenizer

SAE_REPO_DEFAULT = "Qwen/SAE-Res-Qwen3-8B-Base-W64K-L0_50"
BASE_MODEL_DEFAULT = "Qwen/Qwen3-8B-Base"


# ── SAE decoder ───────────────────────────────────────────────────────────────
def load_sae_decoder(repo: str, layer: int, device: str = "cpu"):
    """Return (W_dec [d_model, d_sae], b_dec [d_model]) for one layer."""
    path = hf_hub_download(repo, f"layer{layer}.sae.pt")
    sae = torch.load(path, map_location=device, weights_only=True)
    W_dec = sae["W_dec"].float()            # [4096, 65536] — columns are residual directions
    device = W_dec.device
    b_dec = sae.get("b_dec")
    b_dec = b_dec.float().to(device=device) if b_dec is not None else torch.zeros(
        W_dec.shape[0], device=device, dtype=W_dec.dtype)
    return W_dec, b_dec


# ── Unembed (lazy: two tensors, never the whole 8B) ───────────────────────────
def load_unembed(base_model: str, device: str = "cpu"):
    """Return (lm_head [vocab, d_model], norm_weight [d_model], rms_eps)."""
    cfg = json.load(open(hf_hub_download(base_model, "config.json")))
    eps = float(cfg.get("rms_norm_eps", 1e-6))

    index = json.load(open(hf_hub_download(base_model, "model.safetensors.index.json")))
    wmap = index["weight_map"]
    lm_key = "lm_head.weight" if "lm_head.weight" in wmap else "model.embed_tokens.weight"
    norm_key = "model.norm.weight"

    tensors = {}
    for key in (lm_key, norm_key):
        shard = hf_hub_download(base_model, wmap[key])
        with safe_open(shard, framework="pt", device=device) as f:
            tensors[key] = f.get_tensor(key).float()
    return tensors[lm_key], tensors[norm_key], eps


# ── Qwen3 final norm ──────────────────────────────────────────────────────────
def rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    var = x.pow(2).mean(-1, keepdim=True)
    return x * torch.rsqrt(var + eps) * weight


# ── The bridge ────────────────────────────────────────────────────────────────
def bridge(idxs, mags, W_dec, b_dec, lm_head, norm_w, eps,
           use_rmsnorm=True, add_bdec=False, scale=1.0):
    """Sparse feature activation → residual → (norm) → logits over vocab."""
    mags = torch.as_tensor(mags, dtype=torch.float32, device=W_dec.device)
    recon = W_dec[:, idxs] @ mags                 # [d_model]
    # scale BEFORE the bias: a positive scale on the formed vector is just softmax
    # temperature and never moves top-k ranking (Codex). It only changes ranking
    # relative to b_dec, so apply it here where it can actually matter.
    recon = recon * scale
    if add_bdec:
        recon = recon + b_dec
    h = rmsnorm(recon, norm_w, eps) if use_rmsnorm else recon
    return h @ lm_head.T                          # [vocab]


def show(tag, logits, tok, k):
    probs = torch.softmax(logits, dim=-1)
    vals, ids = probs.topk(k)
    toks = [repr(tok.decode([i])) for i in ids.tolist()]
    print(f"  {tag}: " + "  ".join(f"{t}({v:.3f})" for t, v in zip(toks, vals.tolist())))


# ── Concept selection from the atlas ─────────────────────────────────────────
def atlas_top_features(atlas_path, layer, variant, n):
    """Top-N SAE features by topic_fstat at this layer (the atlas's contribution)."""
    con = sqlite3.connect(f"file:{atlas_path}?mode=ro", uri=True)
    rows = con.execute(
        "SELECT feature_idx, topic_fstat FROM sae_features "
        "WHERE layer=? AND variant=? AND topic_fstat IS NOT NULL "
        "ORDER BY topic_fstat DESC LIMIT ?",
        (layer, variant, n),
    ).fetchall()
    con.close()
    return rows  # [(feature_idx, topic_fstat), ...]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["feature", "centroid", "atlas-scan"], default="feature")
    ap.add_argument("--sae-repo", default=SAE_REPO_DEFAULT)
    ap.add_argument("--base-model", default=BASE_MODEL_DEFAULT)
    ap.add_argument("--layer", type=int, default=35, help="late layer = cleanest logit-lens")
    ap.add_argument("--feature-idx", type=int, default=0, help="mode=feature: which feature")
    ap.add_argument("--atlas", default=None, help="path to atlas.sqlite (atlas-scan/centroid)")
    ap.add_argument("--atlas-variant", default="l0_50")
    ap.add_argument("--top-n-features", type=int, default=20, help="atlas-scan/centroid: how many features")
    ap.add_argument("--top-k-tokens", type=int, default=15)
    # GPT's diagnostic toggles
    ap.add_argument("--rmsnorm", dest="rmsnorm", action="store_true", default=True)
    ap.add_argument("--no-rmsnorm", dest="rmsnorm", action="store_false")
    ap.add_argument("--add-bdec", action="store_true", help="add decoder bias before norm")
    ap.add_argument("--scale", type=float, default=1.0,
                    help="recon magnitude; only affects ranking relative to --add-bdec "
                         "(no-op under rmsnorm without bias). Judge results on rank, not probs.")
    args = ap.parse_args()

    print(f"[load] SAE decoder  {args.sae_repo}  layer {args.layer}")
    W_dec, b_dec = load_sae_decoder(args.sae_repo, args.layer)
    print(f"[load] unembed      {args.base_model}  (lm_head + norm only)")
    lm_head, norm_w, eps = load_unembed(args.base_model)
    tok = AutoTokenizer.from_pretrained(args.base_model)
    print(f"[ok]  W_dec {tuple(W_dec.shape)}  lm_head {tuple(lm_head.shape)}  eps {eps}")
    print(f"[cfg] rmsnorm={args.rmsnorm}  add_bdec={args.add_bdec}  scale={args.scale}\n")

    def run(idxs, mags, tag):
        logits = bridge(idxs, mags, W_dec, b_dec, lm_head, norm_w, eps,
                        use_rmsnorm=args.rmsnorm, add_bdec=args.add_bdec, scale=args.scale)
        show(tag, logits, tok, args.top_k_tokens)

    if args.mode == "feature":
        run([args.feature_idx], [1.0], f"feature {args.feature_idx}")

    elif args.mode in ("atlas-scan", "centroid"):
        assert args.atlas, "--atlas path required for atlas-scan/centroid"
        feats = atlas_top_features(args.atlas, args.layer, args.atlas_variant, args.top_n_features)
        print(f"[atlas] top {len(feats)} features by topic_fstat at layer {args.layer} ({args.atlas_variant})")
        if args.mode == "atlas-scan":
            # decode each individually — read what each selective feature says
            for fid, fstat in feats:
                run([fid], [1.0], f"feat {fid:>6} (F={fstat:.1f})")
        else:
            # GPT's summed centroid — one concept vector from many features
            idxs = [f for f, _ in feats]
            mags = [s for _, s in feats]  # weight by selectivity
            run(idxs, mags, f"centroid of {len(idxs)} features")


if __name__ == "__main__":
    main()
