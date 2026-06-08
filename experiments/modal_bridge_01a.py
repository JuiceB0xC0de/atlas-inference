"""
Modal app — Experiment 01, gated: Decoder → Unembed bridge.

Run:  modal run experiments/modal_bridge_01a.py

STEP 0 — reconstruction gate  (layers 3, 18, 35)
    For a real residual x captured at each layer: encode → TopK → decode to x̂,
    then measure BOTH
        cos(x, x̂)                              residual fidelity
        top_k_overlap(unembed(x), unembed(x̂))  head-relevant fidelity
    tested with and without b_dec. At layer 35, unembed(x) IS the model's real
    next-token distribution (resid_post[35] → final RMSNorm → lm_head), so overlap
    there = how faithfully the SAE reproduces the model's actual output.
    High cos + low overlap = the SAE drops exactly what the head reads (Codex's
    distinct failure mode).

STEP 01A — feature → tokens  (layers 35, 18)
    Decode individual atlas-selected features (top topic_fstat) to top-k tokens,
    against NORM-MATCHED random columns + gaussian directions. The atlas selects
    the meaningful features; the bridge reveals what they say. Concept identity
    emerges from the decode — the atlas has no topic labels (that's gate 2 / 01B).

Judged on token coherence / rank, not softmax probabilities (positive scaling of
the pre-logit vector is just softmax temperature and never moves the ranking).
"""

import os
import sqlite3
from pathlib import Path

import modal

app = modal.App("atlas-inference-bridge-01")
image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "torch", "transformers", "accelerate", "huggingface_hub", "safetensors"
)

SAE_REPO = "Qwen/SAE-Res-Qwen3-8B-Base-W64K-L0_50"
BASE = "Qwen/Qwen3-8B-Base"
VARIANT = "l0_50"
K_SPARSE = 50  # matches the L0_50 SAE


@app.function(gpu="L40S", image=image, timeout=2400)
def run_bridge(atlas_idx: dict, layers_recon, layers_read, probe_texts,
               k_tokens=12, n_sel=30, n_gauss=8):
    import torch
    import torch.nn.functional as F
    from huggingface_hub import hf_hub_download
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dev = "cuda"
    tok = AutoTokenizer.from_pretrained(BASE)
    model = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.bfloat16).to(dev).eval()
    lm_head = model.lm_head.weight.float()        # [vocab, d_model]
    norm_w = model.model.norm.weight.float()      # [d_model]
    eps = model.config.rms_norm_eps

    def rmsnorm(x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * norm_w

    def to_logits(vec, use_rmsnorm=True):
        h = rmsnorm(vec) if use_rmsnorm else vec
        return h @ lm_head.T

    def topk_relu(x, k):
        relu_x = torch.relu(x)
        vals, idx = torch.topk(relu_x, k, dim=-1)
        out = torch.zeros_like(relu_x)
        out.scatter_(-1, idx, vals)
        return out

    def load_sae(layer):
        sae = torch.load(hf_hub_download(SAE_REPO, f"layer{layer}.sae.pt"),
                         map_location=dev, weights_only=True)
        W_enc = sae["W_enc"].T.float().to(dev)    # [d_model, width]
        b_enc = sae["b_enc"].float().to(dev)      # [width]
        W_dec = sae["W_dec"].float().to(dev)      # [d_model, width]
        b_dec = sae.get("b_dec")
        b_dec = b_dec.float().to(dev) if b_dec is not None else torch.zeros(W_dec.shape[0], device=dev)
        return W_enc, b_enc, W_dec, b_dec

    # ── capture resid_post (decoder-layer output) at the requested layers ──
    def capture(layers):
        buf, handles = {}, []

        def mk(L):
            def hook(m, i, o):
                buf[L] = (o[0] if isinstance(o, tuple) else o).detach()
            return hook

        for L in layers:
            handles.append(model.model.layers[L].register_forward_hook(mk(L)))
        out = {L: [] for L in layers}
        with torch.no_grad():
            for txt in probe_texts:
                ids = tok(txt, return_tensors="pt").input_ids.to(dev)
                model(ids)
                for L in layers:
                    out[L].append(buf[L][0].float())   # [seq, d_model]
        for h in handles:
            h.remove()
        return out

    all_layers = sorted(set(list(layers_recon) + list(layers_read)))
    acts = capture(all_layers)
    report = {"recon": {}, "read": {}}

    # ── STEP 0: reconstruction gate ──
    def topk_set(logits, k):
        return set(torch.topk(logits, k, dim=-1).indices.tolist())

    for L in layers_recon:
        W_enc, b_enc, W_dec, b_dec = load_sae(L)
        agg = {k: [] for k in
               ("cos_nobias", "cos_bias", "overlap_nobias", "overlap_bias", "norm_x", "norm_xhat")}
        for x in acts[L]:                          # [seq, d_model]
            f = topk_relu(x @ W_enc + b_enc, K_SPARSE)
            base = f @ W_dec.T                     # [seq, d_model]
            for tag, xhat in (("nobias", base), ("bias", base + b_dec)):
                agg[f"cos_{tag}"].append(F.cosine_similarity(x, xhat, dim=-1).mean().item())
                lx, lxh = to_logits(x), to_logits(xhat)   # [seq, vocab]
                ov = [len(topk_set(lx[p], k_tokens) & topk_set(lxh[p], k_tokens)) / k_tokens
                      for p in range(x.shape[0])]
                agg[f"overlap_{tag}"].append(sum(ov) / len(ov))
            agg["norm_x"].append(x.norm(dim=-1).mean().item())
            agg["norm_xhat"].append(base.norm(dim=-1).mean().item())
        report["recon"][L] = {k: round(sum(v) / len(v), 4) for k, v in agg.items()}

    # ── STEP 01A: decode features → tokens, with norm-matched controls ──
    def decode(W_dec, idx):
        ids = torch.topk(to_logits(W_dec[:, idx]), k_tokens).indices.tolist()
        return [tok.decode([i]) for i in ids]

    for L in layers_read:
        W_enc, b_enc, W_dec, b_dec = load_sae(L)
        sel = atlas_idx[str(L)][:n_sel]
        col_norms = W_dec.norm(dim=0)              # [width]
        order = torch.argsort(col_norms)
        rank = torch.empty_like(order)
        rank[order] = torch.arange(order.numel(), device=order.device)
        width = col_norms.numel()

        # norm-matched random control: for each selected feature, draw a random
        # feature from a nearby norm-rank window (kills the high-norm false positive)
        selset, ctrl = set(sel), []
        for fid in sel:
            r = int(rank[fid].item())
            lo, hi = max(0, r - 200), min(width, r + 200)
            for _ in range(25):
                cand = int(order[torch.randint(lo, hi, (1,)).item()].item())
                if cand not in selset and cand not in ctrl:
                    ctrl.append(cand)
                    break

        gauss = torch.randn(n_gauss, W_dec.shape[0], device=dev)
        report["read"][L] = {
            "selective": {int(f): decode(W_dec, f) for f in sel},
            "norm_matched_random": {int(f): decode(W_dec, f) for f in ctrl},
            "gaussian": [[tok.decode([i]) for i in
                          torch.topk(to_logits(gauss[j]), k_tokens).indices.tolist()]
                         for j in range(n_gauss)],
        }
    return report


@app.local_entrypoint()
def main():
    default_db = Path(__file__).resolve().parent / "data" / "atlas.sqlite"
    db = os.environ.get("ATLAS_DB", str(default_db))
    # Set ATLAS_DB if you need a custom atlas location outside the repo.
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    atlas_idx = {}
    for L in (3, 18, 35):
        rows = con.execute(
            "SELECT feature_idx FROM sae_features WHERE layer=? AND variant=? "
            "AND topic_fstat IS NOT NULL ORDER BY topic_fstat DESC LIMIT 30",
            (L, VARIANT)).fetchall()
        atlas_idx[str(L)] = [r[0] for r in rows]
    con.close()

    probe_texts = [
        "The integral of x squared from zero to one equals one third.",
        "She walked quietly through the old library at dusk.",
        "def fibonacci(n): return n if n < 2 else fibonacci(n-1) + fibonacci(n-2)",
        "The mitochondria is the powerhouse of the cell.",
        "In 1969, humans first set foot on the surface of the moon.",
        "Prices rose three percent while wages stayed flat this quarter.",
    ]

    rep = run_bridge.remote(atlas_idx, layers_recon=[3, 18, 35], layers_read=[35, 18],
                            probe_texts=probe_texts)

    print("\n================  STEP 0: RECON GATE  ================")
    print(f"{'layer':>5} | {'cos_nb':>7} {'cos_b':>7} | {'ovlp_nb':>7} {'ovlp_b':>7} | {'|x|':>7} {'|xhat|':>7}")
    for L, r in rep["recon"].items():
        print(f"{L:>5} | {r['cos_nobias']:>7} {r['cos_bias']:>7} | "
              f"{r['overlap_nobias']:>7} {r['overlap_bias']:>7} | {r['norm_x']:>7} {r['norm_xhat']:>7}")

    for L, blk in rep["read"].items():
        print(f"\n================  STEP 01A: layer {L}  ================")
        print("--- SELECTIVE (atlas top topic_fstat) ---")
        for fid, toks in blk["selective"].items():
            print(f"  {fid:>6}: " + " ".join(repr(t) for t in toks))
        print("--- NORM-MATCHED RANDOM ---")
        for fid, toks in list(blk["norm_matched_random"].items())[:12]:
            print(f"  {fid:>6}: " + " ".join(repr(t) for t in toks))
        print("--- GAUSSIAN ---")
        for toks in blk["gaussian"][:5]:
            print("    rand: " + " ".join(repr(t) for t in toks))

    # small text artifact (orjson per Rick's standing preference; safe fallback)
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts_01a.json")
    try:
        import orjson
        with open(out, "wb") as fh:
            fh.write(orjson.dumps(rep, option=orjson.OPT_INDENT_2))
    except ImportError:
        import json
        with open(out, "w") as fh:
            json.dump(rep, fh, indent=2)
    print(f"\n[saved] {out}")
