"""
Modal app — Experiment 03A (Gate 3): one-hop transition fidelity z_L → z_{L+1}.

Run:  modal run experiments/modal_transition_03a.py

QUESTION
    Can the atlas state MOVE? Fit a linear operator T on per-token SAE feature states
    z_18 → z_19 and test whether it predicts the next layer's state better than the
    marginal (mean) and a shuffled-pairing null.

DESIGN (GPT-ratified, 2026-06-08)
    z = per-token TopK(50) SAE features (BOS excluded). z_18 and z_19 live in DIFFERENT
        SAE bases, so there is NO trivial identity baseline — T maps between two learned
        dictionaries (non-trivial by construction).
    basis  : top-2048 features by activation frequency, computed SEPARATELY per layer.
             ~4M params on ~30k token pairs ≈ 7:1, tractable with L2 ridge.
    T      : linear ridge, λ chosen on a val split. (Linear first; if it can't beat the
             mean-predictor, the chain is dead until the operator improves.)
    baselines (load-bearing):
        mean-predictor : always predict mean(z_19_train) — does z_18 carry per-token info?
        shuffle-null   : T fit on shuffled (z_18, z_19) pairs — does the pairing matter?
    metrics (held-out tokens): cosine(pred, actual), top-k active-feature overlap, R².

    01B → 03A thread: report whether the math-specific features land inside the top-2048
    bases, and whether T preserves the math direction cos(T(math_z18), math_z19). If T
    scatters the math direction, 03B is dead regardless of aggregate cosine.

    ONE HOP ONLY (18→19). If it works, then 18→19→20.
"""

import os
import modal

app = modal.App("atlas-inference-transition-03a")
image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "torch", "transformers", "accelerate", "huggingface_hub", "safetensors", "datasets"
)

SAE_REPO = "Qwen/SAE-Res-Qwen3-8B-Base-W64K-L0_50"
BASE = "Qwen/Qwen3-8B-Base"
K_SPARSE = 50
TOPN = 2048

# compact math/control sets for the math-direction tracking (01B → 03A thread)
MATH = [
    "What is 7 times 8?", "Solve for x: 2x + 3 = 11.", "Compute the integral of x squared dx.",
    "Is 17 a prime number?", "What is the derivative of sin(x)?", "Factor x^2 - 5x + 6.",
    "What is the square root of 144?", "Evaluate the limit of 1/x as x to infinity.",
    "Solve the quadratic x^2 - 4 = 0.", "What is the sum of the first 10 integers?",
    "Differentiate f(x) = 3x^3.", "Greatest common divisor of 24 and 36?",
    "Find the slope of y = 2x + 1.", "Compute the factorial of 5.",
    "What is the cosine of 60 degrees?", "Logarithm base 10 of 1000?",
    "Perimeter of a square with side 7?", "What is the next prime after 13?",
    "Simplify the fraction 18/24.", "Hypotenuse of a 3-4-5 triangle?",
]
CONTROL = [
    "What is your favorite season of the year?", "Describe the smell of fresh bread.",
    "Tell me about the history of the Roman Empire.", "How do you make a good cup of coffee?",
    "What makes a friendship last?", "Describe a walk through a quiet forest.",
    "Who was the first president of the United States?", "What do cats like to do all day?",
    "Explain how to plant a vegetable garden.", "What emotions does rain bring up for you?",
    "Tell me a story about a brave knight.", "What is the capital of France?",
    "How do birds know where to migrate?", "Describe your ideal vacation destination.",
    "Talk about the taste of a ripe peach.", "Why do leaves change color in autumn?",
    "Describe the sound of ocean waves.", "How do you comfort a sad friend?",
    "Tell me about the life of a honeybee.", "What makes a sunset beautiful?",
]


@app.function(gpu="L40S", image=image, timeout=3600)
def run_03a(layers=(18, 19), n_passages=600, k_overlap=10, seed=0):
    import random
    import torch
    import torch.nn.functional as F
    from huggingface_hub import hf_hub_download
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(seed)
    random.seed(seed)
    dev = "cuda"
    Lin, Lout = layers

    tok = AutoTokenizer.from_pretrained(BASE)
    model = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.bfloat16).to(dev).eval()

    def load_sae(L):
        sae = torch.load(hf_hub_download(SAE_REPO, f"layer{L}.sae.pt"),
                         map_location=dev, weights_only=True)
        return sae["W_enc"].T.float().to(dev), sae["b_enc"].float().to(dev)

    Wenc = {L: load_sae(L) for L in layers}
    width = Wenc[Lin][0].shape[1]

    def topk_idx_val(x, W_enc, b_enc, k):       # x:[seq,d] -> (idx[seq,k], val[seq,k])
        pre = torch.relu(x @ W_enc + b_enc)
        val, idx = pre.topk(k, dim=-1)
        return idx, val

    # ── corpus ──
    from datasets import load_dataset
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    passages = [t.strip() for t in ds["text"] if len(t.strip()) > 120][:n_passages]

    buf = {}
    def mk(L):
        def hook(m, i, o):
            buf[L] = (o[0] if isinstance(o, tuple) else o).detach()
        return hook
    handles = [model.model.layers[L].register_forward_hook(mk(L)) for L in layers]

    # collect per-token sparse features + activation frequency, single pass
    sp = {L: {"idx": [], "val": []} for L in layers}     # per token: [k] idx, [k] val
    freq = {L: torch.zeros(width, device=dev) for L in layers}
    with torch.no_grad():
        for txt in passages:
            ids = tok(txt, return_tensors="pt", truncation=True, max_length=128).input_ids.to(dev)
            if ids.shape[1] < 3:
                continue
            model(ids)
            for L in layers:
                x = buf[L][0].float()[1:]                # drop BOS -> [seq-1, d]
                W_enc, b_enc = Wenc[L]
                idx, val = topk_idx_val(x, W_enc, b_enc, K_SPARSE)
                sp[L]["idx"].append(idx); sp[L]["val"].append(val)
                freq[L].scatter_add_(0, idx.reshape(-1),
                                     (val.reshape(-1) > 0).float())
    for h in handles:
        h.remove()

    # ── top-2048 basis per layer (separate sets) + feature->position maps ──
    basis = {L: torch.topk(freq[L], TOPN).indices for L in layers}
    pos = {L: torch.full((width,), -1, dtype=torch.long, device=dev) for L in layers}
    for L in layers:
        pos[L][basis[L]] = torch.arange(TOPN, device=dev)
    basis_overlap = len(set(basis[Lin].tolist()) & set(basis[Lout].tolist()))

    def densify(L):                              # stacked sparse tokens -> dense [N, TOPN]
        idx = torch.cat(sp[L]["idx"], 0)         # [N, k]
        val = torch.cat(sp[L]["val"], 0)
        bp = pos[L][idx]                         # [N, k] basis position or -1
        N = idx.shape[0]
        Z = torch.zeros(N, TOPN, device=dev)
        keep = bp >= 0
        rows = torch.arange(N, device=dev).unsqueeze(1).expand_as(bp)[keep]
        Z[rows, bp[keep]] = val[keep]
        return Z

    Z_in, Z_out = densify(Lin), densify(Lout)    # [N, TOPN] aligned per token
    N = Z_in.shape[0]
    g = torch.Generator(device=dev).manual_seed(seed)
    perm = torch.randperm(N, generator=g, device=dev)
    n_tr, n_va = int(0.7 * N), int(0.15 * N)
    tr, va, te = perm[:n_tr], perm[n_tr:n_tr + n_va], perm[n_tr + n_va:]

    def ridge(X, Y, lam):
        Xc = X - X.mean(0, keepdim=True)
        Yc = Y - Y.mean(0, keepdim=True)
        A = Xc.T @ Xc + lam * torch.eye(X.shape[1], device=dev)
        W = torch.linalg.solve(A, Xc.T @ Yc)
        b = Y.mean(0) - X.mean(0) @ W
        return W, b

    def metrics(pred, Y):
        cos = F.cosine_similarity(pred, Y, dim=1).mean().item()
        ss_res = ((Y - pred) ** 2).sum().item()
        ss_tot = ((Y - Y.mean(0, keepdim=True)) ** 2).sum().item()
        r2 = 1 - ss_res / ss_tot
        pk = pred.topk(k_overlap, dim=1).indices
        yk = Y.topk(k_overlap, dim=1).indices
        ov = [len(set(pk[i].tolist()) & set(yk[i].tolist())) / k_overlap for i in range(pred.shape[0])]
        return {"cosine": round(cos, 4), "r2": round(r2, 4),
                "topk_overlap": round(sum(ov) / len(ov), 4)}

    # λ sweep on val
    best = None
    for lam in [1.0, 10.0, 100.0, 1000.0]:
        W, b = ridge(Z_in[tr], Z_out[tr], lam)
        m = metrics(Z_in[va] @ W + b, Z_out[va])
        if best is None or m["cosine"] > best[1]["cosine"]:
            best = (lam, m, W, b)
    lam, _, W, b = best

    report = {"N_tokens": N, "lambda": lam, "basis_overlap_in_out": basis_overlap,
              "layers": list(layers)}
    report["fitted_T"] = metrics(Z_in[te] @ W + b, Z_out[te])
    # mean-predictor baseline
    mean_pred = Z_out[tr].mean(0, keepdim=True).expand(len(te), -1)
    report["mean_predictor"] = metrics(mean_pred, Z_out[te])
    # shuffle-null: fit on mismatched pairs
    Ws, bs = ridge(Z_in[tr], Z_out[tr][torch.randperm(len(tr), generator=g, device=dev)], lam)
    report["shuffle_null"] = metrics(Z_in[te] @ Ws + bs, Z_out[te])

    # ── 01B → 03A thread: math direction transport ──
    def encode_full(L, txt):                     # mean over tokens (BOS excl) in full width
        ids = tok(txt, return_tensors="pt").input_ids.to(dev)
        model(ids)
        x = buf[L][0].float()[1:]
        W_enc, b_enc = Wenc[L]
        f = torch.zeros(x.shape[0], width, device=dev)
        idx, val = topk_idx_val(x, W_enc, b_enc, K_SPARSE)
        f.scatter_(1, idx, torch.relu(val))
        return f.mean(0)
    handles = [model.model.layers[L].register_forward_hook(mk(L)) for L in layers]
    with torch.no_grad():
        def dir_at(L):
            m = torch.stack([encode_full(L, t) for t in MATH]).mean(0)
            c = torch.stack([encode_full(L, t) for t in CONTROL]).mean(0)
            return m - c
        math_in_full, math_out_full = dir_at(Lin), dir_at(Lout)
    for h in handles:
        h.remove()

    topm = math_in_full.abs().topk(20).indices
    report["math_feats_top20_in"] = topm.tolist()
    report["math_feats_in_basis_Lin"] = int((pos[Lin][topm] >= 0).sum())
    report["math_feats_in_basis_Lout"] = int((pos[Lout][math_out_full.abs().topk(20).indices] >= 0).sum())

    math_in = torch.zeros(TOPN, device=dev); math_out = torch.zeros(TOPN, device=dev)
    mi = pos[Lin][math_in_full.nonzero().squeeze(-1)]
    math_in[mi[mi >= 0]] = math_in_full[math_in_full.nonzero().squeeze(-1)][mi >= 0]
    mo = pos[Lout][math_out_full.nonzero().squeeze(-1)]
    math_out[mo[mo >= 0]] = math_out_full[math_out_full.nonzero().squeeze(-1)][mo >= 0]
    pred_math = math_in @ W + b
    report["math_transport_cos_T"] = round(F.cosine_similarity(pred_math, math_out, dim=0).item(), 4)
    rand_dir = torch.randn(TOPN, device=dev)
    report["math_transport_cos_random_ctrl"] = round(
        F.cosine_similarity(rand_dir @ W + b, math_out, dim=0).item(), 4)
    report["math_dir_cos_in_vs_out_raw"] = round(F.cosine_similarity(math_in, math_out, dim=0).item(), 4)
    return report


@app.local_entrypoint()
def main():
    import json
    rep = run_03a.remote()
    print(f"\n=== 03A  L{rep['layers'][0]}→L{rep['layers'][1]}  (N={rep['N_tokens']} tokens, "
          f"λ={rep['lambda']}, basis overlap={rep['basis_overlap_in_out']}/{2048}) ===")
    for cond in ["fitted_T", "mean_predictor", "shuffle_null"]:
        m = rep[cond]
        print(f"  {cond:16s} cosine={m['cosine']:.4f}  topk_overlap={m['topk_overlap']:.4f}  r2={m['r2']:.4f}")
    print(f"\n  math feats in basis: L{rep['layers'][0]}={rep['math_feats_in_basis_Lin']}/20  "
          f"L{rep['layers'][1]}={rep['math_feats_in_basis_Lout']}/20")
    print(f"  math transport cos(T(math_in), math_out) = {rep['math_transport_cos_T']:.4f}  "
          f"(random ctrl {rep['math_transport_cos_random_ctrl']:.4f}, raw in-vs-out {rep['math_dir_cos_in_vs_out_raw']:.4f})")

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts_03a.json")
    try:
        import orjson
        with open(out, "wb") as fh:
            fh.write(orjson.dumps(rep, option=orjson.OPT_INDENT_2))
    except ImportError:
        with open(out, "w") as fh:
            json.dump(rep, fh, indent=2)
    print(f"\n[saved] {out}")
