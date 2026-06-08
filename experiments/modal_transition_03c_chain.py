"""
Modal app — Experiment 03C (Gate 3, acid test): composition of feature-space transitions.

Run:  modal run experiments/modal_transition_03c_chain.py

QUESTION (the one that can kill the project)
    03A showed one hop z_18→z_19 is linearly recoverable (R²=0.92). But one hop is a
    small residual change, so high R² is partly change-of-basis, not computation. The
    load-bearing test is COMPOSITION: fit T_L for L=18..34, compose them FREE-RUNNING
    (feed each predicted state into the next operator), and ask whether the chain still
    predicts a distant layer. If composed R² holds up, the operators captured real
    computation. If it crashes toward the mean-predictor floor, each hop was just local
    basis-translation and the map cannot run a forward pass.

DESIGN (GPT-ratified)
    z_L = per-token TopK(50) features (BOS excluded); separate top-2048 basis per layer.
    T_L : linear ridge z_L → z_{L+1} (λ=1000, 03A's pick), 17 operators (L=18..34).
    - teacher-forced per hop: T_L(actual z_L) vs actual z_{L+1}      → operator quality
    - free-running compose:   z_{L+1} = T_L(predicted z_L)           → error accumulation
    - decay curve: cos/R² at hop lengths 1,2,…,17 (layers 19…35)
    - direct T_18→35 (one ridge) for comparison: composed vs direct
    - baseline: mean-predictor at the target layer (R²≈0 floor)
"""

import os
import modal

app = modal.App("atlas-inference-transition-03c")
image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "torch", "transformers", "accelerate", "huggingface_hub", "safetensors", "datasets"
)

SAE_REPO = "Qwen/SAE-Res-Qwen3-8B-Base-W64K-L0_50"
BASE = "Qwen/Qwen3-8B-Base"
K_SPARSE = 50
TOPN = 2048
LAM = 1000.0


@app.function(gpu="L40S", image=image, timeout=5400)
def run_chain(n_passages=500, k_overlap=10, seed=0):
    import torch
    import torch.nn.functional as F
    from huggingface_hub import hf_hub_download
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(seed)
    dev = "cuda"
    layers = list(range(18, 36))             # 18..35 inclusive (18 layers)
    fit_L = layers[:-1]                       # 18..34 (17 operators)

    tok = AutoTokenizer.from_pretrained(BASE)
    model = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.bfloat16).to(dev).eval()

    def load_wenc(L):
        sae = torch.load(hf_hub_download(SAE_REPO, f"layer{L}.sae.pt"),
                         map_location=dev, weights_only=True)
        return sae["W_enc"].T.float().to(dev), sae["b_enc"].float().to(dev)
    Wenc = {L: load_wenc(L) for L in layers}
    width = Wenc[layers[0]][0].shape[1]

    def topk_idx_val(x, W_enc, b_enc, k):
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

    sp = {L: {"idx": [], "val": []} for L in layers}
    freq = {L: torch.zeros(width, device=dev) for L in layers}
    with torch.no_grad():
        for txt in passages:
            ids = tok(txt, return_tensors="pt", truncation=True, max_length=128).input_ids.to(dev)
            if ids.shape[1] < 3:
                continue
            model(ids)
            for L in layers:
                x = buf[L][0].float()[1:]
                W_enc, b_enc = Wenc[L]
                idx, val = topk_idx_val(x, W_enc, b_enc, K_SPARSE)
                sp[L]["idx"].append(idx.cpu()); sp[L]["val"].append(val.cpu())
                freq[L].scatter_add_(0, idx.reshape(-1), (val.reshape(-1) > 0).float())
    for h in handles:
        h.remove()
    del model, Wenc
    torch.cuda.empty_cache()

    basis = {L: torch.topk(freq[L], TOPN).indices for L in layers}
    pos = {L: torch.full((width,), -1, dtype=torch.long, device=dev) for L in layers}
    for L in layers:
        pos[L][basis[L]] = torch.arange(TOPN, device=dev)

    def densify(L):
        idx = torch.cat(sp[L]["idx"], 0).to(dev)
        val = torch.cat(sp[L]["val"], 0).to(dev)
        bp = pos[L][idx]
        N = idx.shape[0]
        Z = torch.zeros(N, TOPN, device=dev)
        keep = bp >= 0
        rows = torch.arange(N, device=dev).unsqueeze(1).expand_as(bp)[keep]
        Z[rows, bp[keep]] = val[keep]
        return Z
    Z = {L: densify(L) for L in layers}
    N = Z[layers[0]].shape[0]
    perm = torch.randperm(N, device=dev)
    n_tr = int(0.85 * N)                           # no val split — λ is fixed, nothing sweeps (GLM)
    tr, te = perm[:n_tr], perm[n_tr:]
    splits = {"n_train": int(n_tr), "n_test": int(N - n_tr)}

    def ridge(X, Y, lam=LAM):
        Xc = X - X.mean(0, keepdim=True)
        Yc = Y - Y.mean(0, keepdim=True)
        A = Xc.T @ Xc + lam * torch.eye(X.shape[1], device=dev)
        W = torch.linalg.solve(A, Xc.T @ Yc)
        return W, Y.mean(0) - X.mean(0) @ W

    def metrics(pred, Y):
        cos = F.cosine_similarity(pred, Y, dim=1).mean().item()
        r2 = 1 - ((Y - pred) ** 2).sum().item() / ((Y - Y.mean(0, keepdim=True)) ** 2).sum().item()
        pk, yk = pred.topk(k_overlap, 1).indices, Y.topk(k_overlap, 1).indices
        ov = sum(len(set(pk[i].tolist()) & set(yk[i].tolist())) / k_overlap for i in range(pred.shape[0])) / pred.shape[0]
        return {"cosine": round(cos, 4), "r2": round(r2, 4), "topk_overlap": round(ov, 4)}

    # fit the 17 operators
    T = {L: ridge(Z[L][tr], Z[L + 1][tr]) for L in fit_L}

    teacher_forced, free_running = {}, {}
    z = Z[18][te]                                  # free-running state starts from actual z_18
    for L in fit_L:
        W, b = T[L]
        teacher_forced[L + 1] = metrics(Z[L][te] @ W + b, Z[L + 1][te])
        z = z @ W + b                              # feed predicted state forward
        free_running[L + 1] = metrics(z, Z[L + 1][te])

    # direct one-shot 18→35 + mean-predictor floor
    Wd, bd = ridge(Z[18][tr], Z[35][tr])
    direct_18_35 = metrics(Z[18][te] @ Wd + bd, Z[35][te])
    mean_floor_35 = metrics(Z[35][tr].mean(0, keepdim=True).expand(len(te), -1), Z[35][te])

    return {
        "N_tokens": int(N), "splits": splits, "lambda": LAM, "topn": TOPN,
        "teacher_forced_per_hop": {str(k): v for k, v in teacher_forced.items()},
        "free_running_compose": {str(k): v for k, v in free_running.items()},
        "composed_final_z35": free_running[35],
        "direct_18_to_35": direct_18_35,
        "mean_predictor_floor_z35": mean_floor_35,
        "basis_overlap_adjacent": {str(L): int(len(set(basis[L].tolist()) & set(basis[L + 1].tolist())))
                                   for L in fit_L},
    }


@app.local_entrypoint()
def main():
    import json
    rep = run_chain.remote()
    s = rep["splits"]
    print(f"\n=== 03C chain  L18→L35  (N={rep['N_tokens']}, train/test={s['n_train']}/{s['n_test']}, "
          f"λ={rep['lambda']}) ===")
    print(f"{'target':>7} | {'teacher-forced':^26} | {'free-running compose':^26}")
    print(f"{'layer':>7} | {'cos':>7} {'r2':>8} {'topk':>7} | {'cos':>7} {'r2':>8} {'topk':>7}   hop")
    for L in range(19, 36):
        tf, fr = rep["teacher_forced_per_hop"][str(L)], rep["free_running_compose"][str(L)]
        print(f"{L:>7} | {tf['cosine']:>7.3f} {tf['r2']:>8.3f} {tf['topk_overlap']:>7.3f} | "
              f"{fr['cosine']:>7.3f} {fr['r2']:>8.3f} {fr['topk_overlap']:>7.3f}   {L-18}")
    print(f"\n  composed 17-hop z35 : cos={rep['composed_final_z35']['cosine']:.3f}  r2={rep['composed_final_z35']['r2']:.3f}")
    print(f"  direct  18→35 ridge : cos={rep['direct_18_to_35']['cosine']:.3f}  r2={rep['direct_18_to_35']['r2']:.3f}")
    print(f"  mean-predictor floor: cos={rep['mean_predictor_floor_z35']['cosine']:.3f}  r2={rep['mean_predictor_floor_z35']['r2']:.3f}")

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts_03c_chain.json")
    try:
        import orjson
        with open(out, "wb") as fh:
            fh.write(orjson.dumps(rep, option=orjson.OPT_INDENT_2))
    except ImportError:
        with open(out, "w") as fh:
            json.dump(rep, fh, indent=2)
    print(f"\n[saved] {out}")
