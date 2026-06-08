"""
Modal app — Experiment 03D (Gate 3): functional decode of the composed chain.

Run:  modal run experiments/modal_transition_03d_decode.py

03C showed the 17-hop linear chain predicts z_35 at feature-R²=0.445 (free-running),
well above the mean floor. But that is an internal-state metric. 03D asks the functional
question: does the composed predicted z_35 MEAN anything at the mouth?

(a) FUNCTIONAL DECODE
    predicted z_35 → scatter to 65536 → W_dec_35 + b_dec → final RMSNorm → lm_head → top-k
    Compare each method's tokens to:
      - SAE-decoded ACTUAL z_35  (isolates z-prediction quality; same decode path)
      - the MODEL's real next token (end-to-end; folds in Gate-1 SAE-decode loss ~0.69)
    Methods: mean-predictor, direct 18→35, free-running chain, teacher-forced, actual.
    Pre-registered 4-bucket scorer (math/numeric/code/other) on each method's tokens.

(b) MATH DIRECTION THROUGH THE CHAIN
    math_z18 → 17-hop chain → predicted math_z35 → decode → scorer, vs actual math_z35
    and a random control. Does the concept separable at L18 survive transport to L35?
"""

import os
import modal

app = modal.App("atlas-inference-transition-03d")
image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "torch", "transformers", "accelerate", "huggingface_hub", "safetensors", "datasets"
)

SAE_REPO = "Qwen/SAE-Res-Qwen3-8B-Base-W64K-L0_50"
BASE = "Qwen/Qwen3-8B-Base"
K_SPARSE = 50
TOPN = 2048
LAM = 1000.0
K_TOK = 10

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


@app.function(gpu="L40S", image=image, timeout=5400)
def run_03d(n_passages=500, seed=0, n_decode=2000):
    import random
    import torch
    import torch.nn.functional as F
    from huggingface_hub import hf_hub_download
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(seed); random.seed(seed)
    dev = "cuda"
    layers = list(range(18, 36))
    fit_L = layers[:-1]

    tok = AutoTokenizer.from_pretrained(BASE)
    model = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.bfloat16).to(dev).eval()
    lm_head = model.lm_head.weight.float()
    norm_w = model.model.norm.weight.float()
    eps = model.config.rms_norm_eps

    def load_sae(L, dec=False):
        sae = torch.load(hf_hub_download(SAE_REPO, f"layer{L}.sae.pt"), map_location=dev, weights_only=True)
        W_enc = sae["W_enc"].T.float().to(dev); b_enc = sae["b_enc"].float().to(dev)
        if not dec:
            return W_enc, b_enc
        W_dec = sae["W_dec"].float().to(dev)
        b_dec = sae.get("b_dec")
        b_dec = b_dec.float().to(dev) if b_dec is not None else torch.zeros(W_dec.shape[0], device=dev)
        return W_enc, b_enc, W_dec, b_dec

    Wenc = {L: load_sae(L) for L in layers}
    _, _, W_dec35, b_dec35 = load_sae(35, dec=True)
    width = Wenc[layers[0]][0].shape[1]

    def topk_idx_val(x, W_enc, b_enc, k):
        pre = torch.relu(x @ W_enc + b_enc)
        val, idx = pre.topk(k, dim=-1)
        return idx, val

    # ── 4-bucket scorer (pre-registered, from 01B) ──
    MATH_SYM = set("½⅓¼⅔¾∜√∛∫∑∏π±≤≥≠×÷∞∂∇θαβγδλμσΣΠΔΩ°·⋅−⁰¹²³⁴⁵⁶⁷⁸⁹ⁿ")
    MATH_WORDS = {"integral", "derivative", "prime", "equation", "polynomial", "factorial",
                  "theorem", "cosine", "sine", "tangent", "logarithm", "quadratic", "circumference",
                  "consecutive", "nonzero", "hypotenuse", "calculus", "algebra", "geometry",
                  "quotient", "remainder", "divisor", "fraction", "sqrt", "数学", "方程", "积分",
                  "导数", "质数", "平方", "概率", "定理", "函数", "几何", "代数"}
    CODE_WORDS = {"json", "def", "return", "import", "function", "const", "let", "async", "npm",
                  "git", "select", "class", "print", "echo", "public", "static", "void", "self"}
    CODE_PUNCT = (";", "{", "}", "=>", "::", "->", "//", "();", "()", "[]", ",json", "@app")

    def classify(s):
        t = s.strip(); low = t.lower()
        if not t: return "other"
        if any(c in MATH_SYM for c in t): return "math"
        if low in MATH_WORDS or t in MATH_WORDS: return "math"
        if any(w in low for w in MATH_WORDS if len(w) >= 6): return "math"
        if any(c.isdigit() for c in t) and all(c.isdigit() or c in ".,-+/%:" for c in t): return "numeric"
        if low in CODE_WORDS or any(p in t for p in CODE_PUNCT): return "code"
        return "other"

    def fractions(ids):
        toks = [tok.decode([int(i)]) for i in ids]
        n = max(len(toks), 1); c = {"math": 0, "numeric": 0, "code": 0, "other": 0}
        for t in toks: c[classify(t)] += 1
        return {k + "_frac": round(c[k] / n, 3) for k in c}

    # ── collect per-token features (all layers) + model real top-k next token ──
    buf = {}
    def mk(L):
        def hook(m, i, o): buf[L] = (o[0] if isinstance(o, tuple) else o).detach()
        return hook
    handles = [model.model.layers[L].register_forward_hook(mk(L)) for L in layers]
    sp = {L: {"idx": [], "val": []} for L in layers}
    freq = {L: torch.zeros(width, device=dev) for L in layers}
    real_tok = []                                  # model's real top-k next token per collected token
    with torch.no_grad():
        for txt in passages_iter(n_passages):
            ids = tok(txt, return_tensors="pt", truncation=True, max_length=128).input_ids.to(dev)
            if ids.shape[1] < 3: continue
            out = model(ids)
            real_tok.append(out.logits[0][1:].topk(K_TOK, dim=-1).indices.cpu())
            for L in layers:
                x = buf[L][0].float()[1:]
                W_enc, b_enc = Wenc[L]
                idx, val = topk_idx_val(x, W_enc, b_enc, K_SPARSE)
                sp[L]["idx"].append(idx.cpu()); sp[L]["val"].append(val.cpu())
                freq[L].scatter_add_(0, idx.reshape(-1), (val.reshape(-1) > 0).float())
    for h in handles: h.remove()

    # math/control directions (full width) before freeing model
    handles = [model.model.layers[L].register_forward_hook(mk(L)) for L in layers]
    def enc_full(L, txt):
        ids = tok(txt, return_tensors="pt").input_ids.to(dev); model(ids)
        x = buf[L][0].float()[1:]; W_enc, b_enc = Wenc[L]
        f = torch.zeros(x.shape[0], width, device=dev)
        idx, val = topk_idx_val(x, W_enc, b_enc, K_SPARSE)
        f.scatter_(1, idx, torch.relu(val)); return f.mean(0)
    with torch.no_grad():
        math_dir = {L: torch.stack([enc_full(L, t) for t in MATH]).mean(0)
                    - torch.stack([enc_full(L, t) for t in CONTROL]).mean(0) for L in (18, 35)}
    for h in handles: h.remove()
    del model, Wenc; torch.cuda.empty_cache()

    basis = {L: torch.topk(freq[L], TOPN).indices for L in layers}
    pos = {L: torch.full((width,), -1, dtype=torch.long, device=dev) for L in layers}
    for L in layers: pos[L][basis[L]] = torch.arange(TOPN, device=dev)

    def densify(L):
        idx = torch.cat(sp[L]["idx"], 0).to(dev); val = torch.cat(sp[L]["val"], 0).to(dev)
        bp = pos[L][idx]; N = idx.shape[0]; Z = torch.zeros(N, TOPN, device=dev)
        keep = bp >= 0
        rows = torch.arange(N, device=dev).unsqueeze(1).expand_as(bp)[keep]
        Z[rows, bp[keep]] = val[keep]; return Z
    Z = {L: densify(L) for L in layers}
    real_tok = torch.cat(real_tok, 0).to(dev)      # [N, K_TOK]
    N = Z[18].shape[0]
    perm = torch.randperm(N, device=dev); n_tr = int(0.85 * N)
    tr, te = perm[:n_tr], perm[n_tr:]

    def ridge(X, Y, lam=LAM):
        Xc = X - X.mean(0, keepdim=True); Yc = Y - Y.mean(0, keepdim=True)
        W = torch.linalg.solve(Xc.T @ Xc + lam * torch.eye(X.shape[1], device=dev), Xc.T @ Yc)
        return W, Y.mean(0) - X.mean(0) @ W
    T = {L: ridge(Z[L][tr], Z[L + 1][tr]) for L in fit_L}
    Wd, bd = ridge(Z[18][tr], Z[35][tr])

    # ── decode helpers ──
    def decode_ids(Z2048):                          # [M,2048] -> [M,K_TOK] token ids
        zf = torch.zeros(Z2048.shape[0], width, device=dev); zf[:, basis[35]] = Z2048
        xhat = zf @ W_dec35.T + b_dec35
        h = xhat * torch.rsqrt(xhat.pow(2).mean(-1, keepdim=True) + eps) * norm_w
        return (h @ lm_head.T).topk(K_TOK, dim=-1).indices

    def overlap(a, b):
        return sum(len(set(a[i].tolist()) & set(b[i].tolist())) / K_TOK for i in range(a.shape[0])) / a.shape[0]

    # test subset for decoding
    sub = te[:n_decode]
    z_free = Z[18][sub].clone()
    for L in fit_L: z_free = z_free @ T[L][0] + T[L][1]
    methods = {
        "actual_z35": Z[35][sub],
        "free_chain": z_free,
        "direct_18_35": Z[18][sub] @ Wd + bd,
        "teacher_forced": Z[34][sub] @ T[34][0] + T[34][1],
        "mean_predictor": Z[35][tr].mean(0, keepdim=True).expand(len(sub), -1),
    }
    ids_dec = {k: decode_ids(v) for k, v in methods.items()}
    real_sub = real_tok[sub]
    report = {"N_tokens": int(N), "n_decode": int(sub.shape[0]),
              "splits": {"n_train": int(n_tr), "n_test": int(N - n_tr)}, "decode": {}}
    for k, dids in ids_dec.items():
        report["decode"][k] = {
            "tok_overlap_vs_actual_z35": round(overlap(dids, ids_dec["actual_z35"]), 4),
            "tok_overlap_vs_model_real": round(overlap(dids, real_sub), 4),
            **fractions(dids.reshape(-1)),
        }

    # ── (b) math direction through the chain ──
    def to_basis(full, L):
        v = torch.zeros(TOPN, device=dev); nz = full.nonzero().squeeze(-1)
        bp = pos[L][nz]; v[bp[bp >= 0]] = full[nz][bp >= 0]; return v
    m18 = to_basis(math_dir[18], 18); m35 = to_basis(math_dir[35], 35)
    mc = m18.clone()
    for L in fit_L: mc = mc @ T[L][0] + T[L][1]
    rc = torch.randn(TOPN, device=dev)              # random control through the SAME 17-hop chain (GLM)
    for L in fit_L: rc = rc @ T[L][0] + T[L][1]
    def decode_dir(v):
        return decode_ids(v.unsqueeze(0))[0]
    report["math_through_chain"] = {
        "cos_chain_math_vs_actual_math35": round(F.cosine_similarity(mc, m35, dim=0).item(), 4),
        "cos_random_ctrl": round(F.cosine_similarity(rc, m35, dim=0).item(), 4),
        "math_feats_in_basis_L18": int((pos[18][math_dir[18].abs().topk(20).indices] >= 0).sum()),
        "math_feats_in_basis_L35": int((pos[35][math_dir[35].abs().topk(20).indices] >= 0).sum()),
        "decoded_chain_math": fractions(decode_dir(mc)),
        "decoded_actual_math35": fractions(decode_dir(m35)),
        "tokens_chain_math": [tok.decode([int(i)]) for i in decode_dir(mc)],
        "tokens_actual_math35": [tok.decode([int(i)]) for i in decode_dir(m35)],
    }
    return report


def passages_iter(n_passages):
    from datasets import load_dataset
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    return [t.strip() for t in ds["text"] if len(t.strip()) > 120][:n_passages]


@app.local_entrypoint()
def main():
    import json
    rep = run_03d.remote()
    s = rep["splits"]
    print(f"\n=== 03D functional decode  (N={rep['N_tokens']}, train/test={s['n_train']}/{s['n_test']}, "
          f"decoded {rep['n_decode']} tokens) ===")
    print(f"{'method':>16} | {'ov vs actual_z35':>16} {'ov vs model_real':>16} | math  num  code  other")
    for k in ["actual_z35", "free_chain", "direct_18_35", "teacher_forced", "mean_predictor"]:
        d = rep["decode"][k]
        print(f"{k:>16} | {d['tok_overlap_vs_actual_z35']:>16.3f} {d['tok_overlap_vs_model_real']:>16.3f} | "
              f"{d['math_frac']:.2f}  {d['numeric_frac']:.2f}  {d['code_frac']:.2f}  {d['other_frac']:.2f}")
    mc = rep["math_through_chain"]
    print(f"\n  (b) math through chain: cos(chain(math18), math35)={mc['cos_chain_math_vs_actual_math35']:.3f} "
          f"(random {mc['cos_random_ctrl']:.3f})  math in basis L18={mc['math_feats_in_basis_L18']}/20 L35={mc['math_feats_in_basis_L35']}/20")
    print(f"      chain-math decodes to:  {' '.join(repr(t) for t in mc['tokens_chain_math'])}")
    print(f"      actual-math35 decodes:  {' '.join(repr(t) for t in mc['tokens_actual_math35'])}")

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts_03d_decode.json")
    try:
        import orjson
        with open(out, "wb") as fh: fh.write(orjson.dumps(rep, option=orjson.OPT_INDENT_2))
    except ImportError:
        with open(out, "w") as fh: json.dump(rep, fh, indent=2)
    print(f"\n[saved] {out}")
