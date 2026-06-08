"""
Modal app — Experiment 01B (Gate 2): contrastive concept direction → vocabulary.

Run:  modal run experiments/modal_bridge_01b.py

HEADLINE QUESTION
    Can a contrastive, prompt-derived concept direction in SAE feature space produce
    concept-matched vocabulary evidence?  (NOT "can the atlas reconstruct Qwen's logits.")

METHOD (per layer L, same-layer only — no L→L mapping; that would be Gate 3)
    z(prompt) = mean over tokens (BOS excluded) of TopK(50) SAE features at layer L
    math_delta = mean_z(math prompts) − mean_z(control prompts)        # contrastive
    residual   = W_dec @ math_delta                                    # NO b_dec
    logits     = final_RMSNorm(residual) @ lm_head.T
    classify the top-k tokens into math_specific / numeric / code_syntax / other

PRE-REGISTERED SCORER (ratified by GPT, 2026-06-08)
    math_specific : unambiguous math unicode + LaTeX commands + fixed math-word list
    numeric       : pure digits/decimals — shared by math AND code, NOT math-specific
    code_syntax   : code/API punctuation + code keywords
    other         : prose, foreign words, generic punctuation, bare _ ^ $ ( )
    Digits are NOT math. Bare _, ^, $, parens, punctuation are NOT math. Every decoded
    token is stored for every condition, especially the shuffled-label nulls.

CONTROLS
    shuffled-label nulls  — scramble math/control labels (the key null), tokens stored per run
    feature-shuffled      — permute the delta's feature indices (random-real, magnitude-matched)
    code_delta/poetry_delta — same construction, other concepts (specificity)
    gaussian              — random feature-space vector (floor)

EXPECTATION (GPT): L35 cleaner math notation; L18 richer internal signal (math words),
weaker direct unembed coherence. math/code share the NUMERIC axis; they diverge on
notation (math) vs syntax (code).
"""

import os
import modal

app = modal.App("atlas-inference-bridge-01b")
image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "torch", "transformers", "accelerate", "huggingface_hub", "safetensors"
)

SAE_REPO = "Qwen/SAE-Res-Qwen3-8B-Base-W64K-L0_50"
BASE = "Qwen/Qwen3-8B-Base"
K_SPARSE = 50

MATH = [
    "What is 7 times 8?", "Solve for x: 2x + 3 = 11.", "Compute the integral of x squared dx.",
    "Is 17 a prime number?", "What is the derivative of sin(x)?", "Add 145 and 278.",
    "Factor the polynomial x^2 - 5x + 6.", "What is the square root of 144?",
    "Find the area of a circle with radius 5.", "What is 12 divided by 4?",
    "Evaluate the limit of 1/x as x approaches infinity.", "Solve the quadratic x^2 - 4 = 0.",
    "What is the sum of the first 10 integers?", "Convert 3/4 to a decimal.",
    "What is 2 to the power of 10?", "Probability of rolling a 6 on a die?",
    "Differentiate f(x) = 3x^3.", "Greatest common divisor of 24 and 36?",
    "What is pi to two decimal places?", "Subtract 89 from 200.",
    "Find the slope of the line y = 2x + 1.", "What is 15 percent of 240?",
    "Compute the factorial of 5.", "Solve: x + y = 10, x - y = 2.",
    "What is the cosine of 60 degrees?", "Round 3.14159 to three decimals.",
    "How many degrees are in a triangle?", "Logarithm base 10 of 1000?",
    "Express 0.25 as a fraction.", "Find the mean of 4, 8, and 12.",
    "Perimeter of a square with side 7?", "Calculate 9 squared minus 4 squared.",
    "What is the next prime after 13?", "Divide 100 by 8; give the remainder.",
    "What is the derivative of e^x?", "Simplify the fraction 18/24.",
    "How much is 6 factorial?", "Hypotenuse of a 3-4-5 triangle?",
    "Sum the squares of 1 through 4.", "What is 1000 minus 333?",
]

CONTROL = [
    "What is your favorite season of the year?", "Describe the smell of fresh bread.",
    "Tell me about the history of the Roman Empire.", "How do you make a good cup of coffee?",
    "What makes a friendship last?", "Describe a walk through a quiet forest.",
    "Who was the first president of the United States?", "What do cats like to do all day?",
    "Explain how to plant a vegetable garden.", "What emotions does rain bring up for you?",
    "Tell me a story about a brave knight.", "What is the capital of France?",
    "How do birds know where to migrate?", "Describe your ideal vacation destination.",
    "What are the benefits of regular exercise?", "Talk about the taste of a ripe peach.",
    "Why do leaves change color in autumn?", "What makes a city feel like home?",
    "Describe the sound of ocean waves.", "How do you comfort a sad friend?",
    "What is the plot of Romeo and Juliet?", "Tell me about the life of a honeybee.",
    "What should I cook for a dinner party?", "Describe a warm summer evening.",
    "Who painted the Mona Lisa?", "Tips for a good night's sleep?",
    "Explain why dogs are loyal to humans.", "Describe a bustling morning market.",
    "What is your opinion on modern art?", "How do you stay motivated at work?",
    "Tell me about the culture of Japan.", "What makes a sunset beautiful?",
    "Describe the personality of a golden retriever.", "How do you brew a pot of tea?",
    "What is the story behind Thanksgiving?", "Talk about the joy of reading a book.",
    "Describe a cozy cabin in winter.", "What are the qualities of a good leader?",
    "How do you train a puppy to sit?", "What does a rainbow look like after a storm?",
]

CODE = [
    "def reverse_string(s): return s[::-1]", "for i in range(10): print(i)",
    "import numpy as np; arr = np.zeros(10)", "class Dog:\n    def __init__(self, name):\n        self.name = name",
    "SELECT * FROM users WHERE age > 18;", "const sum = (a, b) => a + b;",
    "git commit -m 'fix bug'", "try:\n    x = 1/0\nexcept ZeroDivisionError:\n    pass",
    "print('Hello, world!')", "x = [i*2 for i in range(5)]",
    "async function fetchData() { await fetch(url); }", "if __name__ == '__main__':\n    main()",
    "npm install react", "public static void main(String[] args) {}",
    "return [x for x in lst if x > 0]", "model.fit(X_train, y_train)",
    "df = pd.read_csv('data.csv')", "let arr = new Array(5).fill(0);",
    "@app.route('/api')", "docker run -p 8080:80 nginx",
    "function add(a, b) { return a + b; }", "with open('file.txt') as f:\n    data = f.read()",
    "const [state, setState] = useState(0);", "echo $PATH",
]

POETRY = [
    "The moon spills silver across the sleeping sea.", "Roses are red, violets are blue.",
    "I wandered lonely as a cloud.", "Shall I compare thee to a summer's day?",
    "The fog comes on little cat feet.", "Two roads diverged in a yellow wood.",
    "Hope is the thing with feathers.", "Do not go gentle into that good night.",
    "Her eyes held the quiet of falling snow.", "The autumn leaves drift past the pane.",
    "A heart that beats in time with distant rain.", "Whispers of dawn upon the dew-soft grass.",
    "The stars are old letters we cannot read.", "Love is a rose that blooms in winter's frost.",
    "Silence wears the color of the moon.", "The river sings its slow and ancient song.",
    "Petals fall like the breath of a sigh.", "Night unfolds her dark and velvet wings.",
    "Time is a thief with gentle, patient hands.", "The wind remembers names the trees forgot.",
    "Grief is an ocean with no farther shore.", "Morning gilds the rooftops with soft fire.",
    "We are but candles in the wind of years.", "The sea returns what the heart sets free.",
]


@app.function(gpu="L40S", image=image, timeout=3600)
def run_01b(prompts: dict, layers=(18, 35), k_tokens=20, n_shuffles=12, seed=42):
    import random
    import torch
    from huggingface_hub import hf_hub_download
    random.seed(seed)
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dev = "cuda"
    tok = AutoTokenizer.from_pretrained(BASE)
    model = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.bfloat16).to(dev).eval()
    lm_head = model.lm_head.weight.float()
    norm_w = model.model.norm.weight.float()
    eps = model.config.rms_norm_eps

    def to_logits(vec):                       # rmsnorm → unembed (the real head path)
        h = vec * torch.rsqrt(vec.pow(2).mean(-1, keepdim=True) + eps) * norm_w
        return h @ lm_head.T

    def topk_relu(x, k):
        relu_x = torch.relu(x)
        vals, idx = torch.topk(relu_x, k, dim=-1)
        out = torch.zeros_like(relu_x)
        out.scatter_(-1, idx, vals)
        return out

    def load_sae(L):
        sae = torch.load(hf_hub_download(SAE_REPO, f"layer{L}.sae.pt"),
                         map_location=dev, weights_only=True)
        return (sae["W_enc"].T.float().to(dev), sae["b_enc"].float().to(dev),
                sae["W_dec"].float().to(dev),
                (sae.get("b_dec").float().to(dev) if sae.get("b_dec") is not None
                 else torch.zeros(sae["W_dec"].shape[0], device=dev)))

    # ── pre-registered 4-bucket token classifier (GPT-ratified) ──
    MATH_SYM = set("½⅓¼⅔¾⅕⅖⅗⅘∜√∛∫∬∮∑∏π±≤≥≠≈×÷∞∂∇∈∀∃⇒⇔θαβγδλμσφω"
                   "ΣΠΔΩ°·⋅−⁰¹²³⁴⁵⁶⁷⁸⁹ⁿ₀₁₂₃₄")
    TEX = ("\\frac", "\\sum", "\\int", "\\sqrt", "\\pi", "\\theta", "\\cdot", "\\times",
           "\\partial", "\\lim", "\\infty", "\\alpha", "\\beta", "\\geq", "\\leq",
           "\\neq", "\\pm", "\\div", "\\sigma", "\\delta", "\\nabla", "\\prod")
    MATH_WORDS = {
        "integral", "integrate", "derivative", "differentiate", "prime", "equation",
        "polynomial", "factorial", "theorem", "matrix", "cosine", "sine", "tangent",
        "logarithm", "quadratic", "circumference", "consecutive", "nonzero", "modulo",
        "numerator", "denominator", "hypotenuse", "exponent", "coefficient", "calculus",
        "algebra", "geometry", "arithmetic", "quotient", "remainder", "divisor",
        "exponential", "logarithmic", "summation", "fraction", "sqrt", "pi",
    }
    MATH_ZH = {"数学", "方程", "方程式", "积分", "导数", "质数", "平方", "概率",
               "定理", "函数", "几何", "代数", "微积分"}          # strict (GPT)
    CODE_WORDS = {"json", "def", "return", "import", "function", "const", "let", "var",
                  "async", "await", "npm", "git", "select", "from", "where", "class",
                  "print", "echo", "true", "false", "null", "none", "public", "static",
                  "void", "app", "route", "self", "lambda", "yield", "struct"}
    CODE_PUNCT = (";", "{", "}", "=>", "::", "->", "&&", "||", "//", "/*", "();", "()",
                  "[]", ",json", "@app", "</", "/>", "():", "));", "});")

    def classify(s):
        t = s.strip()
        low = t.lower()
        if not t:
            return "other"
        if any(c in MATH_SYM for c in t):
            return "math"
        if any(x in low for x in TEX):
            return "math"
        if low in MATH_WORDS or t in MATH_ZH:
            return "math"
        if any(w in low for w in MATH_WORDS if len(w) >= 6):     # long math word contained
            return "math"
        if any(c.isdigit() for c in t) and all(c.isdigit() or c in ".,-+/%:" for c in t):
            return "numeric"                                     # pure number — NOT math
        if low in CODE_WORDS or any(p in t for p in CODE_PUNCT):
            return "code"
        return "other"

    def fractions(toks):
        n = max(len(toks), 1)
        c = {"math": 0, "numeric": 0, "code": 0, "other": 0}
        for t in toks:
            c[classify(t)] += 1
        return {"math_specific_fraction": round(c["math"] / n, 3),
                "numeric_fraction": round(c["numeric"] / n, 3),
                "code_syntax_fraction": round(c["code"] / n, 3),
                "other_fraction": round(c["other"] / n, 3)}

    # ── capture per-prompt mean_z at each layer (BOS excluded) ──
    items = [(lab, t) for lab, texts in prompts.items() for t in texts]
    labels = [lab for lab, _ in items]
    saes = {L: load_sae(L) for L in layers}

    buf = {}
    def mk(L):
        def hook(m, i, o):
            buf[L] = (o[0] if isinstance(o, tuple) else o).detach()
        return hook
    handles = [model.model.layers[L].register_forward_hook(mk(L)) for L in layers]

    mz = {L: [] for L in layers}
    with torch.no_grad():
        for _, txt in items:
            ids = tok(txt, return_tensors="pt").input_ids.to(dev)
            model(ids)
            for L in layers:
                x = buf[L][0].float()
                W_enc, b_enc, W_dec, b_dec = saes[L]
                f = topk_relu(x @ W_enc + b_enc, K_SPARSE)
                z = f[1:].mean(0) if f.shape[0] > 1 else f.mean(0)   # drop BOS
                mz[L].append(z)
    for h in handles:
        h.remove()
    for L in layers:
        mz[L] = torch.stack(mz[L])

    def set_mean(Z, lab):
        idx = [i for i, l in enumerate(labels) if l == lab]
        return Z[idx].mean(0)

    def decode_delta(delta, W_dec, b_dec=None):
        resid = W_dec @ delta
        if b_dec is not None:
            resid = resid + b_dec
        ids = torch.topk(to_logits(resid), k_tokens).indices.tolist()
        return [tok.decode([i]) for i in ids]

    def score(delta, W_dec, b_dec=None):
        toks = decode_delta(delta, W_dec, b_dec)
        out = fractions(toks)
        out["raw_tokens"] = toks
        return out

    report = {}
    for L in layers:
        Z = mz[L]
        W_enc, b_enc, W_dec, b_dec = saes[L]
        math_delta = set_mean(Z, "math") - set_mean(Z, "control")
        code_delta = set_mean(Z, "code") - set_mean(Z, "control")
        poetry_delta = set_mean(Z, "poetry") - set_mean(Z, "control")

        top_vals, top_idx = torch.topk(math_delta.abs(), 20)     # for atlas cross-ref later
        res = {
            "math_delta": score(math_delta, W_dec),                          # HEADLINE (no b_dec)
            "code_delta": score(code_delta, W_dec),                          # specificity
            "poetry_delta": score(poetry_delta, W_dec),                      # specificity
            "random_real_matched": score(math_delta[torch.randperm(math_delta.numel(), device=dev)], W_dec),
            "gaussian_runs": [score(torch.randn(W_dec.shape[1], device=dev), W_dec) for _ in range(5)],
            "math_mean_full_bdec_SANITY": score(set_mean(Z, "math"), W_dec, b_dec),
            "math_delta_top_feats": [
                {"idx": int(top_idx[i]), "delta": round(float(top_vals[i]), 4),
                 "sign": "math" if math_delta[top_idx[i]] > 0 else "control"}
                for i in range(len(top_idx))],
        }

        # shuffled-label null — store EVERY token per shuffle (GPT)
        mc_idx = [i for i, l in enumerate(labels) if l in ("math", "control")]
        n_math = sum(1 for l in labels if l == "math")
        shuffles = []
        for _ in range(n_shuffles):
            p = mc_idx[:]
            random.shuffle(p)
            d = Z[p[:n_math]].mean(0) - Z[p[n_math:]].mean(0)
            shuffles.append(score(d, W_dec))
        ms = [s["math_specific_fraction"] for s in shuffles]
        res["shuffled_label_runs"] = shuffles
        res["shuffled_label_math_mean"] = round(sum(ms) / len(ms), 3)
        res["shuffled_label_math_max"] = round(max(ms), 3)
        report[L] = res
    return report


@app.local_entrypoint()
def main():
    import json
    prompts = {"math": MATH, "control": CONTROL, "code": CODE, "poetry": POETRY}
    print(f"[01B] math={len(MATH)} control={len(CONTROL)} code={len(CODE)} poetry={len(POETRY)}")
    rep = run_01b.remote(prompts, layers=(18, 35), n_shuffles=200)

    def line(name, s):
        return (f"  {name:24s} math={s['math_specific_fraction']:.3f}  "
                f"num={s['numeric_fraction']:.3f}  code={s['code_syntax_fraction']:.3f}  "
                f"other={s['other_fraction']:.3f}")

    for L, r in rep.items():
        head = "   <<< HEADLINE" if L == 35 else ""
        print(f"\n================  LAYER {L}  ================")
        print(line("math_delta", r["math_delta"]) + head)
        print(f"      tokens: {' '.join(repr(t) for t in r['math_delta']['raw_tokens'])}")
        for name in ["code_delta", "poetry_delta", "random_real_matched"]:
            print(line(name, r[name]))
        gmax = max(g["math_specific_fraction"] for g in r["gaussian_runs"])
        print(f"  gaussian (x{len(r['gaussian_runs'])})            math_max={gmax:.3f}")
        print(f"  shuffled_label NULL      math mean={r['shuffled_label_math_mean']:.3f}  "
              f"max={r['shuffled_label_math_max']:.3f}  (n={len(r['shuffled_label_runs'])})")
        feats = " ".join(f"{d['idx']}:{d['delta']:.2f}({d['sign'][0]})" for d in r["math_delta_top_feats"][:8])
        print(f"  top math_delta feats: {feats}")

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts_01b.json")
    try:
        import orjson
        with open(out, "wb") as fh:
            fh.write(orjson.dumps(rep, option=orjson.OPT_INDENT_2))
    except ImportError:
        with open(out, "w") as fh:
            json.dump(rep, fh, indent=2)
    print(f"\n[saved] {out}")
