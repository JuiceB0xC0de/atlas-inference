# Issue: Undefined name `b_dec` in experiments/01_decoder_unembed_bridge.py

While running `ruff check .` as part of repo housekeeping, I found the following error in `experiments/01_decoder_unembed_bridge.py` at line 63:

F821 Undefined name `b_dec`
  --> experiments/01_decoder_unembed_bridge.py:63:13
   |
61 |     W_dec = sae["W_dec"].float()            # [4096, 65536] — columns are residual directions
62 |     device = W_dec.device
63 |     b_dec = b_dec.float().to(device=device) if b_dec is not None else torch.zeros(
   |             ^^^^^
64 |         W_dec.shape[0], device=device, dtype=W_dec.dtype)
65 |     return W_dec, b_dec
   |

F821 Undefined name `b_dec`
  --> experiments/01_decoder_unembed_bridge.py:63:48
   |
61 |     W_dec = sae["W_dec"].float()            # [4096, 65536] — columns are residual directions
62 |     device = W_dec.device
63 |     b_dec = b_dec.float().to(device=device) if b_dec is not None else torch.zeros(
   |                                                ^^^^^
64 |         W_dec.shape[0], device=device, dtype=W_dec.dtype)
65 |     return W_dec, b_dec
   |
```

As per the `AGENTS.md` guidelines, I must not modify anything under the `experiments/` directory or anything touching `b_dec`. Therefore, I am opening this issue for a researcher to address.
