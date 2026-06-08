# Contributing to atlas-inference

This is an open research experiment by a solo dev and a rotating team of AI collaborators. It's early, it's weird, and all are welcome.

## How to get involved

- **Ideas / critique** — open an issue. "Here's why the premise is wrong" is as valuable as "here's a result." We're mapping territory; negative results count.
- **Code** — fork, branch, PR. Keep changes small and legible. Every experiment should state its hypothesis and its success/failure criterion up front.
- **Reproductions** — if you build your own atlas and run a probe, report what you saw. Same numbers or different, we want to know.

## Ground rules

- **No data, weights, or secrets in commits.** The `.gitignore` blocks the obvious cases; double-check before you push. The atlas and model weights stay external.
- **Claims need evidence.** A result is a number against a baseline, not a vibe. Say what you ran and what you measured.
- **Be honest about uncertainty.** "I think, but haven't verified" beats false confidence. This is research, not marketing.

## Style

- Python, kept simple. Readability over cleverness.
- Each experiment is self-contained and documents what it needs — which atlas tables, which external weights.

That's it. Bring a strange question.
