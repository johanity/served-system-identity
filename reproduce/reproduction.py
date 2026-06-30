#!/usr/bin/env python3
"""REPRODUCTION: the gap is not a one-day fluke. It holds on every concurrent run.

Four concurrent runs across three days (June 11, 13, 19), strict acceptance tag, free tier.
The absolute OpenRouter level drifts over time (the drift documented in the paper), but the
divergence from Google never closes: OpenRouter is far above Google on every single run.

  run           OpenRouter   Google   gap
  Jun 11 (1)       57          6       51
  Jun 13 (2)       57          3       54
  Jun 19 (3)       35          4       31
  Jun 19 (4)       39          4       35

Counts the frozen strict_accept label. No API keys.
"""
import json
from _verify import DATA, check, note, done

RUNS = [
    ("Jun 11 (1)", "headline_concurrent.jsonl", 57, 6),
    ("Jun 13 (2)", "headline_run2_jun13.jsonl", 57, 3),
    ("Jun 19 (3)", "headline_run3_jun19.jsonl", 35, 4),
    ("Jun 19 (4)", "headline_run4_jun19.jsonl", 39, 4),
]

print("OpenRouter vs Google, strict tag, free tier, every concurrent run:")
min_gap = 99
for label, fname, exp_or, exp_go in RUNS:
    rows = [json.loads(l) for l in open(DATA / fname)]
    oc = sum(1 for r in rows if r.get("endpoint") == "openrouter" and r.get("side") == "free" and r.get("strict_accept"))
    gc = sum(1 for r in rows if r.get("endpoint") == "google" and r.get("side") == "free" and r.get("strict_accept"))
    gap = oc - gc
    min_gap = min(min_gap, gap)
    note(f"{label}:  OpenRouter {oc:2d}   Google {gc:2d}   gap {gap:2d}")
    check(f"{label} OpenRouter", oc, exp_or)
    check(f"{label} Google", gc, exp_go)

note(f"smallest gap across all four runs = {min_gap} points (OpenRouter always far above Google)")
check("gap holds on every run (min >= 30)", min_gap >= 30, True)
done("REPRODUCTION")
