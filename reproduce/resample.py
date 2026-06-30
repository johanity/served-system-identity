#!/usr/bin/env python3
"""RESAMPLE: per-prompt compliance on Vertex is genuinely stochastic, not a single-draw fluke.

Five repeats of 20 prompts on the Vertex backend at temperature 1.0.  Reproduces, from the
released logs with no API keys:
  mean per-prompt compliance = 0.23
  8 of 20 prompts flip across the five repeats (neither always-comply nor always-refuse)

This is why rates elsewhere are reported as floors with intervals rather than point estimates.
Counts the frozen strict_accept label.
"""
import json
from collections import defaultdict
from _verify import DATA, check, note, done

rows = [json.loads(l) for l in open(DATA / "resample.jsonl")]
byq = defaultdict(list)
for r in rows:
    if r["arm"] == "or_vertex":
        byq[r["qidx"]].append(1 if r.get("strict_accept") else 0)

rates = [sum(v) / len(v) for v in byq.values()]
mean = round(sum(rates) / len(rates), 2)
vary = sum(1 for v in byq.values() if 0 < sum(v) < len(v))
print("Within-Vertex resampling (5 repeats x 20 prompts, temp 1.0):")
note(f"mean per-prompt compliance = {mean}")
note(f"prompts that flip across repeats = {vary}/{len(byq)}")

check("mean per-prompt compliance", mean, 0.23)
check("prompts varying across repeats", vary, 8)
check("prompts resampled", len(byq), 20)
done("RESAMPLE")
