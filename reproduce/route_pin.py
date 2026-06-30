#!/usr/bin/env python3
"""ROUTE PIN: pin OpenRouter's one alias to each Google backend, same prompts and framing.

Reproduces, from the released logs with no API keys, the strict acceptance-tag rate per
pinned backend while both report the identical version string:
  Vertex 27/100   AI Studio 2/100

Both arms return served_model `google/gemini-3.1-pro-preview-20260219`, so the divergence
is not visible in the advertised identity.  Counts the FROZEN `strict_accept` label, which
is why this is 27 and not 28: re-parsing a redacted body would put the accept tag at the
front of every blanked compliance and over-count by one row.
"""
import json
from collections import Counter
from _verify import DATA, check, note, done

rows = [json.loads(l) for l in open(DATA / "route_pin.jsonl")]
print("STRICT acceptance tag, per pinned backend:")
got = {}
for arm in ("or_vertex", "or_aistudio"):
    sub = [r for r in rows if r["arm"] == arm]
    acc = sum(1 for r in sub if r.get("strict_accept"))
    got[arm] = acc
    note(f"{arm:12s} {acc}/{len(sub)} = {100*acc/len(sub):.0f}%")

served = Counter(r.get("served_model") for r in rows)
note(f"served_model reported by both arms: {dict(served)}")

check("route-pin Vertex strict", got["or_vertex"], 27)
check("route-pin AI Studio strict", got["or_aistudio"], 2)
done("ROUTE PIN")
