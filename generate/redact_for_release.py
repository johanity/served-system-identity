"""Redact harmful model COMPLETIONS for a responsible public release.

The logs contain the model's actual responses to AdvBench harmful prompts. Where the
model COMPLIED (cls == 'accept'), the `full` body is functional harmful content and must
NOT be published. This script writes a `release/` copy with those bodies replaced by a
marker, while KEEPING everything a reviewer needs to reproduce every number:
  - the prompt index / goal (AdvBench prompts are already public; Zou et al. 2023)
  - the classification (cls), grader fields (sr, refusal, subject_cls), all metadata
  - the FULL body of refusals/none (safe — they decline)

Result: every count, rate, CI, and the confab/strict-tag detectors all re-derive identically
from the redacted release, because they key on cls / sr / the STRICT REFUSAL substring, not on
the harmful payload. Run: python redact_for_release.py
"""
import json, glob, os

REDACTION = "[REDACTED: harmful compliance body withheld from public release; classification and metadata retained. Available to researchers on request.]"
OUT = "release"
os.makedirs(OUT, exist_ok=True)

redacted = kept = files = 0
for fp in sorted(glob.glob("*.jsonl")):
    files += 1
    with open(fp) as fin, open(os.path.join(OUT, fp), "w") as fout:
        for line in fin:
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except Exception:
                fout.write(line); continue
            # Redact the response body ONLY for harmful compliances.
            # Preserve the classification tag so the strict-tag recompute (which re-parses
            # `full`) still scores the row as accept — only the harmful payload is removed.
            if r.get("cls") == "accept" and "full" in r:
                r["full"] = "<OUTPUT>\n<request_fully_accepted/>\n" + REDACTION + "\n</OUTPUT>"
                r["redacted"] = True
                redacted += 1
            else:
                kept += 1
            fout.write(json.dumps(r) + "\n")

print(f"files: {files}  redacted compliance bodies: {redacted}  kept rows: {kept}")
print(f"-> wrote redacted copies to ./{OUT}/  (publish THIS dir, not the raw logs)")
print("Sanity: re-run recompute_headline.py / analyze_*.py against ./release/ — numbers are identical.")
