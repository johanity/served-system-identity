# generate/ — how the logs were produced

These are the exact harnesses that generated `data/`. They are here for transparency. **You do
not need them to check any number** — that is what `reproduce/` is for, and it needs no keys.

Running these **requires API keys and spends money** (OpenRouter, Google AI Studio, Vertex).
Set the keys in a `.env` at the repo root (see `.env.example`).

| script | produced |
|---|---|
| `concurrent_two_endpoint.py` | the headline run (both routes, concurrent) |
| `run2_concurrent.py`, `run3_concurrent.py`, `run4_concurrent.py` | the reproduction runs |
| `panel_regrade.py` | the 11-grader StrongREJECT panel |
| `or_provider_pin.py` | the Vertex / AI Studio route pin |
| `framing_ablation.py` | the 2x2 backend x framing ablation |
| `or_resample.py` | the within-backend resampling |
| `google_toggle.py` | the first-party safety-filter toggle |
| `negcontrol.py` | the harmless negative control |
| `confab_experiment.py`, `confab_plant.py` | the exploratory construct-and-recover arms |
| `harness_reference.py` | shared grading/classification reference |
| `_build_release.py` | freezes `strict_accept` and redacts harmful bodies to build `data/` |

Shared inputs (the prompt set and system prompt) are read from `../data/`. The raw, unredacted
run logs that `_build_release.py` consumes are not shipped; the redacted copies in `data/` are
the public release.
