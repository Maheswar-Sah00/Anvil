# Figures

Pre-rendered architecture diagrams for the P-04 PCAM submission.

| File | Generator | What it shows |
|---|---|---|
| [`architecture.png`](architecture.png) | `make_architecture.py` | Full system flow: harness → offline precompute → online predict_precision → dynamics → scoring |
| [`uncertainty_gate.png`](uncertainty_gate.png) | `make_uncertainty_gate.py` | The sigmoid gate function over `w_max`, showing the clean separation between Test-2 probes and corrupted queries |
| [`benchmarks.png`](benchmarks.png) | `make_benchmarks.py` | Anti-gaming scoreboard (L1/L2/L3) + reference-adapter comparison |

## Regenerating

```bash
pip install matplotlib              # not a runtime dep — only needed for these figures
python docs/figures/make_architecture.py
python docs/figures/make_uncertainty_gate.py
python docs/figures/make_benchmarks.py
```

All three scripts use only matplotlib + numpy and have no dependencies on the bench code, so they regenerate in a few seconds.
