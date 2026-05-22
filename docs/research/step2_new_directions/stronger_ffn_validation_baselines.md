# Stronger FFN Validation Baselines

Date: 2026-05-07.

This pass used validation only. No lockbox was run.

## Commands

Smoke:

```bash
source .venv/bin/activate
PYTHONUNBUFFERED=1 python benchmarks/step2_transformer_stronger_ffn_validation.py --steps 64 --seeds 1 --hidden-sizes 96 128 --learning-rates 0.05 0.15 --eval-steps 64 --eval-batch-size 32 --final-window 32 --output-dir outputs/step2_new_directions/stronger_ffn_validation_smoke
```

Validation grid:

```bash
source .venv/bin/activate
PYTHONUNBUFFERED=1 python benchmarks/step2_transformer_stronger_ffn_validation.py --steps 10000 --seeds 10 --hidden-sizes 96 128 --learning-rates 0.05 0.10 0.15 0.20 0.30 --eval-steps 4096 --eval-batch-size 512 --final-window 512 --output-dir outputs/step2_new_directions/stronger_ffn_validation_10000_10seed
```

30-seed confirmation for the strongest validation rows:

```bash
source .venv/bin/activate
PYTHONUNBUFFERED=1 python benchmarks/step2_transformer_stronger_ffn_validation.py --steps 10000 --seeds 30 --hidden-sizes 128 96 --learning-rates 0.05 0.10 --eval-steps 4096 --eval-batch-size 512 --final-window 512 --output-dir outputs/step2_new_directions/stronger_ffn_validation_10000_30seed_best2
```

## Result

The stronger FFN baselines erase the transformer-memory validation claim.

Reference 10000-step validation means from the 30-seed confirmatory artifact:

| Method | Final-window NLL | Held-out NLL | Held-out PPL |
|---|---:|---:|---:|
| `baseline_ffn_transformer` H64/LR0.15 | 2.7219 | 3.0697 | 21.94 |
| `advantage_post_ffn_memory` | 2.7212 | 3.0553 | 21.54 |
| `advantage_pre_ffn_kv_memory` | 2.7253 | 3.0468 | 21.33 |

30-seed stronger FFN validation:

| Method | Final-window NLL | Held-out NLL | Held-out PPL |
|---|---:|---:|---:|
| `ffn_h96_lr0p05` | 2.7564 +/- 0.0127 | 2.9317 +/- 0.0188 | 18.86 +/- 0.36 |
| `ffn_h128_lr0p05` | 2.7516 +/- 0.0123 | 2.9351 +/- 0.0208 | 18.94 +/- 0.41 |
| `ffn_h96_lr0p1` | 2.7153 +/- 0.0130 | 2.9989 +/- 0.0246 | 20.24 +/- 0.51 |
| `ffn_h128_lr0p1` | 2.7158 +/- 0.0125 | 3.0103 +/- 0.0277 | 20.53 +/- 0.60 |

Positive paired deltas below mean the memory method has higher NLL than the
stronger FFN and therefore loses.

| Stronger FFN | Memory method | Metric | Paired diff | 95% CI | W/L |
|---|---|---|---:|---:|---:|
| `ffn_h96_lr0p05` | `advantage_post_ffn_memory` | held-out NLL | +0.123620 | [+0.085214, +0.162026] | 29/1 |
| `ffn_h128_lr0p05` | `advantage_post_ffn_memory` | held-out NLL | +0.120251 | [+0.082677, +0.157825] | 29/1 |
| `ffn_h96_lr0p1` | `advantage_post_ffn_memory` | held-out NLL | +0.056463 | [+0.031056, +0.081870] | 24/6 |
| `ffn_h128_lr0p1` | `advantage_post_ffn_memory` | held-out NLL | +0.045009 | [+0.021824, +0.068193] | 24/6 |
| `ffn_h96_lr0p1` | `advantage_post_ffn_memory` | final-window NLL | +0.005895 | [-0.001747, +0.013536] | 21/9 |
| `ffn_h128_lr0p1` | `advantage_post_ffn_memory` | final-window NLL | +0.005400 | [-0.001037, +0.011838] | 21/9 |

## Decision

The replay-capped transformer-memory candidate should not be promoted. It beat
the original H64/LR0.15 FFN on the 10000-step held-out validation split, but it
does not beat validation-tuned H96/H128 FFNs. The fairest next transformer
baseline is at least H96/H128 with validation-selected LR, and possibly a
wall-clock matched FFN.

This does not invalidate the slow/fast thesis. It says the current transformer
memory block is not yet better than a simple tuned FFN.

## Validation

```bash
source .venv/bin/activate
python -m py_compile benchmarks/step2_transformer_stronger_ffn_validation.py
ruff check benchmarks/step2_transformer_stronger_ffn_validation.py
mypy --follow-imports=skip benchmarks/step2_transformer_stronger_ffn_validation.py
```

All passed.
