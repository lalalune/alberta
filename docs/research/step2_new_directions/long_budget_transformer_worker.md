# Long-Budget Transformer Worker

Scope: run existing Tiny Shakespeare transformer scripts without code edits. Outputs
are under `outputs/step2_new_directions/long_budget_transformer_worker/`.

## Commands

```bash
source .venv/bin/activate && PYTHONUNBUFFERED=1 python "examples/The Alberta Plan/Step2/step2_tiny_shakespeare_proto_basis_transformer.py" --steps 2000 --seeds 2 --final-window 512 --eval-steps 512 --baseline-lr 0.10 --proto-lr 0.15 --output-dir outputs/step2_new_directions/long_budget_transformer_worker/proto_b010_p015_s2000
source .venv/bin/activate && PYTHONUNBUFFERED=1 python "examples/The Alberta Plan/Step2/step2_tiny_shakespeare_proto_basis_transformer.py" --steps 2000 --seeds 2 --final-window 512 --eval-steps 512 --baseline-lr 0.07 --proto-lr 0.15 --output-dir outputs/step2_new_directions/long_budget_transformer_worker/proto_b007_p015_s2000
source .venv/bin/activate && PYTHONUNBUFFERED=1 python "examples/The Alberta Plan/Step2/step2_tiny_shakespeare_proto_basis_transformer.py" --steps 2000 --seeds 2 --final-window 512 --eval-steps 512 --baseline-lr 0.15 --proto-lr 0.15 --output-dir outputs/step2_new_directions/long_budget_transformer_worker/proto_b015_p015_s2000
source .venv/bin/activate && PYTHONUNBUFFERED=1 python "examples/The Alberta Plan/Step2/step2_tiny_shakespeare_proto_basis_transformer.py" --steps 2000 --seeds 2 --final-window 512 --eval-steps 512 --baseline-lr 0.10 --proto-lr 0.10 --output-dir outputs/step2_new_directions/long_budget_transformer_worker/proto_b010_p010_s2000
source .venv/bin/activate && PYTHONUNBUFFERED=1 python "examples/The Alberta Plan/Step2/step2_tiny_shakespeare_proto_basis_transformer.py" --steps 2000 --seeds 2 --final-window 512 --eval-steps 512 --baseline-lr 0.10 --proto-lr 0.20 --output-dir outputs/step2_new_directions/long_budget_transformer_worker/proto_b010_p020_s2000
source .venv/bin/activate && PYTHONUNBUFFERED=1 python "examples/The Alberta Plan/Step2/step2_tiny_shakespeare_upgd_ffn_transformer.py" --steps 2000 --seeds 2 --final-window 512 --eval-steps 512 --baseline-lr 0.07 --upgd-lr 0.07 --output-dir outputs/step2_new_directions/long_budget_transformer_worker/upgd_b007_u007_s2000
source .venv/bin/activate && PYTHONUNBUFFERED=1 python "examples/The Alberta Plan/Step2/step2_tiny_shakespeare_upgd_ffn_transformer.py" --steps 2000 --seeds 2 --final-window 512 --eval-steps 512 --baseline-lr 0.10 --upgd-lr 0.10 --output-dir outputs/step2_new_directions/long_budget_transformer_worker/upgd_b010_u010_s2000
source .venv/bin/activate && PYTHONUNBUFFERED=1 python "examples/The Alberta Plan/Step2/step2_tiny_shakespeare_upgd_ffn_transformer.py" --steps 2000 --seeds 2 --final-window 512 --eval-steps 512 --baseline-lr 0.15 --upgd-lr 0.15 --output-dir outputs/step2_new_directions/long_budget_transformer_worker/upgd_b015_u015_s2000
source .venv/bin/activate && PYTHONUNBUFFERED=1 python "examples/The Alberta Plan/Step2/step2_tiny_shakespeare_proto_basis_transformer.py" --steps 5000 --seeds 2 --final-window 512 --eval-steps 512 --baseline-lr 0.10 --proto-lr 0.20 --output-dir outputs/step2_new_directions/long_budget_transformer_worker/proto_b010_p020_s5000
source .venv/bin/activate && PYTHONUNBUFFERED=1 python "examples/The Alberta Plan/Step2/step2_tiny_shakespeare_proto_basis_transformer.py" --steps 5000 --seeds 2 --final-window 512 --eval-steps 512 --baseline-lr 0.15 --proto-lr 0.20 --output-dir outputs/step2_new_directions/long_budget_transformer_worker/proto_b015_p020_s5000
```

## 2000-Step Prototype Sweep

Means are over 2 paired seeds.

| Run | Method | Final NLL | Eval PPL | Steps/s | Active Prototypes |
| --- | --- | ---: | ---: | ---: | ---: |
| `proto_b007_p015_s2000` | FFN baseline | 3.1861 | 23.4970 | 7463.6 | |
| `proto_b007_p015_s2000` | Prototype basis | 3.0895 | 23.4495 | 6460.7 | 3.1 |
| `proto_b007_p015_s2000` | FFN + prototype | 3.0241 | 22.9724 | 5518.5 | 12.8 |
| `proto_b010_p010_s2000` | FFN baseline | 3.0978 | 22.6779 | 7374.8 | |
| `proto_b010_p010_s2000` | Prototype basis | 3.1898 | 23.9688 | 5994.7 | 1.9 |
| `proto_b010_p010_s2000` | FFN + prototype | 3.1049 | 23.0462 | 5207.3 | 5.7 |
| `proto_b010_p015_s2000` | FFN baseline | 3.0978 | 22.6779 | 7434.4 | |
| `proto_b010_p015_s2000` | Prototype basis | 3.0895 | 23.4495 | 6296.8 | 3.1 |
| `proto_b010_p015_s2000` | FFN + prototype | 3.0241 | 22.9724 | 5969.1 | 12.8 |
| `proto_b010_p020_s2000` | FFN baseline | 3.0978 | 22.6779 | 7072.2 | |
| `proto_b010_p020_s2000` | Prototype basis | 3.0332 | 22.8154 | 5674.4 | 6.2 |
| `proto_b010_p020_s2000` | FFN + prototype | 2.9991 | 24.3986 | 5472.8 | 22.5 |
| `proto_b015_p015_s2000` | FFN baseline | 3.0170 | 22.9526 | 7432.6 | |
| `proto_b015_p015_s2000` | Prototype basis | 3.0895 | 23.4495 | 5868.2 | 3.1 |
| `proto_b015_p015_s2000` | FFN + prototype | 3.0241 | 22.9724 | 5115.7 | 12.8 |

Best 2000-step FFN baseline inside the prototype script:

- Final-window NLL: `3.0170` at `baseline_lr=0.15`.
- Eval perplexity: `22.6779` at `baseline_lr=0.10`.

Best 2000-step slow/fast methods:

- Best final-window NLL: FFN + prototype at `proto_lr=0.20`, `2.9991`.
- Best eval perplexity: prototype basis at `proto_lr=0.20`, `22.8154`.

Conclusion at 2000 steps: the hybrid beats the prototype-script tuned FFN on
final-window NLL, but no prototype or hybrid method beats the best tuned FFN on
held-out eval perplexity.

## 5000-Step Follow-Up

Means are over 2 paired seeds.

| Run | Method | Final NLL | Eval PPL | Steps/s | Active Prototypes |
| --- | --- | ---: | ---: | ---: | ---: |
| `proto_b010_p020_s5000` | FFN baseline | 2.8056 | 19.3734 | 10509.8 | |
| `proto_b010_p020_s5000` | Prototype basis | 2.7778 | 19.9615 | 9589.6 | 19.9 |
| `proto_b010_p020_s5000` | FFN + prototype | 2.8146 | 21.1859 | 7980.5 | 47.2 |
| `proto_b015_p020_s5000` | FFN baseline | 2.7866 | 19.6951 | 11119.5 | |
| `proto_b015_p020_s5000` | Prototype basis | 2.7778 | 19.9615 | 8691.2 | 19.9 |
| `proto_b015_p020_s5000` | FFN + prototype | 2.8146 | 21.1859 | 8192.4 | 47.2 |

Best 5000-step FFN baseline:

- Final-window NLL: `2.7866` at `baseline_lr=0.15`.
- Eval perplexity: `19.3734` at `baseline_lr=0.10`.

Best 5000-step slow/fast method:

- Prototype basis at `proto_lr=0.20`: final-window NLL `2.7778`, eval perplexity `19.9615`.

Conclusion at 5000 steps: prototype basis beats tuned FFN on final-window NLL,
but still loses held-out eval perplexity. The hybrid saturates more prototypes
and is worse than the pure prototype basis on both metrics.

## 2000-Step UPGD FFN Sweep

Means are over 2 paired seeds.

| Run | Method | Final NLL | Eval PPL | Steps/s | Max Perturbation |
| --- | --- | ---: | ---: | ---: | ---: |
| `upgd_b007_u007_s2000` | FFN baseline | 3.0784 | 25.5271 | 7293.8 | |
| `upgd_b007_u007_s2000` | UPGD FFN | 3.0783 | 25.5253 | 4992.2 | 0.000089 |
| `upgd_b010_u010_s2000` | FFN baseline | 3.0249 | 25.7150 | 7405.8 | |
| `upgd_b010_u010_s2000` | UPGD FFN | 3.0249 | 25.7123 | 4275.8 | 0.000087 |
| `upgd_b015_u015_s2000` | FFN baseline | 2.9899 | 27.4061 | 7258.9 | |
| `upgd_b015_u015_s2000` | UPGD FFN | 2.9899 | 27.4044 | 5142.7 | 0.000084 |

Conclusion for UPGD FFN: perturbations at the default magnitude are essentially
neutral on NLL/perplexity and slower than the plain FFN. It does not produce a
meaningful win over the tuned FFN baseline in this budget.

## Bottom Line

No slow+fast transformer-memory method beat the best tuned FFN baseline on both
held-out eval perplexity and final-window NLL.

The strongest positive result is narrow: prototype basis with `proto_lr=0.20`
beats tuned FFN final-window NLL at 5000 steps (`2.7778` vs `2.7866`) and the
hybrid beats tuned FFN final-window NLL at 2000 steps inside the prototype
script (`2.9991` vs `3.0170`). Both lose the held-out eval perplexity criterion.
