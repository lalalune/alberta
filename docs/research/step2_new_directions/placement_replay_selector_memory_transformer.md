# Replay-Advantage Placement Selector for Tiny Shakespeare Memory Transformer

Date: 2026-05-07

## Question

The current replay-capped memory transformer has a stable static post-FFN
placement across 3000, 5000, and 10000 online tokens, while static pre-KV can
give better long-horizon held-out perplexity but is less stable on online
final-window loss. This pass tested whether a measured replay signal can manage
that placement tradeoff.

## Implementation

New isolated runner:

- `examples/The Alberta Plan/Step2/step2_tiny_shakespeare_placement_memory_transformer.py`

The runner keeps the current replay-capped scalar memory gate:

- delayed replay advantage signal;
- `replay_size=128`;
- `gate_max=0.15`;
- `gate_lr=0.5`;
- `gate_l2=0.1`;
- `slow_lr=0.1`.

It adds an adaptive placement selector:

- total prototype budget is preserved by default: 64 prototypes split into 32
  post-FFN slots and 32 pre-KV slots;
- both placement memories update every time step;
- the deployed logits are selected by a scalar learned pre-KV weight;
- memory gate signal is replay `base_loss - selected_memory_loss`;
- placement signal is replay `post_loss - pre_loss`;
- positive placement signal opens pre-KV, negative signal closes toward the
  post-FFN prior.

An initial failure mode was fixed before the reported comparison: selector
decay originally drifted the placement logit toward zero, which opened pre-KV
even with no positive replay evidence. The final runner decays toward the
configured prior logit instead.

## Commands

Smoke:

```bash
source .venv/bin/activate && PYTHONUNBUFFERED=1 python "examples/The Alberta Plan/Step2/step2_tiny_shakespeare_placement_memory_transformer.py" --steps 200 --seeds 1 --eval-steps 64 --final-window 64 --output-dir outputs/step2_new_directions/placement_memory_transformer_smoke_prior_decay
```

Conservative selector:

```bash
source .venv/bin/activate && PYTHONUNBUFFERED=1 python "examples/The Alberta Plan/Step2/step2_tiny_shakespeare_placement_memory_transformer.py" --steps 3000 --seeds 2 --eval-steps 512 --final-window 512 --output-dir outputs/step2_new_directions/placement_memory_transformer_3000_2seed_prior_decay
source .venv/bin/activate && PYTHONUNBUFFERED=1 python "examples/The Alberta Plan/Step2/step2_tiny_shakespeare_placement_memory_transformer.py" --steps 5000 --seeds 2 --eval-steps 512 --final-window 512 --output-dir outputs/step2_new_directions/placement_memory_transformer_5000_2seed_prior_decay
```

Aggressive selector:

```bash
source .venv/bin/activate && PYTHONUNBUFFERED=1 python "examples/The Alberta Plan/Step2/step2_tiny_shakespeare_placement_memory_transformer.py" --steps 3000 --seeds 2 --eval-steps 512 --final-window 512 --placement-lr 10.0 --placement-l2 0.0 --output-dir outputs/step2_new_directions/placement_memory_transformer_3000_2seed_aggressive_place
source .venv/bin/activate && PYTHONUNBUFFERED=1 python "examples/The Alberta Plan/Step2/step2_tiny_shakespeare_placement_memory_transformer.py" --steps 5000 --seeds 2 --eval-steps 512 --final-window 512 --placement-lr 10.0 --placement-l2 0.0 --output-dir outputs/step2_new_directions/placement_memory_transformer_5000_2seed_aggressive_place
```

## Results

Profiles are nearly budget-matched:

| Method | Trainable params | Extra state elements |
| --- | ---: | ---: |
| Static post-FFN replay memory | 15617 | 6502 |
| Static pre-KV replay memory | 15617 | 6502 |
| Adaptive placement memory | 15649 | 6505 |

Conservative selector (`placement_lr=0.5`, `placement_l2=0.01`):

| Horizon | Method | Final-window NLL | Held-out NLL | Held-out perplexity | Pre weight |
| ---: | --- | ---: | ---: | ---: | ---: |
| 3000 | Static post-FFN | 2.952083 | 3.044201 | 21.018904 | |
| 3000 | Static pre-KV | 2.952852 | 3.044946 | 21.034595 | |
| 3000 | Adaptive selector | 2.952408 | 3.045254 | 21.040441 | 0.108338 |
| 5000 | Static post-FFN | 2.786456 | 2.975732 | 19.697406 | |
| 5000 | Static pre-KV | 2.787092 | 2.972465 | 19.629089 | |
| 5000 | Adaptive selector | 2.786757 | 2.975754 | 19.697930 | 0.109210 |

Aggressive selector (`placement_lr=10.0`, `placement_l2=0.0`):

| Horizon | Method | Final-window NLL | Held-out NLL | Held-out perplexity | Pre weight |
| ---: | --- | ---: | ---: | ---: | ---: |
| 3000 | Static post-FFN | 2.952083 | 3.044201 | 21.018904 | |
| 3000 | Static pre-KV | 2.952852 | 3.044946 | 21.034595 | |
| 3000 | Adaptive selector | 2.952309 | 3.043702 | 21.008659 | 0.499467 |
| 5000 | Static post-FFN | 2.786456 | 2.975732 | 19.697406 | |
| 5000 | Static pre-KV | 2.787092 | 2.972465 | 19.629089 | |
| 5000 | Adaptive selector | 2.786288 | 2.974273 | 19.665534 | 0.424474 |

## Assessment

The learned placement signal is useful, but not yet a replacement for the
static post-FFN candidate.

Conservative placement simplifies back to post-FFN behavior and avoids the
unstable pre-KV branch, but it misses the 5000-step held-out pre-KV opportunity.
It does not beat static post.

Aggressive placement is more interesting. It improves held-out perplexity over
static post at both 3000 and 5000 steps. At 5000 steps it also improves
final-window NLL over static post (`2.786288` vs `2.786456`), while recovering
about half of static pre-KV's held-out gain. At 3000 steps it slightly loses
final-window NLL to static post while improving held-out perplexity.

Conclusion: do not promote this over the current replay-capped post-FFN static
placement yet. The aggressive selector is a promising follow-up because it
turns the post/pre choice into a measured replay-control problem and gives a
positive 5000-step result, but it is a 2-seed result with tiny margins.

## Failure Modes

- The placement replay signal is very small: aggressive replay placement
  advantage was about `0.000897` at 3000 and `0.001774` at 5000. This is close
  to noise at 2 seeds.
- The online replay signal does not perfectly match held-out behavior. Static
  pre-KV had the best 5000 held-out perplexity, but the adaptive selector only
  moved partway toward it.
- The fair budget split gives each adaptive branch only half the prototypes.
  This keeps total state comparable, but can undertrain the branch that becomes
  useful later.
- The adaptive runner is slower because it evaluates both placement branches:
  aggressive 5000-step throughput was about `2231` steps/s versus static
  post-FFN `2393` steps/s in the 2-seed run.
- The memory resource cap is still static. This pass learns placement, not
  `gate_max`.

## Validation

Passed:

- `python -m py_compile "examples/The Alberta Plan/Step2/step2_tiny_shakespeare_placement_memory_transformer.py"`
- `ruff check "examples/The Alberta Plan/Step2/step2_tiny_shakespeare_placement_memory_transformer.py"`
- `mypy --follow-imports=skip "examples/The Alberta Plan/Step2/step2_tiny_shakespeare_placement_memory_transformer.py"`
- `ruff check .`
- `pytest tests/ -v`: 1295 passed, 36 warnings in 682.17s
