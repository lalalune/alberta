# Learned-Budget Transformer Memory

Date: 2026-05-07

## Question

The replay-capped advantage-memory transformer became the leading Tiny
Shakespeare Step 2 transformer candidate, but its resource budget was still a
static `gate_max=0.15`. This pass replaced that fixed cap with an online
learned budget and compared it directly against the static cap.

## Mechanism

New runner:

- `examples/The Alberta Plan/Step2/step2_tiny_shakespeare_learned_budget_memory_transformer.py`

The mechanism is a scalar learned cap:

- state: `budget_logit`;
- active cap: `budget_min + (budget_max - budget_min) * sigmoid(budget_logit)`;
- utility: delayed replay advantage, `base_loss - memory_loss`;
- uncertainty scale: EMA variance of replay advantage;
- pressure: current gate divided by current budget;
- cost: small normalized budget penalty.

The update is a bounded online controller:

`budget_logit += budget_lr * clip(tanh(replay_utility / replay_std) * pressure_gate - budget_cost * normalized_budget, -1, 1)`

The ordinary gate is still learned from replay advantage, but its maximum logit
is clipped to the current learned budget rather than to a fixed `gate_max`.

## Commands

Smoke:

```bash
source .venv/bin/activate && PYTHONUNBUFFERED=1 python "examples/The Alberta Plan/Step2/step2_tiny_shakespeare_learned_budget_memory_transformer.py" --steps 64 --seeds 1 --eval-steps 64 --final-window 32 --block-size 16 --d-model 16 --mlp-hidden 32 --proto-count 16 --replay-size 16 --output-dir outputs/step2_new_directions/learned_budget_memory_transformer_smoke
```

Small comparison:

```bash
source .venv/bin/activate && PYTHONUNBUFFERED=1 python "examples/The Alberta Plan/Step2/step2_tiny_shakespeare_learned_budget_memory_transformer.py" --steps 3000 --seeds 3 --eval-steps 512 --final-window 512 --budget-lr 0.0005 --budget-cost 0.005 --output-dir outputs/step2_new_directions/learned_budget_memory_transformer_3000_3seed_conservative
```

An earlier aggressive-budget screen was also run:

```bash
source .venv/bin/activate && PYTHONUNBUFFERED=1 python "examples/The Alberta Plan/Step2/step2_tiny_shakespeare_learned_budget_memory_transformer.py" --steps 3000 --seeds 3 --eval-steps 512 --final-window 512 --output-dir outputs/step2_new_directions/learned_budget_memory_transformer_3000_3seed
```

## Results

Smoke passed. The 64-step run traced and wrote artifacts. The budget stayed at
`0.150`, which is expected at that horizon.

The conservative 3000-step, 3-seed comparison:

| Metric | Static `gate_max=0.15` | Learned budget | Learned minus static |
| --- | ---: | ---: | ---: |
| final-window NLL | 2.9218 +/- 0.0345 | 2.9219 +/- 0.0345 | -0.0001 +/- 0.0000 |
| eval NLL | 3.1016 +/- 0.0641 | 3.1020 +/- 0.0638 | -0.0004 +/- 0.0004 |
| eval perplexity | 22.3260 +/- 1.4379 | 22.3344 +/- 1.4308 | -0.0084 +/- 0.0078 |
| eval fast-only perplexity | 22.4924 +/- 1.2662 | 22.4441 +/- 1.3060 | +0.0483 +/- 0.0547 |

Learned-budget diagnostics:

| Metric | Value |
| --- | ---: |
| final budget | 0.115114 +/- 0.000847 |
| final-window budget | 0.116049 +/- 0.000833 |
| final-window gate | 0.109153 +/- 0.001511 |
| final-window budget pressure | 0.940867 +/- 0.019508 |
| final-window replay advantage EMA | 0.000145 +/- 0.001126 |

The aggressive controller collapsed the budget harder, to about `0.0506`, and
lost more of the memory-enabled static-cap edge. It did improve fast-only eval,
which suggests the learned budget can function as a slow-path regularizer, but
that is not the target result here.

## Assessment

The learned-budget mechanism is mechanically clean and did replace static
`gate_max` with online state. It did not improve the replay-capped candidate on
the target memory-enabled metrics in the 3000-step comparison. The static
`gate_max=0.15` remains slightly better for memory-enabled eval NLL/perplexity.

The failure mode is informative: replay advantage is near zero or slightly
negative at 3000 steps, so the controller rationally shrinks the budget. That
protects fast-only deployment but removes the small memory-enabled gain that
the static cap preserved.

## Static Knobs Remaining

Still static after this change:

- `gate_lr`, `gate_decay`, and `gate_l2`;
- replay buffer size and FIFO replay sampling;
- `slow_lr`, `fast_lr`, and gradient clip;
- budget bounds and controller constants: `budget_min`, `budget_max`,
  `budget_lr`, `budget_ema_decay`, `budget_target_utilization`,
  `budget_cost`, and `budget_advantage_floor`;
- prototype count, novelty threshold, bandwidth, and update rate;
- post-FFN versus pre-FFN placement.

Next useful attack: keep the learned budget, but replace FIFO replay with
advantage-stratified replay or learn `slow_lr`/`gate_l2` from the same replay
advantage signal. The current scalar budget alone is too conservative when the
measured replay advantage is weak.
