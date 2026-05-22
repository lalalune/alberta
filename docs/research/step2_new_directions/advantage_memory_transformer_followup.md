# Advantage-Gated Transformer Memory Follow-Up

Date: 2026-05-06

## Question

The previous transformer pass showed objective mismatch: slow prototype memory
could improve online adaptation, but memory-enabled inference usually hurt
held-out Tiny Shakespeare perplexity versus the tuned FFN transformer. This
follow-up tested whether explicit fast-vs-slow advantage gating can turn the
slow path into a reliable resource-managed learner.

## Implementation

New runner:

- `examples/The Alberta Plan/Step2/step2_tiny_shakespeare_advantage_memory_transformer.py`

The runner compares a tuned FFN transformer against two slow-memory placements:

- post-FFN residual memory;
- pre-FFN KV-style memory.

It now supports:

- scalar advantage gate: one gate for the whole slow path;
- per-prototype utility gate: one gate per prototype, updated by activation
  credit assignment;
- delayed replay advantage gates using a fixed-size in-stream replay buffer;
- explicit resource caps with `gate_max`;
- memory-only training loss;
- blended fast/memory training loss;
- fast-only evaluation after memory-regularized training.

The advantage signal is `base_loss - memory_loss`. Positive values open memory;
negative values close memory.

## Hard-Target Update: Replay-Capped Resource Manager

The strongest current candidate is no longer fast-only deployment. It is a
memory-enabled post-FFN replay-gated block with an explicit resource cap.

Config:

- `gate_objective=replay`;
- `replay_size=128`;
- `gate_init_logit=-3.0`;
- `gate_lr=0.5`;
- `gate_decay=0.995`;
- `gate_l2=0.1`;
- `gate_max=0.15`;
- `slow_lr=0.1`;
- `train_loss_mode=memory`;
- `seeds=10`.

This single config was tested at 3000, 5000, and 10000 online steps.

| Horizon | Method | Final-window NLL | Held-out NLL | Held-out perplexity |
| ---: | --- | ---: | ---: | ---: |
| 3000 | FFN baseline | 2.9025 +/- 0.0195 | 3.0999 +/- 0.0492 | 22.4428 +/- 1.1414 |
| 3000 | post-FFN replay memory | 2.9023 +/- 0.0195 | 3.0996 +/- 0.0492 | 22.4360 +/- 1.1393 |
| 5000 | FFN baseline | 2.8686 +/- 0.0263 | 3.0131 +/- 0.0597 | 20.6829 +/- 1.2633 |
| 5000 | post-FFN replay memory | 2.8683 +/- 0.0262 | 3.0115 +/- 0.0595 | 20.6484 +/- 1.2610 |
| 10000 | FFN baseline | 2.7227 +/- 0.0439 | 3.0681 +/- 0.0679 | 21.9730 +/- 1.5942 |
| 10000 | post-FFN replay memory | 2.7217 +/- 0.0442 | 3.0504 +/- 0.0639 | 21.5362 +/- 1.4832 |

The margins are small, especially at 3000 and 5000 steps, but the direction is
consistent across all three horizons. This is the first transformer memory
candidate in this series that beats the tuned FFN on both online final-window
loss and held-out perplexity with the memory path enabled.

The lower cap matters. With `gate_max=0.35`, 3000 and 5000 steps improved, but
10000 steps lost final-window NLL despite better held-out perplexity. With no
cap or a loose cap, the replay objective over-opened memory and damaged the
fast representation. The useful mechanism is therefore not replay alone; it is
replay-gated slow memory under an explicit resource budget.

The best long-horizon held-out metric sometimes came from pre-FFN KV memory.
At 10000 steps with `gate_max=0.15`, pre-KV reached held-out perplexity
`21.0964 +/- 1.4275` versus post-FFN `21.5362 +/- 1.4832`, but it was only
neutral on final-window NLL. Post-FFN is the current canonical placement
because it is the stable all-horizon choice on both target metrics.

## Confirmed Positive Result

The strongest confirmed result is scalar advantage-gated memory used as a
training-time auxiliary path, with fast-only deployment.

Config:

- `steps=3000` and `steps=5000`;
- `seeds=10`;
- `slow_lr=0.1`;
- `gate_init_logit=-3.0`;
- `gate_decay=0.995`;
- `gate_l2=0.05`;
- `train_loss_mode=memory`.

At 3000 steps:

| Metric | Baseline FFN | Post memory, fast-only | Pre-KV memory, fast-only |
| --- | ---: | ---: | ---: |
| final-window base NLL | 2.9025 +/- 0.0195 | 2.9007 +/- 0.0195 | 2.9008 +/- 0.0195 |
| held-out fast NLL | 3.0999 +/- 0.0492 | 3.0967 +/- 0.0463 | 3.0962 +/- 0.0474 |
| held-out fast perplexity | 22.4428 +/- 1.1414 | 22.3420 +/- 1.0610 | 22.3432 +/- 1.0898 |

At 5000 steps:

| Metric | Baseline FFN | Post memory, fast-only | Pre-KV memory, fast-only |
| --- | ---: | ---: | ---: |
| final-window base NLL | 2.8686 +/- 0.0263 | 2.8661 +/- 0.0258 | 2.8677 +/- 0.0262 |
| held-out fast NLL | 3.0131 +/- 0.0597 | 3.0080 +/- 0.0578 | 3.0118 +/- 0.0591 |
| held-out fast perplexity | 20.6829 +/- 1.2633 | 20.5567 +/- 1.2073 | 20.6516 +/- 1.2535 |

This is a real but narrow result: slow memory improves the fast transformer when
the deployed prediction path ignores the memory residual. The effect is small,
but it survived 10 seeds at both horizons.

## What Still Fails

Memory-enabled inference is still not robust for the scalar gate.

At 3000 steps, scalar memory-enabled held-out perplexity loses:

| Metric | Baseline FFN | Post memory | Pre-KV memory |
| --- | ---: | ---: | ---: |
| held-out perplexity | 22.4428 +/- 1.1414 | 22.4740 +/- 1.1423 | 22.4656 +/- 1.1463 |

At 5000 steps, scalar memory-enabled held-out perplexity also loses:

| Metric | Baseline FFN | Post memory | Pre-KV memory |
| --- | ---: | ---: | ---: |
| held-out perplexity | 20.6829 +/- 1.2633 | 20.7385 +/- 1.2789 | 20.6969 +/- 1.2655 |

The learned scalar gate remains partially open even though final-window
advantage is negative. That makes it useful as a regularizer but not yet as an
inference-time memory controller.

## Falsified Knobs

Blended fast/memory loss did not improve the result. Weights `0.25`, `0.50`,
and `0.75` weakened the fast-only gain compared with memory-only training.

Removing gate drift by setting `gate_decay=1.0` closed the memory path almost
entirely. This fixed harmful memory inference but removed the useful training
signal.

Intermediate gate decays `0.998` and `0.999` were safer than `0.995`, but also
too weak to improve held-out fast-only evaluation meaningfully.

Per-prototype gates with `gate_l2=0.05` improved online/base NLL but damaged
held-out fast-only generalization when opened enough to matter.

## Promising But Unstable Result

Penalty-free per-prototype gating gave the first memory-enabled result that can
beat the FFN on both online and held-out metrics, but the winning placement
changed with horizon.

Config:

- `gate_mode=prototype`;
- `gate_init_logit=-2.0`;
- `gate_lr=10.0`;
- `gate_l2=0.0`;
- `slow_lr=0.1`.

At 3000 steps, post-FFN memory-enabled inference was slightly positive:

| Metric | Baseline FFN | Post memory |
| --- | ---: | ---: |
| final-window NLL | 2.9025 +/- 0.0195 | 2.9020 +/- 0.0193 |
| held-out NLL | 3.0999 +/- 0.0492 | 3.0994 +/- 0.0499 |
| held-out perplexity | 22.4428 +/- 1.1414 | 22.4395 +/- 1.1602 |

At 5000 steps, post-FFN lost final-window NLL, while pre-KV became slightly
positive:

| Metric | Baseline FFN | Pre-KV memory |
| --- | ---: | ---: |
| final-window NLL | 2.8686 +/- 0.0263 | 2.8684 +/- 0.0260 |
| held-out NLL | 3.0131 +/- 0.0597 | 3.0118 +/- 0.0594 |
| held-out perplexity | 20.6829 +/- 1.2633 | 20.6534 +/- 1.2574 |

This is not canonical yet. The signs are encouraging, but the margins are tiny
and the correct placement is not stable across training horizon.

## Compute Cost

The memory variants add modest parameters but significant online compute.

Scalar gate:

- baseline trainable parameters: `13537`;
- memory trainable parameters: `15617` (+15.4%);
- memory extra state: `2276` elements / `9104` bytes;
- 3000-step throughput: about `5746`-`5806` steps/s versus baseline `8896`
  steps/s.

Per-prototype gate:

- memory extra state: `2339` elements / `9356` bytes;
- 3000-step throughput: about `4981`-`5233` steps/s versus baseline `8023`
  steps/s.

The current Python/JAX research runner is roughly 1.5x to 1.6x slower than the
small FFN baseline. A fused core implementation is still required before making
production efficiency claims.

## Assessment

The slow+fast thesis is stronger after this pass, and the replay-capped
post-FFN block is now the leading transformer Step 2 candidate.

What is now established:

- slow prototype memory can improve the fast learner as an auxiliary online
  training path;
- the effect survives 10 seeds at 3000 and 5000 steps;
- per-prototype utility gates can sometimes make memory-enabled inference beat
  the FFN directly;
- delayed replay gating plus an explicit resource cap gives memory-enabled
  post-FFN inference that beats the tuned FFN on both final-window NLL and
  held-out perplexity at 3000, 5000, and 10000 online steps.

What is still missing:

- stronger margins; the 3000/5000 wins are real but small;
- a learned resource budget; `gate_max=0.15` is still static;
- a learned placement manager; post-FFN is stable across horizons, while
  pre-KV sometimes gives better held-out perplexity at longer horizons;
- token-budget evidence beyond 10000 online steps;
- evidence beyond Tiny Shakespeare;
- a fused implementation with competitive throughput.

The next concrete research direction is to learn the resource budget itself.
The present result says that replay-gated memory is useful only when compute is
bounded tightly. A stronger universal learner should adapt that cap from
uncertainty, replay advantage, and drift instead of treating `gate_max` as a
fixed hyperparameter.
