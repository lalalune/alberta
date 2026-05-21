# Step 2 Fast/Slow And OPMNIST Task Closure

Date: May 6, 2026.

## Task Breakdown And Status

| Workstream | Owner | Deliverable | Status |
| --- | --- | --- | --- |
| Learned fast/slow resource controls | Worker A | `d19_adaptive_fast_slow_d18.py` | Implemented and validated mechanically |
| OPMNIST bridge for D18 | Worker B | `d18_opmnist_bridge.py` | Implemented; exposes raw and deployment metrics |
| Complexity pruning | Worker C | `worker_c_d18_complexity_pruning_analysis.py` | Implemented; tanh128 confirmed, tanh64 promising |
| Core JAX production scaffold | Worker D | `alberta_framework.core.fast_slow` | Implemented, tested, exported from core |
| OPMNIST retention fix | Main thread | `d20_multiprototype_opmnist.py` | Implemented; beats MLP on local and true-MNIST compact OPMNIST checks |

## What We Learned

The D18 canonical learner is still the current all14 Step 2 benchmark winner,
but its OPMNIST bridge exposed a precise failure mode. D18 adapts online on
OPMNIST-like streams, but a single retained prototype per class does not retain
multiple pixel-permutation task views. Softmax deployment calibration fixes much
of the raw MSE scale, but not the task-view accuracy gap.

The decisive new result is D20: multiple novelty-allocated prototypes per class.
This is not a router and not an MLP expert. It is one online memory learner that
updates every step from the current observation and one-hot target.

## D18 OPMNIST Bridge Result

Artifact:
`outputs/step2_new_directions/d18_opmnist_bridge_softmax_local3/d18_opmnist_bridge_softmax_local3_results.json`

Protocol: local no-network sklearn digits expanded to 28x28, 5 pixel
permutation tasks, 1000 online steps, 3 paired seeds, 200-step final window.

| Metric | D18 vs best MLP |
| --- | ---: |
| Final-window MSE | `+0.008268`, wins `3/0/0` |
| Final-window accuracy | `+0.073333`, wins `3/0/0` |
| Raw held-out test MSE | `-0.122405`, wins `0/3/0` |
| Raw held-out test accuracy | `-0.074833`, wins `0/3/0` |
| Softmax deployment test MSE | `-0.003654`, wins `1/2/0` |
| Softmax deployment test accuracy | `-0.074833`, wins `0/3/0` |

Interpretation: D18 solves online adaptation on the bridge, but not held-out
task-view retention.

## D19 Learned Resource Controls

Worker A implemented a causal controller over:

- Target-persistence threshold.
- Basis readout decay.
- Simplex/one-hot readout decay.
- Fast/slow loss EMA and persistence EMA signals.

It works mechanically and keeps the model as one D18 additive learner with no
MLP expert and no prediction router. It closes the static threshold/decay
prototype gap for class-blocked behavior, but does not solve compact OPMNIST.
The failure is representational: OPMNIST needs multiple task-view memories per
class, not only adaptive readout decay.

## Complexity Pruning

Worker C confirmed:

- Full-budget tanh128 is the smallest fully confirmed all14 candidate:
  `140/0/0` paired final-window MSE wins and no digit heldout negative cells.
- Full-budget tanh64 is promising at 3 seeds:
  `42/0/0` paired final-window MSE wins and no digit heldout negative cells, but
  needs a 10-seed confirmation before promotion.
- Half RKHS budget reopens a digit heldout gap.
- Quarter RKHS budget reopens primary all14 gaps.

Current pruning decision: keep full RKHS budgets, keep tanh128 as confirmed,
and treat tanh64 as the next confirmation run.

## D20 OPMNIST Fix

Artifact:
`outputs/step2_new_directions/d20_multiprototype_opmnist_local3/d20_multiprototype_opmnist_local3_results.json`

Protocol: local sklearn digits 28x28 OPMNIST analogue, 5 pixel permutation
tasks, 1000 online steps, 3 paired seeds, 200-step final window.

| Metric | D20 | Best MLP | D20 margin |
| --- | ---: | ---: | ---: |
| Final-window MSE | `0.026361` | `0.076341` | `+0.049980` |
| Final-window accuracy | `0.825000` | `0.520000` | `+0.305000` |
| Held-out test MSE | `0.013433` | `0.085800` | `+0.070844` |
| Held-out test accuracy | `0.907333` | `0.526333` | `+0.381000` |

All paired comparisons are `3/0/0` in favor of D20.

True-MNIST compact smoke:
`outputs/step2_new_directions/d20_multiprototype_opmnist_openml_smoke/d20_multiprototype_opmnist_openml_smoke_results.json`

Protocol: OpenML MNIST, canonical split, max 2000 train examples, max 500 test
examples, 5 pixel permutation tasks, 1000 online steps, 1 paired seed.

| Metric | D20 | Best MLP | D20 margin |
| --- | ---: | ---: | ---: |
| Final-window MSE | `0.055277` | `0.092701` | `+0.037424` |
| Final-window accuracy | `0.595000` | `0.400000` | `+0.195000` |
| Held-out test MSE | `0.043641` | `0.097477` | `+0.053836` |
| Held-out test accuracy | `0.684000` | `0.346400` | `+0.337600` |

## Core JAX Production Scaffold

Worker D added `alberta_framework.core.fast_slow` and exported it from
`alberta_framework.core`.

The scaffold is intentionally smaller than D18:

- One learned tanh encoder.
- One slow readout.
- One fast decayed readout.
- One learned sigmoid gate.
- `jax.lax.scan` array runner.

It does not yet claim D18 or D20 parity. It is the production track for moving
the fast/slow hypothesis out of Python research runners.

## Remaining Caveats

No empirical gap remains for the current all14 D18 Step 2 benchmark suite.

OPMNIST is now solved on:

- A 3-seed local no-network OPMNIST analogue.
- A 1-seed compact true-MNIST OpenML sanity check.

What is still not done:

- D20 has not been folded into D18 canonical and re-run on all14.
- D20 has not been run on the full Dohare 800-task, 48M-step OPMNIST protocol.
- Tanh64 has not been confirmed at 10 seeds.
- The fused JAX fast/slow learner is a production scaffold, not yet the
  benchmark-winning learner.
- There is still no theorem proving arbitrary recursive feature construction.

The practical next candidate is clear: integrate multi-prototype task-view
memory into the canonical fast/slow learner, then run all14 plus larger true
MNIST OPMNIST scales.
