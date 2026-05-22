# Stratified Learned-Budget Survivability

Date: 2026-05-07

## Question

Does `stratified_budget_post_ffn_memory` survive a paired comparison against
the static replay-capped post-FFN memory candidate and the FFN baseline?

Short answer: no. The stratified learned-budget path is competitive with FFN
and sometimes faster than the static memory candidate, but it should not replace
the static replay-capped post-FFN candidate. At 10000 steps, the 30-seed
comparison gives a paired final-window NLL delta of `-0.000951` versus static
with a 95% CI of `[-0.001876, -0.000026]`, where positive favors stratified.
Held-out NLL also favors static on mean (`-0.005973`) although that CI still
crosses zero.

## Protocol

Runner:

```bash
examples/The Alberta Plan/Step2/step2_tiny_shakespeare_stratified_budget_memory_transformer.py
```

The runner trains three paired methods per seed:

- `baseline_ffn_transformer`
- `static_post_ffn_memory` with post-FFN replay and `gate_max=0.15`
- `stratified_budget_post_ffn_memory`

All runs used `eval_steps=512`, `final_window=512`, `placement=post_ffn`,
`replay_size=128`, `train_loss_mode=memory`, `gate_lr=0.5`, `gate_decay=0.995`,
and `gate_l2=0.1` as the controller initialization.

Output artifacts:

| Horizon | Seeds | Output directory | Elapsed |
| ---: | ---: | --- | ---: |
| 3000 | 10 | `outputs/step2_new_directions/stratified_budget_survivability_3000_10seed` | 102.5 s |
| 5000 | 10 | `outputs/step2_new_directions/stratified_budget_survivability_5000_10seed` | 184.9 s |
| 10000 | 10 | `outputs/step2_new_directions/stratified_budget_survivability_10000_10seed` | 504.5 s |
| 10000 | 30 | `outputs/step2_new_directions/stratified_budget_survivability_10000_30seed` | 768.6 s |

The 10-seed timings estimated a full three-horizon 30-seed repeat at about
`2375.7 s` (`39.6 min`). I ran the decisive 10000-step 30-seed comparison
instead of repeating 3000 and 5000 at 30 seeds, because the 10-seed results did
not clear the static replacement bar and 10000 steps is the target horizon where
the longer-run memory behavior matters.

## Commands

```bash
source .venv/bin/activate && PYTHONUNBUFFERED=1 python "examples/The Alberta Plan/Step2/step2_tiny_shakespeare_stratified_budget_memory_transformer.py" --steps 3000 --seeds 10 --eval-steps 512 --final-window 512 --output-dir outputs/step2_new_directions/stratified_budget_survivability_3000_10seed
source .venv/bin/activate && PYTHONUNBUFFERED=1 python "examples/The Alberta Plan/Step2/step2_tiny_shakespeare_stratified_budget_memory_transformer.py" --steps 5000 --seeds 10 --eval-steps 512 --final-window 512 --output-dir outputs/step2_new_directions/stratified_budget_survivability_5000_10seed
source .venv/bin/activate && PYTHONUNBUFFERED=1 python "examples/The Alberta Plan/Step2/step2_tiny_shakespeare_stratified_budget_memory_transformer.py" --steps 10000 --seeds 10 --eval-steps 512 --final-window 512 --output-dir outputs/step2_new_directions/stratified_budget_survivability_10000_10seed
source .venv/bin/activate && PYTHONUNBUFFERED=1 python "examples/The Alberta Plan/Step2/step2_tiny_shakespeare_stratified_budget_memory_transformer.py" --steps 10000 --seeds 30 --eval-steps 512 --final-window 512 --output-dir outputs/step2_new_directions/stratified_budget_survivability_10000_30seed
```

## Paired Results

Paired deltas are reported as stratified advantage. Positive means stratified is
better: comparator minus stratified for NLL/perplexity/time, stratified minus
comparator for accuracy.

### Ten-Seed Horizons

| Horizon | Metric | Stratified vs static, mean | 95% CI | Wins | Stratified vs FFN, mean | 95% CI | Wins |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 3000 | final-window NLL | -0.000041 | [-0.000211, +0.000129] | 5/10 | +0.000210 | [+0.000032, +0.000388] | 8/10 |
| 3000 | held-out NLL | +0.000634 | [-0.000140, +0.001408] | 8/10 | +0.000916 | [-0.000298, +0.002130] | 7/10 |
| 3000 | held-out perplexity | +0.016946 | [-0.005722, +0.039614] | 8/10 | +0.023668 | [-0.011183, +0.058518] | 7/10 |
| 5000 | final-window NLL | -0.000076 | [-0.000618, +0.000466] | 4/10 | +0.000187 | [-0.000211, +0.000586] | 6/10 |
| 5000 | held-out NLL | -0.000051 | [-0.003673, +0.003571] | 7/10 | +0.001545 | [-0.001028, +0.004117] | 7/10 |
| 5000 | held-out perplexity | +0.000467 | [-0.088799, +0.089732] | 7/10 | +0.035013 | [-0.025339, +0.095366] | 7/10 |
| 10000 | final-window NLL | -0.001129 | [-0.002764, +0.000507] | 4/10 | -0.000152 | [-0.001242, +0.000939] | 3/10 |
| 10000 | held-out NLL | -0.005051 | [-0.014628, +0.004526] | 4/10 | +0.012644 | [-0.009567, +0.034854] | 7/10 |
| 10000 | held-out perplexity | -0.107051 | [-0.303633, +0.089531] | 4/10 | +0.329790 | [-0.285985, +0.945564] | 7/10 |

Interpretation: the 3000-step and 5000-step results are near ties. The
10000-step result moves against stratified versus the static cap, which is the
survival criterion.

### 10000-Step Thirty-Seed Confirmation

| Metric | FFN mean +/- SE | Static mean +/- SE | Stratified mean +/- SE |
| --- | ---: | ---: | ---: |
| final-window NLL | 2.745882 +/- 0.019234 | 2.744856 +/- 0.019329 | 2.745807 +/- 0.019228 |
| held-out NLL | 3.057432 +/- 0.030764 | 3.044200 +/- 0.029542 | 3.050173 +/- 0.029032 |
| held-out perplexity | 21.576266 +/- 0.697793 | 21.269509 +/- 0.662199 | 21.386993 +/- 0.652702 |
| held-out accuracy | 0.221810 +/- 0.006909 | 0.222591 +/- 0.006546 | 0.221940 +/- 0.006603 |
| fast-only held-out NLL | 3.057432 +/- 0.030764 | 3.039211 +/- 0.034824 | 3.038984 +/- 0.030505 |
| train seconds | 3.072890 +/- 0.271356 | 9.509382 +/- 0.519662 | 7.630272 +/- 0.483664 |

| Metric | Stratified vs static, mean | 95% CI | Wins | Stratified vs FFN, mean | 95% CI | Wins |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| final-window NLL | -0.000951 | [-0.001876, -0.000026] | 13/30 | +0.000075 | [-0.000498, +0.000648] | 14/30 |
| held-out NLL | -0.005973 | [-0.012901, +0.000956] | 12/30 | +0.007258 | [-0.001211, +0.015728] | 20/30 |
| held-out perplexity | -0.117484 | [-0.268989, +0.034022] | 12/30 | +0.189273 | [-0.035345, +0.413891] | 20/30 |
| held-out accuracy | -0.000651 | [-0.002199, +0.000897] | 9/30 | +0.000130 | [-0.001365, +0.001625] | 10/30 |
| fast-only held-out NLL | +0.000227 | [-0.019295, +0.019749] | 12/30 | +0.018448 | [+0.005576, +0.031320] | 24/30 |
| train seconds | +1.879110 | [+0.986746, +2.771474] | 26/30 | -4.557382 | [-5.446057, -3.668708] | 1/30 |

The 30-seed run confirms the replacement decision. Stratified is clearly not a
static-cap upgrade on final-window NLL and does not recover a held-out advantage
large enough to compensate. It remains much slower than FFN, though it is
faster than the static memory candidate in this runner.

## Controller Diagnostics

| Horizon | Seeds | Budget | Gate | Learned gate L2 | Slow-LR multiplier | Replay advantage EMA |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 3000 | 10 | 0.125600 +/- 0.000603 | 0.079915 +/- 0.003011 | 0.155883 +/- 0.001154 | 0.844956 +/- 0.004482 | -0.010208 +/- 0.001458 |
| 5000 | 10 | 0.122102 +/- 0.000513 | 0.066944 +/- 0.002975 | 0.163219 +/- 0.001012 | 0.806591 +/- 0.004464 | -0.017867 +/- 0.002732 |
| 10000 | 10 | 0.115215 +/- 0.000524 | 0.062861 +/- 0.002110 | 0.173103 +/- 0.001221 | 0.741841 +/- 0.006066 | -0.038728 +/- 0.008800 |
| 10000 | 30 | 0.115244 +/- 0.000327 | 0.062721 +/- 0.001311 | 0.172670 +/- 0.000738 | 0.742802 +/- 0.003340 | -0.039758 +/- 0.004345 |

The controllers consistently move in the same direction:

- budget shrinks below the static `0.15` cap;
- gate exposure falls with horizon;
- learned `gate_l2` rises toward the upper half of its range;
- the slow-memory LR multiplier falls to about `0.74` by 10000 steps;
- replay advantage EMA is negative by the final window.

This is not evidence of a learned budget finding useful extra memory. It is a
conservative controller learning to suppress a weak memory path.

## Decision

Do not replace the static replay-capped post-FFN candidate with the stratified
learned-budget candidate.

The stratified candidate survives only as an ablation that shows the controller
can reduce harmful exposure and improve fast-only FFN deployment metrics. It
does not survive as the current Step 2 memory candidate because:

- the 10000-step 30-seed final-window NLL comparison significantly favors the
  static cap;
- held-out NLL and perplexity have negative mean deltas versus static;
- the controller reaches its gains by closing memory, not by learning a better
  budget allocation;
- the FFN comparison is not strong enough to matter when the static cap is the
  incumbent memory-enabled candidate.

## Static Or Harmful Knobs

Still static:

- placement remains `post_ffn`;
- budget bounds, budget LR, EMA decay, target utilization, cost, pressure floor,
  and advantage floor are fixed;
- replay size and stratum cycle are fixed;
- gate init, gate LR, gate decay, and advantage margin are fixed;
- base slow LR and slow-LR controller cost/range are fixed;
- prototype count, bandwidth, update rate, novelty threshold, and reset mode are
  fixed;
- model shape, fast LR, gradient clip, and train-loss mode are fixed.

Harmful or suspicious in this result:

- learned `gate_l2` increases while replay advantage is negative, making the
  budget path mostly a gate-closing regularizer;
- slow-LR control decays the memory update rate, which may protect from damage
  but also prevents the memory path from earning utility;
- the hard-negative/positive/uncertainty/recent cycle is deterministic and
  appears dominated by negative replay evidence at long horizon;
- the budget controller opens only when utility survives the pressure/cost
  logic, but measured utility is too weak and noisy to beat a simple static cap;
- lower memory exposure helps fast-only eval, but that is not the target
  deployment mode for replacing the static memory candidate.

Recommended next action: keep the static `gate_max=0.15` post-FFN replay-capped
candidate as the incumbent. Any future learned-budget attempt should first prove
that it can learn useful memory exposure on top of the static cap, not merely
learn to close the memory path.
