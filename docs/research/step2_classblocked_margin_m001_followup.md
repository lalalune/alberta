# Step 2 Class-Blocked Margin M001 Follow-up

Date: 2026-05-06.

## Question

Can we get closer to a universal Step 2 learner on the class-blocked
tracking/retention conflict without a portfolio, replay, task id, or MLP
fallback?

Short answer: closer, but not closed.  A very small readout-margin pressure
combined with the existing repetition/meta branch is the first 30-seed
class-blocked candidate that improves all four mean metrics versus `mlp64`.
It still fails badly against `mlp64_64` on current-block tracking, so it is a
strong lead, not a universal solution.

Candidate:

`upgd_density_sigma1e_4_adaptk035_065_lr06_repx025_meta001_margin_m001_notrunk_tight`

Mechanism:

- target-density one-hot pressure;
- adaptive `kappa` in `[0.35, 0.65]`;
- head-only gradient-alignment meta-plasticity;
- mild repeated-target head pressure, `head_repetition_multiplier=0.25`;
- tiny output-margin update, `readout_margin=0.2`,
  `readout_margin_step_size=0.001`;
- no replay, no router, no task id, no MLP fallback.

## Negative Screens Before This

Stacking existing output-head knobs did not solve the problem:

- margin plus simplex bias centering improved some metrics but did not clear
  final-window tracking;
- slow utility helped held-out MSE but not held-out accuracy;
- linear-MSE classification improved current-block tracking but destroyed the
  retained-class advantage;
- fast/slow residual linear heads worsened the tradeoff in the standalone
  `output/subagents/classblocked_fast_residual` probe.

This narrows the useful mechanism to very mild output margin pressure rather
than heavier output centering or a separate fast residual.

## Thirty-Seed Class-Blocked Confirmation

Artifact:

- `output/subagents/classblocked_margin_grid/margin_m001_30seed_1200/digits_ablation_results.json`
- `output/subagents/classblocked_margin_grid/margin_m001_30seed_1200/digits_ablation_SUMMARY.md`

Command:

```bash
.venv/bin/python "examples/The Alberta Plan/Step2/step2_upgd_ablation.py" \
  --suite digits \
  --upgd-configs upgd_density_sigma1e_4_adaptk035_065_lr06_repx025_meta001_margin_m001_notrunk_tight \
  --digits-regimes class_blocked \
  --steps 1200 \
  --n-seeds 30 \
  --seed 1020 \
  --final-window 300 \
  --output-dir output/subagents/classblocked_margin_grid/margin_m001_30seed_1200
```

Aggregate result:

| Method | Final MSE | Test acc |
|---|---:|---:|
| `mlp64_64` | `0.0029 +/- 0.0000` | `0.1000 +/- 0.0003` |
| `margin_m001` | `0.0049 +/- 0.0001` | `0.1244 +/- 0.0055` |
| `mlp64` | `0.0050 +/- 0.0001` | `0.1187 +/- 0.0052` |

Paired against `mlp64`:

| Metric | Diff favoring UPGD | Wins |
|---|---:|---:|
| final-window MSE | `+0.000138 +/- 0.000045` | `21/30` |
| final-window accuracy | `+0.000444 +/- 0.000499` | `12/30` |
| held-out test MSE | `+0.002527 +/- 0.001290` | `18/30` |
| held-out test accuracy | `+0.005690 +/- 0.004156` | `14/30` |

Paired against `mlp64_64`:

| Metric | Diff favoring UPGD | Wins |
|---|---:|---:|
| final-window MSE | `-0.001959 +/- 0.000077` | `0/30` |
| final-window accuracy | `-0.005000 +/- 0.000634` | `0/30` |
| held-out test MSE | `+0.031757 +/- 0.001716` | `30/30` |
| held-out test accuracy | `+0.024366 +/- 0.005492` | `20/30` |

Read:

- This is the strongest `mlp64`-relative class-blocked candidate so far because
  all four mean metrics move in the right direction.
- The effect is small and not yet robust on accuracy wins.
- `mlp64_64` remains the pure current-block tracking ceiling, but it has poor
  retained accuracy.  This exposes a Pareto conflict rather than a clean
  dominance result.

## Five-Regime Guardrail

Artifact:

- `output/subagents/classblocked_margin_grid/margin_m001_digits_guardrail_5seed/digits_ablation_results.json`
- `output/subagents/classblocked_margin_grid/margin_m001_digits_guardrail_5seed/digits_ablation_SUMMARY.md`

Command:

```bash
.venv/bin/python "examples/The Alberta Plan/Step2/step2_upgd_ablation.py" \
  --suite digits \
  --upgd-configs upgd_density_sigma1e_4_adaptk035_065_lr06_repx025_meta001_margin_m001_notrunk_tight \
  --digits-regimes iid,permuted,class_blocked,label_drift,mask_noise \
  --steps 1200 \
  --n-seeds 5 \
  --seed 2020 \
  --final-window 300 \
  --phase-length 500 \
  --mask-keep-fraction 0.5 \
  --mask-noise-std 0.05 \
  --output-dir output/subagents/classblocked_margin_grid/margin_m001_digits_guardrail_5seed
```

Guardrail read:

| Regime | Final-window MSE vs best MLP | Held-out accuracy vs best MLP |
|---|---:|---:|
| iid | `+0.0062`, `5/5` wins | `+0.0204`, `5/5` wins |
| permuted | `+0.0089`, `5/5` wins | `+0.0612`, `3/5` wins |
| label drift | `+0.0065`, `5/5` wins | `-0.0015`, `2/5` wins |
| mask noise | `+0.0073`, `5/5` wins | `+0.0271`, `4/5` wins |
| class blocked | loses `mlp64_64` tracking | `-0.0019` vs `mlp64` on this 5-seed split |

The candidate does not damage the ordinary digit matrix and remains strong on
MSE.  The class-blocked held-out accuracy result is seed-sensitive: positive
over the 30-seed `1020..1049` confirmation, slightly negative in the 5-seed
`2020..2024` guardrail.

## Current Scientific Position

The closest current Step 2 answer is not "UPGD beats every MLP."  It is:

> Mild output-margin pressure gives a single UPGD learner a better
> tracking/retention compromise than `mlp64` on the class-blocked digits
> stressor while preserving the larger digit-regime wins.  A deeper MLP still
> dominates immediate current-block tracking by sacrificing retained accuracy.

That is close to a useful universal learner, but not a solved universal
replacement.  The remaining gap is to match `mlp64_64` tracking without giving
up the retained-class advantage.

## Next Direction

The next scientifically sensible direction is not another wrapper.  The failed
fast-residual and centering screens suggest the mechanism needs to be integrated
into the output head as a controlled plasticity/retention decomposition:

- one slow calibrated head for retained classes;
- one bounded current-block plasticity component;
- a single prediction path and single update rule;
- no replay, routing, or task-id input;
- explicit resource accounting and head utility diagnostics.

The `margin_m001` candidate is the right baseline for that next mechanism.
