# Step 2 Output-Head Anti-Drift

Date: 2026-05-06.

## Question

Can a single UPGD learner improve the class-blocked digits tracking/retention
tradeoff with a minimal output-head anti-drift mechanism, without portfolios,
replay, or MLP fallback?

Short answer: not enough to promote. Simplex-gated output-bias anti-drift
improved the current UPGD compromise in a small three-seed probe, especially
held-out accuracy and final-window MSE, but it still did not beat `mlp64_64` on
tracking and did not clear `mlp64` on held-out MSE.

## Mechanism

Added two disabled-by-default UPGDLearner knobs:

- `readout_simplex_bias_decay`: per-update decay for active output biases.
- `readout_simplex_bias_centering_rate`: per-update removal of the active-bias
  mean.

The hook runs after the normal head update and optional margin update. It is
gated to non-negative unit-mass targets with more than one active head, so it
targets one-hot/simplex classification streams and does not alter dense
regression targets. NaN-masked inactive heads are left unchanged.

For active biases `b`, decay first applies `b <- (1 - decay) b`, then centering
applies `b <- b - centering_rate * mean(b)`. Defaults are zero, preserving
existing behavior exactly.

This is temporally uniform and single-learner: it does not detect class blocks,
store replay, select between learners, or add a fallback classifier.

## Tests

Added focused coverage in `tests/test_upgd.py`:

- validation rejects out-of-range decay/centering rates;
- config roundtrip preserves both new fields;
- simplex bias decay shrinks active biases without touching head weights;
- simplex centering changes active biases but leaves NaN-inactive biases
  unchanged;
- non-simplex targets skip the anti-drift hook.

Commands run:

```bash
.venv/bin/python -m pytest tests/test_upgd.py -q
.venv/bin/python -m ruff check src/alberta_framework/core/upgd.py tests/test_upgd.py output/subagents/output_head_antidrift/run_classblocked_probe.py
```

Result: `49 passed`; ruff passed.

## Probe

Artifacts:

- `output/subagents/output_head_antidrift/probe_3seed/results.json`
- `output/subagents/output_head_antidrift/probe_3seed/records.csv`
- `output/subagents/output_head_antidrift/run_classblocked_probe.py`

Command:

```bash
.venv/bin/python output/subagents/output_head_antidrift/run_classblocked_probe.py \
  --steps 1200 \
  --n-seeds 3 \
  --seed 1020 \
  --final-window 300 \
  --output-dir output/subagents/output_head_antidrift/probe_3seed
```

The probe used class-blocked sklearn digits with the same 1200-step, seed-1020,
final-window-300 shape as the previous follow-up. It compared `mlp64`,
`mlp64_64`, the current UPGD compromise, and three anti-drift variants.

| Method | Final MSE | Final acc | Test MSE | Test acc |
|---|---:|---:|---:|---:|
| `mlp64_64` | 0.002730 +/- 0.000089 | 0.992222 +/- 0.001111 | 0.159643 +/- 0.010299 | 0.099567 +/- 0.001237 |
| `mlp64` | 0.004472 +/- 0.000060 | 0.986667 +/- 0.001924 | 0.126024 +/- 0.005605 | 0.102041 +/- 0.001855 |
| `base + slowutil999 + bias_decay001 + center1` | 0.004534 +/- 0.000253 | 0.988889 +/- 0.002222 | 0.127522 +/- 0.006194 | 0.103896 +/- 0.003862 |
| `base + bias_center1` | 0.004534 +/- 0.000252 | 0.985556 +/- 0.001111 | 0.128105 +/- 0.008155 | 0.106370 +/- 0.006184 |
| `base + bias_decay001` | 0.004545 +/- 0.000250 | 0.986667 +/- 0.001924 | 0.126144 +/- 0.006292 | 0.102041 +/- 0.002142 |
| `base` | 0.004731 +/- 0.000217 | 0.987778 +/- 0.002940 | 0.129793 +/- 0.008097 | 0.100804 +/- 0.002474 |

Paired deltas are positive when the candidate beats the named MLP baseline.

| Variant | d final MSE vs `mlp64` | d final MSE vs `mlp64_64` | d test MSE vs `mlp64` | d test acc vs `mlp64` |
|---|---:|---:|---:|---:|
| `base` | -0.000259 (1/3) | -0.002001 (0/3) | -0.003769 (1/3) | -0.001237 (1/3) |
| `base + bias_center1` | -0.000062 (1/3) | -0.001804 (0/3) | -0.002081 (1/3) | +0.004329 (2/3) |
| `base + bias_decay001` | -0.000073 (1/3) | -0.001815 (0/3) | -0.000120 (2/3) | +0.000000 (1/3) |
| `base + slowutil999 + bias_decay001 + center1` | -0.000062 (1/3) | -0.001804 (0/3) | -0.001498 (2/3) | +0.001855 (1/3) |

## Read

Output-bias anti-drift moved the right direction relative to the current UPGD
compromise:

- Final-window MSE improved from `0.004731` to about `0.004534`.
- Held-out test MSE improved from `0.129793` to as low as `0.126144`.
- Held-out test accuracy improved from `0.100804` to as high as `0.106370`.

But the mechanism did not close the tradeoff:

- `mlp64_64` remained far ahead on final-window MSE and final-window accuracy.
- No anti-drift variant beat `mlp64` on mean final-window MSE.
- The best held-out MSE variant essentially tied but did not beat `mlp64`.

## Recommendation

Keep the new mechanism disabled by default. It is clean and testable enough to
keep as an experimental UPGD option, but the three-seed evidence is not strong
enough to promote it into `step2_default` or the current class-blocked
candidate. The best next bounded follow-up would tune only two values,
`readout_simplex_bias_decay` in `{0.0003, 0.001, 0.003}` and centering in
`{0, 1}`, against the same baselines before considering any wider run.
