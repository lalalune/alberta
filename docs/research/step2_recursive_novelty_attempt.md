# Step 2 Recursive Novelty Attempt

Date: 2026-05-04

## Question

Signed tanh scaffolds reduced nonlinear final-window MSE from roughly 0.225 to
0.143 in prior probes, but still trailed the fair MLP control. This attempt
tests whether the failure is candidate credit allocation rather than basis
availability.

## Mechanism

Added an opt-in `candidate_scoring_mode="energy_novelty"` path to
`CompositionalFeatureLearner`.

- Active and candidate utilities use online matching-pursuit style residual
  alignment: `mean(|E[error * feature]|) / sqrt(E[feature^2])`.
- Candidate scores are gated by an EMA correlation novelty term against active
  features.
- Defaults remain unchanged with `candidate_scoring_mode="legacy"`.
- The probe variant `single_mechanism_energy_novelty` uses the same signed-tanh
  scaffold budget as `single_mechanism_signed_tanh`, isolating the credit
  allocation change from adding more basis functions.

## Command

```bash
source .venv/bin/activate && python "examples/The Alberta Plan/Step2/step2_recursive_feature_utility_probe.py" --suite --seeds 5 --num-steps 5000 --final-window 500 --methods single_mechanism,single_mechanism_signed_tanh,single_mechanism_energy_novelty,mlp_32x32_no_ln,mlp_64x64_no_ln --output-dir outputs/step2_canonical/recursive_feature_energy_novelty_suite_5seed_5000
```

## Results

Final-window MSE, mean +/- stderr over 5 seeds:

| Task | single | signed tanh | energy novelty | mlp32 | mlp64 |
|---|---:|---:|---:|---:|---:|
| nonlinear | 0.2209 +/- 0.0159 | 0.1124 +/- 0.0175 | 0.4029 +/- 0.0463 | 0.0301 +/- 0.0013 | 0.0618 +/- 0.0041 |
| interaction | 0.1102 +/- 0.0182 | 0.1055 +/- 0.0273 | 0.3693 +/- 0.0848 | 0.1987 +/- 0.0060 | 0.4795 +/- 0.0162 |
| triple | 0.0964 +/- 0.0192 | 0.0857 +/- 0.0124 | 0.0522 +/- 0.0078 | 0.3958 +/- 0.0371 | 0.5484 +/- 0.0161 |
| rare | 0.1282 +/- 0.0329 | 0.1200 +/- 0.0242 | 0.0591 +/- 0.0123 | 0.0661 +/- 0.0126 | 0.0751 +/- 0.0119 |
| polynomial | 0.3719 +/- 0.0534 | 0.4548 +/- 0.0957 | 0.6867 +/- 0.1401 | 0.4264 +/- 0.0518 | 0.7051 +/- 0.0925 |
| frequency | 0.0744 +/- 0.0122 | 0.0776 +/- 0.0224 | 0.1097 +/- 0.0728 | 0.0271 +/- 0.0009 | 0.0631 +/- 0.0029 |

## Decision

Rejected. `single_mechanism_energy_novelty` beat the best fair MLP on 2/6
tasks, but the promotion rule requires beating fair MLP on nonlinear and not
sacrificing interaction, triple, polynomial, or frequency. It failed nonlinear
by a wide margin and sacrificed interaction, polynomial, and frequency.

## Remaining Failure Mode

Energy-normalized residual alignment successfully favors recursive product
structure on triple and rare targets, but it is too local and myopic for
smooth nonlinear targets. The correlation novelty gate also appears to suppress
reusable partially redundant scaffolds that are needed for nonlinear,
polynomial, and frequency tasks. The open failure is therefore not raw
representability; it is stable online assignment of credit to reusable
intermediate features without discarding correlated-but-necessary structure.
