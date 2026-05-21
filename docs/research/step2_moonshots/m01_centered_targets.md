# M01 Centered Targets

Hypothesis: one-hot MSE on 10-class digits wastes most gradient mass on negative
classes. A centered target code with +1 on the true class and -1/(K-1) on every
other class may improve online conditioning without changing the learner
architecture.

## Smoke Protocol

Script: `examples/The Alberta Plan/Step2/moonshots/m01_centered_targets.py`

Command:

```bash
source .venv/bin/activate && python "examples/The Alberta Plan/Step2/moonshots/m01_centered_targets.py"
```

Configuration: 3 seeds, 900 online steps, last-200 final-window metrics, hidden
size 64, LMS step size 0.03, sparsity 0.5, ObGD kappa 2.0, layer norm enabled.

Regimes:

- `iid`: stationary shuffled-epoch sklearn digits stream.
- `label_drift`: class-head meanings are permuted every 300 steps; held-out
  accuracy uses the final-phase mapping.

Both treatments used the same `MultiHeadMLPLearner` architecture and identical
initial parameters per paired seed. Accuracy is the primary code-invariant
metric. MSE is measured against each method's own target code.

## Results

| Regime | Metric | One-hot | Centered | Paired diff |
|---|---|---:|---:|---:|
| iid | final-window accuracy | 0.9033 +/- 0.0073 | 0.9067 +/- 0.0060 | +0.0033 |
| iid | final-window MSE | 0.0322 +/- 0.0013 | 0.0393 +/- 0.0015 | -0.0071 |
| label_drift | final-window accuracy | 0.8600 +/- 0.0076 | 0.8617 +/- 0.0117 | +0.0017 |
| label_drift | final-window MSE | 0.0424 +/- 0.0007 | 0.0517 +/- 0.0008 | -0.0094 |
| overall | final-window accuracy | 0.8817 +/- 0.0108 | 0.8842 +/- 0.0116 | +0.0025 |
| overall | final-window MSE | 0.0373 +/- 0.0024 | 0.0455 +/- 0.0029 | -0.0082 |

Held-out final-phase accuracy also moved slightly toward centered targets:
0.9001 versus 0.8983 overall.

## Decision

Worth scaling, but only as a weak positive. The predeclared positive criterion
was that centered targets beat one-hot MLP on paired final-window accuracy or
MSE. Centered targets cleared that threshold on final-window accuracy
(+0.0025 overall; 3 centered wins, 2 one-hot wins, 1 tie across 6 paired runs),
but one-hot had lower code MSE in every paired run.

Next scaling pass should use more seeds, at least one stronger input-shift
regime such as `permuted_pixels`, and report accuracy as the primary endpoint.
