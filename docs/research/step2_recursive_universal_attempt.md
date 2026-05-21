# Step 2 Recursive Universal Attempt

Date: May 5, 2026.

## Question

The current recursive/self-contained feature mechanism wins the algebraic and
interaction probes but misses the nonlinear probe. This attempt tests two small
online additions:

- `single_mechanism_signed_tanh`: adds six deterministic signed raw-pair
  `tanh(sign_a * x_i + sign_b * x_j)` scaffolds after the product scaffolds.
- `single_mechanism_tanh_shadow`: adds the signed scaffolds and also trains
  candidate `OP_TANH` parameters online through their shadow output heads before
  promotion.

Both additions are self-contained: no task labels, no offline feature search,
and no full-stream selection. The comparison keeps the current
`single_mechanism` baseline and fair MLP controls.

## Commands

```bash
source .venv/bin/activate
pytest tests/test_compositional_features.py tests/test_future_utility.py -q
python "examples/The Alberta Plan/Step2/step2_recursive_feature_utility_probe.py" --suite --seeds 3 --num-steps 2500 --final-window 500 --methods single_mechanism,single_mechanism_signed_tanh,single_mechanism_tanh_shadow,mlp_32x32_no_ln,mlp_64x64_no_ln --output-dir /tmp/alberta_recursive_universal_attempt_3seed_2500
```

## Results

Lower final-window MSE is better. Delta is best fair MLP final-window MSE minus
the recursive method; positive favors the recursive method.

| Probe | Best fair MLP | Current recursive | Signed tanh | Tanh shadow | Best recursive delta |
|---|---:|---:|---:|---:|---:|
| nonlinear | 0.0584 | 0.2251 | 0.1430 | 0.1438 | -0.0845 |
| interaction | 0.4887 | 0.1321 | 0.1275 | 0.1211 | 0.3676 |
| triple product | 0.8138 | 0.0920 | 0.0958 | 0.1574 | 0.7217 |
| rare-head | 0.1063 | 0.1241 | 0.1308 | 0.0948 | 0.0115 |
| polynomial | 0.8329 | 0.3260 | 0.6166 | 0.4781 | 0.5068 |
| frequency | 0.0770 | 0.0831 | 0.0665 | 0.0845 | 0.0105 |

Heldout MSE:

| Probe | Best fair MLP | Current recursive | Signed tanh | Tanh shadow |
|---|---:|---:|---:|---:|
| nonlinear | 0.0769 | 0.3126 | 0.0713 | 0.0889 |
| interaction | 0.4053 | 0.1411 | 0.1367 | 0.1164 |
| triple product | 0.7676 | 0.0622 | 0.1282 | 0.1662 |
| rare-head | 0.5097 | 0.0809 | 0.1265 | 0.0355 |
| polynomial | 0.8215 | 0.2842 | 0.5587 | 0.6368 |
| frequency | 0.0580 | 0.2050 | 0.0592 | 0.0701 |

Suite count by final-window MSE:

- `single_mechanism`: beats best fair MLP on 3/6 tasks and ties within 0.02 on
  2/6.
- `single_mechanism_signed_tanh`: beats best fair MLP on 4/6 tasks and ties
  within 0.02 on 0/6.
- `single_mechanism_tanh_shadow`: beats best fair MLP on 4/6 tasks and ties
  within 0.02 on 1/6.

## Interpretation

The signed tanh scaffold is the cleaner partial mechanism. It reduces nonlinear
final-window MSE from 0.2251 to 0.1430 and matches the nonlinear heldout MLP
control, while preserving interaction, triple-product, polynomial, and
frequency wins in this small suite. It still loses nonlinear online MSE
decisively: best fair MLP 0.0584 versus 0.1430, with 0/3 paired wins.

Candidate tanh-shadow training is mathematically defensible, but it is not the
right promotion candidate. It helps rare-head final-window and heldout loss, but
it worsens triple-product and polynomial relative to the current recursive
baseline.

## Decision

Status: not closed.

Reject both additions as the Step 2 universal single mechanism. Keep them
available only as opt-in experimental knobs:

- `signed_tanh_scaffold_count` is useful evidence that a small local nonlinear
  basis addresses part of the nonlinear miss without task labels.
- `train_candidate_theta` should not be promoted because candidate-side
  nonlinear adaptation competes with algebraic product discovery.

The blocker remains online nonlinear credit/allocation, not representability:
the signed tanh scaffolds can fit the heldout nonlinear target, but the online
replacement/scoring dynamics do not yet make them competitive with the fair MLP
on final-window learning loss.
