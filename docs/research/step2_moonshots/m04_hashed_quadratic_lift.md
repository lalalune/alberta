# M04 Hashed Quadratic Lift

## Setup

Smoke benchmark: `InteractionFeatureDiscoveryStream` with 3 paired seeds, 1200 online steps, and a 240-step final window.
Methods: raw `MultiHeadMLPLearner(hidden_sizes=(64,))` versus linear multi-head readouts on `phi(x) = [x, signed_hash(x_i*x_j)]`.

## Final-Window Loss

| Method | Params | Final-window loss | Mean loss |
|---|---:|---:|---:|
| `raw_mlp_h64` | 834 | 0.638101 +/- 0.311555 | 0.817519 +/- 0.247023 |
| `hashed_quadratic_h128` | 278 | 0.516897 +/- 0.236447 | 0.668903 +/- 0.210655 |
| `hashed_quadratic_h512` | 1046 | 0.453553 +/- 0.216843 | 0.619803 +/- 0.190944 |

## Paired Test

| Method vs raw MLP | Paired diff (MLP - method) | Wins | Losses |
|---|---:|---:|---:|
| `hashed_quadratic_h128` | 0.121204 +/- 0.075255 | 3 | 0 |
| `hashed_quadratic_h512` | 0.184548 +/- 0.097265 | 3 | 0 |

## Conclusion

Best hashed method: `hashed_quadratic_h512`. Positive smoke: scale explicit quadratic features. The paired final-window difference was 0.184548 (positive means the hashed learner beat the MLP), with 3/3 paired wins.

Interpretation caveat: this smoke benchmark is deliberately aligned with the generator oracle, because the stream target is built from pair products. A positive result would justify a larger negative-control suite; it would not by itself show general feature discovery.

JSON: `outputs/step2_moonshots/m04_hashed_quadratic_lift/results.json`
