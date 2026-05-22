# Step 2 Generator-Internal Meta-Resource Smoke

Date: 2026-05-05

This smoke pass checks whether resource management is now inside candidate
feature construction rather than only outside it as a portfolio over learners.

## Implementation

- `GeneratorMetaResourceManager` is a contextual Hedge manager over generator
  policies.
- Each policy controls a compositional generator bundle: operation
  (`product`, `tanh`, or `gated`), parent-selection mode (`uniform`,
  `mutation`, or `residual_imprint`), replacement-rate multiplier, promotion
  margin multiplier, candidate minimum-age multiplier, and residual-imprint
  scale.
- `CompositionalFeatureLearner(learn_generator_resources=True)` samples a
  generator policy before candidate construction, applies those knobs to
  replacement/promotion/refresh decisions, records active/candidate provenance,
  and updates policy preferences from observed feature utility by provenance.

## Commands

```bash
source .venv/bin/activate
pytest tests/test_resource_manager.py tests/test_compositional_features.py -q
python "examples/The Alberta Plan/Step2/feature_discovery_experiments.py" --quick --benchmark nonlinear --num-steps 120 --seeds 2 --output-dir outputs/step2_generator_meta_nonlinear_quick
python "examples/The Alberta Plan/Step2/feature_discovery_experiments.py" --quick --benchmark interaction --num-steps 120 --seeds 2 --output-dir outputs/step2_generator_meta_interaction_quick
```

## Smoke Results

Final-window MSE, 2 seeds:

| Benchmark | Method | Final-window MSE |
|---|---:|---:|
| nonlinear | shadow_candidates_future | 0.404280 |
| nonlinear | compositional_meta_generator | 0.411432 |
| nonlinear | generator_priors | 0.412972 |
| nonlinear | compositional_fixed | 0.414617 |
| nonlinear | linear_multihead | 0.423150 |
| nonlinear | mlp_obgd | 0.485750 |
| interaction | interaction_static | 0.563661 |
| interaction | interaction_generator_priors | 0.567497 |
| interaction | compositional_meta_generator | 0.589595 |
| interaction | compositional_fixed | 0.593261 |
| interaction | mlp_obgd | 0.625881 |
| interaction | linear_multihead | 0.682073 |

## Interpretation

The smoke result closes the architectural gap: the learned manager now directly
controls generator choice, replacement rate, promotion aggressiveness, and
candidate refresh age inside candidate construction.  It is competitive with
fixed compositional settings in the quick nonlinear and interaction runs, but
this 2-seed smoke pass is not a canonical performance claim.
