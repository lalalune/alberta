# Step 2 compositional + budget + future-utility evaluation

Seeds: 10. Steps per run: 5000. Final-window: last 1500 steps.

Tuned compositional learner combines the ``single_mechanism`` recipe from ``step2_recursive_feature_utility_probe`` (contribution-trace future utility, residual imprint, product-biased priors, depth retention, signed-tanh scaffold) with ``learn_generator_resources=True`` so the GeneratorMetaResourceManager allocates a bounded budget across operation/parent-mode policies.

## Stream: `out_of_class_polynomial`

Final-window MSE (mean +/- stderr):

| Method | Final-window MSE | Total mean MSE |
|---|---|---|
| `upgd` | 0.5031 +/- 0.0476 | 0.5153 +/- 0.0406 |
| `compositional_tuned` | 1.0233 +/- 0.0841 | 1.0177 +/- 0.0783 |
| `mlp_64` | 1.0472 +/- 0.0942 | 1.0885 +/- 0.0826 |

- `paired_mlp_minus_compositional`: mean diff +0.0238 +/- 0.0173, Cohen d=+0.435, 6/10 left-wins.
- `paired_upgd_minus_compositional`: mean diff -0.5203 +/- 0.0379, Cohen d=-4.336, 0/10 left-wins.
- `paired_mlp_minus_upgd`: mean diff +0.5441 +/- 0.0467, Cohen d=+3.685, 10/10 left-wins.

## Stream: `compositional`

Final-window MSE (mean +/- stderr):

| Method | Final-window MSE | Total mean MSE |
|---|---|---|
| `upgd` | 0.1828 +/- 0.0258 | 0.1871 +/- 0.0226 |
| `mlp_64` | 0.2087 +/- 0.0215 | 0.2154 +/- 0.0189 |
| `compositional_tuned` | 1.1897 +/- 0.1576 | 1.1609 +/- 0.1285 |

- `paired_mlp_minus_compositional`: mean diff -0.9811 +/- 0.1381, Cohen d=-2.247, 0/10 left-wins.
- `paired_upgd_minus_compositional`: mean diff -1.0069 +/- 0.1336, Cohen d=-2.383, 0/10 left-wins.
- `paired_mlp_minus_upgd`: mean diff +0.0259 +/- 0.0062, Cohen d=+1.318, 9/10 left-wins.

## Stream: `triple_product`

Final-window MSE (mean +/- stderr):

| Method | Final-window MSE | Total mean MSE |
|---|---|---|
| `compositional_tuned` | 0.1321 +/- 0.0360 | 0.1161 +/- 0.0246 |
| `upgd` | 0.2494 +/- 0.0139 | 0.2727 +/- 0.0059 |
| `mlp_64` | 0.3491 +/- 0.0251 | 0.4802 +/- 0.0128 |

- `paired_mlp_minus_compositional`: mean diff +0.2171 +/- 0.0477, Cohen d=+1.438, 9/10 left-wins.
- `paired_upgd_minus_compositional`: mean diff +0.1174 +/- 0.0382, Cohen d=+0.973, 8/10 left-wins.
- `paired_mlp_minus_upgd`: mean diff +0.0997 +/- 0.0129, Cohen d=+2.450, 10/10 left-wins.

## Verdict

**Positive on `triple_product`.** With ``learn_generator_resources=True`` and the ``single_mechanism`` future-utility recipe, ``CompositionalFeatureLearner`` beats the fair ``MultiHeadMLP(64)`` baseline at Cohen d=+1.44 (9/10 paired wins) and also beats ``UPGD`` at Cohen d=+0.97 (8/10 paired wins) on the recursive triple-product target ``y = x0 * x1 * x2``. This is the first stream on which the compositional path reaches the d > 1.0 threshold against a fair MLP -- the canonical out-of-class audit (``out_of_class_SUMMARY.md``) had compositional only at d=+0.25 there. Wall time: 1.8 minutes for 90 runs.

**Negative on `out_of_class_polynomial` and `compositional`.** Compositional only ties MLP on the polynomial stream (d=+0.44, 6/10 wins) and loses badly on the 2-layer-tanh ``CompositionalStream`` (d=-2.25, 0/10 wins). UPGD remains the strongest method on both streams. Plausible explanation: triple_product has a single fixed cubic target where contribution-trace utility plus product-biased priors find ``(x0*x1)*x2`` quickly, but the polynomial and 2-layer streams add non-stationary contexts every 500 steps and target-shapes that the depth-3 ``OP_PRODUCT/OP_TANH/OP_GATED`` DAG does not match efficiently. The 1.5k final-window also exposes context-shift instability for the smaller compositional bank (n_features=36) compared to MLP(64)'s smooth approximator.

**Promotion recommendation: keep as research with a positive triple-product entry.** UPGD remains the canonical Step 2 promotion (``out_of_class_SUMMARY.md`` is unchanged). This run is additive: it shows the literal-feature-construction path can deliver d > 1.0 wins when the target structure aligns with the operation set, even though it does not yet generalise to the full out-of-class suite.