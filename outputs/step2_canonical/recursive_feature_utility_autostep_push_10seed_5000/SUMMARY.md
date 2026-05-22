# Step 2 Recursive Feature Utility Probe

Seeds: 10; steps: 5000; final-window: 1000.

| Method | Final MSE | Active depth>=2 count | Seeds with depth>=2 |
|---|---:|---:|---:|
| `mlp_32x32_ln` | 0.5404 +/- 0.0332 | 0.00 | 0/10 |
| `mlp_32x32_no_ln` | 0.3649 +/- 0.0274 | 0.00 | 0/10 |
| `one_step` | 1.2159 +/- 0.1626 | 11.30 | 10/10 |
| `recursive_marginal_h8` | 0.4340 +/- 0.1450 | 8.70 | 10/10 |
| `recursive_one_step` | 0.2783 +/- 0.0612 | 9.20 | 10/10 |

| Paired comparison | Delta | Right wins |
|---|---:|---:|
| `one_step - recursive_one_step` | 0.9375 | 9/10 |
| `one_step - recursive_marginal_h8` | 0.7819 | 9/10 |
| `recursive_one_step - recursive_marginal_h8` | -0.1556 | 4/10 |
| `mlp_32x32_no_ln - recursive_one_step` | 0.0866 | 5/10 |
| `mlp_32x32_no_ln - recursive_marginal_h8` | -0.0690 | 5/10 |
| `mlp_32x32_ln - recursive_one_step` | 0.2620 | 9/10 |
| `mlp_32x32_ln - recursive_marginal_h8` | 0.1064 | 6/10 |

Interpretation: this probe checks whether a utility-tested compositional learner can turn depth-1 product scaffolds into recursive depth-2+ features on a triple-product target. `recursive_product` is an opt-in product-biased generator, not a broad Step 2 closure claim. Trace utility should be judged against both one-step residual-imprint utility and the same recursive generator with one-step utility.