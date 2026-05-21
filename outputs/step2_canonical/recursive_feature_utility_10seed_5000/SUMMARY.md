# Step 2 Recursive Feature Utility Probe

Seeds: 10; steps: 5000; final-window: 500.

| Method | Final MSE | Active depth>=2 count | Seeds with depth>=2 |
|---|---:|---:|---:|
| `marginal_h8` | 1.2648 +/- 0.0894 | 13.20 | 10/10 |
| `mlp_32x32_ln` | 0.4771 +/- 0.0270 | 0.00 | 0/10 |
| `mlp_32x32_no_ln` | 0.3032 +/- 0.0226 | 0.00 | 0/10 |
| `one_step` | 1.1951 +/- 0.2833 | 11.30 | 10/10 |
| `recursive_marginal_h8` | 0.6197 +/- 0.1917 | 13.10 | 10/10 |
| `recursive_one_step` | 0.4950 +/- 0.2111 | 14.00 | 10/10 |

| Paired comparison | Delta | Right wins |
|---|---:|---:|
| `one_step - marginal_h8` | -0.0697 | 2/10 |
| `one_step - recursive_one_step` | 0.7001 | 8/10 |
| `one_step - recursive_marginal_h8` | 0.5754 | 6/10 |
| `recursive_one_step - recursive_marginal_h8` | -0.1247 | 4/10 |
| `mlp_32x32_no_ln - recursive_one_step` | -0.1918 | 5/10 |
| `mlp_32x32_no_ln - recursive_marginal_h8` | -0.3165 | 5/10 |
| `mlp_32x32_ln - recursive_one_step` | -0.0179 | 6/10 |
| `mlp_32x32_ln - recursive_marginal_h8` | -0.1426 | 5/10 |

Interpretation: this probe checks whether a utility-tested compositional learner can turn depth-1 product scaffolds into recursive depth-2+ features on a triple-product target. `recursive_product` is an opt-in product-biased generator, not a broad Step 2 closure claim. Trace utility should be judged against both one-step residual-imprint utility and the same recursive generator with one-step utility.