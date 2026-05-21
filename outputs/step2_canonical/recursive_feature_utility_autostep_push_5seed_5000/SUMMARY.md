# Step 2 Recursive Feature Utility Probe

Seeds: 5; steps: 5000; final-window: 1000.

| Method | Final MSE | Active depth>=2 count | Seeds with depth>=2 |
|---|---:|---:|---:|
| `mlp_32x32_ln` | 0.5333 +/- 0.0432 | 0.00 | 0/5 |
| `mlp_32x32_no_ln` | 0.3550 +/- 0.0278 | 0.00 | 0/5 |
| `one_step` | 1.3824 +/- 0.2775 | 11.40 | 5/5 |
| `recursive_marginal_h8` | 0.2104 +/- 0.1037 | 8.60 | 5/5 |
| `recursive_one_step` | 0.1495 +/- 0.0677 | 8.80 | 5/5 |

| Paired comparison | Delta | Right wins |
|---|---:|---:|
| `one_step - recursive_one_step` | 1.2329 | 5/5 |
| `one_step - recursive_marginal_h8` | 1.1720 | 5/5 |
| `recursive_one_step - recursive_marginal_h8` | -0.0610 | 2/5 |
| `mlp_32x32_no_ln - recursive_one_step` | 0.2055 | 4/5 |
| `mlp_32x32_no_ln - recursive_marginal_h8` | 0.1445 | 3/5 |
| `mlp_32x32_ln - recursive_one_step` | 0.3838 | 5/5 |
| `mlp_32x32_ln - recursive_marginal_h8` | 0.3229 | 4/5 |

Interpretation: this probe checks whether a utility-tested compositional learner can turn depth-1 product scaffolds into recursive depth-2+ features on a triple-product target. `recursive_product` is an opt-in product-biased generator, not a broad Step 2 closure claim. Trace utility should be judged against both one-step residual-imprint utility and the same recursive generator with one-step utility.