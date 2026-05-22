# Step 2 All-Fronts Portfolio Attempt

This note is generated from existing artifacts only. It does not rerun large experiments, does not import Step 3 harnesses into Step 2, and does not treat missing scale evidence as closure.

Decision: **PARTIAL**.

| Front | Status | Claim | Evidence |
|---|---|---|---|
| `strict_supervised` | `closed` | strict supervised matrix | Checked 13 portfolio-vs-best-MLP comparisons; failures=0. |
| `recursive_controlled` | `closed` | controlled recursive feature-construction suite | `recursive_mlp_router` beats best fair MLP on 6/6 tasks; ties=0. |
| `opmnist` | `partial` | published-scale Online Permuted MNIST | Core protocol=True; completed full 60k blocks=40/800; primary nonnegative=True. |
| `scr` | `closed` | published-scale Slowly-Changing Regression | Best router `slow_meta`; published-scale SCR closed=True; public protocol=True. |
| `td_gvf_bridge` | `partial` | TD/GVF feature-discovery bridge | Best discovery `step2_interaction_features_linear_gvf` beats linear=True, beats MLP=True; kept partial because this is Step 3 bridge evidence, not a Step 2 portfolio route. |

## Route Audit

- Strict supervised matrix: already represented inside `step2_universal_portfolio.py` by live MLP/UPGD/dynamic-sparse experts, discounted Hedge, MLP-only guard routes, online class-imbalance MSE guard, and held-out retention deployment guard.
- Controlled recursive suite: evidence exists for `recursive_mlp_router`, but this is a separate runner family. Adding it directly to the strict supervised portfolio would require routing over different state shapes and task suites, so the honest integration is artifact-level or a higher-level conclusive runner.
- External OPMNIST/SCR: SCR has a narrowed million-step router closure. OPMNIST is positive on completed true-MNIST blocks but remains incomplete until the full 800 x 60,000 task-count gate is met.
- TD/GVF bridge: positive bridge evidence exists, but it belongs at the Step 2/3 boundary. Importing the Step 3 harness into this Step 2 portfolio runner would create coupling without making TD/GVF a live Step 2 deployment route.

## Interpretation

A temporally uniform portfolio is acceptable as the current portfolio-level Step 2 answer only in a partial sense: it closes the strict supervised matrix, the controlled recursive suite is closed by a separate causal router, and SCR has published-scale evidence. It is weaker than a single feature-construction algorithm because coverage comes from guarded allocation among known mechanisms, not from one mechanism that discovers the right representation across all fronts.

The all-fronts claim should therefore remain **partial**, not promoted, until full published-scale OPMNIST completes and TD/GVF feature finding is either integrated through a clean boundary or reported as a separate Step 3 bridge rather than Step 2 closure.

## Artifact Paths

- `strict_supervised`: `outputs/step2_canonical/universal_portfolio_strict_results.json`
- `recursive_controlled`: `outputs/step2_canonical/recursive_feature_router_suite_10seed_5000/recursive_feature_utility_results.json`
- `opmnist_partial`: `outputs/step2_opmnist_scale_800task/opmnist_true_mnist_800task_partial20block_mse_results.json`
- `opmnist_10block`: `outputs/step2_opmnist_scale_10block/opmnist_true_mnist_10block_results.json`
- `scr_million`: `outputs/step2_scr_million_slow_meta_3seed/scr_million_slow_meta_3seed_results.json`
- `td_gvf_bridge`: `output/td_gvf_ar1_squares_5seed/summary.json`
