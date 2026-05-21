# Step 2 Digits Simple Solution Search

Date: 2026-05-04

## Question

Can we beat the fair MLP / MLP-width ensemble on sklearn-digits with something simpler, faster, or lower overhead?

The answer is mixed but useful:

- Yes, simple methods beat the MLP portfolio on several digits regimes.
- No single simple method dominates all regimes.
- The best new ingredients are `softmax_raw` for fast drift tracking and `dual_knn_guard512` for memory/retention.
- The most plausible next system is not a bigger neural net; it is a small causal portfolio:

`uniform_mlp_width_ensemble + softmax_raw + dual_knn_guard512`

## What Was Tried

I added an isolated screen script:

`output/local_simplify/digits_simple_candidates.py`

It evaluates online prequential learners on the same five sklearn-digits regimes:

- `iid`
- `class_blocked`
- `permuted_pixels`
- `mask_noise`
- `label_drift`

Families tested:

- normalized LMS readouts: raw, square, hashed quadratic
- online softmax readouts: raw, square, hashed quadratic
- RLS readouts: raw, square, random tanh, random Fourier
- prototypes: cumulative and EMA class centroids
- memory: reservoir kNN, FIFO/ring kNN
- memory routing: reservoir + ring kNN with Hedge and a class-imbalance retention guard

Primary artifacts:

- `output/local_simplify/simple_digits_candidates_3seed/`
- `output/local_simplify/simple_digits_memory_router_3seed/`
- `output/local_simplify/simple_digits_candidates_10seed/`
- `output/local_simplify/simple_digits_softmax_10seed/`
- `output/local_simplify/simple_digits_softmax_memory_10seed/`

## Best 10-Seed Results

Reference strict portfolio, 10 seeds:

| Regime | Strict Final MSE | Strict Test Acc |
|---|---:|---:|
| `digits_class_blocked` | 0.0029 | 0.2245 |
| `digits_iid` | 0.0228 | 0.9449 |
| `digits_label_drift` | 0.0334 | 0.9163 |
| `digits_mask_noise` | 0.0385 | 0.8616 |
| `digits_permuted_pixels` | 0.0407 | 0.8931 |

### `softmax_raw`

Online multinomial logistic regression on standardized raw pixels. No hidden layer, no feature search, no memory.

| Regime | Final MSE | Test Acc | us/step |
|---|---:|---:|---:|
| `class_blocked` | 0.0119 | 0.7981 | 25.8 |
| `iid` | 0.0091 | 0.9503 | 22.2 |
| `label_drift` | 0.0222 | 0.9078 | 24.2 |
| `mask_noise` | 0.0214 | 0.8718 | 22.4 |
| `permuted_pixels` | 0.0250 | 0.8917 | 27.0 |

Read:

- Beats strict final-window MSE on 4/5 regimes.
- Loses class-blocked tracking MSE because it does not overfit the final class block as hard as the MLP.
- Beats strict held-out test on class-blocked, iid, and mask-noise.
- Nearly ties strict on permuted pixels.
- Slightly loses label-drift held-out test.
- Best simplicity/speed tradeoff found so far.

### `knn_ring512`

FIFO memory of the most recent 512 examples, 5-nearest-neighbor prediction.

| Regime | Final MSE | Test Acc | us/step |
|---|---:|---:|---:|
| `class_blocked` | 0.0123 | 0.4900 | 35.8 |
| `iid` | 0.0071 | 0.9581 | 33.1 |
| `label_drift` | 0.0732 | 0.8080 | 38.0 |
| `mask_noise` | 0.0184 | 0.9134 | 34.5 |
| `permuted_pixels` | 0.0124 | 0.9495 | 36.4 |

Read:

- Very strong for iid, permuted pixels, and mask noise.
- Fails class-blocked held-out retention because FIFO memory forgets older classes.
- Fails label drift relative to MLP/softmax because nearest-neighbor labels are too path-dependent.

### `dual_knn_guard512`

Reservoir kNN + FIFO kNN, with a class-imbalance guard that deploys reservoir memory when recent labels cover too few historically seen classes.

| Regime | Final MSE | Test Acc | us/step |
|---|---:|---:|---:|
| `class_blocked` | 0.0121 | 0.9532 | 132.8 |
| `iid` | 0.0056 | 0.9623 | 128.9 |
| `label_drift` | 0.0722 | 0.8076 | 135.7 |
| `mask_noise` | 0.0182 | 0.9160 | 132.4 |
| `permuted_pixels` | 0.0124 | 0.9508 | 138.7 |

Read:

- Crushes strict portfolio on held-out class-blocked retention: 0.9532 vs 0.2245.
- Beats strict final-window MSE and test accuracy on iid, mask-noise, and permuted-pixels.
- Fails label drift.
- Higher overhead than softmax because it performs two kNN predictions; still conceptually simple and bounded memory.

### `softmax_quadhash256`

Online softmax on raw pixels plus hashed quadratic features.

| Regime | Final MSE | Test Acc | us/step |
|---|---:|---:|---:|
| `class_blocked` | 0.0194 | 0.8770 | 49.0 |
| `iid` | 0.0096 | 0.9525 | 47.1 |
| `label_drift` | 0.0360 | 0.8961 | 48.6 |
| `mask_noise` | 0.0228 | 0.8840 | 47.1 |
| `permuted_pixels` | 0.0322 | 0.8939 | 52.9 |

Read:

- Useful but not better than `softmax_raw` overall.
- Quadratic features help class-blocked test relative to raw softmax but hurt drift tracking and add overhead.
- Do not promote yet.

### `softmax_memory_hedge`

Causal Hedge mixture of `softmax_raw` and `dual_knn_guard512`.

| Regime | Final MSE | Test Acc | us/step |
|---|---:|---:|---:|
| `class_blocked` | 0.0100 | 0.9165 | 258.6 |
| `iid` | 0.0055 | 0.9659 | 254.7 |
| `label_drift` | 0.0222 | 0.9078 | 239.6 |
| `mask_noise` | 0.0171 | 0.9204 | 249.0 |
| `permuted_pixels` | 0.0123 | 0.9505 | 236.4 |

Read:

- Best non-MLP simple system found in this pass.
- Beats strict final-window MSE on 4/5 regimes.
- Beats strict held-out test accuracy on class-blocked, iid, mask-noise, and permuted-pixels.
- Still slightly below strict on label-drift held-out accuracy.
- Still loses to strict on class-blocked final-window MSE because the strict MLP overfits the final class block more sharply.
- Overhead is higher than the components because the prototype is pure Python and evaluates softmax + two kNN branches. It is still mathematically simple and can be optimized.

## Mechanistic Takeaways

1. The MLP is not special on digits because of deep features.
   A raw online softmax readout beats its final-window MSE in most regimes.

2. Memory is the missing piece for class-blocked retention.
   UPGD helped retention weakly; kNN memory solves it directly.

3. Recent memory and reservoir memory solve opposite problems.
   FIFO adapts to current pixel/mask/permutation regimes. Reservoir protects old classes. A guard gets both for class-blocked, but not for label drift.

4. Label drift is the hardest regime for memory.
   When labels are remapped, stored examples become actively misleading. Parametric readouts with plastic weights handle it better.

5. Hashed quadratic features are not the main answer.
   They are sometimes strong in held-out accuracy, but raw softmax is cleaner and usually better on tracking.

## Ranking By Practical Value

1. `softmax_memory_hedge`
   Best non-MLP digits system in the screen. It is not the cheapest, but it combines drift tracking and memory.

2. `softmax_raw`
   Best simplicity/speed. Strong enough to be a permanent baseline and portfolio member.

3. `dual_knn_guard512`
   Best retention/memory module. It should replace UPGD for digits retention if exemplar memory is allowed.

4. `knn_ring512`
   Strong low-complexity tracker for iid, mask-noise, and pixel permutations. Use as the cheap version of memory if class-blocked retention is not required.

5. `softmax_quadhash256`
   Worth keeping as a negative/secondary feature-construction baseline, but not a main solution.

6. RLS/prototypes/NLMS
   Mostly dominated. RLS raw has good class-blocked retention but poor tracking and more matrix overhead. Prototypes are elegant but underperform kNN.

## Proposed Next System

The next serious digits system should be:

`digits_simple_portfolio = slow_convex(mlp_width_ensemble, softmax_memory_hedge)`

Routing:

- Use convex prediction averaging for tracking, not hard selection.
- Use the class-imbalance guard for held-out deployment:
  - normal regimes: deploy tracking mixture
  - class-blocked hazard: deploy memory-heavy mixture or reservoir branch
- Add a label-drift detector:
  - if recent prediction loss is high but class coverage is broad, downweight memory and upweight `softmax_raw`.

Why this should beat the current strict portfolio:

- MLP ensemble handles class-blocked final-window MSE.
- `softmax_memory_hedge` improves cheap drift tracking and adds bounded memory.
- The memory branch massively improves class-blocked retention and improves iid/mask/permutation test accuracy.

Why it is still scientifically honest:

- It is not a universal Step 2 feature-discovery solution.
- It is a digits-specific memory/readout solution.
- If exemplar memory is disallowed, keep `softmax_raw` as the main simple addition and treat kNN as an oracle/upper-bound memory baseline.

## Open Problems

1. Implement the actual three-way causal portfolio and verify paired 10-seed metrics.
2. Measure all candidates in one runner with the same implementation substrate. The simple candidates report internal NumPy loop time; the JAX MLP runner has compilation overhead and is not directly comparable.
3. Decide whether exemplar memory is allowed under the Step 2 representation budget. If yes, kNN memory is a major improvement. If no, it should be reported as a bounded-memory baseline rather than the Step 2 solution.
4. Add a drift detector that distinguishes class imbalance from label remapping.
