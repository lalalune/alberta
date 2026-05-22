# Step 2 External / Non-Synthetic Benchmarks

This note records Worker S2C's external-data Step 2 benchmark expansion. It is
intended to prevent the Step 2 synthetic UPGD result from being overgeneralized.
All data are locally available through scikit-learn; no network datasets are
used.

## 2026-05-06 Completion Reruns

The external matrix has now been rerun on all ten local domains with 10 seeds
and a stronger MLP width check. These runs do not support an unconditional
"UPGD beats MLP across domains" claim. They do support a narrower claim:
target-structure UPGD is stronger than fair MLPs on image-like classification
shifts, exact-zero stress, and sparse multilabel targets, while scalar
regression remains a blocker.

Completed outputs:

- `output/subagents/external_breadth_sklearn/promoted_step2_default_10seed_all/external_suite_SUMMARY.md`
- `output/subagents/external_breadth_sklearn/promoted_step2_default_10seed_all_mlpdeep/external_suite_SUMMARY.md`
- `output/subagents/external_breadth_sklearn/regression_multilabel_tune_10seed/external_suite_SUMMARY.md`

### Against `MLP(64)` on All Ten Domains

`UPGDLearner.step2_default` wins final-window MSE against `MLP(64)` on seven
of ten domains:

| Domain | UPGD final-window result vs MLP64 | Held-out/test result |
|---|---:|---:|
| `digits_shuffled` | `+0.0165`, `10/10` wins | test accuracy `+0.0078`, `8/10` wins |
| `digits_class_blocked` | `-0.0008`, `3/10` wins | test accuracy `+0.6881`, `10/10` wins |
| `digits_permuted` | `+0.0183`, `10/10` wins | test accuracy `+0.2898`, `10/10` wins |
| `digits_mask_noise` | `+0.0245`, `10/10` wins | test accuracy `+0.0506`, `10/10` wins |
| `wine_shuffled` | `+0.0026`, `10/10` wins | test accuracy near tie |
| `breast_cancer_shuffled` | `+0.0155`, `10/10` wins | test accuracy `+0.0076`, `7/10` wins |
| `diabetes_regression` | `-0.0845`, `0/10` wins | test MSE also worse |
| `dense_exact_zero` | approximately tied-positive, `10/10` wins | test MSE best/tied |
| `sparse_multilabel` | `+0.0282`, `10/10` wins | test accuracy `+0.0219`, `9/10` wins |
| `temporal_delayed_history` | `-0.0659`, `0/10` wins | test MSE also worse |

### Stronger MLP Width Check

Adding `mlp_deep = MLP(64,64)` does not erase the UPGD wins on image-like and
multilabel tasks:

| Domain | Best reading with `MLP(64,64)` included |
|---|---|
| `digits_shuffled` | UPGD still best on final-window MSE and test accuracy. |
| `digits_class_blocked` | `MLP(64,64)` has best final-window MSE; UPGD has far better retained balanced test accuracy. |
| `digits_permuted` | UPGD still best on final-window MSE and test accuracy. |
| `digits_mask_noise` | UPGD still best on final-window MSE and test accuracy. |
| `wine_shuffled` | UPGD best on MSE; accuracy is saturated/tied. |
| `breast_cancer_shuffled` | UPGD best on final-window MSE and mean test accuracy. |
| `diabetes_regression` | `MLP(64,64)` is best; UPGD default loses. |
| `dense_exact_zero` | UPGD best/tied; `MLP(64,64)` leaves a small residual. |
| `sparse_multilabel` | UPGD best on final-window MSE and test accuracy. |
| `temporal_delayed_history` | `MLP(64)` is best; UPGD default loses. |

### Regression Tuning Attempt

The regression/multilabel tuning sweep tested `upgd`, `upgd_fast`,
`upgd_mean`, and `upgd_wide` against `mlp`, `mlp_deep`, and `cbp` on
`diabetes_regression`, `temporal_delayed_history`, `dense_exact_zero`, and
`sparse_multilabel`.

| Domain | Best UPGD variant | Result |
|---|---|---|
| `diabetes_regression` | `upgd_mean` / `upgd_fast` | Beats `MLP(64)` by mean final-window MSE, but loses to `MLP(64,64)`. |
| `temporal_delayed_history` | `upgd_fast` | Nearly ties `MLP(64)` but still loses by mean and wins only `3/10` seeds. |
| `dense_exact_zero` | `upgd` / `upgd_mean` | Best/tied; validates the target-structure exact-zero behavior. |
| `sparse_multilabel` | `upgd_wide` final-window, `upgd_mean` test | Beats both MLP baselines strongly. |

Interpretation: scalar regression is the remaining external hole. The next
mechanism should add a regression-specific adaptive readout or normalizer
rather than claiming the current UPGD family is universally superior.

## 2026-05-06 Passthrough Rescue Reruns

The scalar regression blockers were rerun with passthrough/deep readout UPGD
controls. These are still single learners, not routers or portfolios, but they
change the readout structure: the output head can see `hidden_plus_input`, and
the regression branch can use the same 64-64 depth as `MLP(64,64)`.

Completed outputs:

- `output/subagents/external_breadth_sklearn/regression_temporal_passthrough_rescue_10seed/external_suite_SUMMARY.md`
- `output/subagents/external_breadth_sklearn/passthrough_rescue_full_10seed_all/external_suite_SUMMARY.md`

Focused blocker rerun:

| Domain | Best rescue row | Result |
|---|---|---|
| `diabetes_regression` | `upgd_reg_passthrough_deep` | final-window MSE `0.3461 +/- 0.0113` vs `MLP(64)` `0.5228` and `MLP(64,64)` `0.4831`; `10/10` wins vs MLP64 |
| `temporal_delayed_history` | `upgd_temporal_passthrough_no_mutation` final-window, `upgd_temporal_fast_passthrough` test | final-window MSE `0.4311 +/- 0.0085` vs `MLP(64)` `0.5475`; `10/10` wins vs MLP64 |

The all-domain rerun confirms the rescue is specialized, not a new universal
default:

| Domain | Best row in the passthrough full run |
|---|---|
| `digits_shuffled` | existing `upgd` |
| `digits_class_blocked` | `MLP(64,64)` for online MSE; UPGD family for retained test accuracy |
| `digits_permuted` | existing `upgd` |
| `digits_mask_noise` | existing `upgd` |
| `wine_shuffled` | existing `upgd` |
| `breast_cancer_shuffled` | existing `upgd` |
| `diabetes_regression` | `upgd_reg_passthrough_deep` |
| `dense_exact_zero` | existing `upgd` |
| `sparse_multilabel` | existing `upgd` |
| `temporal_delayed_history` | `upgd_temporal_fast_passthrough` |

Interpretation: the previous scalar blockers were not evidence against UPGD
feature utility alone; they exposed a readout bottleneck. Direct input
passthrough plus scalar readout normalization fixes diabetes and delayed
history regression. However, the same passthrough variants are poor on
multiclass/multilabel digit-style streams, so the repo should not simply
replace `UPGDLearner.step2_default` with a passthrough branch. The next
canonical simplification target is a causal, target-structure-conditioned
readout mode: simplex/vector classification should keep the current hidden
readout, while scalar regression can use normalized `hidden_plus_input`
readout without becoming a dataset router.

## Protocol

Script:

```bash
source .venv/bin/activate
python "examples/The Alberta Plan/Step2/step2_external_suite.py" \
  --output-dir output/worker_s2c_external
```

## Worker C Domain Matrix

`step2_external_suite.py` now has a single cheap matrix that separates target
and domain shape from learner changes. The matrix is:

| Domain slice | Benchmark key | Source | Task shape |
|---|---|---|---|
| Image-like stationary pixels | `digits_shuffled` | sklearn digits | multiclass one-hot |
| Image-like label blocks | `digits_class_blocked` | sklearn digits | multiclass one-hot |
| Image-like nonstationary pixels | `digits_permuted` | sklearn digits | multiclass one-hot |
| Image-like masked/noisy pixels | `digits_mask_noise` | sklearn digits | multiclass one-hot |
| Tabular classification | `wine_shuffled` | sklearn wine | multiclass one-hot |
| Tabular classification | `breast_cancer_shuffled` | sklearn breast cancer | binary one-hot |
| Tabular regression | `diabetes_regression` | sklearn diabetes | scalar regression |
| Dense exact-zero target | `dense_exact_zero` | generated local stream | scalar exact-zero regression |
| Sparse multilabel | `sparse_multilabel` | generated local stream | sparse multi-hot vector |
| Temporal delayed/history target | `temporal_delayed_history` | generated local sequence | scalar delayed regression |

Exact full matrix command:

```bash
source .venv/bin/activate
python "examples/The Alberta Plan/Step2/step2_external_suite.py" \
  --benchmarks digits_shuffled,digits_class_blocked,digits_permuted,digits_mask_noise,wine_shuffled,breast_cancer_shuffled,diabetes_regression,dense_exact_zero,sparse_multilabel,temporal_delayed_history \
  --methods linear,mlp,upgd,cbp \
  --steps 3000 \
  --n-seeds 5 \
  --final-window 500 \
  --n-permutations 5 \
  --permutation-block-size 500 \
  --output-dir outputs/step2_domain_matrix/full
```

Exact smoke command for the newly added slices:

```bash
source .venv/bin/activate
python "examples/The Alberta Plan/Step2/step2_external_suite.py" \
  --benchmarks digits_mask_noise,diabetes_regression,dense_exact_zero,sparse_multilabel,temporal_delayed_history \
  --methods mlp \
  --steps 40 \
  --n-seeds 1 \
  --final-window 10 \
  --output-dir output/worker_c_domain_matrix_new_stressors_smoke
```

The published-style matrix remains in `step2_published_stressors.py` for
Online Permuted MNIST style tasks and Slowly-Changing Regression, including
OpenML/torchvision MNIST support when downloads are explicitly enabled.

Output:

- `output/worker_s2c_external/external_suite_results.json`
- `output/worker_s2c_external/external_suite_SUMMARY.md`

Configuration:

- `5` seeds, seeds `0..4`.
- `3000` online training steps per benchmark.
- Final-window metrics use the last `500` prequential steps.
- Train/test split is stratified with train fraction `0.7`.
- Inputs are standardized using train-split statistics only.
- Targets are one-hot class vectors and loss is mean squared error.
- At each online step, the learner predicts the current example, records
  pre-update loss/accuracy, then updates on the same target.
- Held-out test splits are never used for updates.

Methods:

- `linear`: `MultiHeadMLPLearner(hidden_sizes=())`
- `mlp`: `MultiHeadMLPLearner(hidden_sizes=(64,))`
- `upgd`: `UPGDLearner(hidden_sizes=(64,))`
- `cbp`: `CBPMultiHeadMLPLearner(hidden_sizes=(64,))`

Shared hyperparameters:

- step size `0.03`
- sparsity `0.5`
- layer norm enabled where hidden layers exist
- `ObGDBounding(kappa=2.0)`

UPGD-specific:

- utility decay `0.995`
- perturbation sigma `0.001`

CBP-specific:

- utility decay `0.99`
- replacement rate `1e-4`
- maturity threshold `100`

## Benchmarks

### `digits_shuffled`

Bundled `sklearn.datasets.load_digits`; each repeated training epoch is
shuffled. This is the least non-stationary digits protocol and most closely
matches the earlier single external benchmark.

| Method | Final-window MSE | Final-window acc | Test MSE | Test acc |
|---|---:|---:|---:|---:|
| linear | 0.0878 +/- 0.0126 | 0.8556 +/- 0.0059 | 0.1454 +/- 0.0487 | 0.8646 +/- 0.0099 |
| mlp | **0.0200 +/- 0.0003** | **0.9656 +/- 0.0035** | 0.0229 +/- 0.0008 | 0.9421 +/- 0.0052 |
| upgd | 0.0243 +/- 0.0003 | 0.9428 +/- 0.0040 | 0.0262 +/- 0.0006 | 0.9302 +/- 0.0065 |
| cbp | 0.0202 +/- 0.0003 | 0.9648 +/- 0.0048 | **0.0220 +/- 0.0004** | **0.9429 +/- 0.0049** |

Paired against MLP:

- UPGD loses final-window MSE on `0/5` seeds and loses test accuracy on `1/5`
  seeds won.
- CBP is essentially tied with MLP; it has slightly better mean test MSE/test
  accuracy, but only `2/5` paired test-accuracy wins.

Conclusion: MLP remains the strongest current-tracking learner on shuffled
digits. CBP is a near tie. UPGD does not improve over MLP here.

### `digits_class_blocked`

Bundled digits, but each repeated epoch is ordered into class blocks. This is a
deliberately non-stationary label-distribution protocol. The final-window metric
mostly measures how well the learner tracks the current class block, while the
held-out test metric measures balanced retention/generalization after the final
state.

| Method | Final-window MSE | Final-window acc | Test MSE | Test acc |
|---|---:|---:|---:|---:|
| linear | 0.0573 +/- 0.0127 | 0.9608 +/- 0.0040 | 0.1767 +/- 0.0519 | **0.2549 +/- 0.0221** |
| mlp | 0.0040 +/- 0.0001 | **0.9848 +/- 0.0005** | 0.1314 +/- 0.0064 | 0.1128 +/- 0.0093 |
| upgd | 0.0148 +/- 0.0007 | 0.9472 +/- 0.0051 | **0.0926 +/- 0.0046** | 0.2367 +/- 0.0447 |
| cbp | **0.0040 +/- 0.0002** | **0.9848 +/- 0.0014** | 0.1311 +/- 0.0076 | 0.1206 +/- 0.0151 |

Paired against MLP:

- UPGD loses final-window MSE on `0/5` paired wins, but wins balanced held-out
  test accuracy on `5/5` seeds with mean advantage `+0.1239`.
- CBP ties final-window MSE and wins held-out test accuracy on `4/5` seeds,
  but its mean test-accuracy gain is small (`+0.0078`).
- Linear has the best held-out test accuracy because it is less able to
  over-specialize to the current class block.

Conclusion: UPGD is not better at tracking the current block, but it is much
more robust than MLP on the balanced held-out state after blocked training. This
is a useful anti-forgetting signal, not a clean Step 2 final-window win.

### `digits_permuted`

Bundled digits with `5` fixed pixel permutations. The online stream switches
permutation every `500` examples. Held-out test metrics average over all fixed
permutations.

| Method | Final-window MSE | Final-window acc | Test MSE | Test acc |
|---|---:|---:|---:|---:|
| linear | 0.0783 +/- 0.0044 | 0.7624 +/- 0.0077 | 0.1585 +/- 0.0086 | 0.2798 +/- 0.0152 |
| mlp | **0.0470 +/- 0.0010** | **0.8064 +/- 0.0099** | 0.0971 +/- 0.0007 | 0.3374 +/- 0.0095 |
| upgd | 0.0547 +/- 0.0008 | 0.7292 +/- 0.0045 | **0.0765 +/- 0.0015** | **0.4847 +/- 0.0200** |
| cbp | 0.0497 +/- 0.0010 | 0.7928 +/- 0.0100 | 0.1011 +/- 0.0024 | 0.3316 +/- 0.0148 |

Paired against MLP:

- UPGD loses final-window MSE on `0/5` paired wins, but wins held-out
  permutation-averaged test accuracy on `5/5` seeds with mean advantage
  `+0.1473`.
- CBP loses both final-window MSE and test accuracy to MLP.

Conclusion: this is the clearest external UPGD robustness result in this pass.
MLP tracks the currently active permutation better, but UPGD retains a more
useful representation across all permutations.

### `wine_shuffled`

Bundled `sklearn.datasets.load_wine`; repeated shuffled epochs.

| Method | Final-window MSE | Final-window acc | Test MSE | Test acc |
|---|---:|---:|---:|---:|
| linear | 0.0447 +/- 0.0009 | 0.9804 +/- 0.0032 | 0.0444 +/- 0.0025 | 0.9887 +/- 0.0046 |
| mlp | **0.0025 +/- 0.0002** | **1.0000 +/- 0.0000** | **0.0124 +/- 0.0022** | **0.9962 +/- 0.0038** |
| upgd | 0.0073 +/- 0.0002 | **1.0000 +/- 0.0000** | 0.0179 +/- 0.0024 | 0.9849 +/- 0.0092 |
| cbp | 0.0027 +/- 0.0003 | **1.0000 +/- 0.0000** | 0.0143 +/- 0.0018 | 0.9925 +/- 0.0046 |

Conclusion: MLP is best. This dataset is nearly saturated and too easy to be a
strong Step 2 discriminator.

### `breast_cancer_shuffled`

Bundled `sklearn.datasets.load_breast_cancer`; repeated shuffled epochs.

| Method | Final-window MSE | Final-window acc | Test MSE | Test acc |
|---|---:|---:|---:|---:|
| linear | 0.0832 +/- 0.0035 | 0.9408 +/- 0.0062 | 0.0768 +/- 0.0037 | 0.9368 +/- 0.0062 |
| mlp | 0.0218 +/- 0.0017 | 0.9832 +/- 0.0016 | **0.0295 +/- 0.0030** | 0.9696 +/- 0.0039 |
| upgd | **0.0212 +/- 0.0013** | 0.9808 +/- 0.0023 | 0.0306 +/- 0.0028 | 0.9684 +/- 0.0048 |
| cbp | 0.0215 +/- 0.0021 | **0.9844 +/- 0.0026** | 0.0303 +/- 0.0032 | **0.9743 +/- 0.0040** |

Paired against MLP:

- UPGD has a tiny final-window MSE edge (`+0.0006`, `3/5` wins), but slightly
  worse test accuracy.
- CBP has a tiny final-window MSE edge (`+0.0003`, `3/5` wins) and the best
  mean test accuracy, but only `2/5` paired test-accuracy wins.

Conclusion: breast-cancer is a near tie among MLP/UPGD/CBP. It does not provide
strong evidence that either UPGD or CBP beats MLP.

## Overall Assessment

The external suite does not close Step 2. It adds nuance:

1. **Shuffled external classification still favors plain MLP.** On shuffled
   digits and wine, MLP is the best current-tracking learner. On breast-cancer,
   MLP/UPGD/CBP are within small seed-level differences.
2. **UPGD has a real robustness signal under digit distribution shifts.** On
   class-blocked and permuted digits, UPGD loses immediate final-window MSE but
   wins held-out balanced/transformation-averaged test accuracy on `5/5` seeds.
   This suggests utility perturbation may preserve more generally useful
   features, even when it hurts fast tracking of the current regime.
3. **CBP is mostly MLP-like in this suite.** It near-ties shuffled digits,
   class-blocked final-window MSE, and breast-cancer. It does not reproduce the
   UPGD robustness signal on permuted digits.
4. **The metric matters.** Final-window MSE answers "who tracks the current
   stream state best?" Held-out transformed/balanced accuracy answers "who kept
   a representation that still works away from the final regime?" Step 2 needs
   both, but a method should not be declared canonical from only one.

## What This Leaves Open

- This is still not a full external continual-control benchmark.
- The suite is classification-as-vector-regression with one-hot MSE, not a
  natural streaming regression problem.
- UPGD hyperparameters were not swept. The current result may understate or
  overstate UPGD depending on perturbation sigma, decay, and block length.
- CBP likely needs longer horizons than `3000` steps for replacement to matter
  strongly.
- Class-blocked digits exposed severe forgetting for MLP, but linear also beat
  UPGD on held-out accuracy, so the robustness signal is not yet uniquely a
  feature-construction success.

Recommended next external experiments:

- Run a longer permuted-digits horizon with more permutation cycles and several
  block sizes (`250`, `500`, `1000`) to separate fast adaptation from retention.
- Sweep UPGD perturbation sigma and utility decay on permuted/class-blocked
  digits; treat test accuracy across all transformations/classes as a retention
  metric and final-window MSE as a plasticity metric.
- Run CBP with larger hidden layers and longer horizons so unit replacement is
  not under-sampled.
- Add a small replay-free retention probe: after online training, evaluate each
  saved regime transform/class group without further updates.
- Add external streaming regression once a local bundled dataset with continuous
  targets is chosen, because Step 2's original framing is supervised prediction
  rather than classification.

## Bottom Line

The external evidence for UPGD alone remains mixed and does not justify saying
UPGD solves Step 2. MLP remains the best default on ordinary shuffled external
data. UPGD has the most promising non-synthetic robustness signal on shifted
digits, but that signal is a retention/generalization advantage rather than a
final-window tracking advantage.

The follow-up retention-aware low-noise MLP/UPGD expert mixture addresses this
specific external gap on the current probe. Online tracking still uses ordinary
Hedge and therefore avoids the UPGD current-window regression. Held-out
deployment adds a class-imbalance trigger: when lifetime class coverage is broad
but the recent window is class-narrow, deployment shifts to UPGD. This closes
the class-blocked balanced-retention failure while leaving IID, permuted,
mask-noise, and label-drift digits on their ordinary tracking weights.

This should be read as a benchmark-closing router, not a proof that UPGD alone
or the current compositional feature learner solves Step 2 universally.
