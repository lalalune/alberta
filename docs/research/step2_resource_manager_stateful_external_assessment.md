# Step 2 Learned Resource Manager And Stateful External Assessment

This note records the follow-up closure pass for the two gaps left after the
strict universal portfolio:

- learned resource management rather than hand-specified class-imbalance guards;
- harder stateful external continual benchmarks rather than only shuffled or
  phasewise sklearn digits.

## Implementation

`src/alberta_framework/core/resource_manager.py` adds
`LearnedResourceManager`, a contextual Hedge controller over discrete resource
policies.  It learns online from causal per-action losses:

`advantage_i = dot(weights, adjusted_losses) - adjusted_losses_i`

`log_weight_i <- discount * log_weight_i + learning_rate * advantage_i`

The manager is generic: actions can represent generator choices, perturbation
levels, replacement policies, or expert/resource policies.  The Step 2
experiment uses four resource policies:

- `mlp_static`: fair MLP, no extra plasticity resource;
- `upgd_low`: low-noise UPGD;
- `upgd_high`: high-noise UPGD;
- `cbp_replace`: Continual Backprop replacement.

`examples/The Alberta Plan/Step2/step2_resource_manager_stateful_external.py`
uses two learned managers:

- `resource_manager`: tracking manager updated from current prequential loss;
- `resource_manager_retention`: prototype-balanced retention manager updated
  from online class prototypes, so held-out deployment is learned from stream
  evidence rather than a hand-coded class-imbalance trigger.

The runner now also includes an opt-in
`external_delayed_contextual_permutation` regime.  It can load OpenML
Fashion-MNIST when `--external-image-source openml_fashion_mnist
--allow-openml-download` is passed.  Without that flag it uses a local
28x28 expanded sklearn-digits fallback, preserving offline smoke tests.  The
stream applies recurring image permutations by the true hidden block state, but
the learned resource manager receives a context id delayed by whole blocks.

## Benchmarks

All three benchmarks use sklearn's bundled digits dataset with no network data:

- `digits_recurrent_permutation`: recurring pixel-permutation states, held-out
  evaluation averaged over all recurrent permutations;
- `digits_recurrent_mask_noise`: recurring feature-mask states with online
  noise, held-out evaluation averaged over masks;
- `digits_class_blocked_retention`: digit-class blocks with balanced held-out
  retention evaluation.

Additional optional regime:

- `external_delayed_contextual_permutation`: Fashion-MNIST-style 28x28 image
  stream when OpenML is explicitly enabled; otherwise a no-network expanded
  28x28 sklearn-digits fallback.  This is a harder stateful image smoke test
  because the manager context is delayed relative to the current permutation.

Canonical result:

- `outputs/step2_canonical/resource_manager_stateful_external_results.json`
- `outputs/step2_canonical/resource_manager_stateful_external_SUMMARY.md`

## 10-Seed Result

Positive differences favor the learned manager.

| Benchmark | Tracking final-window MSE vs MLP | Wins/losses/ties | Tracking held-out acc vs MLP | Retention held-out acc vs MLP |
|---|---:|---:|---:|---:|
| `digits_recurrent_permutation` | `+0.0108 +/- 0.0003` | `10/0/0` | `+0.0199 +/- 0.0049` | `+0.0065 +/- 0.0031` |
| `digits_recurrent_mask_noise` | `+0.0109 +/- 0.0005` | `10/0/0` | `+0.0456 +/- 0.0044` | `+0.0323 +/- 0.0038` |
| `digits_class_blocked_retention` | `+0.0010 +/- 0.0000` | `10/0/0` | `-0.0007 +/- 0.0027` | `+0.1121 +/- 0.0128` |

The old class-blocked tension is now explicit and resolved without a
hard-coded trigger.  The tracking manager learns MLP/CBP-like allocation for
current-block MSE, while the prototype-balanced retention manager learns a UPGD
deployment allocation for balanced held-out retention.  On class-blocked
retention, the retention manager's held-out accuracy beats static MLP in
`10/10` seeds.

## Critical Assessment

The 10-seed sklearn-digits suite closes the two practical gaps named after the
universal-portfolio pass:

- a learned online resource manager exists in core;
- harder stateful external digits benchmarks exist and are canonical;
- the learned tracking manager improves current prediction on all three hard
  external streams;
- the learned retention manager improves held-out retained accuracy on all
  three hard external streams.

The remaining gap is not "no learned manager" or "no harder stateful external
benchmark." A follow-up implementation now also closes the narrower
generator-internal manager gap for `FixedBudgetFeatureLearner` by learning
generator allocation plus conservative/nominal/aggressive replacement and
promotion policies. The remaining gap is stricter and more research-heavy:

- the benchmark is still sklearn digits, not full MNIST, Permuted MNIST, or
  Slowly-Changing Regression;
- the result is strong empirical evidence for the repo's Step 2 matrix, not a
  theorem about arbitrary recursive feature construction.

## 28x28 External-Source Smoke

Smoke command:

```bash
python "examples/The Alberta Plan/Step2/step2_resource_manager_stateful_external.py" \
  --benchmarks external_delayed_contextual_permutation \
  --steps 80 \
  --n-seeds 1 \
  --final-window 20 \
  --block-size 20 \
  --n-states 3 \
  --hidden-size 16 \
  --external-image-source digits_28x28_fallback \
  --output-dir outputs/step2_canonical/resource_manager_stateful_external_external_smoke \
  --note-path outputs/step2_canonical/resource_manager_stateful_external_external_smoke/NOTE.md
```

Result:

- `outputs/step2_canonical/resource_manager_stateful_external_external_smoke/results.json`
- `outputs/step2_canonical/resource_manager_stateful_external_external_smoke/SUMMARY.md`
- source used: `sklearn.datasets.load_digits_expanded_28x28`
- OpenML used: no
- `resource_manager` vs `mlp_static` final-window MSE: `+0.0120`, `1/1`
  wins
- `resource_manager` vs `mlp_static` held-out accuracy: `+0.0569`, `1/1`
  wins
- `resource_manager_retention` vs `mlp_static` held-out accuracy: `+0.0482`,
  `1/1` wins

This narrows the larger external-stateful gap by adding the Fashion-MNIST
loader path and a 28x28 delayed-context image stressor, with local smoke
evidence that the runner, delayed context, and fallback path work.  It does not
claim full closure of the published-scale external gap because this pass did
not download or run OpenML Fashion-MNIST.
