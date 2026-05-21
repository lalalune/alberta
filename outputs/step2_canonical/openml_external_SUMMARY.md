# Step 2 True OpenML External Image Follow-up

Date run: 2026-05-05.

Purpose: replace fallback-only external image evidence with verified true
OpenML runs, and characterize whether the remaining published-scale Step 2
external gap can honestly be closed.

## Commands

OpenML MNIST smoke:

```bash
source .venv/bin/activate && python "examples/The Alberta Plan/Step2/step2_published_stressors.py" --benchmarks permuted_mnist_like --smoke --mnist-source openml --allow-openml-download --openml-data-home outputs/step2_canonical/openml_external_cache --output-dir outputs/step2_canonical/openml_external_published_smoke --result-prefix published_stressors_openml_smoke
```

OpenML MNIST 3-seed compact:

```bash
source .venv/bin/activate && python "examples/The Alberta Plan/Step2/step2_published_stressors.py" --benchmarks permuted_mnist_like --steps 600 --n-seeds 3 --seed 0 --final-window 150 --mnist-source openml --allow-openml-download --openml-data-home outputs/step2_canonical/openml_external_cache --max-train-examples 1000 --max-test-examples 300 --n-permutations 3 --task-block-size 200 --output-dir outputs/step2_canonical/openml_external_published_3seed_small --result-prefix published_stressors_openml_3seed_small
```

OpenML MNIST 5-seed canonical-ish:

```bash
source .venv/bin/activate && python "examples/The Alberta Plan/Step2/step2_published_stressors.py" --benchmarks permuted_mnist_like --canonical-ish --mnist-source openml --allow-openml-download --openml-data-home outputs/step2_canonical/openml_external_cache --output-dir outputs/step2_canonical/openml_external_published_5seed --result-prefix published_stressors_openml_5seed
```

OpenML Fashion-MNIST smoke:

```bash
source .venv/bin/activate && python "examples/The Alberta Plan/Step2/step2_resource_manager_stateful_external.py" --benchmarks external_delayed_contextual_permutation --steps 80 --n-seeds 1 --final-window 20 --block-size 20 --n-states 3 --hidden-size 16 --external-image-source openml_fashion_mnist --allow-openml-download --external-sample-limit 300 --output-dir outputs/step2_canonical/openml_external_resource_smoke --note-path outputs/step2_canonical/openml_external_resource_smoke/NOTE.md
```

OpenML Fashion-MNIST 3-seed compact:

```bash
source .venv/bin/activate && python "examples/The Alberta Plan/Step2/step2_resource_manager_stateful_external.py" --benchmarks external_delayed_contextual_permutation --steps 300 --n-seeds 3 --seed 0 --final-window 100 --block-size 60 --n-states 5 --hidden-size 32 --external-image-source openml_fashion_mnist --allow-openml-download --external-sample-limit 1000 --output-dir outputs/step2_canonical/openml_external_resource_3seed_small --note-path outputs/step2_canonical/openml_external_resource_3seed_small/NOTE.md
```

OpenML Fashion-MNIST 5-seed default external-image run:

```bash
source .venv/bin/activate && python "examples/The Alberta Plan/Step2/step2_resource_manager_stateful_external.py" --benchmarks external_delayed_contextual_permutation --external-image-source openml_fashion_mnist --allow-openml-download --output-dir outputs/step2_canonical/openml_external_resource_5seed --note-path outputs/step2_canonical/openml_external_resource_5seed/NOTE.md
```

## Source Metadata

MNIST 5-seed canonical-ish metadata:

- Dataset: `sklearn.datasets.fetch_openml('mnist_784', version=1)`
- `source_kind=openml_mnist_784`
- `is_true_mnist=true`
- `n_total=70000`, `n_train=4000`, `n_test=1000`, `feature_dim=784`
- Split: stratified capped split, `is_full_mnist_split=false`
- Protocol: 5 random pixel permutations, 1,500 online steps, 300-step task
  blocks, no task id provided to the learner
- Published-scale flags: `uses_full_mnist_task_blocks=false`,
  `matches_dohare_opmnist_core_protocol=false`,
  `matches_dohare_opmnist_published_task_count=false`,
  `published_scale_external_claim_supported=false`

Fashion-MNIST 5-seed metadata:

- Dataset: `OpenML Fashion-MNIST`
- `requested_external_source=openml_fashion_mnist`
- `used_fallback=false`
- `openml_name=Fashion-MNIST`, `openml_version=1`
- `n_total=70000`, `n_train=2100`, `n_test=900`, `feature_dim=784`
- Stream: `external_delayed_contextual_permutation`, 5 states, block size
  240, one-block delayed manager context, 1,200 online steps

## Numeric Results

OpenML MNIST 5-seed canonical-ish, portfolio vs best fair MLP:

| Metric | Diff | Wins/losses/ties |
|---|---:|---:|
| Final-window MSE | `+0.0131 +/- 0.0010` | `5/0/0` |
| Held-out test accuracy | `+0.1408 +/- 0.0117` | `5/0/0` |

OpenML Fashion-MNIST 5-seed delayed contextual permutation, resource manager
vs `mlp_static`:

| Metric | Diff | Wins/losses/ties |
|---|---:|---:|
| Final-window MSE | `+0.0102 +/- 0.0015` | `5/0/0` |
| Held-out test accuracy | `+0.0246 +/- 0.0073` | `4/1/0` |

OpenML Fashion-MNIST 5-seed delayed contextual permutation,
`resource_manager_retention` vs `mlp_static`:

| Metric | Diff | Wins/losses/ties |
|---|---:|---:|
| Held-out test accuracy | `+0.0232 +/- 0.0058` | `5/0/0` |

## Assessment

The Fashion-MNIST fallback-only gap is closed: the delayed-context external
image stream now has true OpenML, multi-seed evidence.

The published-scale Online Permuted MNIST gap is not closed. The compact true
OpenML MNIST result is positive, but it is capped and short; it does not use
the full source split, 60,000-example task blocks, or the published task count.
