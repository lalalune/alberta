# Step 2 Published MNIST Closure

Superseded status: this note records the older `step2_published_stressors.py`
runner state. The current Step 2 OPMNIST task-count/protocol gate is superseded
by
`outputs/step2_canonical/upgd_memory_opmnist_latest_best_800block_1seed_results.json`,
which completes one 800-task / 48,000,000-update OpenML MNIST seed with all 800
held-out permutation views. The remaining gap is multi-seed and metric-level
performance closure.

## 2026-05-06 OPMNIST Runner Status

The current true OpenML MNIST OPMNIST runner is
`examples/The Alberta Plan/Step2/step2_published_stressors.py`. Its highest
durable checkpoint is:

`outputs/step2_opmnist_scale_800task/opmnist_true_mnist_800task_partial40block_mse_seed0_opmnist_resume.pkl`

Despite the historical `partial40block` filename, the checkpoint sidecar now
reports **47/800 full 60,000-example task blocks** and **2,820,000/48,000,000
steps** complete. The status artifact is:

`outputs/step2_opmnist_scale_800task/opmnist_true_mnist_800task_full_mse_status.json`

The 47-block evaluated snapshot is not closed: final-window MSE favors the
portfolio by `+0.003795`, but held-out test accuracy trails the same-run best
fair MLP by `-0.001798`, and the published task-count gate is still false.

Canonical full resume command:

```bash
source .venv/bin/activate && python "examples/The Alberta Plan/Step2/step2_published_stressors.py" --benchmarks permuted_mnist_like --n-seeds 1 --seed 0 --steps 48000000 --final-window 60000 --mnist-source openml --allow-openml-download --openml-data-home outputs/step2_published_mnist_openml_cache --mnist-published-scale --n-permutations 800 --max-test-permutation-views 800 --opmnist-chunk-size 20000 --opmnist-resume --opmnist-resume-path outputs/step2_opmnist_scale_800task/opmnist_true_mnist_800task_partial40block_mse_seed0_opmnist_resume.pkl --digits-deployment-objective mse --output-dir outputs/step2_opmnist_scale_800task --result-prefix opmnist_true_mnist_800task_full_mse
```

## Protocol audit

Dohare et al. define Online Permuted MNIST as true MNIST with pixels scaled to
`[0, 1]`, a sequence of random pixel permutations, `60,000` examples per task,
one-by-one presentation in random order, a single pass with no mini-batches, and
no task-switch indication to the learner. The main Online Permuted MNIST study
uses `800` permuted tasks, so full published task-count scale is `48,000,000`
online examples. Sources: [Nature article](https://www.nature.com/articles/s41586-024-07711-7)
and [public reproduction repository](https://github.com/shibhansh/loss-of-plasticity).

The runner now records these gates explicitly:

| Gate | Runner flag |
|---|---|
| True MNIST source | `uses_true_mnist` |
| OpenML canonical `60k/10k` split | `uses_full_openml_mnist_split` |
| `60,000`-example task blocks | `uses_full_mnist_task_blocks` |
| No task id to learner | `task_id_provided_to_learner=false` |
| Sequential single-pass task order | `single_pass_examples_within_task` |
| Random pixel permutation for every task | `uses_random_pixel_permutations_for_all_tasks` |
| Core source/split/block/order protocol | `matches_dohare_opmnist_core_protocol` |
| Full `800` task / `48M` example scale | `matches_dohare_opmnist_published_task_count` |

`published_scale_external_claim_supported` is true only if the Online Permuted
MNIST gates pass, the `800` task-count gate passes, and the portfolio is
nonnegative against the best MLP on primary comparisons. The sklearn fallback
cannot set this flag.

## Commands

Focused tests:

```bash
source .venv/bin/activate
pytest tests/test_step2_published_stressors.py -v
```

OpenML/cache smoke:

```bash
source .venv/bin/activate
python "examples/The Alberta Plan/Step2/step2_published_stressors.py" \
  --benchmarks permuted_mnist_like \
  --mnist-source openml \
  --allow-openml-download \
  --openml-data-home outputs/step2_published_mnist_openml_cache \
  --smoke \
  --output-dir outputs/step2_published_mnist_openml_smoke \
  --result-prefix openml_smoke
```

One full `60,000`-example OpenML MNIST task block:

```bash
source .venv/bin/activate
python "examples/The Alberta Plan/Step2/step2_published_stressors.py" \
  --benchmarks permuted_mnist_like \
  --mnist-source openml \
  --allow-openml-download \
  --openml-data-home outputs/step2_published_mnist_openml_cache \
  --mnist-published-scale \
  --steps 60000 \
  --n-seeds 1 \
  --final-window 10000 \
  --n-permutations 5 \
  --max-test-permutation-views 5 \
  --output-dir outputs/step2_published_mnist_fullsplit_60k_1seed \
  --result-prefix fullsplit_60k_1seed
```

## Results

### OpenML smoke

Output: `outputs/step2_published_mnist_openml_smoke/openml_smoke_results.json`

Runtime: `19.776s`

Status flags:

```json
{
  "uses_true_mnist": true,
  "uses_true_openml_mnist": true,
  "uses_full_mnist_split": false,
  "uses_full_mnist_task_blocks": false,
  "task_id_provided_to_learner": false,
  "single_pass_examples_within_task": false,
  "uses_random_pixel_permutations_for_all_tasks": true,
  "matches_dohare_opmnist_core_protocol": false,
  "matches_dohare_opmnist_published_task_count": false,
  "published_scale_external_claim_supported": false
}
```

Metrics:

| Method | Final MSE | Final acc | Test MSE | Test acc |
|---|---:|---:|---:|---:|
| `mixture` | `0.082922` | `0.3500` | `0.085238` | `0.3125` |
| best MLP by final MSE, `mlp_h128` | `0.099846` | `0.3750` | `0.112952` | `0.3125` |

Comparison flags: final-window MSE favors the portfolio by `+0.016924`; held-out
test accuracy trails the best MLP by `-0.006250`.

### Full-split one-block run

Output:
`outputs/step2_published_mnist_fullsplit_60k_1seed/fullsplit_60k_1seed_results.json`

Runtime: `175.996s`

Status flags:

```json
{
  "uses_true_mnist": true,
  "uses_true_openml_mnist": true,
  "uses_full_mnist_split": true,
  "uses_full_openml_mnist_split": true,
  "uses_full_mnist_task_blocks": true,
  "task_id_provided_to_learner": false,
  "single_pass_examples_within_task": true,
  "uses_random_pixel_permutations_for_all_tasks": true,
  "matches_dohare_opmnist_core_protocol": true,
  "matches_dohare_opmnist_published_task_count": false,
  "opmnist_steps": 60000,
  "opmnist_n_permutations": 5,
  "published_scale_external_claim_supported": false
}
```

Only task `0` was observed and evaluated on its held-out test view. The run
configured `5` permutations but did not reach tasks `1` through `4`.

Metrics:

| Method | Final MSE | Final acc | Test MSE | Test acc |
|---|---:|---:|---:|---:|
| `mixture` | `0.014551` | `0.9307` | `0.013617` | `0.9361` |
| `mlp_h64` | `0.020543` | `0.9116` | `0.018592` | `0.9161` |
| `mlp_h128` | `0.018481` | `0.9249` | `0.016712` | `0.9292` |
| `mlp_h64_64` | `0.016749` | `0.9269` | `0.013895` | `0.9367` |
| `upgd_low_noise` | `0.015832` | `0.9252` | `0.015124` | `0.9304` |
| `dynamic_sparse` | `0.015846` | `0.9288` | `0.015110` | `0.9322` |

Comparison flags: final-window MSE favors the portfolio by `+0.002199`;
held-out test accuracy trails the best MLP by `-0.000600`.

## Claim boundary

Full published-scale external evidence is not closed.

What is closed: the runner can now run true OpenML MNIST with the canonical
`60k/10k` split, `60,000`-example task blocks, randomized pixel permutations,
single-pass per-task order, and no task id. The one-block run satisfies
`matches_dohare_opmnist_core_protocol=true`.

What remains open: the run is not the published `800` task / `48M` example
Online Permuted MNIST scale, and the portfolio is not nonnegative on every
primary comparison because held-out test accuracy is slightly below the best MLP.
No outputs were copied to `outputs/step2_canonical/published_mnist_*` because
`published_scale_external_claim_supported=false`.
