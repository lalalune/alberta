# DiffEML DiffLogic Benchmark Plan

This plan defines fair external comparisons for a production DiffEML paper push.
It is deliberately conservative: current DiffEML artifacts demonstrate nontrivial
hard Boolean-circuit learning, not superiority over DiffLogic, LogicTreeNet, or
same-feature neural baselines.

## Current Evidence

The strongest current internal artifact is:

| Artifact | Dataset | Split | Features | Circuit | Hard test acc. | Same-feature baseline |
| --- | --- | --- | --- | --- | ---: | ---: |
| `outputs/diffeml_image_demo/cifar_detector3_random_w2048_l6_train20000_vs_mlp512.json` | CIFAR-10 | 20k train / 5k test, seed 0 | detector thresholds, 16,384 selected bits | random, 6 x 2048 | 42.24% | MLP-512: 46.70% |
| `outputs/diffeml_image_demo/cifar_detector4_random_w2048_l6_train20000_hard08_packed.json` | CIFAR-10 | 20k train / 5k test, seed 0 | detector thresholds, 24,576 selected bits | random, 6 x 2048 | 44.22% packed hard | MLP-512: 49.32% |
| `outputs/diffeml_image_demo/cifar_detector4_random_w2048_l6_train20000_dropout_packed.json` | CIFAR-10 | 20k train / 5k test, seed 0 | detector thresholds, 24,576 selected bits, dropout regularization | random, 6 x 2048 | 46.40% packed hard | MLP-512: 49.32% |
| `outputs/diffeml_image_demo/cifar_detector4_random_w2048_l6_train20000_dropout030_packed.json` | CIFAR-10 | 20k train / 5k test, seed 0 | detector thresholds, 24,576 selected bits, input drop 0.02, feature drop 0.30 | random, 6 x 2048 | 47.00% packed hard | MLP-512: 49.32% |
| `outputs/diffeml_image_demo/cifar_detector4_random_w2048_l6_train20000_dropout030_groupsum_tau10_packed.json` | CIFAR-10 | 20k train / 5k test, seed 0 | same detector4 bits and dropout | random, 6 x 2048, `group_sum` readout | 31.78% packed hard | fixed Boolean/count readout |
| `outputs/diffeml_image_demo/cifar_detector4_random_w2048_l6_train20000_dropout030_classvote_tau10_packed.json` | CIFAR-10 | 20k train / 5k test, seed 0 | same detector4 bits and dropout | random, 6 x 2048, `class_vote` readout | 35.50% packed hard | learned discrete readout metadata |
| `outputs/diffeml_image_demo/cifar_detector4_random_w4096_l8_train20000_hard08_packed.json` | CIFAR-10 | 20k train / 5k test, seed 0 | detector thresholds, 24,576 selected bits | random, 8 x 4096 | 43.82% packed hard | pending |

These rows are useful evidence that DiffEML can harden a learned EML-template
selector into a nontrivial CIFAR-10 classifier and compile that classifier to
packed Boolean evaluation. They are not evidence that DiffEML beats same-feature
MLPs or external logic-network baselines.

## External Targets

Use two tiers of external comparison:

| Family | Source | Dataset rows to include | Why it matters |
| --- | --- | --- | --- |
| DiffLogic | Petersen et al., *Deep Differentiable Logic Gate Networks*, NeurIPS 2022: <https://papers.nips.cc/paper_files/paper/2022/file/0d3496dd0cec77a999c98d35003203ca-Paper-Conference.pdf> | MNIST small/largest; CIFAR-10 small/medium/large x4 | Original differentiable logic-gate baseline and official `difflogic` package target. |
| LogicTreeNet | Petersen et al., *Convolutional Differentiable Logic Gate Networks*, NeurIPS 2024: <https://papers.neurips.cc/paper_files/paper/2024/file/db988b089d8d97d0f159c15ed0be6a71-Paper-Conference.pdf> | MNIST LogicTreeNet-M; CIFAR-10 S/M/B/G | Newer convolutional/tree topology baseline with much stronger CIFAR results. |
| Official code | `difflogic`: <https://github.com/Felix-Petersen/difflogic> | local reproduction pending | Required before claims that depend on reproduced external baselines. |

Paper-reported rows are comparison targets. Local reproduced rows must be kept
separate with `provenance = "local_reproduced"`. Until the official baseline is
installed and reproduced, local reproduction rows remain `provenance = "pending"`.

## Dataset Matrix

| Dataset | DiffEML split | External split | Required use |
| --- | --- | --- | --- |
| digits | sklearn digits smoke, 128/64 or 1200/300 | none | CI and script smoke only; never a headline external comparison. |
| MNIST | OpenML MNIST, 60k/10k, seeds `0..4` | DiffLogic/LogicTreeNet MNIST paper defaults | Sanity comparison for binarized image logic. |
| CIFAR-10 threshold | Toronto CIFAR-10, 50k/10k, seeds `0..4` | DiffLogic CIFAR threshold/quantized input setting | Closest current fair feature comparison to original DiffLogic. |
| CIFAR-10 detector | Toronto CIFAR-10, 20k/5k and then 50k/10k, seeds `0..4` | no exact external match | Internal DiffEML feature-engineering row; compare to same-feature MLP first. |

No test-set tuning is allowed. Any hyperparameter selection must use a validation
split or a predeclared seed-0 pilot, then freeze the matrix for all seeds.

## Budget Matching

Report at least four budgets for every DiffEML row:

| Budget | Definition | Purpose |
| --- | --- | --- |
| selector nodes | `layers * width` | Direct count of hard selected two-input Boolean functions. |
| selected Boolean gates | one selected two-input gate per selector node | Closest abstraction to DiffLogic gate count when `gate_mode = "eml_template"`. |
| expanded EML thresholds | `selector_nodes * (2^template_depth - 1)` | Conservative EML implementation cost for executable templates. |
| non-Boolean head parameters | `width * classes + classes` for `head_mode = "linear"` | Separates the current linear classifier head from hard logic cost. |
| train-time discrete readout parameters | `width * classes` for `class_vote`; `2 * width * classes` for `signed_class_vote` | Tracks the soft readout learned during training. |
| deployed readout metadata | zero bytes for `group_sum`, `ceil(width * log2(classes) / 8)` bytes for `class_vote`, `ceil(width * (log2(classes) + 1) / 8)` bytes for `signed_class_vote`, int8 weights plus scales for linear-head ablations | Prevents int8 linear heads from being mixed with pure Boolean/count readouts. |

DiffLogic and LogicTreeNet report gates/binary operations directly. DiffEML
tables must not hide the linear head. Pure Boolean/count readouts (`group_sum`,
`class_vote`, and `signed_class_vote`) must be reported as separate topology
rows rather than mixed with linear-head rows.

## Planned DiffEML Runs

The matrix script emits commands for:

```bash
python "examples/The Alberta Plan/Step2/step2_diffeml_logic_benchmark.py" \
  --scale paper \
  --seeds 0 1 2 3 4 \
  --output outputs/diffeml_image_demo/logic_benchmark_matrix.json
```

Paper-scale planned rows:

| Run family | Dataset | Features | Topology | Required metrics |
| --- | --- | --- | --- | --- |
| MNIST threshold random | MNIST 60k/10k | 784 threshold bits | random, 6 x 2048 | soft, hard, packed hard, same-feature MLP |
| CIFAR threshold random | CIFAR-10 50k/10k | 9,216 threshold bits | random, 6 x 2048 | soft, hard, packed hard, same-feature MLP |
| CIFAR detector random linear | CIFAR-10 20k/5k | 24,576 detector-threshold bits, input drop 0.02, feature drop 0.30 | random, 6 x 2048 | soft, hard, packed hard, int8-linear-head ablation, same-feature MLP |
| CIFAR detector random Boolean readout | CIFAR-10 20k/5k | same detector bits and gate budget | random, 6 x 2048 with `group_sum` and `class_vote` | soft, hard, packed hard, deployed readout metadata bytes |

The 20k/5k detector row preserves continuity with current artifacts. A 50k/10k
detector row can be added after the 20k/5k five-seed result is stable.

## External Reproduction Plan

1. Install official `difflogic` in a separate CUDA-capable environment and record
   Python, PyTorch, CUDA toolkit, GPU, commit, and package versions.
2. Reproduce the smallest MNIST and CIFAR rows first to validate installation and
   result parsing.
3. Reproduce the strongest original DiffLogic CIFAR row feasible on local
   hardware. If the large x4 row is too expensive, keep it as paper-reported and
   mark local reproduction as pending.
4. For LogicTreeNet, start with paper-reported rows. Reproduction is optional
   unless the paper makes a direct claim against LogicTreeNet rather than using
   it as a literature reference.
5. Store raw logs, configs, checkpoints where available, and JSON summaries with
   `provenance = "local_reproduced"` only after the command has run locally.

## Acceptance Criteria

A DiffEML row can be described as competitive with an external row only if all
conditions hold:

1. DiffEML has local artifacts for seeds `0..4`.
2. The external row has either paper-reported metrics or local reproduced
   metrics, clearly labeled.
3. DiffEML reports `test_hard_accuracy` and `packed_hard_test_accuracy`.
4. Packed hard accuracy matches JAX hard accuracy within `1e-12` for every seed,
   or the packed mismatch is explained and the claim is limited to JAX hard eval.
5. The mean hard accuracy is at least the external baseline mean/report.
6. The table includes confidence intervals or per-seed values.
7. The same-feature MLP row is present and DiffEML is not described as better
   than it unless the same acceptance test passes against that MLP.

Current status against these criteria: **not accepted**. The best current CIFAR
artifact is single-seed and includes packed eval, but it still trails the
smallest published DiffLogic CIFAR row by 4.27 percentage points, trails the
exact same-feature MLP control by 2.32 points, and is far below the stronger
published LogicTreeNet CIFAR rows. Under the corrected pure Boolean readout
goal, the best current `class_vote` row trails the same DiffLogic-small target
by 15.77 points. A larger random-sparse run with 32,768 gates underperformed
the 12,288-gate detector4 run, so current evidence does not support closing the
gap by width/depth scaling alone.

## Reporting Rules

- Use `paper_reported` for numbers copied from papers.
- Use `local_reproduced` only for commands run in this repository or a recorded
  sibling environment with saved artifacts.
- Use `pending` for intended official-code reproductions not yet run.
- Do not merge detector-threshold rows with threshold-pixel rows in aggregate
  means; they answer different questions.
- Always report hard/packed metrics separately from soft relaxed metrics.
- Always include gate budget, topology, feature mode, train/test sizes, seed,
  and provenance in tables.
