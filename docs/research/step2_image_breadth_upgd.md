# Step 2 Image Breadth UPGD

Date: 2026-05-05

## Scope

This workstream tested the current promoted Step 2 learner as a single learner:
`UPGDLearner.step2_default(n_heads=10, readout_mode="softmax_ce")`.

The comparison baselines were same-run `MultiHeadMLPLearner` models with hidden
sizes 32 and 64, `LMS(step_size=0.03)`, `ObGDBounding(kappa=0.5)`, 50% sparse
initialization, and layer norm. There was no portfolio, no router, no retained
MLP head, and no replay buffer.

The existing scripts were inspected:

- `examples/The Alberta Plan/Step2/step2_cifar_stream.py` already supports real
  CIFAR-10 through the local Python archive path and compares
  `UPGDLearner.step2_default` to fair MLP baselines.
- `examples/The Alberta Plan/Step2/step2_published_stressors.py` supports
  OpenML/torchvision/sklearn MNIST-like sources and OPMNIST/SCR protocol flags,
  but its learner path is explicitly wired through `step2_universal_portfolio.py`.
  Its portfolio metrics are therefore not used as single-learner proof here.

## Commands

Smoke, real CIFAR wiring:

```bash
.venv/bin/python output/subagents/external_breadth_image/run_no_portfolio_image_breadth.py \
  --scenarios cifar \
  --steps 20 \
  --n-seeds 1 \
  --final-window 5 \
  --cifar-max-train 100 \
  --cifar-max-test 50 \
  --cifar-regimes iid \
  --output-dir output/subagents/external_breadth_image/smoke \
  --result-prefix smoke
```

Smoke, cached OpenML MNIST OPMNIST path:

```bash
.venv/bin/python output/subagents/external_breadth_image/run_no_portfolio_image_breadth.py \
  --scenarios opmnist \
  --steps 20 \
  --n-seeds 1 \
  --final-window 5 \
  --mnist-allow-openml-download \
  --mnist-max-train 200 \
  --mnist-max-test 80 \
  --mnist-n-permutations 2 \
  --mnist-task-block-size 10 \
  --output-dir output/subagents/external_breadth_image/smoke_opmnist \
  --result-prefix smoke_opmnist
```

Main bounded run:

```bash
.venv/bin/python output/subagents/external_breadth_image/run_no_portfolio_image_breadth.py \
  --scenarios cifar,opmnist \
  --steps 600 \
  --n-seeds 3 \
  --final-window 200 \
  --cifar-max-train 2000 \
  --cifar-max-test 500 \
  --cifar-regimes iid,class_blocked \
  --mnist-allow-openml-download \
  --mnist-max-train 2000 \
  --mnist-max-test 500 \
  --mnist-n-permutations 5 \
  --mnist-task-block-size 200 \
  --output-dir output/subagents/external_breadth_image/main_3seed_600 \
  --result-prefix no_portfolio_image_breadth_3seed_600
```

## Data

- CIFAR source: `data/cifar-10-batches-py`, loaded by the existing direct archive
  path as `cifar10_python_archive`; `real_cifar=true`.
- OPMNIST source: cached OpenML `mnist_784`, loaded from
  `outputs/step2_published_mnist_openml_cache`; `is_true_mnist=true`.
- OPMNIST was compact, not published-scale: 2,000 train examples, 500 test
  examples, 5 configured permutations, 600 online steps, 200-step task blocks,
  and 3 observed permutation views.

## Smoke Results

| Scenario | UPGD final acc | Best MLP final acc | UPGD test acc | Best MLP test acc | Notes |
| --- | ---: | ---: | ---: | ---: | --- |
| CIFAR iid, 20 steps | 0.4000 | 0.4000 | 0.1000 | 0.2200 | Real CIFAR loader smoke passed; UPGD lost held-out accuracy. |
| OPMNIST, 20 steps | 0.4000 | 0.4000 | 0.1313 | 0.2813 | Cached OpenML path passed; UPGD lost held-out accuracy. |

Smoke results validate wiring only.

## Main Results

Positive paired differences favor UPGD. MSE and NLL are lower-is-better;
accuracy is higher-is-better.

| Scenario | UPGD final MSE | Best MLP final MSE | Diff | UPGD final NLL | Best MLP final NLL | Diff | UPGD test acc | Best MLP test acc | Diff |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| CIFAR iid | 0.0857 | 0.1073 | +0.0215 | 2.1172 | 2.2243 | +0.1071 | 0.2440 | 0.1633 | +0.0807 |
| CIFAR class-blocked | 0.0686 | 0.0020 | -0.0666 | 1.5800 | 1.4661 | -0.1138 | 0.1673 | 0.1180 | +0.0493 |
| OPMNIST compact | 0.0732 | 0.0781 | +0.0050 | 1.7385 | 2.0823 | +0.3438 | 0.5911 | 0.4842 | +0.1069 |

Win/loss/tie counts against the best MLP:

| Scenario | Final MSE | Final NLL | Test accuracy |
| --- | ---: | ---: | ---: |
| CIFAR iid | 3/0/0 | 3/0/0 | 3/0/0 |
| CIFAR class-blocked | 0/3/0 | 1/2/0 | 3/0/0 |
| OPMNIST compact | 3/0/0 | 3/0/0 | 3/0/0 |

Mean online update times:

| Scenario | UPGD seconds | MLP h32 seconds | MLP h64 seconds |
| --- | ---: | ---: | ---: |
| CIFAR iid | 3.62 | 4.17 | 3.52 |
| CIFAR class-blocked | 2.43 | 2.61 | 2.55 |
| OPMNIST compact | 2.72 | 3.19 | 2.04 |

Raw results are in
`output/subagents/external_breadth_image/main_3seed_600/no_portfolio_image_breadth_3seed_600_results.json`.

## Interpretation

The no-portfolio promoted UPGD learner is competitive to better on this bounded
real-image probe:

- Better than the best MLP on all tracked CIFAR iid metrics.
- Better than the best MLP on compact OpenML OPMNIST final-window MSE, NLL, and
  held-out accuracy.
- Mixed on CIFAR class-blocked: UPGD has better held-out accuracy, but clearly
  worse final-window online MSE and NLL because the MLPs overfit/current-block
  track almost perfectly.

This is real image evidence, but it is not enough for a strict paper-quality
external-breadth proof. The run is only 3 seeds, 600 steps, 2,000 train examples,
and 500 held-out examples. OPMNIST is true MNIST, but not full 60k/10k and not
the Dohare 800-task published-scale protocol.

## Blockers And Caveats

- The published-stressor runner cannot be reported as single-learner evidence as
  written because it routes through `step2_universal_portfolio.py`.
- `torchvision` and `torch` are not installed in the local virtualenv, so the
  torchvision MNIST path is unavailable.
- OpenML MNIST was available through the local sklearn cache, but the loader
  still requires the download permission flag because the underlying script
  guards `--mnist-source openml` that way.
- CIFAR loading used the local Python archive and emitted a NumPy 2.4 pickle
  deprecation warning from the existing loader; it did not block the run.
- No core learner files or existing Step 2 scripts were edited.

## Next Strongest Experiment

Run the same no-portfolio harness at a stronger but still bounded scale:
10 seeds, 5,000 to 10,000 online steps, full local CIFAR train/test subsets, and
OpenML MNIST with more observed permutation blocks. Keep `UPGDLearner.step2_default`
fixed, keep the MLP comparator set fixed, and pre-register primary metrics as
final-window NLL plus held-out accuracy. A later published-scale OPMNIST claim
requires a minimal non-portfolio mode in `step2_published_stressors.py` or a
separate streaming no-portfolio runner for the full 60k block protocol.
