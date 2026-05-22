# Step 2 UPGD-Memory Image-Scale Assessment

This note records the May 2026 image-scale pass for the packaged single
`UPGDMemoryLearner` default.  The goal was to move beyond digit stressors
without reverting to a portfolio/router: one UPGD path plus one fixed-budget
prototype memory, both updated every step, with the update-time target-trace
prior enabled.

## Runner Changes

- `examples/The Alberta Plan/Step2/step2_cifar_stream.py` now treats
  `step2_hybrid_memory_trace` as the primary method and keeps
  `upgd_step2_default`, `mlp_h32`, and `mlp_h64` as same-run baselines.
- `examples/The Alberta Plan/Step2/step2_upgd_memory_opmnist.py` adds a
  chunked, resumable, single-learner OPMNIST runner.  It trains
  `step2_hybrid_memory_trace`, `mlp_h64`, and `mlp_h128` from identical stream
  chunks, writes atomic checkpoints, and records progress/status sidecars.
- The same OPMNIST runner now has an explicit 1% published-scale screening
  mode through `--opmnist-fraction 0.01`, plus opt-in sharpened MLP and
  centroid candidate readouts for cheap final-window MSE triage before running
  full 800-task candidates.
- `examples/The Alberta Plan/Step2/new_directions/d22_upgd_prototype_hybrid_opmnist.py`
  now includes the exact promoted core variant:
  `core_upgdmem_h64_s20_alloc18_mem0_trace80_thr50`.

## True OpenML MNIST Compact OPMNIST

Command:

```bash
source .venv/bin/activate && python "examples/The Alberta Plan/Step2/step2_upgd_memory_opmnist.py" \
  --mnist-source openml \
  --allow-openml-download \
  --mnist-split canonical \
  --steps 1000 \
  --n-seeds 1 \
  --final-window 200 \
  --max-train-examples 2000 \
  --max-test-examples 500 \
  --n-permutations 5 \
  --task-block-size 200 \
  --chunk-size 200 \
  --max-test-permutation-views 5 \
  --output-dir outputs/step2_upgd_memory_opmnist_true_mnist_compact \
  --result-prefix true_mnist_compact_1seed_1000 \
  --note-path docs/research/step2_upgd_memory_opmnist_true_mnist_compact_1seed_1000.md \
  --force-restart
```

Result path:
`outputs/step2_upgd_memory_opmnist_true_mnist_compact/true_mnist_compact_1seed_1000_results.json`.

| Method | Final MSE | Final Acc | Test MSE | Test Acc |
|---|---:|---:|---:|---:|
| `step2_hybrid_memory_trace` | 0.050627 | 0.660000 | 0.045636 | 0.694400 |
| `mlp_h64` | 0.078163 | 0.495000 | 0.082736 | 0.354800 |
| `mlp_h128` | 0.093170 | 0.450000 | 0.090014 | 0.352000 |

The primary learner beats the best same-run MLP on all six tracked online,
final-window, and held-out metrics.

## Published-Protocol OPMNIST, 10 Blocks

Command:

```bash
source .venv/bin/activate && python "examples/The Alberta Plan/Step2/step2_upgd_memory_opmnist.py" \
  --mnist-published-scale \
  --allow-openml-download \
  --steps 600000 \
  --n-seeds 1 \
  --final-window 5000 \
  --chunk-size 6000 \
  --max-test-permutation-views 10 \
  --output-dir outputs/step2_upgd_memory_opmnist_published_scale \
  --result-prefix upgdmem_opmnist_1block_1seed \
  --note-path docs/research/step2_upgd_memory_opmnist_10block_1seed.md \
  --status-path outputs/step2_upgd_memory_opmnist_published_scale/upgdmem_opmnist_10block_1seed_status.json
```

Result paths:

- `outputs/step2_upgd_memory_opmnist_published_scale/upgdmem_opmnist_10block_1seed_results.json`
- `outputs/step2_upgd_memory_opmnist_published_scale/upgdmem_opmnist_10block_1seed_status.json`
- checkpoint:
  `outputs/step2_upgd_memory_opmnist_published_scale/upgdmem_opmnist_1block_1seed_seed0_resume.pkl`

Protocol status:

- true OpenML MNIST source: yes;
- full canonical 60,000/10,000 MNIST split: yes;
- configured OPMNIST tasks: 800;
- task block size: 60,000;
- completed examples: 600,000;
- completed full task blocks: 10/800;
- held-out permutation views evaluated: 10.

| Method | Online MSE | Online Acc | Final MSE | Final Acc | Test MSE | Test Acc |
|---|---:|---:|---:|---:|---:|---:|
| `step2_hybrid_memory_trace` | 0.012373 | 0.919703 | 0.008422 | 0.948400 | 0.051540 | 0.650970 |
| `mlp_h64` | 0.022846 | 0.901603 | 0.017371 | 0.931200 | 0.077128 | 0.398200 |
| `mlp_h128` | 0.022589 | 0.906392 | 0.016182 | 0.936800 | 0.075334 | 0.417370 |

Primary-vs-best-MLP paired deltas, positive favoring the primary learner:

| Metric | Best MLP | Delta |
|---|---|---:|
| Online MSE | `mlp_h128` | +0.010216 |
| Online accuracy | `mlp_h128` | +0.013312 |
| Final-window MSE | `mlp_h128` | +0.007760 |
| Final-window accuracy | `mlp_h128` | +0.011600 |
| Held-out MSE | `mlp_h128` | +0.023794 |
| Held-out accuracy | `mlp_h128` | +0.233600 |

The 10-block result is a strong scale signal for the single UPGD-memory learner.
It has now been superseded by the completed latest-best 800-block run in
`docs/research/step2_upgd_memory_opmnist_latest_best_800block_1seed.md`.
At full task count, UPGD-memory still wins online MSE, online accuracy, and
final-window MSE against the same-run best fair MLP, but no longer wins the
all-permutation held-out test metrics.

## Published-Protocol OPMNIST, 1% Candidate Screen

Command:

```bash
source .venv/bin/activate && python -u "examples/The Alberta Plan/Step2/step2_upgd_memory_opmnist.py" \
  --mnist-published-scale \
  --allow-openml-download \
  --opmnist-fraction 0.01 \
  --n-seeds 1 \
  --final-window 5000 \
  --chunk-size 60000 \
  --max-test-permutation-views 8 \
  --include-centroid-candidates \
  --output-dir outputs/step2_upgd_memory_opmnist_1pct_candidates \
  --result-prefix opmnist_1pct_centroid_candidates \
  --note-path docs/research/step2_upgd_memory_opmnist_1pct_centroid_candidates.md \
  --status-path outputs/step2_upgd_memory_opmnist_1pct_candidates/opmnist_1pct_centroid_candidates_status.json \
  --force-restart
```

Result path:
`outputs/step2_upgd_memory_opmnist_1pct_candidates/opmnist_1pct_centroid_candidates_results.json`.

Protocol status:

- true OpenML MNIST source: yes;
- full canonical 60,000/10,000 MNIST split: yes;
- configured OPMNIST tasks: 800;
- task block size: 60,000;
- completed examples: 480,000;
- completed full task blocks: 8/800;
- held-out permutation views evaluated: 8.

| Method | Online MSE | Online Acc | Final MSE | Final Acc | Test MSE | Test Acc |
|---|---:|---:|---:|---:|---:|---:|
| `step2_hybrid_memory_trace` | 0.012347 | 0.919706 | 0.008546 | 0.944200 | 0.044990 | 0.691950 |
| `mlp_h64` | 0.022729 | 0.902352 | 0.017364 | 0.929800 | 0.082161 | 0.401650 |
| `mlp_h128` | 0.022547 | 0.906617 | 0.016728 | 0.939200 | 0.086684 | 0.405100 |
| `mlp_h64_sharp` | 0.016703 | 0.902125 | 0.011707 | 0.931400 | 0.084441 | 0.457712 |
| `mlp_h128_sharp` | 0.016111 | 0.905590 | 0.010612 | 0.940600 | 0.128652 | 0.320537 |
| `centroid_hysteretic64_center_c030` | 0.014424 | 0.916112 | 0.010790 | 0.938800 | 0.101122 | 0.348725 |

Primary-vs-best-MLP paired deltas, positive favoring the primary learner:

| Metric | Best MLP | Delta |
|---|---|---:|
| Online MSE | `mlp_h128_sharp` | +0.003764 |
| Online accuracy | `mlp_h128` | +0.013090 |
| Final-window MSE | `mlp_h128_sharp` | +0.002066 |
| Final-window accuracy | `mlp_h128_sharp` | +0.003600 |
| Held-out MSE | `mlp_h64` | +0.037170 |
| Held-out accuracy | `mlp_h64_sharp` | +0.234237 |

Centroid candidate reading:

- The one-centroid hysteretic residual readout is useful enough to keep as a
  diagnostic: it beats the sharpened best MLP on online MSE and online
  accuracy.
- It should not be promoted for OPMNIST.  It loses final-window MSE by
  `0.000178` against the best sharpened MLP and loses held-out test MSE and
  held-out accuracy by much larger margins.
- For this stream, the currently promoted `step2_hybrid_memory_trace` is the
  better simple learner: it beats every MLP comparator and the centroid
  candidate on all tracked metrics.

### Sharpened Primary Readout Screen

Command:

```bash
source .venv/bin/activate && python -u "examples/The Alberta Plan/Step2/step2_upgd_memory_opmnist.py" \
  --mnist-published-scale \
  --allow-openml-download \
  --opmnist-fraction 0.01 \
  --n-seeds 1 \
  --final-window 5000 \
  --chunk-size 60000 \
  --max-test-permutation-views 8 \
  --include-primary-sharpened \
  --include-sharpened-mlp \
  --output-dir outputs/step2_upgd_memory_opmnist_1pct_primary_sharp \
  --result-prefix opmnist_1pct_primary_sharp \
  --note-path docs/research/step2_upgd_memory_opmnist_1pct_primary_sharp.md \
  --status-path outputs/step2_upgd_memory_opmnist_1pct_primary_sharp/opmnist_1pct_primary_sharp_status.json \
  --force-restart
```

Result path:
`outputs/step2_upgd_memory_opmnist_1pct_primary_sharp/opmnist_1pct_primary_sharp_results.json`.

| Method | Online MSE | Online Acc | Final MSE | Final Acc | Test MSE | Test Acc |
|---|---:|---:|---:|---:|---:|---:|
| `step2_hybrid_memory_trace` | 0.012347 | 0.919706 | 0.008546 | 0.944200 | 0.044990 | 0.691950 |
| `step2_hybrid_memory_trace_sharp` | 0.015230 | 0.920015 | 0.010566 | 0.945000 | 0.057244 | 0.700187 |
| Best fair MLP | 0.016063 | 0.906492 | 0.011130 | 0.939200 | 0.075937 | 0.429425 |

This rejects confidence sharpening as the final-window MSE fix.  It improves
accuracy slightly, including held-out accuracy, but raises online, final-window,
and held-out MSE relative to the raw `step2_hybrid_memory_trace` predictions.
For MSE-focused claims, the calibrated raw readout should remain canonical.

## CIFAR-10

Command:

```bash
source .venv/bin/activate && python "examples/The Alberta Plan/Step2/step2_cifar_stream.py" \
  --steps 2000 \
  --n-seeds 3 \
  --final-window 400 \
  --max-train 3000 \
  --max-test 1000 \
  --allow-download \
  --data-dir data \
  --output-dir outputs/step2_cifar_stream_upgdmem_real \
  --result-prefix cifar_upgdmem_real_3seed_2000 \
  --note-path docs/research/step2_cifar_stream_upgdmem_real_3seed_2000.md
```

Result path:
`outputs/step2_cifar_stream_upgdmem_real/cifar_upgdmem_real_3seed_2000_results.json`.

Real CIFAR-10 source was the local public Python archive:
`cifar10_python_archive`, with 3,000 train examples and 1,000 held-out test
examples.

### IID CIFAR

| Method | Final MSE | Final Acc | Test MSE | Test Acc |
|---|---:|---:|---:|---:|
| `step2_hybrid_memory_trace` | 0.078847 | 0.344167 | 0.085905 | 0.239667 |
| `upgd_step2_default` | 0.086024 | 0.318333 | 0.090538 | 0.261667 |
| Best fair MLP | 0.106235 | 0.199167 | 0.108111 | 0.196333 |

The primary learner beats the best fair MLP on all four tracked metrics across
3/3 paired seeds.  Plain UPGD has slightly higher held-out accuracy than the
hybrid on this small IID CIFAR slice, so this row argues for keeping both the
plain UPGD and memory traces visible in ablation tables.

### Class-Blocked CIFAR

| Method | Final MSE | Final Acc | Test MSE | Test Acc |
|---|---:|---:|---:|---:|
| `step2_hybrid_memory_trace` | 0.004404 | 0.997500 | 0.114472 | 0.186333 |
| `upgd_step2_default` | 0.015336 | 0.919167 | 0.171924 | 0.095333 |
| Best fair MLP | 0.001631 | 0.993333 | 0.175533 | 0.095000 |

The primary learner wins final-window accuracy, held-out MSE, and held-out
accuracy across 3/3 paired seeds, but loses final-window MSE to the best fair
MLP.  This is not a deployment failure: the MLP has very low current-block MSE
while retaining almost no balanced held-out class knowledge.  It is still a
metric caveat for any blanket "wins every metric" claim.

### CIFAR Adaptive Readout Probe

Command:

```bash
source .venv/bin/activate && python -u "examples/The Alberta Plan/Step2/step2_cifar_stream.py" \
  --steps 2000 \
  --n-seeds 3 \
  --final-window 400 \
  --max-train 3000 \
  --max-test 1000 \
  --allow-download \
  --data-dir data \
  --regimes iid class_blocked \
  --include-adaptive-primary-sharpened \
  --include-primary-sharpened \
  --include-sharpened-mlp \
  --output-dir outputs/step2_cifar_stream_upgdmem_adaptive_sharp_probe \
  --result-prefix cifar_upgdmem_adaptive_sharp_v2_2regime_3seed_2000 \
  --note-path docs/research/step2_cifar_stream_upgdmem_adaptive_sharp_v2_2regime_3seed_2000.md
```

Result path:
`outputs/step2_cifar_stream_upgdmem_adaptive_sharp_probe/cifar_upgdmem_adaptive_sharp_v2_2regime_3seed_2000_results.json`.

The adaptive readout is a scalar causal gate between the raw UPGD-memory
prediction and the same prediction after confidence sharpening.  It updates
from the recent one-step MSE advantage of sharpening and does not select among
separate feature learners.

| Regime | Method | Final MSE | Final Acc | Test MSE | Test Acc |
|---|---|---:|---:|---:|---:|
| IID | `step2_hybrid_memory_trace` | 0.078847 | 0.344167 | 0.085905 | 0.239667 |
| IID | `step2_hybrid_memory_trace_adaptive_sharp` | 0.078847 | 0.344167 | 0.085905 | 0.239667 |
| IID | Best fair MLP | 0.106235 | 0.199167 | 0.108111 | 0.196333 |
| Class-blocked | `step2_hybrid_memory_trace` | 0.004404 | 0.997500 | 0.114472 | 0.186333 |
| Class-blocked | `step2_hybrid_memory_trace_adaptive_sharp` | 0.000500 | 0.997500 | 0.157583 | 0.186333 |
| Class-blocked | Best fair MLP | 0.001217 | 0.993333 | 0.181000 | 0.095000 |

Paired deltas for `step2_hybrid_memory_trace_adaptive_sharp`, positive
favoring the adaptive readout:

| Regime | Final-window MSE | Final-window accuracy | Held-out MSE | Held-out accuracy |
|---|---:|---:|---:|---:|
| IID | +0.027387 | +0.145000 | +0.022207 | +0.043333 |
| Class-blocked | +0.000717 | +0.004167 | +0.017951 | +0.091333 |

This closes the CIFAR class-blocked final-window MSE caveat for the tested
3-seed, 2,000-step image-scale protocol.  The fixed sharpened readout is not
promotable by itself because it loses IID MSE; the adaptive readout matters
because it remains identical to the raw readout on IID and switches only when
the causal MSE evidence favors sharpening.

## Claim Boundary

The image-scale evidence now supports a stronger empirical claim:

> The single packaged UPGD-memory trace learner scales from digit stressors to
> true OpenML MNIST OPMNIST chunks and real CIFAR-10 streams, and it beats fair
> same-run MLP baselines on the most deployment-relevant held-out image metrics.

What is still not claimable:

- An unqualified OPMNIST win is still open: the final 800-task result exists,
  but fair MLP baselines still win final-window accuracy and all-permutation
  held-out test metrics.
- CIFAR class-blocked final-window MSE is closed by the adaptive raw/sharp
  readout on the current 3-seed, 2,000-step probe, but this should be rerun in
  the broader canonical image matrix before replacing the raw readout in the
  package default.
- This is empirical evidence, not a universality theorem.
