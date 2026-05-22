# Step 2 Image Block Promotion Plan

Date: 2026-05-07.

This note assesses the current best Step 2 learners as single blocks for
MNIST-like and CIFAR-style supervised streams.  The promotion target is not a
portfolio.  A promoted block must update causally on every step, keep a fixed
resource budget, expose constructed features or predictions through a stable
API, and beat the strongest same-run MLP comparator before moving to larger
OPMNIST runs.

## Current Block Ranking

| Rank | Block | Status | Why |
|---:|---|---|---|
| 1 | `step2_hybrid_memory_trace` | Primary image candidate | It is already packaged in `UPGDMemoryLearner`: target-structure UPGD plus fixed-budget prototype memory, both updated every step, with learned scalar memory mixing. It beats fair MLPs on real CIFAR iid and on true-MNIST OPMNIST partial runs. |
| 2 | `proto_mem_s20/s32` | Standalone retention candidate | D24 shows large wins over MLP on compact permuted sklearn-digits-28x28. It is simple, fast, and resource explicit. It is classification-specific and must now prove itself on real CIFAR and true MNIST before OPMNIST promotion. |
| 3 | `step2_hybrid_memory_trace_sharp` / adaptive sharp | Deployment-readout ablation | Sharpening fixes class-blocked final-window MSE on CIFAR, but hurts iid MSE when always on. Keep as explicit ablation or adaptive gate, not as default. |
| 4 | `centroid_hysteretic64_center_c030` / fixed scalar centroid | Class-blocked diagnostic | Useful for understanding anti-drift, but the OPMNIST 1 percent candidate run loses to the hybrid and to best MLP on several held-out metrics. Do not promote as an image default. |
| 5 | `upgd_step2_default` | Differentiable baseline | Stronger than MLP on some CIFAR iid metrics and external rows, but still weak on class-blocked tracking and retention alone. Use as baseline and as the UPGD half of the hybrid. |
| 6 | `PrototypeBasisBlock` | Universal substrate only | It is the clean recursive block abstraction, but D24 basis variants did not beat MLP online. Keep it for recursive-feature and transformer experiments, not current image promotion. |
| 7 | learned centroid mix/temperature | Ablation only | Capped learned mix helps class-blocked MSE but gives back label-drift performance. The signal is real but not robust enough to replace fixed defaults. |
| 8 | regression local target prototypes | Non-image branch | Promising on diabetes regression, neutral on Friedman1, and not relevant to CIFAR/MNIST classification promotion except as evidence that prototype charts can generalize beyond one-hot tasks. |

## Existing Evidence

### MNIST-like / Compact OPMNIST

D24 (`outputs/step2_new_directions/d24_prototype_universal_blocks_local3/`) is the strongest standalone prototype-memory evidence:

| Method | Final MSE | Final Acc | Test MSE | Test Acc | Steps/s |
|---|---:|---:|---:|---:|---:|
| best MLP, `mlp_h128` | `0.082160` | `0.505000` | `0.080295` | `0.552500` | `1750` |
| `proto_mem_s20` | `0.026361` | `0.825000` | `0.013433` | `0.907333` | `4399` |
| `proto_mem_s32` | `0.026214` | `0.826667` | `0.011955` | `0.917000` | `4699` |

Interpretation: `proto_mem_s32` is not just better than MLP here; it is faster
and substantially more accurate.  The weakness is scope: this is sklearn digits
upsampled to 28x28 and five permutation tasks, not full true MNIST OPMNIST.

True-MNIST compact hybrid evidence:

- `outputs/step2_upgd_memory_opmnist_true_mnist_compact/`: at 1000 steps,
  `step2_hybrid_memory_trace` beats `mlp_h64/h128` on final MSE, final accuracy,
  test MSE, and test accuracy.
- `outputs/step2_upgd_memory_opmnist_published_scale/`: at 10 full true-MNIST
  OPMNIST blocks, the hybrid beats `mlp_h64/h128` on final MSE and held-out
  accuracy. This is now historical early-scale evidence; the later
  latest-best run completed 800/800 blocks and kept the online/final-MSE win
  while losing retained all-permutation test metrics.

### Real CIFAR-10

Current real-CIFAR 3-seed, 2000-step artifacts test `step2_hybrid_memory_trace`,
plain UPGD, and MLP h32/h64:

| Regime | Hybrid final MSE vs best MLP | Hybrid final acc vs best MLP | Hybrid held-out acc vs best MLP | Assessment |
|---|---:|---:|---:|---|
| iid | `+0.027387`, 3/0 wins | `+0.145000`, 3/0 wins | `+0.043333`, 3/0 wins | Pass against current MLP set. |
| class-blocked | `-0.002774`, 0/3 wins | `+0.004167`, 3/0 wins | `+0.091333`, 3/0 wins | Mixed: tracking accuracy and retention pass, raw MSE fails. |

Sharpened hybrid fixes the class-blocked final-window MSE against sharpened MLP
in the existing probe (`+0.000717`, 3/0 wins) but worsens iid MSE.  This makes
sharpening a conditional or learned-readout lead, not a universal default.

The old CIFAR runner did not test `proto_mem_s20/s32` and did not include MLP
h128/h256/h128_128.  That hole is now fixed in the runner, but real-CIFAR
results with these blocks still need to be run.

### OPMNIST Boundary

Existing true-MNIST OPMNIST scale artifacts are positive for the hybrid but not
decisive:

- `step2_hybrid_memory_trace` has strong 1-block, 8-block, and 10-block
  evidence against MLP h64/h128.
- A separate 47-block strict portfolio artifact is not a single-learner proof
  and does not clear all external claims.
- Full published-scale OPMNIST is 800 tasks x 60,000 examples = 48,000,000
  online updates. Current long-running split jobs are still partial snapshots.

## Runner Changes

The image runners now expose the promising blocks explicitly:

- CIFAR runner: `--include-prototype-memory` adds `proto_mem_s20` and
  `proto_mem_s32`; `--include-wide-mlp` adds `mlp_h128`, `mlp_h256`, and
  `mlp_h128_128`.
- OPMNIST runner: `--include-prototype-memory` adds standalone `proto_mem_s20`
  and `proto_mem_s32`
  to the existing hybrid, MLP, sharpened MLP, and centroid candidate set.

Smoke validation:

```bash
.venv/bin/python -m pytest tests/test_step2_cifar_stream.py \
  tests/test_step2_upgd_memory_opmnist.py tests/test_upgd_memory.py \
  tests/test_prototype_memory.py -q
```

Result: `29 passed`.

```bash
.venv/bin/python "examples/The Alberta Plan/Step2/step2_cifar_stream.py" \
  --steps 12 --n-seeds 1 --final-window 4 --max-train 40 --max-test 20 \
  --data-dir /tmp/alberta-framework-no-cifar --regimes iid class_blocked \
  --include-prototype-memory --include-wide-mlp \
  --include-primary-sharpened --include-adaptive-primary-sharpened \
  --include-sharpened-mlp \
  --output-dir outputs/step2_image_block_promotion_smoke \
  --result-prefix cifar_block_smoke \
  --note-path docs/research/step2_image_block_promotion_cifar_smoke.md
```

Result: wiring smoke passed and wrote
`outputs/step2_image_block_promotion_smoke/cifar_block_smoke_results.json`.
This is synthetic fallback only, not CIFAR evidence.

## Promotion Gates

### Gate A: MNIST-like Block Gate

Run D24-style compact OPMNIST with `proto_mem_s20`, `proto_mem_s32`, the hybrid,
UPGD, and MLP h64/h128/h64_64.  Require:

- 3 paired seeds minimum;
- final-window MSE and accuracy better than best MLP on at least 2/3 seeds;
- held-out test accuracy better than best MLP on at least 2/3 seeds;
- state budget reported in floats and prototype slots;
- no test-set tuning.

### Gate B: Real CIFAR Block Gate

Run real CIFAR-10 with the now-expanded runner:

```bash
.venv/bin/python "examples/The Alberta Plan/Step2/step2_cifar_stream.py" \
  --steps 2000 --n-seeds 3 --final-window 400 \
  --max-train 3000 --max-test 1000 --data-dir data \
  --regimes iid class_blocked \
  --include-prototype-memory --include-wide-mlp \
  --include-primary-sharpened --include-adaptive-primary-sharpened \
  --include-sharpened-mlp \
  --output-dir outputs/step2_image_block_promotion_cifar_real_3seed_2000 \
  --result-prefix cifar_block_promotion_real_3seed_2000 \
  --note-path docs/research/step2_image_block_promotion_cifar_real_3seed_2000.md
```

Require:

- compare to best MLP across h32/h64/h128/h256/h128_128 and sharpened h32/h64;
- report iid and class-blocked separately;
- a candidate may pass class-blocked on tracking accuracy plus held-out
  accuracy even if raw MSE is worse only when a predeclared sharpened/adaptive
  readout also clears MSE;
- if `proto_mem_s20` matches `proto_mem_s32` within noise, prefer s20 for lower
  memory.

### Gate C: True-MNIST OPMNIST Promotion

Only blocks passing Gate A and not failing Gate B should move to true-MNIST
OPMNIST:

```bash
.venv/bin/python "examples/The Alberta Plan/Step2/step2_upgd_memory_opmnist.py" \
  --mnist-published-scale --allow-openml-download --opmnist-fraction 0.01 \
  --include-prototype-memory --include-primary-sharpened \
  --include-adaptive-primary-sharpened --include-sharpened-mlp \
  --output-dir outputs/step2_image_block_promotion_opmnist_1pct \
  --result-prefix opmnist_block_promotion_1pct \
  --note-path docs/research/step2_image_block_promotion_opmnist_1pct.md \
  --force-restart
```

If the 1 percent run passes, scale to 10, 40, then 800 blocks.  Do not describe
anything below 800 complete 60k-example tasks as a full published-scale OPMNIST
solution.

Post-completion update: the latest-best UPGD-memory/MLP comparison has now
reached 800 complete task blocks for one seed. Gate C is therefore no longer a
task-count gate; it is a retained-view gate, because the completed run still
loses all-permutation held-out test metrics to the best fair MLP.

## Critical Read

The best current image answer is the hybrid block, not pure UPGD, standalone
prototype memory, or the universal prototype basis.  The strongest scientific
risk is that the winning mechanism is partly nearest-neighbor retention rather
than general recursive feature discovery.  That is still Step 2 relevant
because it constructs and maintains features under a fixed budget, but it is
classification-biased and does not close arbitrary vector-target representation
learning.

## Executed Image Gates

### True-MNIST OPMNIST Compact Gate

Artifact:

- `outputs/step2_image_block_promotion_opmnist_compact/opmnist_block_promotion_compact_1seed_2000_results.json`
- `docs/research/step2_image_block_promotion_opmnist_compact_1seed_2000.md`

Protocol: true OpenML MNIST, 1 seed, 2000 online updates, 5 permutation blocks,
400 examples per block, 5 held-out permutation views.  This is an OPMNIST
functionality and early-signal gate, not published-scale OPMNIST.

| Method | Final MSE | Final Acc | Test MSE | Test Acc |
|---|---:|---:|---:|---:|
| `step2_hybrid_memory_trace` | `0.043549` | `0.717500` | `0.034687` | `0.774600` |
| `step2_hybrid_memory_trace_adaptive_sharp` | `0.043388` | `0.710000` | `0.038821` | `0.769600` |
| `proto_mem_s20` | `0.047559` | `0.652500` | `0.040776` | `0.718200` |
| `proto_mem_s32` | `0.044059` | `0.682500` | `0.039611` | `0.727000` |
| `mlp_h64` | `0.070756` | `0.537500` | `0.077155` | `0.378400` |
| `mlp_h128` | `0.075977` | `0.547500` | `0.083658` | `0.360800` |

Result: all non-MLP candidates beat the best MLP on all reported metrics in
this compact true-MNIST OPMNIST run.  The hybrid remains best overall.

### Real CIFAR-10 Expanded Gate

Artifact:

- `outputs/step2_image_block_promotion_cifar_real_3seed_2000/cifar_block_promotion_real_3seed_2000_results.json`
- `docs/research/step2_image_block_promotion_cifar_real_3seed_2000.md`

Protocol: real CIFAR-10 Python archive, 3 paired seeds, 2000 updates, final
window 400, 3000 train examples, 1000 held-out examples.  Baselines include
MLP h32/h64/h128/h256/h128_128 plus sharpened h32/h64.

Positive values favor the candidate versus the best same-run MLP for that
metric.

| Regime | Method | Final MSE diff | Final Acc diff | Test MSE diff | Test Acc diff | Decision |
|---|---|---:|---:|---:|---:|---|
| iid | `step2_hybrid_memory_trace` | `+0.027546` | `+0.131667` | `+0.021563` | `+0.044333` | Pass |
| iid | `step2_hybrid_memory_trace_adaptive_sharp` | `+0.027546` | `+0.131667` | `+0.021563` | `+0.044333` | Pass |
| iid | `upgd_step2_default` | `+0.020877` | `+0.130833` | `+0.016979` | `+0.079000` | Pass as baseline, not retention solution |
| iid | `proto_mem_s20` | `-0.032103` | `+0.050833` | `-0.042578` | `+0.000333` | Fail |
| iid | `proto_mem_s32` | `-0.025293` | `+0.077500` | `-0.039616` | `+0.008000` | Fail |
| class-blocked | `step2_hybrid_memory_trace` | `-0.003201` | `+0.000833` | `+0.067622` | `+0.084667` | Mixed |
| class-blocked | `step2_hybrid_memory_trace_adaptive_sharp` | `+0.000833` | `+0.000833` | `+0.022833` | `+0.084667` | Pass |
| class-blocked | `proto_mem_s20` | `-0.152098` | `-0.811667` | `+0.024765` | `+0.073667` | Fail tracking |
| class-blocked | `proto_mem_s32` | `-0.155313` | `-0.828333` | `+0.026569` | `+0.075000` | Fail tracking |

Result: standalone prototype memory should not be promoted to real-CIFAR image
default.  It is useful for compact OPMNIST retention but collapses current-block
tracking on CIFAR class-blocked streams and loses iid CIFAR MSE.  The best
single image block is `step2_hybrid_memory_trace_adaptive_sharp`: it clears iid
and class-blocked CIFAR against the expanded MLP set and also clears the compact
true-MNIST OPMNIST gate.

## Updated Decision

Promote for the next OPMNIST scale step:

- `step2_hybrid_memory_trace`
- `step2_hybrid_memory_trace_adaptive_sharp`

Keep as OPMNIST-specific ablations, not defaults:

- `proto_mem_s20`
- `proto_mem_s32`

Do not move standalone prototype memory to production image default unless a
new mechanism fixes CIFAR current-block tracking without sacrificing its
OPMNIST retention advantage.

## Follow-up Runs

### CIFAR Hybrid-Focused 5-Seed Confirmation

Artifact:

- `outputs/step2_image_block_promotion_cifar_focused_5seed_2000/cifar_hybrid_focused_5seed_2000_results.json`
- `docs/research/step2_image_block_promotion_cifar_hybrid_focused_5seed_2000.md`

Protocol: real CIFAR-10 Python archive, 5 paired seeds, 2000 updates, final
window 400, 3000 train examples, 1000 held-out examples.  Methods were focused
to raw hybrid, adaptive-sharp hybrid, always-sharp hybrid, and expanded MLP
comparators.

| Regime | Method | Final MSE diff | Test MSE diff | Test Acc diff | Wins |
|---|---|---:|---:|---:|---|
| iid | `step2_hybrid_memory_trace` | `+0.027650` | `+0.022337` | `+0.055000` | 5/5 final MSE, 5/5 test acc |
| iid | `step2_hybrid_memory_trace_adaptive_sharp` | `+0.027650` | `+0.022337` | `+0.055000` | 5/5 final MSE, 5/5 test acc |
| iid | `step2_hybrid_memory_trace_sharp` | `-0.012567` | `-0.027502` | `+0.055000` | 0/5 final MSE, 5/5 test acc |
| class-blocked | `step2_hybrid_memory_trace` | `-0.003134` | `+0.065495` | `+0.078600` | 0/5 final MSE, 5/5 test acc |
| class-blocked | `step2_hybrid_memory_trace_adaptive_sharp` | `+0.000800` | `+0.021292` | `+0.078600` | 5/5 final MSE, 5/5 test acc |
| class-blocked | `step2_hybrid_memory_trace_sharp` | `+0.000800` | `+0.021292` | `+0.078600` | 5/5 final MSE, 5/5 test acc |

Decision: `step2_hybrid_memory_trace_adaptive_sharp` is the promoted CIFAR
winner.  It behaves like raw hybrid on iid CIFAR, and like sharpened hybrid
when class-blocked confidence makes sharpening useful.  Always-sharp is not
acceptable as the default because it loses iid MSE.

### OPMNIST Hybrid-Only 1 Percent and 10-Block Runs

Artifacts:

- `outputs/step2_image_block_promotion_opmnist_1pct_hybrid_only/opmnist_hybrid_only_1pct_results.json`
- `docs/research/step2_image_block_promotion_opmnist_1pct_hybrid_only.md`
- `outputs/step2_image_block_promotion_opmnist_10block_hybrid_only/opmnist_hybrid_only_10block_results.json`
- `docs/research/step2_image_block_promotion_opmnist_10block_hybrid_only.md`

Protocol: true OpenML MNIST, 800 permutations, 60,000-example task blocks,
single seed, hybrid-only candidate set.  The 1 percent run is 480,000 updates
or 8 complete task blocks; the 10-block run is 600,000 updates.

| Scale | Method | Final MSE | Final Acc | Test MSE | Test Acc |
|---|---|---:|---:|---:|---:|
| 1 percent / 8 blocks | `step2_hybrid_memory_trace` | `0.008751` | `0.944800` | `0.044436` | `0.693912` |
| 1 percent / 8 blocks | `step2_hybrid_memory_trace_adaptive_sharp` | `0.008814` | `0.944000` | `0.045496` | `0.684150` |
| 10 blocks | `step2_hybrid_memory_trace` | `0.008136` | `0.950200` | `0.052534` | `0.642940` |
| 10 blocks | `step2_hybrid_memory_trace_adaptive_sharp` | `0.008059` | `0.949400` | `0.054608` | `0.630040` |

Decision: raw hybrid remains the stronger OPMNIST held-out performer at these
scales, while adaptive-sharp remains the stronger CIFAR-safe learner.  The
current best presentation claim is therefore:

- `step2_hybrid_memory_trace_adaptive_sharp` for CIFAR-safe image deployment;
- `step2_hybrid_memory_trace` as the OPMNIST scale-front runner;
- both are the same underlying Step 2 hybrid learner, differing only in a
  causal deployment-readout gate.
