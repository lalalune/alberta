# Step 2 OPMNIST Candidate Screen: Latest Promising Learners

Date: 2026-05-08

## Why This Was Run

The latest full 800-task OPMNIST run closed the task-count scale gate for the
packaged UPGD-memory learner, but it also exposed a retained-view gap: the
UPGD-memory variants beat fair MLPs on online MSE, online accuracy, and
final-window MSE, while fair MLPs still won final-window accuracy and
all-permutation held-out test metrics.

I re-read the latest Step 2 notes and identified the OPMNIST-compatible
promising learners that were not all present in the full 800-block latest-best
merge:

- `step2_hybrid_memory_trace_sharp`: direct sharpened UPGD-memory readout.
- `centroid_hysteretic64_center_c030`: one-centroid-per-class hysteretic
  candidate from the class-blocked prototype work.
- `proto_mem_s20` and `proto_mem_s32`: fixed-budget JAX prototype memory,
  the production replacement for the old D20 multi-prototype OPMNIST path.

D18 remains rejected for the OPMNIST production path in the latest notes, and
the transformer/prototype-memory branches are sequence-model candidates rather
than direct image OPMNIST learners. I did not force those into this runner.

## Command

```bash
source .venv/bin/activate && python -u "examples/The Alberta Plan/Step2/step2_upgd_memory_opmnist.py" \
  --mnist-published-scale \
  --allow-openml-download \
  --opmnist-fraction 0.01 \
  --n-seeds 1 \
  --final-window 5000 \
  --chunk-size 60000 \
  --include-primary-sharpened \
  --include-adaptive-primary-sharpened \
  --include-sharpened-mlp \
  --include-centroid-candidates \
  --include-prototype-memory \
  --output-dir outputs/step2_upgd_memory_opmnist_candidate_screen \
  --result-prefix candidate_screen_1pct_all \
  --note-path docs/research/step2_upgd_memory_opmnist_candidate_screen_1pct_all.md \
  --status-path outputs/step2_upgd_memory_opmnist_candidate_screen/candidate_screen_1pct_all_status.json \
  --force-restart
```

Artifacts:

- `outputs/step2_upgd_memory_opmnist_candidate_screen/candidate_screen_1pct_all_results.json`
- `outputs/step2_upgd_memory_opmnist_candidate_screen/candidate_screen_1pct_all_SUMMARY.md`
- `docs/research/step2_upgd_memory_opmnist_candidate_screen_1pct_all.md`

Protocol status:

- true OpenML MNIST: yes;
- canonical 60,000/10,000 split: yes;
- configured OPMNIST permutations: 800;
- completed task blocks: 8/800;
- completed online updates: 480,000;
- test views cover observed permutations: yes;
- test views cover all 800 permutations: no;
- published task-count scale: no.

This is a true-MNIST 1% screen, not full 800-task evidence.

## Results

| Method | Final MSE | Final Acc | Test MSE | Test Acc |
| --- | ---: | ---: | ---: | ---: |
| `step2_hybrid_memory_trace` | 0.008751 | 0.944800 | 0.044436 | 0.693912 |
| `step2_hybrid_memory_trace_sharp` | 0.010936 | 0.944000 | 0.060282 | 0.684150 |
| `step2_hybrid_memory_trace_adaptive_sharp` | 0.008735 | 0.944200 | 0.045235 | 0.690275 |
| `mlp_h64` | 0.017376 | 0.931800 | 0.072024 | 0.424325 |
| `mlp_h128` | 0.016759 | 0.940200 | 0.084707 | 0.394175 |
| `mlp_h64_sharp` | 0.011841 | 0.933400 | 0.093491 | 0.427975 |
| `mlp_h128_sharp` | 0.010863 | 0.936200 | 0.107797 | 0.381212 |
| `centroid_hysteretic64_center_c030` | 0.010668 | 0.939800 | 0.090787 | 0.363587 |
| `proto_mem_s20` | 0.074303 | 0.461800 | 0.075300 | 0.473475 |
| `proto_mem_s32` | 0.053823 | 0.603200 | 0.057363 | 0.590837 |

Positive deltas below favor the candidate over the best fair MLP for that
metric:

| Candidate | Online MSE | Online Acc | Final MSE | Final Acc | Test MSE | Test Acc |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `step2_hybrid_memory_trace` | +0.003521 | +0.011950 | +0.002112 | +0.004600 | +0.027587 | +0.265937 |
| `step2_hybrid_memory_trace_sharp` | +0.000718 | +0.012540 | -0.000074 | +0.003800 | +0.011741 | +0.256175 |
| `step2_hybrid_memory_trace_adaptive_sharp` | +0.003623 | +0.012473 | +0.002128 | +0.004000 | +0.026789 | +0.262300 |
| `centroid_hysteretic64_center_c030` | +0.001428 | +0.008515 | +0.000195 | -0.000400 | -0.018763 | -0.064387 |
| `proto_mem_s20` | -0.044175 | -0.342208 | -0.063440 | -0.478400 | -0.003277 | +0.045500 |
| `proto_mem_s32` | -0.033634 | -0.259423 | -0.042961 | -0.337000 | +0.014660 | +0.162862 |

## Interpretation

`step2_hybrid_memory_trace` remains the best all-around 1% OPMNIST learner in
this screen. It wins every metric against the best fair MLP and also beats the
newly tested prototype-memory rows on final-window tracking and observed-view
held-out accuracy.

`step2_hybrid_memory_trace_adaptive_sharp` is almost tied with raw UPGD-memory
on this screen and has the best final-window MSE, but raw UPGD-memory keeps the
slightly better held-out test accuracy. This matches the existing image-block
promotion notes: adaptive sharpening is CIFAR-safe, while raw UPGD-memory is
still the stronger OPMNIST runner.

`step2_hybrid_memory_trace_sharp` should not be promoted for OPMNIST. It helps
accuracy relative to MLP, but it gives up too much MSE against the raw and
adaptive UPGD-memory variants.

`centroid_hysteretic64_center_c030` is not an OPMNIST retention solution. It
barely clears best MLP on final-window MSE, loses final-window accuracy, and
loses held-out test MSE/accuracy. This confirms the earlier concern that a
single averaged class centroid is too coarse for permutation-view retention.

`proto_mem_s32` is the only newly tested candidate with a useful retained-view
signal: it beats best fair MLP on observed-view held-out test MSE and test
accuracy. However, it badly loses online and final-window tracking. That makes
it a retained-memory component, not a standalone OPMNIST learner.

`proto_mem_s20` is weaker than `proto_mem_s32` on this true-MNIST 1% run. The
old compact local-Digits D20 result favored 20 slots as enough, but true MNIST
already benefits from the larger 32-slot budget.

## Decision

Do not replace the current OPMNIST primary with centroid or direct sharpening.

Keep `proto_mem_s32` as the only promising new retained-view component. The
next useful experiment is not standalone prototype memory; it is a full or
larger fractional OPMNIST run of a fused UPGD-memory plus prototype-memory
deployment rule that preserves UPGD-memory online tracking and uses prototype
memory only for retained-view correction.

This screen does not resolve the full 800-task retained all-permutation gap.
It does narrow the next search: larger-budget prototype memory is the only
newly evaluated candidate with a positive held-out-retention signal, but it
needs a learned or confidence-gated fusion with UPGD-memory before it can be a
serious full-scale candidate.
