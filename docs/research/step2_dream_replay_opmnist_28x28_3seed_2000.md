# Step 2 UPGD-Memory OPMNIST

This note records the resumable OPMNIST run for the packaged UPGD-memory trace learner, optional simple candidates, and fair MLP baselines.

- Primary method: `step2_hybrid_memory_trace`
- MNIST source: `sklearn_digits_28x28`
- Steps: `2000`
- Seeds: `3`
- Permutations: `5`
- Task block size: `400`

| Method | Final MSE | Final Acc | Test MSE | Test Acc |
| --- | ---: | ---: | ---: | ---: |
| `step2_hybrid_memory_trace` | 0.024236 +/- 0.002808 | 0.830833 +/- 0.023467 | 0.011934 +/- 0.000920 | 0.930000 +/- 0.007687 |
| `step2_hybrid_memory_trace_dream_surprise` | 0.031467 +/- 0.000491 | 0.783333 +/- 0.000833 | 0.013455 +/- 0.000797 | 0.913833 +/- 0.007748 |
| `upgd_structure_softmax_h64` | 0.055780 +/- 0.000141 | 0.645833 +/- 0.016223 | 0.039326 +/- 0.001049 | 0.800500 +/- 0.017474 |
| `upgd_structure_softmax_h64_dream_surprise` | 0.039469 +/- 0.000448 | 0.746667 +/- 0.003632 | 0.023495 +/- 0.001516 | 0.871000 +/- 0.008386 |
| `upgd_structure_softmax_h64_dream_progress` | 0.040015 +/- 0.001533 | 0.731667 +/- 0.016853 | 0.022984 +/- 0.001264 | 0.863833 +/- 0.007618 |
| `upgd_structure_softmax_h64_dream_random` | 0.042634 +/- 0.000685 | 0.688333 +/- 0.005833 | 0.024698 +/- 0.000566 | 0.828500 +/- 0.005575 |
| `mlp_h64` | 0.059746 +/- 0.000350 | 0.655000 +/- 0.006614 | 0.067212 +/- 0.000742 | 0.565167 +/- 0.010553 |

## Primary vs Best MLP

- `online_mean_mse` vs `mlp_h64`: +0.042421 +/- 0.000326; wins/losses/ties 3/0/0.
- `online_mean_accuracy` vs `mlp_h64`: +0.228833 +/- 0.006119; wins/losses/ties 3/0/0.
- `final_window_mse` vs `mlp_h64`: +0.035510 +/- 0.002614; wins/losses/ties 3/0/0.
- `final_window_accuracy` vs `mlp_h64`: +0.175833 +/- 0.029309; wins/losses/ties 3/0/0.
- `test_mse` vs `mlp_h64`: +0.055278 +/- 0.001533; wins/losses/ties 3/0/0.
- `test_accuracy` vs `mlp_h64`: +0.364833 +/- 0.018075; wins/losses/ties 3/0/0.

## Additional Candidate vs Best MLP

- `step2_hybrid_memory_trace_dream_surprise` `online_mean_mse` vs `mlp_h64`: +0.038772 +/- 0.000145; wins/losses/ties 3/0/0.
- `upgd_structure_softmax_h64` `online_mean_mse` vs `mlp_h64`: +0.001482 +/- 0.000637; wins/losses/ties 3/0/0.
- `upgd_structure_softmax_h64_dream_surprise` `online_mean_mse` vs `mlp_h64`: +0.015845 +/- 0.000765; wins/losses/ties 3/0/0.
- `upgd_structure_softmax_h64_dream_progress` `online_mean_mse` vs `mlp_h64`: +0.014929 +/- 0.000870; wins/losses/ties 3/0/0.
- `upgd_structure_softmax_h64_dream_random` `online_mean_mse` vs `mlp_h64`: +0.014549 +/- 0.001054; wins/losses/ties 3/0/0.
- `step2_hybrid_memory_trace_dream_surprise` `online_mean_accuracy` vs `mlp_h64`: +0.204833 +/- 0.005085; wins/losses/ties 3/0/0.
- `upgd_structure_softmax_h64` `online_mean_accuracy` vs `mlp_h64`: -0.008000 +/- 0.005299; wins/losses/ties 1/2/0.
- `upgd_structure_softmax_h64_dream_surprise` `online_mean_accuracy` vs `mlp_h64`: +0.086667 +/- 0.001093; wins/losses/ties 3/0/0.
- `upgd_structure_softmax_h64_dream_progress` `online_mean_accuracy` vs `mlp_h64`: +0.037667 +/- 0.007356; wins/losses/ties 3/0/0.
- `upgd_structure_softmax_h64_dream_random` `online_mean_accuracy` vs `mlp_h64`: +0.032667 +/- 0.001364; wins/losses/ties 3/0/0.
- `step2_hybrid_memory_trace_dream_surprise` `final_window_mse` vs `mlp_h64`: +0.028279 +/- 0.000392; wins/losses/ties 3/0/0.
- `upgd_structure_softmax_h64` `final_window_mse` vs `mlp_h64`: +0.003966 +/- 0.000335; wins/losses/ties 3/0/0.
- `upgd_structure_softmax_h64_dream_surprise` `final_window_mse` vs `mlp_h64`: +0.020277 +/- 0.000786; wins/losses/ties 3/0/0.
- `upgd_structure_softmax_h64_dream_progress` `final_window_mse` vs `mlp_h64`: +0.019731 +/- 0.001313; wins/losses/ties 3/0/0.
- `upgd_structure_softmax_h64_dream_random` `final_window_mse` vs `mlp_h64`: +0.017112 +/- 0.000940; wins/losses/ties 3/0/0.
- `step2_hybrid_memory_trace_dream_surprise` `final_window_accuracy` vs `mlp_h64`: +0.128333 +/- 0.006821; wins/losses/ties 3/0/0.
- `upgd_structure_softmax_h64` `final_window_accuracy` vs `mlp_h64`: -0.009167 +/- 0.021858; wins/losses/ties 2/1/0.
- `upgd_structure_softmax_h64_dream_surprise` `final_window_accuracy` vs `mlp_h64`: +0.091667 +/- 0.008207; wins/losses/ties 3/0/0.
- `upgd_structure_softmax_h64_dream_progress` `final_window_accuracy` vs `mlp_h64`: +0.076667 +/- 0.022376; wins/losses/ties 3/0/0.
- `upgd_structure_softmax_h64_dream_random` `final_window_accuracy` vs `mlp_h64`: +0.033333 +/- 0.012444; wins/losses/ties 3/0/0.
- `step2_hybrid_memory_trace_dream_surprise` `test_mse` vs `mlp_h64`: +0.053757 +/- 0.001364; wins/losses/ties 3/0/0.
- `upgd_structure_softmax_h64` `test_mse` vs `mlp_h64`: +0.027886 +/- 0.001650; wins/losses/ties 3/0/0.
- `upgd_structure_softmax_h64_dream_surprise` `test_mse` vs `mlp_h64`: +0.043717 +/- 0.002039; wins/losses/ties 3/0/0.
- `upgd_structure_softmax_h64_dream_progress` `test_mse` vs `mlp_h64`: +0.044228 +/- 0.001548; wins/losses/ties 3/0/0.
- `upgd_structure_softmax_h64_dream_random` `test_mse` vs `mlp_h64`: +0.042514 +/- 0.000664; wins/losses/ties 3/0/0.
- `step2_hybrid_memory_trace_dream_surprise` `test_accuracy` vs `mlp_h64`: +0.348667 +/- 0.017763; wins/losses/ties 3/0/0.
- `upgd_structure_softmax_h64` `test_accuracy` vs `mlp_h64`: +0.235333 +/- 0.027983; wins/losses/ties 3/0/0.
- `upgd_structure_softmax_h64_dream_surprise` `test_accuracy` vs `mlp_h64`: +0.305833 +/- 0.018908; wins/losses/ties 3/0/0.
- `upgd_structure_softmax_h64_dream_progress` `test_accuracy` vs `mlp_h64`: +0.298667 +/- 0.017372; wins/losses/ties 3/0/0.
- `upgd_structure_softmax_h64_dream_random` `test_accuracy` vs `mlp_h64`: +0.263333 +/- 0.012736; wins/losses/ties 3/0/0.

## Scale Status

A full published-scale OPMNIST result requires 800 completed 60,000-example task blocks, or 48,000,000 online updates. This runner reports exact completed blocks and leaves a checkpoint/status sidecar for continuation rather than treating partial runs as full closure.

## 2026-05-09 Dream-Replay Assessment

This run tests causal dreaming as repeated experience of already-observed
examples. The dream buffer stores real observations and labels after their
current-step prediction has been scored, prioritizes them by surprise or
learning progress, and updates priorities from post-dream loss so examples are
rehearsed until they become lower-surprise. No current label can alter its own
deployed prediction.

Paired deltas versus `upgd_structure_softmax_h64`, where positive favors the
dream variant:

| Candidate | Online MSE | Online Acc | Final MSE | Final Acc | Test MSE | Test Acc |
|---|---:|---:|---:|---:|---:|---:|
| `dream_surprise` | +0.014362 | +0.094667 | +0.016311 | +0.100833 | +0.015832 | +0.070500 |
| `dream_progress` | +0.013446 | +0.045667 | +0.015765 | +0.085833 | +0.016342 | +0.063333 |
| `dream_random` | +0.013067 | +0.040667 | +0.013146 | +0.042500 | +0.014628 | +0.028000 |

The same surprise dream replay around `step2_hybrid_memory_trace` is harmful:
online MSE `-0.003649`, online accuracy `-0.024000`, final-window MSE
`-0.007230`, and held-out test accuracy `-0.016167`. Interpretation:
dreaming is a useful plasticity amplifier for the undertrained bare UPGD h64
branch, but it conflicts with the stronger primary UPGD-memory hybrid.
