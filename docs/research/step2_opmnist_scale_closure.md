# Step 2 OPMNIST Published-Scale Closure

Date: 2026-05-04

Update: This note records the earlier 5-block run. The current OPMNIST
hard-blocker status is superseded by
`docs/research/step2_upgd_memory_opmnist_latest_best_800block_1seed.md`, which
records the completed 800-block / 48,000,000-example latest-best run with 800
configured permutations.

## Conclusion

Status: **historical partial result; superseded**.

The OPMNIST runner now has explicit protocol gates and a chunked/resumable path
that avoids materializing the full online stream. The staged true-MNIST evidence
uses the canonical 60,000/10,000 split, no task id, random pixel permutations,
prediction-before-update, every expert updated every step, sequential single-pass
60,000-example task epochs, and 800 configured task permutations.

The best completed run in this note covers **5 full task blocks / 300,000
online examples**. It is useful historical evidence only. The later latest-best
UPGD-memory run reaches the Dohare-style task-count scale of 800 task blocks /
48,000,000 online examples, but the retained held-out metrics remain
MLP-favored, so this older note should not be cited as the current status.

## Implementation Changes

- Added chunked/resumable OPMNIST execution in
  `examples/The Alberta Plan/Step2/step2_published_stressors.py`.
- Added explicit protocol/status flags:
  `prediction_before_update_every_step`, `all_experts_update_every_step`,
  `observed_task_blocks`, `completed_full_task_blocks`,
  `opmnist_completed_full_60000_task_blocks`, `streaming_runner`,
  `resumable_runner`, and stricter published-task-count gating.
- Added OPMNIST-specific tests for protocol gates, chunk determinism, and
  checkpoint round-trip behavior in `tests/test_step2_published_stressors.py`.

## Commands

```bash
source .venv/bin/activate

python "examples/The Alberta Plan/Step2/step2_published_stressors.py" \
  --benchmarks permuted_mnist_like \
  --smoke \
  --opmnist-streaming \
  --no-opmnist-resume \
  --output-dir outputs/step2_opmnist_scale_smoke \
  --result-prefix opmnist_streaming_smoke

python "examples/The Alberta Plan/Step2/step2_published_stressors.py" \
  --benchmarks permuted_mnist_like \
  --mnist-source openml \
  --allow-openml-download \
  --mnist-published-scale \
  --n-permutations 800 \
  --steps 60000 \
  --n-seeds 1 \
  --final-window 10000 \
  --opmnist-chunk-size 10000 \
  --max-test-permutation-views 1 \
  --output-dir outputs/step2_opmnist_scale_1block \
  --result-prefix opmnist_true_mnist_1block \
  --opmnist-force-restart

python "examples/The Alberta Plan/Step2/step2_published_stressors.py" \
  --benchmarks permuted_mnist_like \
  --mnist-source openml \
  --allow-openml-download \
  --mnist-published-scale \
  --n-permutations 800 \
  --steps 300000 \
  --n-seeds 1 \
  --final-window 60000 \
  --opmnist-chunk-size 20000 \
  --max-test-permutation-views 5 \
  --output-dir outputs/step2_opmnist_scale_5block \
  --result-prefix opmnist_true_mnist_5block \
  --opmnist-force-restart
```

Post-run deployment variants were evaluated from the saved 5-block final state
and online tracking weights, without retraining. Output:
`outputs/step2_opmnist_scale_5block/opmnist_true_mnist_5block_deployment_variants.json`.

## Outputs

- `outputs/step2_opmnist_scale_smoke/opmnist_streaming_smoke_results.json`
- `outputs/step2_opmnist_scale_1block/opmnist_true_mnist_1block_results.json`
- `outputs/step2_opmnist_scale_1block/opmnist_true_mnist_1block_SUMMARY.md`
- `outputs/step2_opmnist_scale_5block/opmnist_true_mnist_5block_results.json`
- `outputs/step2_opmnist_scale_5block/opmnist_true_mnist_5block_SUMMARY.md`
- `outputs/step2_opmnist_scale_5block/opmnist_true_mnist_5block_deployment_variants.json`

## Protocol Flags

One-block and five-block true-MNIST runs both report:

- `uses_true_mnist=true`
- `uses_true_openml_mnist=true`
- `uses_full_mnist_split=true`
- `uses_full_mnist_task_blocks=true`
- `task_id_provided_to_learner=false`
- `single_pass_examples_within_task=true`
- `uses_random_pixel_permutations_for_all_tasks=true`
- `prediction_before_update_every_step=true`
- `all_experts_update_every_step=true`
- `matches_dohare_opmnist_core_protocol=true`
- `matches_dohare_opmnist_published_task_count=false`
- `published_scale_external_claim_supported=false`

The failed published-task-count flag is intentional and correct for these runs:
the best staged evidence is 5 completed 60k task blocks, not 800.

## Metrics

### True MNIST, 1 Block

Output: `outputs/step2_opmnist_scale_1block/opmnist_true_mnist_1block_results.json`

- Wall clock: 272.40 s
- Steps: 60,000
- Completed 60k task blocks: 1
- Held-out views: 1 observed permutation view
- Final-window MSE: portfolio 0.014683, best fair MLP 0.016448
- Portfolio-vs-best-MLP final-window MSE diff: +0.001765
- Held-out accuracy: portfolio 0.935100, best fair MLP 0.933100
- Portfolio-vs-best-MLP held-out accuracy diff: +0.002000

### True MNIST, 5 Blocks

Output: `outputs/step2_opmnist_scale_5block/opmnist_true_mnist_5block_results.json`

- Wall clock: 793.64 s
- Steps: 300,000
- Completed 60k task blocks: 5
- Held-out views: 5 observed permutation views
- Final-window MSE: portfolio 0.019856, best fair MLP 0.026167
- Portfolio-vs-best-MLP final-window MSE diff: +0.006311
- Held-out accuracy: portfolio 0.606160, best fair MLP 0.416400
- Portfolio-vs-best-MLP held-out accuracy diff: +0.189760

Per-method 5-block held-out accuracy:

| Method | Test Accuracy |
|---|---:|
| portfolio | 0.606160 |
| mlp_h64 | 0.416400 |
| mlp_h128 | 0.412280 |
| mlp_h64_64 | 0.336840 |
| upgd_low_noise | 0.573200 |
| dynamic_sparse | 0.629940 |

## Deployment Variants

The 5-block final state was reused to compare held-out deployment objectives
without changing the fair MLP comparator:

| Deployment | Test MSE | Test Accuracy |
|---|---:|---:|
| MSE-tracking portfolio | 0.060628 | 0.606160 |
| Accuracy-tracking portfolio | 0.055604 | 0.620320 |
| MLP-H128 only | 0.127919 | 0.412280 |
| Dynamic sparse only | 0.053788 | 0.629940 |

The accuracy-tracking deployment improves held-out accuracy over the default
MSE-tracking portfolio, and the dynamic sparse expert is strongest on the
5-block held-out views. This historical result did not close the blocker
because the strict published task-count scale was still missing at the time.
That task-count blocker is now superseded by the one-seed 800-block
UPGD-memory artifact; the remaining gap is multi-seed and metric-level
performance closure.

## Verification

Passing:

```bash
pytest tests/test_step2_published_stressors.py -q
ruff check "examples/The Alberta Plan/Step2/step2_published_stressors.py" \
  tests/test_step2_published_stressors.py
mypy "examples/The Alberta Plan/Step2/step2_published_stressors.py" \
  tests/test_step2_published_stressors.py
pytest tests/ -v
```

Repo-wide `pytest tests/ -v`: 970 passed, 36 warnings, 505.86 s.

Known out-of-scope verification failures in the dirty worktree:

- `ruff check .` fails on `src/alberta_framework/core/optimizers.py` import
  ordering, outside this worker's ownership scope.
- Bare `mypy` exits with "Missing target module, package, files, or command".
- `mypy src tests` reports 1373 errors across pre-existing/out-of-scope tests
  and modules. The OPMNIST-owned file and tests pass targeted mypy.
