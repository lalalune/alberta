# Step 2 OPMNIST Protocol Audit

Date: 2026-05-08

## Bottom Line

Status: **task-count closed for one seed; retained-view claim not closed**.

The latest evaluated OPMNIST evidence is a true OpenML MNIST, canonical
60,000/10,000 split, 800-task-block result:

- `outputs/step2_upgd_memory_opmnist_latest_best_split/combined_latest_best_800block_1seed_results.json`
- `outputs/step2_canonical/upgd_memory_opmnist_latest_best_800block_1seed_results.json`
- `outputs/step2_canonical/upgd_memory_opmnist_latest_best_800block_1seed_SUMMARY.md`
- `outputs/step2_canonical/upgd_memory_opmnist_single_upgd_h128_800block_1seed_results.json`
- `outputs/step2_canonical/upgd_memory_opmnist_single_upgd_h128_800block_1seed_SUMMARY.md`

It satisfies the core online protocol gates: true MNIST, random pixel
permutations, 60,000-example sequential task blocks, no task id,
prediction-before-update, and one pass through each task block. It also
satisfies the published task-count/update budget: 800 completed 60,000-example
blocks / 48,000,000 online examples in the main Online Permuted MNIST protocol.

The result is mixed. UPGD-memory beats the same-run best fair MLP on online
MSE, online accuracy, and final-window MSE. Fair MLP baselines still beat it on
final-window accuracy and all-permutation held-out test MSE/accuracy.

A follow-up single-UPGD H128 artifact is also mixed but improves one important
metric: `upgd_structure_softmax_h128` beats `mlp_h128` on online MSE, online
accuracy, final-window MSE, and all-permutation held-out test accuracy. It
still loses final-window accuracy and all-permutation held-out test MSE.

## Reference Protocol

Dohare et al. 2024 describe Online Permuted MNIST as follows:

- MNIST has 60,000 28x28 training images, normalized to [0, 1].
- Each task applies one randomly selected pixel permutation to all 60,000
  training images.
- The main sequence has 800 permuted MNIST tasks.
- Within a task, each image is presented once, one by one, in random order.
- There are no mini-batches and no task-switch indication.
- The learner predicts class probabilities and is trained with cross-entropy.
- The primary online performance measure is per-task classification accuracy
  over the 60,000 online examples.

Source: [Dohare et al., Nature 2024](https://www.nature.com/articles/s41586-024-07711-7).

## Comparability Audit

| Dimension | Current evaluated artifact | Published-scale requirement | Audit |
|---|---|---|---|
| Data source | OpenML MNIST, canonical 60k/10k split | True MNIST 60k training images | Comparable |
| Task count | 800 completed blocks, 800 configured permutations | 800 completed tasks | Comparable for one seed |
| Update budget | 48M online examples | 48M online examples | Comparable for one seed |
| Examples per task | 60,000 | 60,000 | Comparable for completed tasks |
| Task sampling | Sequential shuffled epoch, no replacement | Single pass in random order | Comparable |
| Task id | Not provided | Not provided | Comparable |
| Train/test split | 60k train, 10k held-out test | Published online metric is over train-stream examples; paper also uses task accuracy | Split is valid, but held-out test is extra |
| Target/loss | One-hot targets, MSE router/objective metrics | Class probabilities with cross-entropy | Not directly comparable |
| Primary metric | MSE plus online/final/test accuracy | Online per-task classification accuracy | Needs accuracy-first reporting |
| Baseline fairness | Same-run fair MLP widths: h64/h128 and sharpened variants | Published baselines include much wider 3-hidden-layer networks and tuned methods | Useful internal comparator, not published baseline parity |
| Held-out evaluation | All 800 permutation views over the 10,000-image held-out set | If using held-out test, evaluate all completed tasks and preferably all 800 configured views | Comparable |
| Seeds | 1 evaluated seed | Multiple independent runs expected for robust evidence | Underpowered |

## Ways The Current Evidence Can Overstate Closure

- **Compact blocks**: earlier compact/local runs use fewer examples, fewer
  permutations, and sometimes sklearn digits. Those are useful smoke tests but
  cannot support an OPMNIST closure claim.
- **Repeated examples**: any run with task blocks larger than the source train
  set, random sampling with replacement, or fallback datasets repeats examples.
  The promoted 800-block artifact avoids this, but compact artifacts do not.
- **Small or capped held-out test set**: compact defaults cap test examples.
  The promoted 800-block artifact uses the full 10,000-image held-out set over
  all 800 permutation views; earlier partial artifacts did not.
- **Missing or secondary accuracy**: the paper's classification measure is
  accuracy. The artifact now reports online/final/test accuracy, but the result
  prefix and default deployment objective are MSE-oriented, and the status gate
  emphasizes final-window MSE plus test accuracy rather than per-task online
  accuracy across 800 tasks.
- **Target/loss mismatch**: one-hot MSE is not the published cross-entropy
  classification loss. Positive MSE evidence can be directionally useful but is
  not a strict reproduction.
- **Same-run MLP width selection**: `mixture_vs_best_mlp` selects the best MLP
  width after observing each metric. This is acceptable only if the MLP set and
  metrics are preregistered. It does not replace a separately tuned published
  baseline suite.
- **Nonstreaming leakage risk**: the compact path materializes the full stream
  and test-view tensor. The 800-block artifact uses the chunked streaming path,
  but any compact artifact should be labelled as a protocol smoke test.
- **Split-merge and metric selection risk**: the current strongest evaluated
  JSON is a split-merged latest-best run. It is acceptable as a one-seed
  protocol artifact, but not as an inferential performance claim across
  post-hoc candidate variants and metrics.
- **Best expert caveat**: the current latest-best artifact compares
  UPGD-memory variants against fair MLP comparators. Earlier broader portfolio
  artifacts included non-MLP experts; those should be cited separately when the
  claim is about the best available Step 2 expert rather than fair MLP parity.
- **Single-seed uncertainty**: all paired standard errors are zero because
  `n=1`. The win/loss counts are descriptive, not inferential.

## Remaining Protocol And Performance Fixes

1. **Keep the closure run full-scale**: the promoted single-seed artifact now
   reaches 48,000,000 online examples with 800 completed 60,000-example blocks,
   true MNIST, no replacement, random pixel permutations, no task id, and
   chunked/resumable execution.
2. **Keep all relevant views in promoted held-out results**: the promoted
   artifact evaluates all 800 held-out permutation views. Future promoted
   artifacts should retain that requirement.
3. **Promote accuracy to the primary performance metric**: require nonnegative
   `online_mean_accuracy`, `final_window_accuracy`, and held-out
   `test_accuracy` versus the preregistered best fair MLP comparator. Continue
   reporting MSE as diagnostic only unless the research claim is MSE-specific.
4. **Add a cross-entropy or NLL track**: either train/evaluate a cross-entropy
   classifier variant, or label the current run as an MSE-surrogate OPMNIST
   variant rather than a strict published reproduction.
5. **Predeclare comparators and deployment objective**: freeze MLP widths,
   portfolio settings, `digits_deployment_objective`, and acceptance metrics
   before the full run. Do not choose between MSE, accuracy, dynamic-sparse, or
   MLP-only deployment after seeing held-out results.
6. **Use multiple seeds for closure**: at minimum 5 independent seeds; stronger
   evidence should use 10 or more if compute permits. Report paired mean,
   standard error, wins/losses/ties, and all seed-level diffs.
7. **Publish progress and failures**: keep checkpoint sidecars and progress logs
   for interrupted runs, but do not promote partial checkpoints as closure.

## Tests Added

Added `tests/test_step2_opmnist_protocol.py` with artifact-level checks that:

- the older 40-block canonical OPMNIST artifact cannot close published scale;
- the promoted latest-best single-seed artifact closes the published-scale
  protocol gates;
- the older partial artifact discloses limited held-out permutation-view
  coverage;
- accuracy and MSE comparisons are both present;
- the portfolio does not beat the best non-MLP expert on held-out test metrics.

These tests are intentionally read-only over the canonical artifact and avoid
touching the main runner.

## Acceptance Criteria For Calling OPMNIST Performance Closed

All of the following must be true in the promoted result JSON:

- `uses_true_mnist=true`
- `uses_full_mnist_split=true`
- `n_train=60000` and `n_test=10000`
- `n_permutations=800`
- `steps=48000000`
- `task_block_size=60000`
- `sample_with_replacement=false`
- `task_sampling=sequential_epoch`
- `include_identity_permutation=false`
- `task_id_provided_to_learner=false`
- `prediction_before_update_every_step=true`
- `all_experts_update_every_step=true`
- `opmnist_completed_full_60000_task_blocks=800`
- `matches_dohare_opmnist_core_protocol=true`
- `matches_dohare_opmnist_published_task_count=true`
- `matches_dohare_opmnist_core_protocol=true` and
  `matches_dohare_opmnist_published_task_count=true`; older runners may also
  expose `published_scale_external_claim_supported=true`
- `test_views_cover_observed_permutations=true`
- if held-out test is used for the closure claim,
  `test_views_cover_all_permutations=true`
- `n_seeds >= 5` for performance closure; the current one-seed artifact is
  protocol-scale evidence only
- nonnegative portfolio-vs-best-fair-MLP differences on:
  `online_mean_accuracy`, `final_window_accuracy`, `test_accuracy`, and the
  selected primary loss metric
- no post-hoc deployment objective or checkpoint cherry-picking

## Exact Commands And Artifacts

Current promoted single-seed protocol-scale artifacts:

- `outputs/step2_canonical/upgd_memory_opmnist_latest_best_800block_1seed_results.json`
- `outputs/step2_canonical/upgd_memory_opmnist_latest_best_800block_1seed_SUMMARY.md`
- `outputs/step2_canonical/upgd_memory_opmnist_single_upgd_h128_800block_1seed_results.json`
- `outputs/step2_canonical/upgd_memory_opmnist_single_upgd_h128_800block_1seed_SUMMARY.md`

The promoted latest-best result was produced as method-filtered resumable
splits from a common checkpoint family and merged by method name. The split
provenance is recorded in the canonical summary and result payload under
`split_results`.

Historical single-seed full-scale continuation command for the older
`step2_published_stressors.py` runner:

```bash
source .venv/bin/activate
python "examples/The Alberta Plan/Step2/step2_published_stressors.py" \
  --benchmarks permuted_mnist_like \
  --n-seeds 1 \
  --seed 0 \
  --steps 48000000 \
  --final-window 60000 \
  --mnist-source openml \
  --allow-openml-download \
  --openml-data-home outputs/step2_published_mnist_openml_cache \
  --mnist-published-scale \
  --n-permutations 800 \
  --max-test-permutation-views 800 \
  --opmnist-chunk-size 20000 \
  --opmnist-resume \
  --opmnist-resume-path outputs/step2_opmnist_scale_800task/opmnist_true_mnist_800task_partial40block_mse_seed0_opmnist_resume.pkl \
  --digits-deployment-objective accuracy \
  --output-dir outputs/step2_opmnist_scale_800task \
  --result-prefix opmnist_true_mnist_800task_full_accuracy
```

A stronger performance-closure run should repeat the promoted protocol across
at least 5 seeds, using distinct per-seed checkpoint paths and a preregistered
result prefix such as `upgd_memory_opmnist_latest_best_800block_5seed`.

Expected older-run closure artifacts:

- `outputs/step2_opmnist_scale_800task/opmnist_true_mnist_800task_full_accuracy_results.json`
- `outputs/step2_opmnist_scale_800task/opmnist_true_mnist_800task_full_accuracy_SUMMARY.md`
- per-seed `*_opmnist_resume.pkl`
- per-seed `*_opmnist_resume.pkl.json`
- per-seed `*_opmnist_resume.pkl.progress.jsonl`

Verification commands:

```bash
source .venv/bin/activate
pytest tests/test_step2_opmnist_protocol.py -q
pytest tests/test_step2_published_stressors.py -q
ruff check .
mypy
```

## Blockers

- The evaluated latest-best result is single-seed.
- The current primary learner optimizes an MSE-style vector target rather than
  the published cross-entropy classifier.
- The retained all-permutation held-out metrics are still MLP-favored.
- The promoted evidence is single-seed.
- The latest-best artifact is positive versus best fair MLP on online MSE,
  online accuracy, and final-window MSE, but not on final-window accuracy or
  held-out all-permutation test MSE/accuracy.
