# Step 2 OPMNIST 800-Task Closure Attempt

Date: 2026-05-05

Superseded status: this note records the earlier `step2_published_stressors.py`
800-task resume attempts. The current Step 2 OPMNIST task-count/protocol gate
is superseded by
`outputs/step2_canonical/upgd_memory_opmnist_latest_best_800block_1seed_results.json`,
which completes one 800-task / 48,000,000-update OpenML MNIST seed with all 800
held-out permutation views. The remaining OPMNIST gap is multi-seed and
metric-level performance closure, not task-count completion.

## 2026-05-06 Worker O1 Status Artifact Update

Status: **partial, still open**. Worker O1 hardened the OPMNIST runner with
fsync-backed atomic writes, a run manifest, an atomically rewritten periodic
status JSON, and a status-only JSON output option. The current durable
checkpoint is still:

`outputs/step2_opmnist_scale_800task/opmnist_true_mnist_800task_partial40block_mse_seed0_opmnist_resume.pkl`

The checkpoint has now advanced to **2,820,000 online examples / 47 full
60,000-example OPMNIST blocks**. It remains **47/800 blocks**, not a completed
published-scale run.

Artifacts from this update:

- Full-run status artifact:
  `outputs/step2_opmnist_scale_800task/opmnist_true_mnist_800task_full_mse_status.json`
- 47-block evaluated result:
  `outputs/step2_opmnist_scale_800task/opmnist_true_mnist_800task_partial47block_mse_results.json`
- 47-block summary:
  `outputs/step2_opmnist_scale_800task/opmnist_true_mnist_800task_partial47block_mse_SUMMARY.md`
- 47-block run manifest:
  `outputs/step2_opmnist_scale_800task/opmnist_true_mnist_800task_partial47block_mse_manifest.json`
- 47-block periodic status:
  `outputs/step2_opmnist_scale_800task/opmnist_true_mnist_800task_partial47block_mse_opmnist_status.json`

Latest checkpoint status:

- Completed steps: 2,820,000 / 48,000,000.
- Completed full 60,000-example task blocks: 47 / 800.
- Remaining steps: 45,180,000.
- Remaining full task blocks: 753.
- Progress fraction: 0.05875.
- Checkpoint elapsed scan time: 7,798.259 s.
- Overall throughput: 361.619 learner steps/s.
- Latest chunk throughput: 205.828 learner steps/s.
- Latest ETA to 48,000,000 steps: 219,503.750 s, or about 2d 12h 58m.

47-block evaluated metrics:

| Metric | Portfolio | Best fair MLP | Difference |
|---|---:|---:|---:|
| Final-window MSE | 0.024955 | 0.028750 (`mlp_h64`) | +0.003795 favors portfolio |
| Final-window accuracy | 0.887683 | 0.878783 (`mlp_h128`) | +0.008900 favors portfolio |
| Held-out test MSE over 47 observed permutation views | 0.095747 | 0.111081 (`mlp_h64_64`) | +0.015334 favors portfolio |
| Held-out test accuracy over 47 observed permutation views | 0.211996 | 0.213794 (`mlp_h128`) | -0.001798 trails best MLP |

Because held-out test accuracy is negative against the best same-run fair MLP,
`all_primary_nonnegative_vs_best_mlp=false` on the 47-block evaluated snapshot.
Because only 47 of 800 task blocks are complete,
`matches_dohare_opmnist_published_task_count=false` and
`published_scale_external_claim_supported=false`.

Canonical full resume command:

```bash
source .venv/bin/activate && python "examples/The Alberta Plan/Step2/step2_published_stressors.py" --benchmarks permuted_mnist_like --n-seeds 1 --seed 0 --steps 48000000 --final-window 60000 --mnist-source openml --allow-openml-download --openml-data-home outputs/step2_published_mnist_openml_cache --mnist-published-scale --n-permutations 800 --max-test-permutation-views 800 --opmnist-chunk-size 20000 --opmnist-resume --opmnist-resume-path outputs/step2_opmnist_scale_800task/opmnist_true_mnist_800task_partial40block_mse_seed0_opmnist_resume.pkl --digits-deployment-objective mse --output-dir outputs/step2_opmnist_scale_800task --result-prefix opmnist_true_mnist_800task_full_mse
```

Status refresh command:

```bash
source .venv/bin/activate && python "examples/The Alberta Plan/Step2/step2_published_stressors.py" --opmnist-status-checkpoint outputs/step2_opmnist_scale_800task/opmnist_true_mnist_800task_partial40block_mse_seed0_opmnist_resume.pkl --opmnist-status-target-steps 48000000 --opmnist-status-output outputs/step2_opmnist_scale_800task/opmnist_true_mnist_800task_full_mse_status.json
```

## 2026-05-06 Worker OPMNIST-800 Resume Update

Status: **partial, still open**. The current highest durable checkpoint is:

`outputs/step2_opmnist_scale_800task/opmnist_true_mnist_800task_partial40block_mse_seed0_opmnist_resume.pkl`

The checkpoint was validated by its sidecar/progress log, then resumed with the
full 48,000,000-step target. The run advanced from **2,400,000 online examples /
40 full 60,000-example OPMNIST blocks** to **2,760,000 online examples / 46 full
blocks**. Full published-scale completion still requires 800 blocks /
48,000,000 examples, so `published_scale_external_claim_supported=false`.

Exact full resume command used:

```bash
source .venv/bin/activate && python "examples/The Alberta Plan/Step2/step2_published_stressors.py" --benchmarks permuted_mnist_like --n-seeds 1 --seed 0 --steps 48000000 --final-window 60000 --mnist-source openml --allow-openml-download --openml-data-home outputs/step2_published_mnist_openml_cache --mnist-published-scale --n-permutations 800 --max-test-permutation-views 800 --opmnist-chunk-size 20000 --opmnist-resume --opmnist-resume-path outputs/step2_opmnist_scale_800task/opmnist_true_mnist_800task_partial40block_mse_seed0_opmnist_resume.pkl --digits-deployment-objective mse --output-dir outputs/step2_opmnist_scale_800task --result-prefix opmnist_true_mnist_800task_full_mse
```

Durable status from `--opmnist-status-checkpoint` after interruption:

- Completed steps: 2,760,000 / 48,000,000.
- Completed full 60,000-example task blocks: 46 / 800.
- Remaining steps: 45,240,000.
- Remaining full task blocks: 754.
- Progress fraction: 0.0575.
- Checkpoint elapsed scan time: 7,544.324 s.
- Overall throughput: 365.838 learner steps/s.
- Latest chunk throughput: 532.970 learner steps/s.
- Latest ETA to 48,000,000 steps: 84,882.833 s, or 23h 34m 43s.
- Mean throughput across the 18 new chunks in this resume attempt: 404.283
  learner steps/s, implying about 31.1h remaining from the 46-block checkpoint.

No `opmnist_true_mnist_800task_full_mse_*results.json` or full-run summary was
produced because the 48,000,000-step run did not complete. The current reported
evaluation metrics therefore remain the last completed aggregation from the
40-block result below; the 46-block checkpoint is progress evidence only until
a full aggregation/evaluation run completes.

## 2026-05-06 Worker OPMNIST-FULL Update

Status: **partial, still open**. The current highest resumable checkpoint is:

`outputs/step2_opmnist_scale_800task/opmnist_true_mnist_800task_partial40block_mse_seed0_opmnist_resume.pkl`

This checkpoint completed **2,400,000 online examples / 40 full 60,000-example
OPMNIST blocks** against a requested 800-block / 48,000,000-example target. The
full published-scale boundary is not closed:
`published_scale_external_claim_supported=false`.

Exact resume command:

```bash
source .venv/bin/activate && python "examples/The Alberta Plan/Step2/step2_published_stressors.py" \
  --benchmarks permuted_mnist_like \
  --n-seeds 1 \
  --seed 0 \
  --steps 2400000 \
  --final-window 60000 \
  --mnist-source openml \
  --allow-openml-download \
  --openml-data-home outputs/step2_published_mnist_openml_cache \
  --mnist-published-scale \
  --n-permutations 800 \
  --max-test-permutation-views 40 \
  --opmnist-chunk-size 20000 \
  --opmnist-resume \
  --opmnist-resume-path outputs/step2_opmnist_scale_800task/opmnist_true_mnist_800task_partial40block_mse_seed0_opmnist_resume.pkl \
  --digits-deployment-objective mse \
  --output-dir outputs/step2_opmnist_scale_800task \
  --result-prefix opmnist_true_mnist_800task_partial40block_mse
```

Before resuming, the previous highest checkpoint was confirmed at
`outputs/step2_worker_m_opmnist_30block/step2_worker_m_opmnist_30block_seed0_opmnist_resume.pkl`
with **1,860,000 steps / 31 full blocks**. That checkpoint was copied into the
owned OPMNIST output directory under the 40-block result prefix, then resumed.

Artifacts:

- Results:
  `outputs/step2_opmnist_scale_800task/opmnist_true_mnist_800task_partial40block_mse_results.json`
- Summary:
  `outputs/step2_opmnist_scale_800task/opmnist_true_mnist_800task_partial40block_mse_SUMMARY.md`
- Checkpoint sidecar:
  `outputs/step2_opmnist_scale_800task/opmnist_true_mnist_800task_partial40block_mse_seed0_opmnist_resume.pkl.json`
- Progress log:
  `outputs/step2_opmnist_scale_800task/opmnist_true_mnist_800task_partial40block_mse_seed0_opmnist_resume.pkl.progress.jsonl`

Current 40-block metrics:

| Metric | Portfolio | Best fair MLP | Difference |
|---|---:|---:|---:|
| Final-window MSE | 0.025684 | 0.027934 (`mlp_h128`) | +0.002250 favors portfolio |
| Final-window accuracy | 0.885467 | 0.881917 (`mlp_h128`) | +0.003550 favors portfolio |
| Held-out test accuracy over 40 observed permutation views | 0.211182 | 0.200170 (`mlp_h128`) | +0.011013 favors portfolio |
| Held-out test MSE over 40 observed permutation views | 0.113824 | 0.133766 (`mlp_h64`) | +0.019942 favors portfolio |

Throughput and ETA from the completed checkpoint:

- Checkpoint scan elapsed time: 6,624.279 s.
- Overall throughput: 362.304 learner steps/s.
- Last chunk throughput: 363.065 learner steps/s.
- Remaining to 800 blocks: 45,600,000 examples / 760 full blocks.
- ETA to 800 blocks from last-chunk rate: 125,597.475 s, about 1d 10h 53m.
- Current progress fraction: 0.05.

No throughput or resume correctness bug was found during this run. The copied
checkpoint resumed cleanly, wrote atomic `.pkl` and `.pkl.json` checkpoints
after each 20,000-example chunk, and appended progress rows through the
2,400,000-step target.

## Status

Status: **partial, still open**.

The runner is now capable of safely attempting the full 800-task / 48,000,000
example OPMNIST protocol: it streams chunks, writes an atomic checkpoint and
JSON sidecar after every chunk, appends JSONL progress rows, validates
training-affecting resume config, and has a status/ETA command that does not
need to wait for final aggregation.

The highest durable checkpoint in this session is now **46 full OPMNIST task
blocks / 2,760,000 online examples** on true OpenML MNIST with the canonical
60,000/10,000 split and 800 configured random pixel permutations. The strongest
completed evaluation aggregation remains the 40-block result, which is positive
against the fair MLP comparator by final-window MSE and held-out test accuracy
over the observed permutation views. This is still not the published 800-task
count, so the published-scale blocker remains open.

## Strictness Audit

- Prediction-before-update: yes. Each chunk scan computes all expert
  predictions and mixture/router losses before calling each expert update.
- Temporal uniformity: yes. `mlp_h64`, `mlp_h128`, `mlp_h64_64`,
  `upgd_low_noise`, and `dynamic_sparse` update on every online example.
- No task id at deployment: yes. The learner receives only permuted pixels;
  held-out deployment uses fixed post-run weights, not task ids.
- Deterministic permutations: patched and verified. Pixel orders are generated
  from the stream seed, checkpointed, and now validated against the expected
  deterministic orders on resume; chunks regenerate examples deterministically
  by global step.
- Fair MLP baselines: yes. The comparator set is the three fair MLP experts
  trained online in the same scan as the portfolio.
- Resume safety: patched. Checkpoints are atomic, include elapsed time and
  chunk history, validate training-affecting config and checkpointed pixel
  orders, and migrate older OPMNIST UPGD states that predate current
  `unit_long_utilities`, `unit_gradient_emas`, and loss EMA fields. Deployment
  objective changes remain evaluation-only and are allowed on replay.
- Chunked writes: patched. Each chunk writes `.pkl`, `.pkl.json`, and
  `.pkl.progress.jsonl` state/progress artifacts.

## Commands Run

```bash
source .venv/bin/activate

pytest tests/test_step2_published_stressors.py -q
ruff check "examples/The Alberta Plan/Step2/step2_published_stressors.py" \
  tests/test_step2_published_stressors.py
mypy "examples/The Alberta Plan/Step2/step2_published_stressors.py" \
  tests/test_step2_published_stressors.py

python "examples/The Alberta Plan/Step2/step2_published_stressors.py" \
  --benchmarks permuted_mnist_like \
  --smoke \
  --opmnist-streaming \
  --no-opmnist-resume \
  --output-dir outputs/step2_published_stressors_opmnist800_smoke \
  --result-prefix opmnist800_smoke

python "examples/The Alberta Plan/Step2/step2_published_stressors.py" \
  --opmnist-status-results \
  outputs/step2_opmnist_scale_5block/opmnist_true_mnist_5block_results.json

python "examples/The Alberta Plan/Step2/step2_published_stressors.py" \
  --benchmarks permuted_mnist_like \
  --mnist-source openml \
  --allow-openml-download \
  --mnist-published-scale \
  --n-permutations 800 \
  --steps 600000 \
  --n-seeds 1 \
  --final-window 60000 \
  --opmnist-chunk-size 20000 \
  --max-test-permutation-views 10 \
  --output-dir outputs/step2_opmnist_scale_10block \
  --result-prefix opmnist_true_mnist_10block \
  --opmnist-force-restart

python "examples/The Alberta Plan/Step2/step2_published_stressors.py" \
  --benchmarks permuted_mnist_like \
  --mnist-source openml \
  --allow-openml-download \
  --mnist-published-scale \
  --n-permutations 800 \
  --steps 600000 \
  --n-seeds 1 \
  --final-window 60000 \
  --opmnist-chunk-size 20000 \
  --max-test-permutation-views 10 \
  --output-dir outputs/step2_opmnist_scale_10block_dynamic_sparse \
  --result-prefix opmnist_true_mnist_10block_dynamic_sparse \
  --opmnist-resume-path \
  outputs/step2_opmnist_scale_10block/opmnist_true_mnist_10block_seed0_opmnist_resume.pkl \
  --digits-deployment-objective dynamic_sparse

python "examples/The Alberta Plan/Step2/step2_published_stressors.py" \
  --benchmarks permuted_mnist_like \
  --mnist-source openml \
  --allow-openml-download \
  --mnist-published-scale \
  --n-permutations 800 \
  --steps 600000 \
  --n-seeds 1 \
  --final-window 60000 \
  --opmnist-chunk-size 20000 \
  --max-test-permutation-views 10 \
  --output-dir outputs/step2_opmnist_scale_10block_accuracy \
  --result-prefix opmnist_true_mnist_10block_accuracy \
  --opmnist-resume-path \
  outputs/step2_opmnist_scale_10block/opmnist_true_mnist_10block_seed0_opmnist_resume.pkl \
  --digits-deployment-objective accuracy

mkdir -p outputs/step2_opmnist_scale_800task
cp outputs/step2_opmnist_scale_10block/opmnist_true_mnist_10block_seed0_opmnist_resume.pkl \
  outputs/step2_opmnist_scale_800task/opmnist_true_mnist_800task_partial20block_seed0_opmnist_resume.pkl
cp outputs/step2_opmnist_scale_10block/opmnist_true_mnist_10block_seed0_opmnist_resume.pkl.json \
  outputs/step2_opmnist_scale_800task/opmnist_true_mnist_800task_partial20block_seed0_opmnist_resume.pkl.json
cp outputs/step2_opmnist_scale_10block/opmnist_true_mnist_10block_seed0_opmnist_resume.pkl.progress.jsonl \
  outputs/step2_opmnist_scale_800task/opmnist_true_mnist_800task_partial20block_seed0_opmnist_resume.pkl.progress.jsonl

python "examples/The Alberta Plan/Step2/step2_published_stressors.py" \
  --benchmarks permuted_mnist_like \
  --mnist-source openml \
  --allow-openml-download \
  --mnist-published-scale \
  --steps 1200000 \
  --n-seeds 1 \
  --seed 0 \
  --final-window 60000 \
  --n-permutations 800 \
  --opmnist-chunk-size 20000 \
  --max-test-permutation-views 20 \
  --output-dir outputs/step2_opmnist_scale_800task \
  --result-prefix opmnist_true_mnist_800task_partial20block_mse \
  --opmnist-resume-path \
  outputs/step2_opmnist_scale_800task/opmnist_true_mnist_800task_partial20block_seed0_opmnist_resume.pkl

python "examples/The Alberta Plan/Step2/step2_published_stressors.py" \
  --benchmarks permuted_mnist_like \
  --mnist-source openml \
  --allow-openml-download \
  --mnist-published-scale \
  --steps 1200000 \
  --n-seeds 1 \
  --seed 0 \
  --final-window 60000 \
  --n-permutations 800 \
  --opmnist-chunk-size 20000 \
  --max-test-permutation-views 20 \
  --output-dir outputs/step2_opmnist_scale_800task \
  --result-prefix opmnist_true_mnist_800task_partial20block_accuracy \
  --opmnist-resume-path \
  outputs/step2_opmnist_scale_800task/opmnist_true_mnist_800task_partial20block_seed0_opmnist_resume.pkl \
  --digits-deployment-objective accuracy

python "examples/The Alberta Plan/Step2/step2_published_stressors.py" \
  --benchmarks permuted_mnist_like \
  --mnist-source openml \
  --allow-openml-download \
  --mnist-published-scale \
  --steps 1200000 \
  --n-seeds 1 \
  --seed 0 \
  --final-window 60000 \
  --n-permutations 800 \
  --opmnist-chunk-size 20000 \
  --max-test-permutation-views 20 \
  --output-dir outputs/step2_opmnist_scale_800task \
  --result-prefix opmnist_true_mnist_800task_partial20block_dynamic_sparse \
  --opmnist-resume-path \
  outputs/step2_opmnist_scale_800task/opmnist_true_mnist_800task_partial20block_seed0_opmnist_resume.pkl \
  --digits-deployment-objective dynamic_sparse
```

The first 20-block resume attempt exposed that the 10-block checkpoint also
predated newer `UPGDState` utility/loss fields. The runner now migrates those
fields in the OPMNIST checkpoint carry on load; the copied 20-block checkpoint
has been rewritten in the current format.

## Numeric Results

Main 10-block result:
`outputs/step2_opmnist_scale_10block/opmnist_true_mnist_10block_results.json`.

- Wall clock: 1595.53 s.
- Checkpointed scan elapsed time: 1582.83 s.
- Steps: 600,000.
- Completed 60k task blocks: 10.
- Held-out evaluation views: 10 observed permutation views.
- Final-window MSE: portfolio 0.021351, best fair MLP 0.025517.
- Portfolio-vs-best-MLP final-window MSE diff: +0.004166.
- Held-out accuracy: portfolio 0.493010, best fair MLP 0.355650.
- Portfolio-vs-best-MLP held-out accuracy diff: +0.137360.

Per-method 10-block held-out accuracy:

| Method | Test Accuracy | Test MSE |
|---|---:|---:|
| portfolio MSE-tracking | 0.493010 | 0.067939 |
| mlp_h64 | 0.311090 | 0.111127 |
| mlp_h128 | 0.330620 | 0.101016 |
| mlp_h64_64 | 0.355650 | 0.082377 |
| upgd_low_noise | 0.411240 | 0.082539 |
| dynamic_sparse | 0.513740 | 0.064873 |

Deployment variants from the same final checkpoint:
`outputs/step2_opmnist_scale_10block/opmnist_true_mnist_10block_deployment_variants.json`.

| Deployment | Test Accuracy | Test MSE |
|---|---:|---:|
| MSE-tracking portfolio | 0.493010 | 0.067939 |
| Accuracy-tracking portfolio | 0.454690 | 0.070344 |
| Dynamic sparse only | 0.513740 | 0.064873 |

The grounded improvement that held up at 10 blocks is sparse expert-only
deployment. It changes only held-out deployment weights, not training, and it
beats the MSE-tracking portfolio and all fair MLPs on the 10 observed held-out
permutation views.

Main continuation outputs:
`outputs/step2_opmnist_scale_800task/opmnist_true_mnist_800task_partial20block_mse_results.json`.
`outputs/step2_opmnist_scale_800task/opmnist_true_mnist_800task_partial40block_mse_results.json`.

| Checkpoint | Blocks | Examples | Throughput | ETA to 800 | Portfolio vs best fair MLP final-window MSE | Held-out accuracy | Route/deployment drift |
|---|---:|---:|---:|---:|---:|---:|---|
| 10-block MSE-tracking | 10 | 600,000 | 379.07 overall / 701.62 last chunk steps/s | 18.8h last-chunk; 34.8h overall | +0.004166 | 0.493010 | route fixed convex, final deployment: dynamic_sparse 0.300, mlp_h64 0.120, mlp_h128 0.184, mlp_h64_64 0.164, upgd 0.232 |
| 20-block MSE-tracking | 20 | 1,200,000 | 323.00 overall / 429.07 last chunk steps/s | 30.3h last-chunk; 40.3h overall | +0.004075 | 0.327580 | route fixed convex, final deployment: dynamic_sparse 0.274, mlp_h64 0.212, mlp_h128 0.234, mlp_h64_64 0.169, upgd 0.111 |
| 40-block MSE-tracking | 40 | 2,400,000 | 362.30 overall / 363.07 last chunk steps/s | 34.9h last-chunk | +0.002250 | 0.211182 | route fixed convex; published-scale flag remains false |
| 46-block checkpoint | 46 | 2,760,000 | 365.84 overall / 532.97 last chunk steps/s | 23.6h latest chunk | n/e | n/e | durable checkpoint only; no full aggregation yet |

20-block deployment variants from the same final checkpoint:
`outputs/step2_opmnist_scale_800task/opmnist_true_mnist_800task_partial20block_deployment_variants.json`.

| Deployment | Held-out views | Test Accuracy | Test MSE |
|---|---:|---:|---:|
| MSE-tracking portfolio | 20 observed | 0.327580 | 0.084924 |
| Accuracy-tracking portfolio | 20 observed | 0.269080 | 0.102125 |
| Dynamic sparse only | 20 observed | 0.402620 | 0.075841 |

Dynamic-sparse-only deployment was best at the earlier 20-block replay
checkpoint, but the current main run has advanced to 46/800 tasks. The primary
claim is not changed until a deployment rule is reproducibly better at the full
target or across a predeclared replication slice.

## ETA Model

Measured 20-block checkpoint throughput:

- Overall checkpoint throughput: 323.00 steps/s.
- Latest chunk throughput: 429.07 steps/s.
- Full 800-task target: 48,000,000 steps.
- Remaining after 20 blocks: 46,800,000 steps.
- Overall-throughput ETA for remaining run: about **40.3 hours**.
- Last-chunk ETA for remaining run: about **30.3 hours**.
- Conservative observed chunk range in the continuation: about 146 to
  456 steps/s, implying roughly **28.5 to 89 hours** from the 20-block
  checkpoint. The sidecar ETA should be treated as the live source of truth.

## Resume Command

Continue the strict run from the current 20-block checkpoint copy to the full
800-task target:

```bash
source .venv/bin/activate
python "examples/The Alberta Plan/Step2/step2_published_stressors.py" \
  --benchmarks permuted_mnist_like \
  --mnist-source openml \
  --allow-openml-download \
  --mnist-published-scale \
  --n-permutations 800 \
  --steps 48000000 \
  --n-seeds 1 \
  --final-window 60000 \
  --opmnist-chunk-size 20000 \
  --max-test-permutation-views 800 \
  --evaluate-all-permutation-views \
  --output-dir outputs/step2_opmnist_scale_800task \
  --result-prefix opmnist_true_mnist_800task \
  --opmnist-resume-path \
  outputs/step2_opmnist_scale_800task/opmnist_true_mnist_800task_partial20block_seed0_opmnist_resume.pkl
```

Monitor without loading the binary checkpoint:

```bash
python "examples/The Alberta Plan/Step2/step2_published_stressors.py" \
  --opmnist-status-checkpoint \
  outputs/step2_opmnist_scale_800task/opmnist_true_mnist_800task_partial20block_seed0_opmnist_resume.pkl
```

The blocker is closed only when that full run completes with
`matches_dohare_opmnist_published_task_count=true` and nonnegative
portfolio-vs-best-fair-MLP comparisons.
