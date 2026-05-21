# Step 2 Associative OPMNIST Confirmation Protocol

This is the guarded confirmation surface for the Step 2 associative/core status.
It does not make a published-scale claim by default. The full Dohare-style
OPMNIST setting is 800 task permutations x 60,000 examples per task, or
48,000,000 online examples. The runner records enough manifest metadata for
smoke, partial, and full-scale runs to be compared without changing schemas.

## Runner

```bash
source .venv/bin/activate
python benchmarks/step2_associative_opmnist_confirmation.py \
  --scale smoke \
  --output-dir outputs/step2_associative_opmnist_confirmation_smoke
```

Smoke mode uses bundled `sklearn.datasets.load_digits` when scikit-learn is
available and falls back to a deterministic synthetic digit stream otherwise.
Both are protocol probes only. They verify prequential prediction-before-update,
per-step associative-core updates, manifest writing, and held-out view plumbing.
Use `--adaptive-feature-family`, `--adaptive-window`, and `--adaptive-budget`
to test the learned-scope variant; the manifest records these switches and
their learning-rate/resource-control values.

## Scale Modes

| Mode | Default steps | Default seeds | Purpose |
|---|---:|---:|---|
| `smoke` | 128 | 1 | Fast local CI/protocol probe. Never counts as published confirmation. |
| `partial` | 60,000 | 3 | One 60k task-block equivalent with OPMNIST task metadata. Still not published confirmation. |
| `full` | 48,000,000 | 5 | Published-scale plan/execution path. Guarded by `--allow-published-scale`. |

Full-scale dry run:

```bash
python benchmarks/step2_associative_opmnist_confirmation.py \
  --scale full \
  --dry-run \
  --output-dir outputs/step2_associative_opmnist_confirmation_full_plan
```

Actual full-scale execution requires an explicit guard override and a true MNIST
source:

```bash
python benchmarks/step2_associative_opmnist_confirmation.py \
  --scale full \
  --allow-published-scale \
  --allow-openml-download \
  --mnist-source openml \
  --evaluate-all-permutation-views \
  --output-dir outputs/step2_associative_opmnist_confirmation_full
```

## Manifest And Claim Flags

Each run writes:

- `<prefix>_results.json`
- `<prefix>_manifest.json`

The manifest captures argv, git commit/dirty state when available, Python/JAX
environment basics, dataset metadata, protocol metadata, seed list, scale flag,
and the published-scale guard state.

The claim-support field is:

```json
"published_scale_external_claim_supported": false
```

for all smoke, partial, and dry-run outputs. It can become `true` only when all
of these are true:

- `--scale full` was executed with `--allow-published-scale`.
- The run completed at least 48,000,000 examples.
- The data source is true MNIST with the canonical 60,000/10,000 split.
- The protocol used 800 random pixel permutations, 60,000-example task blocks,
  no task id to the learner, and prediction before update at every step.
- Held-out evaluation covers all configured permutation views.

This separation is deliberate: a run can be configured for full scale, planned
as a dry run, or partially completed without being marked as published-scale
external confirmation.
