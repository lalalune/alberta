# Step 2 Confirmatory Held-Out Eval Streaming

The frozen confirmatory runner now passes `--eval-batch-size` to the Tiny
Shakespeare advantage-memory transformer runner. This changes only how held-out
metrics are computed: every seed still evaluates the same `eval_steps` contexts
from the same deterministic offset, and the reported NLL, accuracy, and
perplexity are aggregated over all held-out examples without subsampling.

The practical default for `validation` and `lockbox` is:

```bash
source .venv/bin/activate
python benchmarks/step2_transformer_confirmatory_paperworthy_runner.py \
  --preset validation \
  --eval-batch-size 512
```

This keeps the full 4096 held-out contexts per seed while limiting each JAX
evaluation compile to 512 contexts. A smoke run can use the preset default:

```bash
source .venv/bin/activate
python benchmarks/step2_transformer_confirmatory_paperworthy_runner.py \
  --preset smoke
```

The wrapper manifest records `eval_steps`, `requested_eval_batch_size`,
`eval_batch_size`, `eval_batches_per_seed`, and the aggregation rule under
`protocol`. Each underlying transformer `results.json` also records the
effective evaluation batch size in `config`, `manifest.config.raw_args`, seed
offset metadata, and the `manifest.evaluation` block.

Use `--eval-batch-size 0` only to reproduce the legacy full-context eval compile.
That path still evaluates the same examples, but it is not practical for the
30-seed, 4096-context confirmatory evaluation on ordinary local machines.
