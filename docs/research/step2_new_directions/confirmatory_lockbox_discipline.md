# Step 2 Confirmatory Lockbox Discipline

Date added: 2026-05-07

The lockbox preset is now gated in
`benchmarks/step2_transformer_confirmatory_paperworthy_runner.py`. A real
lockbox run is refused unless the caller supplies a validation decision summary
whose JSON field `clears_confirmatory_bar` is exactly `true`.

Allowed lockbox evaluation:

```bash
source .venv/bin/activate
python benchmarks/step2_transformer_confirmatory_paperworthy_runner.py \
  --preset lockbox \
  --validation-decision-summary \
  outputs/step2_new_directions/advantage_memory_transformer_confirmatory_validation_30seed/confirmatory_decision_summary.json
```

Planning-only exception:

```bash
source .venv/bin/activate
python benchmarks/step2_transformer_confirmatory_paperworthy_runner.py \
  --preset lockbox \
  --dry-run \
  --allow-lockbox-without-validation
```

The planning exception manifests commands and split hashes only. It must not run
the transformer runner, the report generator, or the decision summary.

The wrapper manifest schema records command status (`planned`, `completed`,
`failed`, or `skipped_existing`), return codes when a command is executed,
skipped-existing status, hashes for runner result artifacts, report input and
output artifacts, decision outputs, and source/train/eval/derived data split
hashes. The decision summary no longer contains the stale
`original_runner_missing_offsets` flag; promotion now gates on nested per-seed
offsets in both `records[].data_offsets` and `manifest.seed_runs[].methods`.
