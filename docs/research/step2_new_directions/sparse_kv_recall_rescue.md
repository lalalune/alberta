# Sparse KV Recall Rescue

Date: 2026-05-07.

This pass stayed on the existing `sparse_kv_recall` external probe. No new
benchmark was added.

## Commands

```bash
source .venv/bin/activate
PYTHONUNBUFFERED=1 python "examples/The Alberta Plan/Step2/step2_sequence_external_memory_benchmarks.py" --benchmarks sparse_kv_recall --memory-variants rescue_split_update_utility --steps 900 --seeds 3 --eval-steps 256 --final-window 256 --proto-count 128 --min-update-gate 0.2 --utility-replace-threshold 0.0 --utility-novelty-bonus 0.0 --output-dir outputs/step2_new_directions/sparse_kv_recall_rescue_p128_gate02_thr0
```

```bash
source .venv/bin/activate
PYTHONUNBUFFERED=1 python "examples/The Alberta Plan/Step2/step2_sequence_external_memory_benchmarks.py" --benchmarks sparse_kv_recall --memory-variants rescue_split_update_utility --steps 900 --seeds 3 --eval-steps 256 --final-window 256 --proto-count 256 --min-update-gate 0.2 --utility-replace-threshold 0.0 --utility-novelty-bonus 0.0 --output-dir outputs/step2_new_directions/sparse_kv_recall_rescue_p256_gate02_thr0
```

## Result

Increasing prototype budget and forcing a stronger update gate did not fix
sparse key-value recall.

| Variant | FFN held-out NLL | Static replay held-out NLL | Rescue held-out NLL | Rescue gate |
|---|---:|---:|---:|---:|
| P128, update gate 0.2 | about 2.819 | about 2.819 | about 2.820 | about 0.046 |
| P256, update gate 0.2 | about 2.819 | about 2.819 | about 2.820 | about 0.046 |

The larger budgets reproduced the previous pattern: static replay stays capped
open and does not help; rescue closes the prediction gate and avoids most harm,
but does not create a useful associative binding.

## Diagnosis

Sparse KV recall is not blocked only by prototype count. The current residual
prototype memory is still a smooth feature memory, not a clean key-value binding
mechanism. It needs a value-update rule that binds queried keys to labels or
latent values directly enough to be retrieved later.

Promising next mechanisms:

- prototype-key / value-table update with a direct associative readout;
- separate write gate and read gate, where write can stay open under uncertainty
  while read remains closed until utility is proven;
- replacement by key collision and stale value utility, not only distance or
  negative advantage;
- local Hebbian or recursive least-squares value update on active prototypes.

The current rescue rules are still useful as safety rules, but they are not a
universal associative memory.
