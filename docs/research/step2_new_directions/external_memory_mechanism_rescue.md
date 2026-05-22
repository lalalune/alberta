# External Memory Mechanism Rescue

Date: 2026-05-07

## Scope

Worker D changed only the external sequence benchmark runner and this report.
The goal was to rescue the external benchmark failure through gate and
replacement dynamics, not by adding probes. The benchmark set stayed fixed:
`block_shift_markov`, `delayed_copy`, `sparse_kv_recall`, and
`local_text_motif`.

## Mechanism Variants Tried

All variants keep the same post-FFN prototype residual memory interface and the
same replay buffer comparator.

| Variant | Mechanism change |
| --- | --- |
| `static_replay` | Original replay-capped scalar gate from the Tiny Shakespeare follow-up: `gate_objective=replay`, `train_loss_mode=memory`, `reset_mode=meta_ema`, scalar prediction/update gate, least-used/oldest replacement. |
| `rescue_close_gate` | Scalar gate decay is anchored back to `gate_init_logit` instead of drifting toward zero; negative gate signals are multiplied by `8`; replacement reset uses zero rows. Replacement dynamics are otherwise unchanged. |
| `rescue_proto_utility` | Per-prototype prediction gates plus per-prototype utility EMA. Novel replacement uses the lowest-utility slot only when utility is below `-0.002`; otherwise the nearest center is updated. Utility also raises the novelty threshold by `0.05 * max(nearest_utility, 0)`. Reset uses zero rows. |
| `rescue_split_update_utility` | `rescue_proto_utility` plus a separate update/probe gate. Prediction uses the learned low gate, but gradient updates and replay advantage measurement use `max(prediction_gate, 0.08)`. |

The exact short-suite command was:

```bash
source .venv/bin/activate && PYTHONUNBUFFERED=1 python "examples/The Alberta Plan/Step2/step2_sequence_external_memory_benchmarks.py" \
  --benchmarks all \
  --memory-variants all \
  --steps 300 \
  --seeds 3 \
  --eval-steps 128 \
  --final-window 128 \
  --block-size 32 \
  --d-model 16 \
  --mlp-hidden 32 \
  --proto-count 32 \
  --replay-size 32 \
  --output-dir outputs/step2_new_directions/external_memory_mechanism_rescue_short_3seed
```

Because `rescue_split_update_utility` won two of four benchmarks in the
3-seed run, it was escalated:

```bash
source .venv/bin/activate && PYTHONUNBUFFERED=1 python "examples/The Alberta Plan/Step2/step2_sequence_external_memory_benchmarks.py" \
  --benchmarks all \
  --memory-variants static_replay rescue_split_update_utility \
  --steps 300 \
  --seeds 10 \
  --eval-steps 128 \
  --final-window 128 \
  --block-size 32 \
  --d-model 16 \
  --mlp-hidden 32 \
  --proto-count 32 \
  --replay-size 32 \
  --output-dir outputs/step2_new_directions/external_memory_mechanism_rescue_split_10seed
```

Full artifacts:

- `outputs/step2_new_directions/external_memory_mechanism_rescue_compile_smoke/`
- `outputs/step2_new_directions/external_memory_mechanism_rescue_short_3seed/`
- `outputs/step2_new_directions/external_memory_mechanism_rescue_split_10seed/`

## Three-Seed Sweep

Positive diffs favor the row method over the FFN. The static replay comparator
is included to show the original failure mode.

| Benchmark | Variant | Final-window NLL diff vs FFN | Held-out NLL diff vs FFN | Gate | Allocation rate |
| --- | --- | ---: | ---: | ---: | ---: |
| `block_shift_markov` | `static_replay` | -0.000151 | -0.000121 | 0.146575 | 1.000000 |
| `block_shift_markov` | `rescue_close_gate` | -0.000004 | -0.000002 | 0.035976 | 1.000000 |
| `block_shift_markov` | `rescue_proto_utility` | -0.000008 | -0.000003 | 0.046450 | 0.000000 |
| `block_shift_markov` | `rescue_split_update_utility` | +0.000081 | +0.000120 | 0.046413 | 0.002604 |
| `delayed_copy` | `static_replay` | -0.000065 | -0.000062 | 0.146756 | 1.000000 |
| `delayed_copy` | `rescue_close_gate` | -0.000004 | +0.000001 | 0.035770 | 1.000000 |
| `delayed_copy` | `rescue_proto_utility` | -0.000010 | -0.000003 | 0.046909 | 0.000000 |
| `delayed_copy` | `rescue_split_update_utility` | +0.000081 | +0.000153 | 0.046934 | 0.015625 |
| `sparse_kv_recall` | `static_replay` | -0.000095 | +0.000085 | 0.146944 | 1.000000 |
| `sparse_kv_recall` | `rescue_close_gate` | -0.000015 | +0.000002 | 0.036067 | 1.000000 |
| `sparse_kv_recall` | `rescue_proto_utility` | -0.000018 | +0.000003 | 0.046908 | 0.000000 |
| `sparse_kv_recall` | `rescue_split_update_utility` | +0.000005 | -0.000164 | 0.046817 | 0.000000 |
| `local_text_motif` | `static_replay` | -0.000292 | -0.000379 | 0.146590 | 1.000000 |
| `local_text_motif` | `rescue_close_gate` | -0.000028 | -0.000034 | 0.035136 | 1.000000 |
| `local_text_motif` | `rescue_proto_utility` | -0.000041 | -0.000049 | 0.046538 | 0.000000 |
| `local_text_motif` | `rescue_split_update_utility` | -0.000038 | -0.000093 | 0.046382 | 0.010417 |

The 3-seed sweep fixed the open-gate symptom for every rescue variant.
`rescue_proto_utility` and `rescue_split_update_utility` also stopped the
replacement churn: replacement was blocked in roughly all full-dictionary steps.
Only `rescue_split_update_utility` produced a predictive win, and only on two
benchmarks.

## Ten-Seed Follow-Up

Positive diffs favor the method named in the column.

| Benchmark | Static final diff vs FFN | Split final diff vs FFN | Split final diff vs static | Static eval diff vs FFN | Split eval diff vs FFN | Split eval diff vs static | Split gate | Split allocation |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `block_shift_markov` | -0.000187 | +0.000066 | +0.000253 | -0.000138 | +0.000110 | +0.000248 | 0.046477 | 0.006250 |
| `delayed_copy` | -0.000055 | +0.000079 | +0.000134 | -0.000076 | +0.000091 | +0.000167 | 0.046955 | 0.011719 |
| `sparse_kv_recall` | -0.000134 | -0.000017 | +0.000117 | +0.000002 | -0.000125 | -0.000127 | 0.046808 | 0.000781 |
| `local_text_motif` | -0.000305 | -0.000045 | +0.000260 | -0.000357 | -0.000124 | +0.000233 | 0.046447 | 0.005469 |

Aggregate over all 40 seed-benchmark cells:

| Method | Final-window NLL diff vs FFN | Held-out NLL diff vs FFN | Benchmark wins/losses on final NLL | Benchmark wins/losses on held-out NLL |
| --- | ---: | ---: | ---: | ---: |
| `static_replay` | -0.000170 +/- 0.000017 | -0.000142 +/- 0.000027 | 0 / 4 | 1 / 3 |
| `rescue_split_update_utility` | +0.000021 +/- 0.000011 | -0.000012 +/- 0.000028 | 2 / 2 | 2 / 2 |

## Assessment

The mechanism failure is fixed, but the external benchmark failure is not.

Fixed:

- Static replay kept its prediction gate open near the cap despite negative
  final-window advantage: gate around `0.146-0.147`.
- The rescue mechanisms closed the prediction gate under negative advantage:
  scalar close gate around `0.035`, utility variants around `0.046-0.047`.
- Utility replacement stopped full-budget churn. In the 10-seed split run,
  allocation rates were `0.000781` to `0.011719` instead of `1.0`.
- Zero reset on replacement removed inherited stale value rows for the local
  rescue variants.
- The split update/prediction gate allowed a minimum update/probe gate of `0.08`
  while keeping prediction gated low.

Not fixed:

- Final-window advantage remains negative for the split-gate utility variant on
  every benchmark, around `-0.00004` to `-0.00011`.
- The 10-seed split run does not robustly beat the FFN. It wins
  `block_shift_markov` and `delayed_copy`, but loses `sparse_kv_recall` and
  `local_text_motif` on held-out NLL.
- `sparse_kv_recall` remains the important unresolved failure: closing the gate
  and stopping replacement churn did not make the residual prototype block act
  like an addressable key/value memory.

## Promotion

Promote the following mechanism rules into the next candidate search:

- gate decay anchored to the closed initialization, not toward zero;
- explicit negative-advantage closure;
- per-prototype utility for replacement eligibility;
- reset replaced value rows to zero;
- separate update/probe gate from prediction gate.

Do not promote the current external memory candidate as solved. The best
variant to carry forward is `rescue_split_update_utility`, but only as a
mechanism rescue baseline for longer or better-addressed memory experiments,
not as a confirmed replacement for the FFN/static replay external benchmark.
