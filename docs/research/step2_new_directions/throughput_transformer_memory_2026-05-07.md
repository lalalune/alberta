# Replay-Capped Transformer Memory Throughput

Date: 2026-05-07.

Scope: inspected `examples/The Alberta Plan/Step2/step2_tiny_shakespeare_advantage_memory_transformer.py`
and `src/alberta_framework/core/prototype_basis.py`, then added
`benchmarks/step2_transformer_memory_throughput.py`.

## Summary

The replay-capped advantage-memory transformer is slower than the FFN baseline
for two separate reasons.

1. The replay objective adds a second `model_parts` forward pass every token.
   In the post-FFN placement this doubles attention/FFN/readout forward work
   outside the gradient path; in pre-FFN KV placement it adds another two FFN
   applications.
2. The prototype path traverses the `P x D` center matrix repeatedly. The
   current-token path does activation, reset-slot selection, and center update
   as three separate distance reductions. Replay adds a fourth prototype lookup.

The benchmark shows that a production-style reusable JIT implementation is less
bad than short research-run wall-clock timings that include compilation, but the
steady-state replay path is still materially slower than FFN and becomes
memory/control dominated as the prototype budget grows.

## Commands

```bash
source .venv/bin/activate && python benchmarks/step2_transformer_memory_throughput.py --steps 256 --repeats 3 --eval-steps 128 --final-window 128 --output-dir outputs/benchmarks/step2_transformer_memory_throughput_run
source .venv/bin/activate && python benchmarks/step2_transformer_memory_throughput.py --steps 256 --repeats 3 --eval-steps 128 --final-window 128 --proto-count 256 --output-dir outputs/benchmarks/step2_transformer_memory_throughput_p256
source .venv/bin/activate && python benchmarks/step2_transformer_memory_throughput.py --steps 256 --repeats 3 --eval-steps 128 --final-window 128 --proto-count 512 --output-dir outputs/benchmarks/step2_transformer_memory_throughput_p512
```

Device: CPU (`cpu:0`). Shape unless varied: block size 32, `d_model=32`,
FFN hidden 64, vocab 65, replay size 128, scalar gate.

## Timings

Steady timings exclude compile+first-run. The benchmark hoists JIT functions so
this estimates a production kernel better than the current runner, which creates
fresh jitted scan functions inside each method call.

| Prototype count | Variant | Steps/s | Relative to FFN | State bytes | Forward MACs/step | Prototype distance passes |
|---:|---|---:|---:|---:|---:|---:|
| 64 | `baseline_ffn` | 11486.6 | 1.000x | 0 | 202784 | 0 |
| 64 | `advantage_post_current` | 7191.6 | 0.626x | 9104 | 208960 | 3 |
| 64 | `advantage_post_replay` | 6968.6 | 0.607x | 26008 | 417920 | 4 |
| 64 | `advantage_post_replay_fused_center` | 5918.7 | 0.515x | 26008 | 417920 | 3 |
| 256 | `baseline_ffn` | 10867.6 | 1.000x | 0 | 202784 | 0 |
| 256 | `advantage_post_current` | 5327.5 | 0.490x | 35984 | 221248 | 3 |
| 256 | `advantage_post_replay` | 4086.7 | 0.376x | 52888 | 442496 | 4 |
| 256 | `advantage_post_replay_fused_center` | 4319.3 | 0.397x | 52888 | 442496 | 3 |
| 512 | `baseline_ffn` | 11210.7 | 1.000x | 0 | 202784 | 0 |
| 512 | `advantage_post_current` | 3602.1 | 0.321x | 71824 | 237632 | 3 |
| 512 | `advantage_post_replay` | 2856.1 | 0.255x | 88728 | 475264 | 4 |
| 512 | `advantage_post_replay_fused_center` | 3016.3 | 0.269x | 88728 | 475264 | 3 |

The isolated fused-center update removes one duplicate center pass. At `P=64`
it is slower on CPU, likely because scatter/control overhead dominates the small
center matrix. At `P=256` and `P=512`, it improves replay post-FFN throughput by
about 5.7% and 5.6%. This is a valid low-risk patch, but it is not the main fix.

## Bottlenecks

The precise hot path in the current replay runner is:

- `model_parts(...)` computes the current-token fast and memory logits.
- `block.activations(...)` reads all prototype centers once for the memory
  residual.
- `select_center_slot(...)` reads all centers again to decide the reset slot.
- `block.update_centers(...)` reads all centers a third time and performs the
  actual center update.
- With `--gate-objective replay`, `prediction_advantage(...)` calls
  `model_parts(...)` on a replay buffer sample, adding another full transformer
  forward and another prototype activation.
- `memory_loss = cross_entropy_from_logits(logits, label)` is recomputed after
  the gradient call. This is only a small logsumexp duplicate, not a duplicate
  model forward.

The benchmark's MAC model only counts matmul-style work and intentionally omits
softmax, exponentials, scatter updates, and memory traffic. The measured
slowdown grows faster than MAC count as `P` increases, which is evidence that
prototype center loads, reductions, and dynamic updates are the scaling
bottleneck.

## Parameter And State Overhead

For this shape, the FFN baseline has 13537 trainable parameters and no extra
learner state.

The advantage-memory path adds `P * D + D` trainable parameters for prototype
values and bias. With `D=32`:

| P | Extra trainable params | Total trainable params |
|---:|---:|---:|
| 64 | 2080 | 15617 |
| 256 | 8224 | 21761 |
| 512 | 16416 | 29953 |

Scalar-gate memory state adds centers, bandwidths, counts, last-update indices,
gate/EMA scalars, and `init_value`. Replay adds a ring buffer of
`replay_size * block_size + replay_size + 2` int32 elements. With replay size
128 and block size 32 this adds 16904 bytes.

| P | Non-replay state bytes | Replay state bytes | Total replay state bytes |
|---:|---:|---:|---:|
| 64 | 9104 | 16904 | 26008 |
| 256 | 35984 | 16904 | 52888 |
| 512 | 71824 | 16904 | 88728 |

## Production Throughput Estimate

On this CPU, a reusable-JIT production implementation should expect roughly:

- `P=64`: about 7.0k steps/s for post-FFN replay memory versus 11.5k for FFN,
  or 1.65x slower.
- `P=256`: about 4.1k steps/s unfused and 4.3k steps/s with fused center-slot
  update versus 10.9k for FFN, or 2.5-2.7x slower.
- `P=512`: about 2.9k steps/s unfused and 3.0k steps/s with fused center-slot
  update versus 11.2k for FFN, or 3.7-3.9x slower.

Short research runs can look worse because the current runner pays
compile+first-run cost per method call/seed. In these measurements that cold
cost is 0.4-1.7 seconds for only 256 training steps, while hot execution is
0.02-0.09 seconds.

## What A Fused Production Implementation Should Change

1. Hoist JIT kernels out of per-method runner functions. Build one reusable
   scan per variant and shape.
2. Add a `PrototypeBasisBlock.update_centers_with_slot(...)` style primitive
   that returns `(new_state, center_metrics, slot, novel)` from one distance
   pass. This removes `select_center_slot(...)` as a separate full-center pass.
3. Return `memory_loss` from the training loss auxiliary instead of recomputing
   it after `value_and_grad`. This is small but free.
4. If the research objective allows stale replay features, store replay
   `basis_input` and base loss/logits in the ring buffer instead of replaying
   the raw context through attention and FFN. That would preserve the delayed
   gate update while avoiding the duplicate transformer forward. If exact
   latest-parameter replay is required, lower the replay-gate frequency or batch
   replay evaluations every `K` steps.

## Recommended Next Patch

Patch the runner behind a flag, for example `--fused-center-update`, by moving
the benchmark's fused center-slot update into the runner or into
`PrototypeBasisBlock`. Validate against the current path on fixed seeds by
checking identical slot/reset decisions and matching online NLL/gate summaries.
After that, add an experimental `--replay-cache basis_input` mode to quantify
the larger duplicate-forward opportunity separately from the center-update
cleanup.
