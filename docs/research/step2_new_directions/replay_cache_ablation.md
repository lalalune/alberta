# Replay Cache Ablation

Date: 2026-05-07.

Scope: `benchmarks/step2_transformer_memory_throughput.py` only. The
confirmatory runner, external benchmark, and stratified runner were not edited.

## Question

Two changes must stay separated:

1. **Exact fused center update**: fuse slot selection and center update while
   preserving the current replay runner output exactly.
2. **Cached replay features**: store replay `basis_input` and base loss instead
   of replaying raw context through the latest-parameter model. This is faster,
   but it changes the gate signal and therefore the science contract.

## Commands

```bash
source .venv/bin/activate && pytest tests/test_prototype_basis.py -v
source .venv/bin/activate && ruff check benchmarks/step2_transformer_memory_throughput.py tests/test_prototype_basis.py
source .venv/bin/activate && mypy benchmarks/step2_transformer_memory_throughput.py tests/test_prototype_basis.py
source .venv/bin/activate && python benchmarks/step2_transformer_memory_throughput.py --steps 8 --repeats 1 --eval-steps 8 --final-window 8 --output-dir outputs/step2_new_directions/replay_cache_ablation_smoke
source .venv/bin/activate && python benchmarks/step2_transformer_memory_throughput.py --steps 512 --repeats 3 --eval-steps 128 --final-window 128 --proto-count 64 --output-dir outputs/step2_new_directions/replay_cache_ablation_p64
source .venv/bin/activate && python benchmarks/step2_transformer_memory_throughput.py --steps 512 --repeats 5 --eval-steps 128 --final-window 128 --proto-count 256 --output-dir outputs/step2_new_directions/replay_cache_ablation_p256
```

## Variant Manifest

The benchmark artifacts now label the behavior contract directly:

| Variant | Contract | Replay source | Status |
|---|---|---|---|
| `advantage_post_replay_exact_reference` | `exact_current_replay` | raw context, latest params | Current behavior reference |
| `advantage_post_replay_exact_fused_center` | `exact_current_replay` | raw context, latest params | Exact production path |
| `advantage_post_replay_cached_basis_ablation` | `cached_replay_ablation_changes_behavior` | stale `basis_input` and base loss | Science-changing ablation |

Exactness checks passed for both P=64 and P=256 runs:

| proto_count | Steps checked | Pack max abs diff | Metrics max abs diff |
|---:|---:|---:|---:|
| 64 | 512 | 0.0 | 0.0 |
| 256 | 512 | 0.0 | 0.0 |

The check compares the full runner output for the separate-slot/update reference
against the fused-center path: params, learner state, replay buffers, and online
metrics.

## Throughput

Device: CPU `cpu:0`. Shape: block size 32, `d_model=32`, FFN hidden 64, vocab
65, replay size 128, scalar gate. Steady timings exclude compile+first-run.

| proto_count | Variant | Contract | Steps/s | Steady s +/- stderr | Distance passes |
|---:|---|---|---:|---:|---:|
| 64 | `baseline_ffn` | baseline | 11528.6 | 0.0444 +/- 0.0007 | 0 |
| 64 | `advantage_post_replay_exact_reference` | exact | 3778.4 | 0.1355 +/- 0.0195 | 4 |
| 64 | `advantage_post_replay_exact_fused_center` | exact | 5113.9 | 0.1001 +/- 0.0026 | 3 |
| 64 | `advantage_post_replay_cached_basis_ablation` | cached ablation | 5913.2 | 0.0866 +/- 0.0057 | 3 |
| 256 | `baseline_ffn` | baseline | 1824.9 | 0.2806 +/- 0.1179 | 0 |
| 256 | `advantage_post_replay_exact_reference` | exact | 1506.9 | 0.3398 +/- 0.0392 | 4 |
| 256 | `advantage_post_replay_exact_fused_center` | exact | 1354.2 | 0.3781 +/- 0.0804 | 3 |
| 256 | `advantage_post_replay_cached_basis_ablation` | cached ablation | 2513.6 | 0.2037 +/- 0.0170 | 3 |

The P=256 run had high CPU timing variance, including the FFN baseline. The
behavior result is still decisive: exact fused center matched the current output
bit-for-bit, while cached replay did not preserve quality/gate metrics.

## Quality Deltas

Deltas are cached replay minus exact fused center. Positive NLL/perplexity is
worse. The exact reference and exact fused-center rows had identical metrics.

| proto_count | Final NLL delta | Eval NLL delta | Eval PPL delta | Gate delta | Advantage delta | Gate-update delta |
|---:|---:|---:|---:|---:|---:|---:|
| 64 | +0.0089 | +0.0263 | +1.0139 | +0.3569 | -0.0274 | +0.1276 |
| 256 | +0.0087 | +0.0202 | +0.7798 | +0.4097 | -0.0234 | +0.1218 |

The cached path makes the gate much more open (`~0.996` versus exact
`0.586-0.639`) because it updates from stale replay features and stale base
losses. That is not an implementation-only speedup; it changes the control
signal the learner sees.

## Recommendation

Keep `advantage_post_replay_exact_fused_center` as the exact-behavior production
path. It is safe to promote because the benchmark and unit test both verify that
it matches the current separate-slot/update output.

Keep `advantage_post_replay_cached_basis_ablation` out of confirmatory and
paperworthy runs unless it is explicitly named as a cached-replay ablation. It
can be useful for throughput science, but the quality and gate deltas show it
changes the method being evaluated.
