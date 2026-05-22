# Transformer Compute-Fairness Pareto Plan

Date: 2026-05-07.

Scope: Worker J. Owned outputs are
`outputs/step2_new_directions/compute_fairness_pareto_smoke/` and
`benchmarks/step2_transformer_compute_fairness_pareto.py`. The throughput
benchmark was read but not edited.

## Inputs

Existing quality artifacts:

- `outputs/step2_new_directions/advantage_memory_transformer_confirmatory_validation_30seed/validation_3000_30seed_eval4096_fw512_eb512_replay128_scalar_glr05_l2_01_gmax015/results.json`
- `outputs/step2_new_directions/advantage_memory_transformer_3000_10seed_replay128_scalar_glr05_l2_01_gmax015/results.json`
- `outputs/step2_new_directions/advantage_memory_transformer_5000_10seed_replay128_scalar_glr05_l2_01_gmax015/results.json`
- `outputs/step2_new_directions/advantage_memory_transformer_10000_10seed_replay128_scalar_glr05_l2_01_gmax015/results.json`

Existing throughput artifacts:

- `outputs/step2_new_directions/replay_cache_ablation_p64/results.json`
- `outputs/step2_new_directions/replay_cache_ablation_p256/results.json`

Stronger FFN validation artifact:

- `outputs/step2_new_directions/stronger_ffn_validation_10000_30seed_best2/results.json`

Command:

```bash
source .venv/bin/activate && python benchmarks/step2_transformer_compute_fairness_pareto.py --output-dir outputs/step2_new_directions/compute_fairness_pareto_with_stronger_ffn
```

## Fairness Frames

Positive NLL diffs favor replay memory.

| Frame | Smoke answer | Production status |
|---|---|---|
| Equal token budget | Available from existing paired result artifacts. Post-FFN exact replay has held-out NLL diffs of `+0.000540` at 3000 steps/30 seeds, `+0.000282` at 3000 steps/10 seeds, `+0.001596` at 5000 steps/10 seeds, and `+0.017694` at 10000 steps/10 seeds. | Needs frozen train/validation/lockbox and larger eval for all horizons. |
| Equal params | Not answered by existing artifacts. The P=64 memory candidate has `15617` trainable params versus FFN `13537` (`+2080`). A parameter-matched FFN is straightforward: with this shape, FFN hidden `96` gives `15617` params. | Must run tuned FFN h=64, parameter-matched FFN h=96, and memory with identical paired streams. |
| Equal state bytes | Not answered by existing artifacts. The exact replay memory candidate carries `26008` extra state bytes versus FFN `0`. | Must add a state-matched control or explicitly report that the claim is not state-fair. |
| Equal hot-loop wall clock | Existing throughput artifacts separate compile+first from steady hot time. At P=64, exact fused replay runs `5113.9` steps/s versus FFN `11528.6`, so FFN gets about `1154` tokens in the same hot time as memory gets `512`. | Needs quality curves/checkpoints so FFN quality at those additional tokens is measured, not inferred. |
| Quality to threshold | Using `0.005` held-out NLL as the practical effect threshold, 3000-step 30-seed and 3000/5000-step 10-seed post-FFN memory do not clear the threshold. The 10000-step 10-seed run clears the mean threshold, but with only 512 eval contexts and high uncertainty. | Must preregister quality thresholds and record first-token and first-hot-second crossing. |

## Stronger FFN Update

The compute-fairness runner now reads the 30-seed stronger-FFN validation
artifact and writes:

- `outputs/step2_new_directions/compute_fairness_pareto_with_stronger_ffn/stronger_ffn_quality.csv`
- `outputs/step2_new_directions/compute_fairness_pareto_with_stronger_ffn/SUMMARY.md`

The stronger FFN rows overturn the earlier narrow replay-memory claim:

| Method | Final NLL | Eval NLL | Eval PPL | Post-memory eval advantage | Post-memory final advantage |
|---|---:|---:|---:|---:|---:|
| `ffn_h96_lr0p05` | `2.7564 +/- 0.0127` | `2.9317 +/- 0.0188` | `18.86 +/- 0.36` | `+0.123620` | `-0.035110` |
| `ffn_h128_lr0p05` | `2.7516 +/- 0.0123` | `2.9351 +/- 0.0208` | `18.94 +/- 0.41` | `+0.120251` | `-0.030322` |
| `ffn_h96_lr0p1` | `2.7153 +/- 0.0130` | `2.9989 +/- 0.0246` | `20.24 +/- 0.51` | `+0.056463` | `+0.005895` |
| `ffn_h128_lr0p1` | `2.7158 +/- 0.0125` | `3.0103 +/- 0.0277` | `20.53 +/- 0.60` | `+0.045009` | `+0.005400` |

Positive memory-advantage values mean the FFN row has lower NLL than the
post-FFN replay-memory candidate. The tuned h=96/h=128 FFNs are materially
better on held-out NLL, and the lr=0.1 variants also match or slightly beat
final-window NLL. Therefore the replay-memory transformer should not be treated
as a promoted candidate, and no lockbox run is justified from these results.

## Fused Center And Cached Replay

The exact fused-center path and cached replay path must remain separate.

The throughput benchmark's exactness check shows fused center preserves the
current raw-context replay output exactly. It is a production implementation
cleanup: same science contract, fewer center-distance passes.

Cached replay is different. It stores stale `basis_input` and base loss instead
of replaying raw context through latest parameters. It is a valid ablation, but
not a substitute for exact replay. In the P=64 smoke, cached replay is faster
than exact fused replay (`5913.2` versus `5113.9` steps/s) but has worse
same-token held-out NLL by `0.0263`; the earlier cache-ablation note also shows
it changes the gate behavior materially.

## Pareto Conclusions

- Baseline FFN remains a resource Pareto point: fewer params, zero extra state,
  and faster hot loop.
- Exact replay memory remains only a tentative quality Pareto point under equal
  token budget. Its quality edge is tiny at 3000/5000 steps and is not
  parameter-, state-, or hot-time-fair.
- At P=64, exact fused replay dominates the exact unfused replay reference in
  the smoke table because quality/params/state are identical and hot time is
  lower. At P=256, CPU timing variance reverses that row, so the production
  benchmark should use more repeats and report confidence intervals.
- Cached replay must not be used in the production Pareto frontier. It changes
  behavior and is dominated by FFN in the P=64 smoke quality/resource table.

## Output Files

- `outputs/step2_new_directions/compute_fairness_pareto_smoke/SUMMARY.md`
- `outputs/step2_new_directions/compute_fairness_pareto_smoke/report.json`
- `outputs/step2_new_directions/compute_fairness_pareto_smoke/token_budget_quality.csv`
- `outputs/step2_new_directions/compute_fairness_pareto_smoke/resource_fairness_gaps.csv`
- `outputs/step2_new_directions/compute_fairness_pareto_smoke/hot_loop_quality.csv`
- `outputs/step2_new_directions/compute_fairness_pareto_smoke/pareto_smoke.csv`
- `outputs/step2_new_directions/compute_fairness_pareto_with_stronger_ffn/SUMMARY.md`
- `outputs/step2_new_directions/compute_fairness_pareto_with_stronger_ffn/report.json`
- `outputs/step2_new_directions/compute_fairness_pareto_with_stronger_ffn/stronger_ffn_quality.csv`

## Exact Production Benchmark Remaining

The exact production benchmark should run one frozen family:

1. Methods: tuned FFN h=64, parameter-matched FFN h=96, exact fused-center
   post-FFN replay memory, exact fused-center pre-FFN KV replay memory,
   no-replay memory, no-cap/fixed-gate memory, and cached replay as a separate
   ablation.
2. Fairness stops: equal token budget and equal hot-loop wall-clock budget.
   Compile+first-run must be reported separately and excluded from hot-loop
   stopping.
3. Resource reporting: trainable params/bytes, runtime state bytes including
   replay buffers, peak memory if available, forward work model, device,
   JAX/JAXLIB versions, git status, data hash, and exact command.
4. Quality reporting: rolling prequential final-window NLL, held-out validation
   checkpoints, first token and first hot second to each preregistered threshold,
   and one untouched lockbox report after selection.
5. Pairing: at least 30 paired seeds with stored train/eval offsets and the same
   stream for all methods in a fairness row.

Until that benchmark exists, the defensible claim is narrow: exact replay memory
has small exploratory equal-token wins, while compute-fair Pareto superiority is
not established.
