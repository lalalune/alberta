# Fast/Slow Transformer Worker

## Scope

This worker added an isolated Tiny Shakespeare experiment:

- `examples/The Alberta Plan/Step2/step2_tiny_shakespeare_fast_slow_transformer.py`
- Outputs under `outputs/step2_new_directions/fast_slow_transformer_worker/`

The candidate keeps the existing tiny one-block causal attention stem and token
readout, then replaces the residual FFN block with:

`hidden -> fast GELU FFN residual`

`hidden -> online prototype slow residual`

`[hidden, previous loss EMA, current novelty] -> learned sigmoid gate`

The fast path and slow prototype-value path use separate learning rates.

## Promoted 800-Step Run

Run:

```bash
source .venv/bin/activate
python "examples/The Alberta Plan/Step2/step2_tiny_shakespeare_fast_slow_transformer.py" \
  --steps 800 \
  --seeds 2 \
  --eval-steps 256 \
  --final-window 256 \
  --output-dir outputs/step2_new_directions/fast_slow_transformer_worker/run_800_promoted
```

Results:

| Metric | Baseline FFN | Fast/slow | Diff favoring fast/slow |
|---|---:|---:|---:|
| final-window NLL | 3.503801 | 3.502008 | +0.001793 |
| final-window perplexity | 33.241624 | 33.182062 | +0.059562 |
| eval NLL | 3.548977 | 3.546871 | +0.002107 |
| eval perplexity | 34.894415 | 34.835220 | +0.059195 |
| train steps/s | 2125.262563 | 1391.630652 | -733.631911 |

Seed-level final-window NLL:

| Seed | Baseline FFN | Fast/slow | Diff favoring fast/slow |
|---|---:|---:|---:|
| 0 | 3.501879 | 3.500160 | +0.001719 |
| 1 | 3.505723 | 3.503856 | +0.001868 |

Seed-level eval perplexity:

| Seed | Baseline FFN | Fast/slow | Diff favoring fast/slow |
|---|---:|---:|---:|
| 0 | 37.745617 | 37.849739 | -0.104122 |
| 1 | 32.043213 | 31.820702 | +0.222511 |

Verdict for the required 800-step run: the fast/slow transformer beat the tuned
FFN baseline on mean final-window NLL and mean eval perplexity, but the eval
perplexity win was not per-seed robust.

## Longer Check

The promoted setting was then run for 2000 steps with 2 seeds and a 512-step
final/eval window.

| Metric | Baseline FFN | Fast/slow | Diff favoring fast/slow |
|---|---:|---:|---:|
| final-window NLL | 3.323867 | 3.330712 | -0.006845 |
| final-window perplexity | 27.834510 | 28.029233 | -0.194723 |
| eval NLL | 3.272579 | 3.282361 | -0.009782 |
| eval perplexity | 26.681372 | 26.928391 | -0.247020 |

A conservative 2000-step setting (`slow_lr=0.1`, `gate_lr=0.05`,
`proto_bandwidth=2.0`) reduced but did not eliminate the loss:

| Metric | Baseline FFN | Fast/slow | Diff favoring fast/slow |
|---|---:|---:|---:|
| final-window NLL | 3.323867 | 3.326436 | -0.002568 |
| eval perplexity | 26.681372 | 26.796351 | -0.114980 |

## Assessment

This is a real fast/slow transformer-memory signal at 800 online steps, but it
is not yet a robust Tiny Shakespeare result. The slow path helps early enough
to beat final-window NLL in the required 2-seed 800-step run, but longer runs
show the tuned FFN baseline catching up and overtaking it.

The next likely fixes are:

- Learn the slow-path trust schedule instead of using a fixed gate bias and LR.
- Decay or regularize stale prototype values after the first online phase.
- Let the gate see a richer slow-path reliability statistic, not only novelty
  and previous loss EMA.
- Reduce the compute overhead; the 800-step promoted run was about 35% slower
  in steps/s than the baseline.
