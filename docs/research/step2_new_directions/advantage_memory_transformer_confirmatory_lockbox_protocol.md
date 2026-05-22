# Replay-Capped Advantage-Memory Transformer Confirmatory Protocol

Date frozen: 2026-05-06

## Status

This document freezes the confirmatory benchmark protocol for the current Step 2
transformer candidate:

- runner: `examples/The Alberta Plan/Step2/step2_tiny_shakespeare_advantage_memory_transformer.py`;
- primary candidate: `advantage_post_ffn_memory`;
- mechanism: post-FFN residual prototype memory with replay-gated scalar resource
  cap;
- exploratory predecessor: `gate_objective=replay`, `replay_size=128`,
  `gate_lr=0.5`, `gate_l2=0.1`, `gate_max=0.15`.

The previous 10-seed Tiny Shakespeare runs remain exploratory. A paperworthy
claim requires a new run under this frozen protocol.

## Frozen Data Protocol

Use the exact Tiny Shakespeare bytes from:

`https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt`

The confirmatory split is byte-indexed and must be recorded with sha256 hashes:

| Shard | Byte range | Role |
| --- | ---: | --- |
| train | `[0, 800000)` | Online updates only |
| validation | `[800000, 950000)` | Confirmatory validation report |
| lockbox | `[950000, EOF)` | One-shot final report after validation passes |

The wrapper materializes two derived corpora so the unchanged runner can keep
using its `train_fraction` interface:

- validation corpus: `train + validation`;
- lockbox corpus: `train + lockbox`.

The wrapper records the derived corpus sha256, train fraction, source byte
ranges, git status, and exact commands before running.

## Primary Family

Positive paired differences favor the candidate.

| Item | Frozen choice |
| --- | --- |
| Primary method | `advantage_post_ffn_memory` |
| Baseline | `baseline_ffn_transformer` with tuned FFN config from the runner |
| Primary metrics | `final_window_nll`, `eval_nll` |
| Horizons | `3000`, `5000`, `10000` online updates |
| Final online window | `512` examples |
| Held-out contexts | `4096` contexts per seed |
| Paired seeds | `30` seeds, same seed and stream construction per method |
| Multiplicity correction | Holm correction across `3 horizons x 2 primary metrics` |
| Smallest effect of interest | held-out `eval_nll` improvement of at least `0.005` at the main horizon (`10000`) |
| Alpha | `0.05` two-sided paired t-test, with Wilcoxon and sign tests reported |

Perplexity is report-only because it is a monotone transform of NLL.

## Candidate Configuration

All candidate and baseline shape/training flags are frozen:

```bash
--block-size 32
--d-model 32
--mlp-hidden 64
--proto-count 64
--baseline-lr 0.15
--fast-lr 0.15
--slow-lr 0.1
--grad-clip 1.0
--proto-update-rate 0.3
--proto-novelty-threshold 0.0002
--proto-bandwidth 0.01
--gate-init-logit -3.0
--gate-lr 0.5
--gate-decay 0.995
--gate-max 0.15
--advantage-margin 0.0
--gate-l2 0.1
--gate-mode scalar
--gate-objective replay
--replay-size 128
--train-loss-mode memory
--memory-loss-weight 1.0
--reset-mode meta_ema
--seed 0
```

The runner also emits `advantage_pre_ffn_memory`; that placement is secondary
only and must not replace the post-FFN primary method after seeing validation or
lockbox outcomes.

## Baseline Family

The current wrapper runs the available tuned FFN baseline emitted by the runner.
For a full paper claim, the final report must also include these additional
controls or explicitly narrow the claim:

- parameter-matched FFN transformer;
- compute-matched or wall-clock-matched FFN transformer;
- wider tuned FFN transformer under the same online protocol;
- no-replay memory;
- no-cap replay memory;
- fixed-gate or cap-only memory;
- pre-FFN KV placement as a secondary placement check.

The present confirmatory wrapper is sufficient to decide whether the leading
candidate survives a stronger paired-seed rerun. It is not sufficient by itself
to prove the mechanism cannot be explained by parameters or compute.

## Failure Flags

The report must flag, and the promotion decision must account for:

- any NaN or non-finite metric in any primary record;
- negative paired mean difference on either primary metric at any horizon;
- Holm-corrected primary p-value `>= 0.05`;
- 95% CI lower bound for `eval_nll` at `10000` below the `0.005` smallest
  effect of interest;
- mean gate open above `0.05` while mean final-window advantage is negative;
- resource saturation, defined as mean gate within 90% of `gate_max` or mean
  active prototypes above 95% of `proto_count`;
- paired seed outliers beyond 3 standard deviations from the paired mean,
  reported but not dropped;
- missing train/eval offset manifest fields in the original runner artifact.

## Exact Commands

Smoke run used to verify the wrapper path:

```bash
source .venv/bin/activate
python benchmarks/step2_transformer_confirmatory_paperworthy_runner.py \
  --preset smoke \
  --seeds 2 \
  --steps 300 \
  --eval-steps 128 \
  --final-window 128 \
  --output-root outputs/step2_new_directions/advantage_memory_transformer_confirmatory_smoke
```

Frozen 30-seed validation run:

```bash
source .venv/bin/activate
python benchmarks/step2_transformer_confirmatory_paperworthy_runner.py \
  --preset validation \
  --seeds 30 \
  --steps 3000 5000 10000 \
  --eval-steps 4096 \
  --final-window 512 \
  --output-root outputs/step2_new_directions/advantage_memory_transformer_confirmatory_validation_30seed
```

One-shot lockbox command, to run only after the validation report clears the
predeclared bar:

```bash
source .venv/bin/activate
python benchmarks/step2_transformer_confirmatory_paperworthy_runner.py \
  --preset lockbox \
  --seeds 30 \
  --steps 3000 5000 10000 \
  --eval-steps 4096 \
  --final-window 512 \
  --output-root outputs/step2_new_directions/advantage_memory_transformer_confirmatory_lockbox_30seed
```

## Promotion Decision Rule

The replay-capped post-FFN candidate clears this confirmatory protocol only if:

- all six primary paired means are positive;
- all six primary Holm-corrected p-values support the directional claim or the
  paper explicitly downgrades to descriptive evidence;
- the `10000`-step held-out `eval_nll` 95% CI lower bound is at least `0.005`;
- the lockbox run is positive on both primary metrics at all horizons;
- failure flags are absent or explained without dropping seeds;
- parameter-matched and compute-matched baselines do not explain the result.

Until then, the correct wording is that replay-capped memory remains a promising
exploratory Step 2 candidate, not a confirmed paperworthy advantage.
