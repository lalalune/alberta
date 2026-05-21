# Step 2 UPGD Transformer-Scale Check

Date: 2026-05-05

## Question

Can the current Step 2 UPGD learner replace transformer MLP/FFN blocks in a
small language-model demo, fairly compared to an MLP transformer?

Short answer: not yet. The existing Tiny Shakespeare integration is useful
as an integration smoke test, but it is not strong transformer-scale evidence.
In the most fair bounded run here, UPGD was parameter matched to the MLP
FFN+readout and lost on cross-entropy/perplexity after 2,000 online examples,
while carrying more state and running substantially slower.

## What Is Implemented

### `step2_tiny_shakespeare_upgd_transformer.py`

This is the only current language-modeling script. It implements char-level
online next-token prediction on Tiny Shakespeare.

- `mlp_transformer`: trainable one-head causal attention, a residual GELU
  FFN (`d_model -> mlp_hidden -> d_model`), and a linear softmax head.
  It is trained by manual clipped SGD on next-token cross-entropy.
- `upgd_transformer`: trainable one-head causal attention, then an observation
  formed by concatenating the last token state and mean sequence state
  (`2 * d_model`). The next-token learner/readout is
  `UPGDLearner.step2_default(..., readout_mode="softmax_ce")`.
- The UPGD attention front-end is trained with clipped SGD through
  `learner.predict`. The UPGD learner itself updates with its own online Step 2
  utility/perturbation rule.
- This does not isolate an FFN-only replacement. It replaces the MLP/readout
  learner with UPGD, and the UPGD model sees a different feature vector from
  the MLP baseline's FFN path.

UPGD factory settings for this path:

- `loss_normalization="target_structure"`
- `readout_mode="softmax_ce"`
- `ObGDBounding(kappa=0.5)`
- `sparsity=0.5`
- `use_layer_norm=True`
- `perturbation_sigma=1e-4`
- `perturbation_noise="rademacher"`
- `perturbation_interval=16`
- `track_unit_utilities=False`
- `track_gradient_history=False`

### `step2_upgd_transformer_demo.py`

This is not a language-modeling benchmark. It is a synthetic online sequence
matching task with a frozen deterministic one-head attention stem. The baseline
sees the same frozen attention features and uses `MultiHeadMLPLearner`; the
UPGD branch uses `UPGDLearner`. Targets are two-head one-hot vectors and
metrics are MSE/accuracy. This script validates attention-feature integration,
not transformer LM quality.

## Harness Added

Added:

- `output/subagents/transformer_scale/run_tiny_shakespeare_compare.py`

The harness imports the existing Tiny Shakespeare script, leaves it unchanged,
and adds:

- per-method train/eval wall-clock
- trainable parameter counts
- carried state element/byte counts
- parameter-matched UPGD hidden size

The parameter match used here is:

- MLP: `d_model=32`, `mlp_hidden=64`
- UPGD: `upgd_hidden=48`

At vocab size 65 and block size 32, this gives almost identical trainable
counts:

| Method | Attention trainable | Non-attention trainable | Total trainable | Total state elements | State bytes |
|---|---:|---:|---:|---:|---:|
| `mlp_transformer` | 7,200 | 6,337 | 13,537 | 13,537 | 54,148 |
| `upgd_transformer` | 7,200 | 6,305 | 13,505 | 16,652 | 66,612 |

UPGD has 32 fewer trainable parameters but 3,147 extra bookkeeping/state
elements in this setting.

## Commands

Smoke for harness correctness:

```bash
.venv/bin/python output/subagents/transformer_scale/run_tiny_shakespeare_compare.py \
  --steps 20 \
  --seeds 1 \
  --eval-steps 16 \
  --final-window 16 \
  --block-size 16 \
  --d-model 16 \
  --mlp-hidden 32 \
  --upgd-hidden 24 \
  --output-dir output/subagents/transformer_scale/script_smoke
```

Parameter-matched bounded run:

```bash
.venv/bin/python output/subagents/transformer_scale/run_tiny_shakespeare_compare.py \
  --steps 800 \
  --seeds 3 \
  --eval-steps 256 \
  --final-window 256 \
  --block-size 32 \
  --d-model 32 \
  --mlp-hidden 64 \
  --upgd-hidden 48 \
  --mlp-lr 0.03 \
  --upgd-lr 0.03 \
  --attention-lr 0.003 \
  --grad-clip 1.0 \
  --output-dir output/subagents/transformer_scale/tiny_shakespeare_param_matched_800x3
```

Longer bounded run:

```bash
.venv/bin/python output/subagents/transformer_scale/run_tiny_shakespeare_compare.py \
  --steps 2000 \
  --seeds 3 \
  --eval-steps 512 \
  --final-window 512 \
  --block-size 32 \
  --d-model 32 \
  --mlp-hidden 64 \
  --upgd-hidden 48 \
  --mlp-lr 0.03 \
  --upgd-lr 0.03 \
  --attention-lr 0.003 \
  --grad-clip 1.0 \
  --output-dir output/subagents/transformer_scale/tiny_shakespeare_param_matched_2000x3
```

Lint/check:

```bash
.venv/bin/ruff check output/subagents/transformer_scale/run_tiny_shakespeare_compare.py
```

## Results

### 800 steps x 3 seeds

Raw output:

- `output/subagents/transformer_scale/tiny_shakespeare_param_matched_800x3/results.json`
- `output/subagents/transformer_scale/tiny_shakespeare_param_matched_800x3/SUMMARY.md`

| Metric | MLP transformer | UPGD transformer | Diff favoring UPGD |
|---|---:|---:|---:|
| `final_window_nll` | 3.4427 +/- 0.0709 | 3.4433 +/- 0.0600 | -0.0006 +/- 0.0119 |
| `final_window_accuracy` | 0.1589 +/- 0.0094 | 0.1263 +/- 0.0057 | -0.0326 +/- 0.0116 |
| `final_window_perplexity` | 31.4277 +/- 2.2096 | 31.4032 +/- 1.8594 | +0.0245 +/- 0.3874 |
| `eval_nll` | 3.4239 +/- 0.1459 | 3.4126 +/- 0.1563 | +0.0112 +/- 0.0147 |
| `eval_accuracy` | 0.1432 +/- 0.0154 | 0.1432 +/- 0.0154 | +0.0000 +/- 0.0000 |
| `eval_perplexity` | 31.3515 +/- 4.6120 | 31.0921 +/- 4.8267 | +0.2594 +/- 0.3560 |
| `train_s` | 1.6773 +/- 0.2533 | 12.9635 +/- 2.5326 | -11.2862 +/- 2.6949 |
| `train_steps_per_s` | 502.2192 +/- 84.2330 | 67.1860 +/- 14.2571 | -435.0332 +/- 92.9370 |

Interpretation: CE/perplexity is effectively tied within the tiny 3-seed
smoke. UPGD loses final-window accuracy and is much slower.

### 2,000 steps x 3 seeds

Raw output:

- `output/subagents/transformer_scale/tiny_shakespeare_param_matched_2000x3/results.json`
- `output/subagents/transformer_scale/tiny_shakespeare_param_matched_2000x3/SUMMARY.md`

| Metric | MLP transformer | UPGD transformer | Diff favoring UPGD |
|---|---:|---:|---:|
| `final_window_nll` | 3.2572 +/- 0.0671 | 3.3453 +/- 0.0710 | -0.0881 +/- 0.0067 |
| `final_window_accuracy` | 0.1589 +/- 0.0073 | 0.1361 +/- 0.0095 | -0.0228 +/- 0.0047 |
| `final_window_perplexity` | 26.0910 +/- 1.6941 | 28.5085 +/- 1.9652 | -2.4175 +/- 0.3138 |
| `eval_nll` | 3.3438 +/- 0.1067 | 3.4306 +/- 0.1052 | -0.0868 +/- 0.0050 |
| `eval_accuracy` | 0.1582 +/- 0.0137 | 0.1562 +/- 0.0148 | -0.0020 +/- 0.0011 |
| `eval_perplexity` | 28.6427 +/- 2.9496 | 31.2324 +/- 3.1941 | -2.5897 +/- 0.2850 |
| `train_s` | 2.6548 +/- 1.3610 | 10.6214 +/- 1.3137 | -7.9666 +/- 2.6240 |
| `train_steps_per_s` | 1165.6382 +/- 409.2628 | 194.5023 +/- 25.2502 | -971.1359 +/- 434.4256 |

Per-seed paired diffs favoring UPGD:

| Metric | Seed diffs | Mean |
|---|---:|---:|
| `final_window_nll` | -0.1014, -0.0804, -0.0824 | -0.0881 |
| `eval_nll` | -0.0769, -0.0928, -0.0907 | -0.0868 |
| `final_window_accuracy` | -0.0293, -0.0137, -0.0254 | -0.0228 |
| `eval_accuracy` | -0.0039, 0.0000, -0.0020 | -0.0020 |
| `train_s` | -9.3688, -11.6452, -2.8859 | -7.9666 |

Interpretation: after 2,000 online updates, the fair parameter-matched run
clearly favors the MLP transformer on CE/perplexity and final-window accuracy.
UPGD is also about 5.5x slower by the reported mean train steps/s in this
JAX-compile-included harness.

## Critical Assessment

This does not support replacing transformer FFN/MLP blocks with the current
Step 2 UPGD learner.

Primary blockers:

1. Architecture mismatch. The Tiny Shakespeare UPGD path replaces the
   next-token learner/readout, not just the transformer FFN sublayer. The MLP
   baseline has a normal residual FFN followed by a linear head; the UPGD
   branch consumes `last_state || mean_state` and directly emits softmax token
   probabilities.
2. Optimization mismatch. The MLP branch is end-to-end SGD on CE. The UPGD
   branch updates attention by SGD through current predictions, but updates the
   learner through `UPGDLearner.update` with utility perturbation and ObGD.
   This may be the intended Step 2 online rule, but it is not an apples-to-
   apples optimizer comparison.
3. State/compute overhead. At essentially matched trainable parameter count,
   UPGD carries 23% more state elements here and is materially slower in the
   measured scans. The current implementation pays for learner-update
   bookkeeping and many-head softmax CE gradients.
4. LM evidence is small. 2,000 online contexts x 3 seeds is still a bounded
   smoke relative to Tiny Shakespeare. It is enough to reject a strong proof
   claim, not enough to tune all possible UPGD variants.

The most likely issue is not parameter count. The parameter match is close:
13,505 UPGD trainable parameters vs 13,537 MLP trainable parameters. The loss
comes from the current UPGD-readout architecture and update dynamics under
next-token CE, plus state/update overhead.

## Recommendation

Do not present the current Step 2 UPGD learner as a transformer-scale FFN
replacement. Present the current Tiny Shakespeare result only as an integration
smoke: UPGD can be placed in a transformer-shaped online LM loop and train
without crashing, but the fairer bounded comparison favors the MLP transformer.

Concrete next patch:

1. Patch `examples/The Alberta Plan/Step2/step2_tiny_shakespeare_upgd_transformer.py`
   to report per-method wall-clock and resource counts directly.
2. Change its fair default from `upgd_hidden=32` to a parameter-matched
   `upgd_hidden=48` when `d_model=32`, or add an explicit `--match-params`
   option that computes the closest UPGD hidden width.
3. Rename the reported UPGD method to `upgd_readout_transformer` unless a true
   FFN-only replacement is implemented.
4. Only after that, add a real scale run: at least 10 seeds, 10k-50k online
   contexts, fixed train/eval offsets, compile-excluded timing, and the same
   resource-count table.

If the goal remains a true FFN replacement, the next design step is larger:
the current supervised `UPGDLearner` API needs an example-level adapter that
can update an internal hidden transform from the downstream LM CE gradient, or
the claim must be narrowed to "UPGD readout/head replacement" rather than
"UPGD FFN replacement."
