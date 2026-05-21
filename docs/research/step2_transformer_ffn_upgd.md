# Step 2 Transformer FFN-only UPGD Experiment

Date: 2026-05-05

## Question

Can a Step 2 UPGD-style mechanism replace only the transformer FFN sublayer,
while the baseline and UPGD model keep the same attention stem and same linear
token readout?

Short answer: not yet as a positive result. The clean FFN-only experiment now
exists and removes the readout-replacement confound. In the bounded 2,000-step
Tiny Shakespeare run, the UPGD FFN tied the ordinary FFN on NLL/accuracy within
noise-level differences, while taking about twice the wall-clock time and
carrying extra utility state.

## Implementation

Added:

- `examples/The Alberta Plan/Step2/step2_tiny_shakespeare_upgd_ffn_transformer.py`

Outputs from this run:

- `output/subagents/transformer_ffn/smoke/results.json`
- `output/subagents/transformer_ffn/smoke/SUMMARY.md`
- `output/subagents/transformer_ffn/param_matched_2000x3/results.json`
- `output/subagents/transformer_ffn/param_matched_2000x3/SUMMARY.md`

## Architecture

The old `step2_tiny_shakespeare_upgd_transformer.py` does not isolate the FFN:
the MLP branch uses attention, residual FFN, and a linear softmax readout, while
the UPGD branch feeds attention observations into `UPGDLearner` and lets UPGD
own the next-token readout.

The new experiment uses this shared path for both methods:

`character context -> token/position embedding -> one causal self-attention residual stem -> last-token hidden state -> residual FFN module -> same-shape linear vocab readout -> softmax CE`

The baseline FFN is:

`h_out = h + gelu(h W1 + b1) W2 + b2`

The UPGD FFN uses the same forward equation and trainable tensor shapes. After
the clipped SGD CE update, it updates a per-feature utility trace:

`u_i <- decay u_i + (1 - decay) * 0.5 * (mean |W1_col_i * grad_W1_col_i| + mean |W2_row_i * grad_W2_row_i|)`

Every perturbation interval, low-utility FFN feature slots receive
utility-scaled rademacher noise:

`delta_i = sigma * (1 - u_i / max(u))^beta * noise`

The perturbation is applied only inside the FFN feature slot weights. The
attention stem and token readout are ordinary differentiable parameters in both
models.

## Why Not Exact `UPGDLearner`

The public `UPGDLearner` API is a supervised learner with its own prediction
heads and target loss. Using it directly in this transformer language-modeling
loop would again replace the token readout, or require an artificial hidden
target that is not the next-token CE objective. That would fail the isolation
requirement. The new module is therefore UPGD-inspired rather than an exact
`UPGDLearner`: it keeps utility tracking and low-utility perturbation local to
FFN features while leaving the LM readout untouched.

## Commands

Smoke:

```bash
.venv/bin/python 'examples/The Alberta Plan/Step2/step2_tiny_shakespeare_upgd_ffn_transformer.py' \
  --steps 20 \
  --seeds 1 \
  --eval-steps 16 \
  --final-window 16 \
  --block-size 16 \
  --d-model 16 \
  --mlp-hidden 32 \
  --upgd-hidden 32 \
  --output-dir output/subagents/transformer_ffn/smoke
```

Bounded parameter-matched comparison:

```bash
.venv/bin/python 'examples/The Alberta Plan/Step2/step2_tiny_shakespeare_upgd_ffn_transformer.py' \
  --steps 2000 \
  --seeds 3 \
  --eval-steps 512 \
  --final-window 512 \
  --block-size 32 \
  --d-model 32 \
  --mlp-hidden 64 \
  --upgd-hidden 64 \
  --baseline-lr 0.03 \
  --upgd-lr 0.03 \
  --grad-clip 1.0 \
  --perturbation-sigma 0.0001 \
  --perturbation-interval 16 \
  --output-dir output/subagents/transformer_ffn/param_matched_2000x3
```

Lint:

```bash
.venv/bin/ruff check 'examples/The Alberta Plan/Step2/step2_tiny_shakespeare_upgd_ffn_transformer.py'
```

## Parameter and State Profile

For the bounded run, both methods use `d_model=32`, `mlp_hidden=64`,
`upgd_hidden=64`, block size 32, and vocabulary size 65.

| Method | Trainable params | Extra state elements | Extra state bytes |
|---|---:|---:|---:|
| `baseline_ffn_transformer` | 13,537 | 0 | 0 |
| `upgd_ffn_transformer` | 13,537 | 66 | 268 |

The trainable parameter count is exactly matched. The UPGD FFN state is the
64-element feature utility vector plus PRNG/step bookkeeping.

## Results

Smoke output:

- Baseline final-window NLL 4.158, eval perplexity 62.25, train 1.52 s.
- UPGD FFN final-window NLL 4.158, eval perplexity 62.25, train 2.69 s.
- Max perturbation was 0.000063.

Bounded 2,000 steps x 3 seeds:

| Metric | Baseline FFN | UPGD FFN | Diff favoring UPGD |
|---|---:|---:|---:|
| `final_window_nll` | 3.2621 +/- 0.0550 | 3.2621 +/- 0.0550 | +0.0000 +/- 0.0000 |
| `final_window_accuracy` | 0.1699 +/- 0.0030 | 0.1699 +/- 0.0030 | +0.0000 +/- 0.0000 |
| `final_window_perplexity` | 26.1840 +/- 1.4353 | 26.1839 +/- 1.4357 | +0.0001 +/- 0.0004 |
| `eval_nll` | 3.3470 +/- 0.1066 | 3.3470 +/- 0.1066 | -0.0000 +/- 0.0000 |
| `eval_accuracy` | 0.1595 +/- 0.0131 | 0.1595 +/- 0.0131 | +0.0000 +/- 0.0000 |
| `eval_perplexity` | 28.7289 +/- 2.8893 | 28.7293 +/- 2.8897 | -0.0004 +/- 0.0005 |
| `train_s` | 0.9121 +/- 0.0581 | 1.8560 +/- 0.1798 | -0.9439 +/- 0.1527 |
| `train_steps_per_s` | 2211.9060 +/- 150.4725 | 1096.4549 +/- 97.3481 | -1115.4511 +/- 118.6818 |

UPGD FFN diagnostics:

| Metric | Mean +/- stderr |
|---|---:|
| `final_window_mean_utility` | 0.000374 +/- 0.000002 |
| `final_window_min_utility` | 0.000098 +/- 0.000019 |
| `max_perturbation` | 0.000090 +/- 0.000003 |
| `mean_perturbation` | 0.000005 +/- 0.000000 |

## Interpretation

This is a clean FFN isolation test. The attention stem, residual topology,
linear vocab readout, train stream, eval stream, initialization seed, and
trainable parameter count are matched. The earlier architecture/readout mismatch
is no longer the explanation.

The quality result is neutral: UPGD FFN does not improve final-window or eval
NLL/accuracy in this bounded Tiny Shakespeare setting. It also does not
materially hurt quality at `sigma=1e-4`, because the perturbations are tiny
relative to the FFN weights and gradients.

The concrete loss is efficiency: UPGD FFN adds utility-state bookkeeping and
roughly halves throughput in this JAX scan implementation. The blocker is
therefore not readout mismatch; it is that the local UPGD-style perturbation
mechanism has not produced useful optimization benefit in this transformer FFN
setting, while it has measurable runtime overhead.

## Strict Conclusion

Do not claim transformer FFN replacement evidence from this run. Claim only
that the clean FFN-only harness exists and that the first parameter-matched
Tiny Shakespeare result is neutral on quality and negative on efficiency. The
next worthwhile test is a controlled schedule/scale study of perturbation
strength or a nonstationary sequence setting where resource management has a
reason to help, still keeping the readout fixed and avoiding portfolio methods.
