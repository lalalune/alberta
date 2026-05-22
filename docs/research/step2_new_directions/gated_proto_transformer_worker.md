# Gated Prototype Transformer Worker

This worker added a Tiny Shakespeare transformer-block probe with a learned
residual gate over the prototype-basis residual. The implementation is isolated
to `examples/The Alberta Plan/Step2/step2_tiny_shakespeare_gated_proto_transformer.py`.

The gate input vector is `[bias, novelty_binary, novelty_distance_ratio,
recent_loss_log1p, entropy_norm, one_minus_max_prob, margin]`. The default gate
is per-channel. Scalar and per-prototype-group gate modes are also available.
Learning rates are split across the fast transformer parameters, prototype
values, and gate parameters.

## Runs

Artifacts are under
`outputs/step2_new_directions/gated_proto_transformer_worker/`.

| Run | Baseline final-window NLL | Ungated final-window NLL | Gated final-window NLL | Baseline eval PPL | Ungated eval PPL | Gated eval PPL |
|---|---:|---:|---:|---:|---:|---:|
| `smoke`, 1 seed, 32 steps | 4.122009 | 4.112744 | 4.101108 | 62.437126 | 61.845776 | 62.450970 |
| `run_800_2seed`, 2 seeds, 800 steps | 3.503801 +/- 0.001922 | 3.508608 +/- 0.000639 | 3.504763 +/- 0.002598 | 34.894415 +/- 2.851202 | 35.072241 +/- 2.983770 | 33.564642 +/- 2.871027 |
| `run_3000_2seed`, 2 seeds, 3000 steps | 3.274380 +/- 0.000221 | 3.288820 +/- 0.004421 | 3.297891 +/- 0.014166 | 25.057534 +/- 3.590858 | 25.403293 +/- 3.514413 | 25.412567 +/- 2.881323 |

## Assessment

At 800 steps, the gated hybrid improved mean held-out eval perplexity over the
baseline by 1.329773, but its final-window NLL was slightly worse by 0.000962.
At 3000 steps, the default gated hybrid did not beat the tuned FFN baseline:
final-window NLL was worse by 0.023511 and eval perplexity was worse by
0.355033 on average across two seeds.

The default gate stayed near its initial operating point
(`final_window_gate_mean` about 0.379). This suggests the current prototype
residual is too weak or too sparse for the learned gate to discover a robust
advantage at this scale.
