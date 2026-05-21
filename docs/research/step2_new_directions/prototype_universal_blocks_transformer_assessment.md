# Prototype Universal Blocks and Transformer Follow-Up

Date: 2026-05-06

## What Was Built

This pass promoted the strongest D20/D22 lesson into reusable blocks:

- `PrototypeBasisBlock`: a JAX global novelty-allocated prototype basis with trainable value rows.
- `d24_prototype_universal_blocks.py`: a compact OPMNIST sweep comparing class-conditioned prototype memory, global prototype-basis readouts, adaptive bandwidth, and a two-layer recursive basis.
- `step2_tiny_shakespeare_proto_basis_transformer.py`: a transformer experiment where the prototype basis is trained as an actual residual sublayer.

The transformer runner tests three methods:

- `baseline_ffn_transformer`: attention, GELU FFN, linear readout.
- `proto_basis_transformer`: attention, prototype-basis residual, linear readout.
- `hybrid_proto_transformer`: attention, GELU FFN, prototype-basis residual, linear readout.

The prototype centers are slow online state. Attention, readout, and prototype value rows are trained by cross-entropy gradients.

## D24 OPMNIST Result

Artifact: `docs/research/step2_new_directions/d24_prototype_universal_blocks_local3.md`.

The class-conditioned prototype memory remains the strongest OPMNIST mechanism:

| Method | Final MSE | Final Acc | Test MSE | Test Acc | Steps/s | Float State |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `mlp_h128` | 0.082160 | 0.505000 | 0.080295 | 0.552500 | 1750.49 | - |
| `proto_mem_s20` | 0.026361 | 0.825000 | 0.013433 | 0.907333 | 4399.03 | 157000 |
| `proto_mem_s32` | 0.026214 | 0.826667 | 0.011955 | 0.917000 | 4698.55 | 251200 |
| `basis_p128_bw001` | 0.087397 | 0.566667 | 0.084483 | 0.811667 | 7234.28 | 101898 |

Interpretation:

- `proto_mem_s20/s32` are still the strongest practical Step 2 OPMNIST candidates.
- The global prototype basis is much more reusable as a neural block, and it is fast, but it does not replace class-conditioned memory on OPMNIST.
- The simple recursive basis failed on this protocol, so recursion cannot just be "basis over basis" without a better training/resource rule.

## Transformer Result

The transformer block is implemented and trainable, but the current evidence does not prove superiority over a tuned FFN.

Key artifacts:

- `outputs/step2_new_directions/proto_basis_transformer_hybrid_800_2seed_blr015_plr015_thr0002/SUMMARY.md`
- `outputs/step2_new_directions/proto_basis_transformer_hybrid_800_2seed_blr010_plr015_thr0002/SUMMARY.md`
- `outputs/step2_new_directions/upgd_ffn_transformer_800_2seed_reference/SUMMARY.md`

At matched 0.15 learning rate:

| Method | Mean final-window NLL | Mean eval perplexity | Mean train s |
| --- | ---: | ---: | ---: |
| Tuned FFN baseline | 3.2565 | 30.1611 | 0.2359 |
| Pure prototype basis | 3.3254 | 30.8705 | 0.3033 |
| FFN + prototype basis | 3.2783 | 30.6210 | 0.3390 |

With `baseline_lr=0.10` and `proto_lr=0.15`, the hybrid prototype block improves final-window NLL but does not clearly improve held-out perplexity:

| Method | Mean final-window NLL | Mean eval perplexity |
| --- | ---: | ---: |
| FFN baseline | 3.3500 | 30.6006 |
| Pure prototype basis | 3.3254 | 30.8705 |
| FFN + prototype basis | 3.2783 | 30.6210 |

UPGD-FFN on the same tiny Shakespeare setup is neutral to displayed precision and slower:

| Method | Mean final-window NLL | Mean eval perplexity |
| --- | ---: | ---: |
| FFN baseline | 3.4483 | 34.1638 |
| UPGD FFN | 3.4483 | 34.1636 |

## Critical Assessment

The result does not yet justify "almost certainly better in transformer context." The prototype-basis block is now in a transformer and trainable, but matched-tuning runs still favor the standard FFN on held-out perplexity.

What did work:

- Tight novelty thresholds are required in transformer hidden space; the default `0.08` threshold collapses to one prototype.
- The hybrid fast+slow block is consistently better than pure prototype basis.
- Higher prototype-block learning rates can improve online/final-window NLL quickly.
- The prototype block has a credible transformer interface: slow centers plus gradient-trained values.

What did not work yet:

- Pure prototype residuals underperform a tuned FFN.
- Hybrid prototype residuals have not beaten a same-rate tuned FFN on held-out perplexity.
- Many active prototypes improve online fit only when learning rate is high; they do not yet give robust eval gains.
- The current block adds state and wall time.

## Next Experiments

The most promising next path is not more class-conditioned memory in a transformer. It is a learned fast/slow residual block:

- Add a learned residual gate per channel or per prototype group.
- Let the gate depend on uncertainty, novelty, and recent token loss.
- Decouple FFN learning rate from prototype-value learning rate.
- Reset or meta-update prototype value rows on replacement with a learned initializer, not zeros.
- Use the prototype basis as a KV-memory-like adapter inside attention, not only as a post-FFN residual.
- Test on a larger token budget where slow memory has time to matter.

Current conclusion: prototype memory is still a strong Step 2 online external benchmark mechanism, and the transformer block exists, but the transformer claim remains open.
