# Step 2 Recursive Retention Attempt

Date: 2026-05-05

## Question

Prior recursive feature probes diagnosed a specific failure: novelty-gated
candidate scoring improved algebraic triple/rare tasks but suppressed
correlated, partially redundant smooth nonlinear scaffolds. This attempt tests
redundancy-tolerant retention instead of another novelty penalty.

## Mechanism

Opt-in retention fields were added to `CompositionalFeatureLearner` with
defaults preserving existing behavior:

- `retention_slow_utility_decay`: maintains a slow utility EMA.
- `retention_tanh_min_count`: protects the last N active `OP_TANH` features.
- `retention_product_min_count`: protects the last N active `OP_PRODUCT`
  features.

When slow retention is enabled, active replacement scores use
`max(fast_utility, slow_utility)`, so replacement requires both the fast and
slow estimates to be low. The final probe variant,
`single_mechanism_retention`, used signed-tanh scaffolds, tanh family
protection, and slow utility hysteresis. It did not use a novelty penalty.

## Six-Probe Result

Command:

```bash
source .venv/bin/activate && python "examples/The Alberta Plan/Step2/step2_recursive_feature_utility_probe.py" --suite --seeds 5 --num-steps 2500 --final-window 500 --methods single_mechanism,single_mechanism_signed_tanh,single_mechanism_retention,mlp_32x32_no_ln,mlp_64x64_no_ln --output-dir outputs/step2_recursive_retention_5seed_2500_tanh12
```

Mean final-window MSE, lower is better:

| Task | single_mechanism | signed_tanh | retention | best fair MLP | Retention delta vs best MLP |
|---|---:|---:|---:|---:|---:|
| nonlinear | 0.2215 | 0.1291 | 0.1002 | 0.0597 | -0.0405 |
| interaction | 0.1306 | 0.1281 | 0.0337 | 0.4847 | +0.4510 |
| triple | 0.1424 | 0.1171 | 0.0795 | 0.8624 | +0.7829 |
| rare | 0.1065 | 0.1391 | 0.0550 | 0.0951 | +0.0401 |
| polynomial | 0.3583 | 0.7303 | 0.3932 | 0.9590 | +0.5658 |
| frequency | 0.0787 | 0.0718 | 0.0570 | 0.0785 | +0.0214 |

Retention beat the best fair MLP on 5/6 tasks. It preserved algebraic wins
and improved the nonlinear final-window mean versus signed tanh
(`0.1291 -> 0.1002`), but it still lost nonlinear to the best fair MLP
(`0.0597`).

## Negative Control

An earlier retention variant combined slow retention with energy-normalized
candidate scoring and no novelty penalty:

```bash
source .venv/bin/activate && python "examples/The Alberta Plan/Step2/step2_recursive_feature_utility_probe.py" --suite --seeds 5 --num-steps 2500 --final-window 500 --methods single_mechanism,single_mechanism_signed_tanh,single_mechanism_retention,mlp_32x32_no_ln,mlp_64x64_no_ln --output-dir outputs/step2_recursive_retention_5seed_2500
```

That version was rejected before finalization. It over-protected stale features
and made nonlinear/algebraic performance unstable. The failure was not lack of
novelty; it was using residual-energy scores as the active replacement utility,
which kept high-correlation scaffolds alive without enough evidence that they
were useful in the output head.

## Decision

Partial, not promoted.

The mechanism is directionally useful: it directly attacks the smooth-scaffold
retention failure, reduces nonlinear loss relative to signed tanh, and keeps
the algebraic wins. It does not satisfy the promotion criterion because the
nonlinear task still trails the best fair MLP by `0.0405` final-window MSE and
wins only `1/5` paired nonlinear seeds.

Exact remaining failure: the retained tanh scaffold basis is representationally
useful but still adapts too slowly online. Hysteresis preserves the basis; it
does not make the output/theta credit assignment competitive with the fair MLP
on the smooth nonlinear final window.
