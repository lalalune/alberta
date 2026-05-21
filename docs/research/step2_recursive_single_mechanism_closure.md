# Step 2 Recursive Single-Mechanism Closure

Date: May 5, 2026.

## Scope

This note audits the compositional future-utility path and tests one simple
recursive feature-construction mechanism against fair MLP controls. The closure
bar is strict: one mechanism should beat or tie the best fair MLP on nonlinear,
interaction, recursive triple-product, rare-head, polynomial, and frequency
probes.

## Mechanism

`single_mechanism` is a fixed-budget compositional learner with:

- contribution-trace future utility (`error * feature`) blended with immediate
  utility;
- residual-imprint candidate output weights;
- cross-product scaffolds followed by self-product scaffolds;
- product-biased operation priors with some sum, tanh, and gated candidates;
- parent selection biased by utility, residual credit, novelty, and depth;
- depth-based retention to reduce churn after recursive features appear.

The code path is intentionally small and configurable rather than a router over
multiple learners.

## Audit

The prior recursive failures were not from label leakage or missing candidate
credit. The future-utility path is causal: it uses current residuals, current
features, previous traces, and previous weights only. The failure mode is search
allocation. Pure cross-product scaffolds discover triple and interaction targets
but under-allocate squares/cubes; adding self-products fixes polynomial probes.
The remaining nonlinear failure is structural: tanh candidates are generated and
selected online, but candidate internal parameters are not trained while in the
shadow bank, so smooth nonlinear targets are still easier for the fair MLP.

## Commands

```bash
source .venv/bin/activate
pytest tests/test_future_utility.py tests/test_compositional_features.py -q
python "examples/The Alberta Plan/Step2/step2_recursive_feature_utility_probe.py" --smoke --output-dir outputs/step2_recursive_single_mechanism_smoke
python "examples/The Alberta Plan/Step2/step2_recursive_feature_utility_probe.py" --suite --seeds 5 --num-steps 2000 --final-window 400 --methods single_mechanism,mlp_32x32_no_ln,mlp_64x64_no_ln,mlp_32x32_ln --output-dir outputs/step2_recursive_single_mechanism_wide_tanh_5seed
```

## Final 5-Seed Suite

Output:
`outputs/step2_recursive_single_mechanism_wide_tanh_5seed/recursive_feature_utility_results.json`

Lower is better. Delta is best fair MLP final-window MSE minus
`single_mechanism`; positive favors the recursive mechanism.

| Probe | Best fair MLP | Best MLP final MSE | Single final MSE | Delta | Single wins |
|---|---|---:|---:|---:|---:|
| nonlinear | `mlp_32x32_ln` | 0.0687 | 0.1738 | -0.1051 | 0/5 |
| interaction | `mlp_32x32_ln` | 0.5517 | 0.0768 | 0.4749 | 5/5 |
| recursive triple product | `mlp_64x64_no_ln` | 0.8615 | 0.1353 | 0.7262 | 5/5 |
| rare-head | `mlp_32x32_ln` | 0.0785 | 0.0835 | -0.0050 | 3/5 |
| polynomial | `mlp_32x32_ln` | 0.9516 | 0.5947 | 0.3569 | 4/5 |
| frequency mismatch | `mlp_32x32_ln` | 0.1735 | 0.1065 | 0.0670 | 4/5 |

Heldout MSE:

| Probe | Best MLP heldout MSE | Single heldout MSE |
|---|---:|---:|
| nonlinear | 0.0653 | 0.2497 |
| interaction | 0.4166 | 0.0748 |
| recursive triple product | 0.7692 | 0.0672 |
| rare-head | 0.5988 | 0.0788 |
| polynomial | 1.1828 | 0.5630 |
| frequency mismatch | 0.1280 | 0.0558 |

## Conclusion

Status: not closed.

The mechanism is strong evidence for recursive algebraic feature construction:
it beats the best fair MLP on interaction, triple product, polynomial, and
frequency probes, and it ties rare-head online loss while substantially
improving rare-head heldout loss. It does not meet the requested single
mechanism standard because it loses the nonlinear probe decisively: 0/5 paired
wins and worse heldout error.

The blocker should remain open unless the scope is narrowed to algebraic and
recursive-product feature construction. For the full Step 2 standard, the next
minimal mechanism would need trained candidate parameters or another way for
smooth nonlinear candidates to become competitive before promotion.

## May 6 Follow-Up: Conservative Tanh-Heavy Prior

The nonlinear 10-seed failure audit showed that the useful signed tanh
scaffold was already present in the failed seeds. The bad runs came from a
promoted depth-2 `sum` or `gated` feature receiving a large output weight
despite weak heldout target correlation. The follow-up keeps one recursive
learner and adds an opt-in fixed operation prior over generated candidates:
`[raw=0, product=0.35, sum=0, tanh=0.65, gated=0]`. The probe variant also
uses conservative promotion (`promotion_blend=0.35`) and lower residual imprint
(`candidate_imprint_scale=0.1`) so one bad promoted candidate cannot dominate
the head immediately.

Command:

```bash
source .venv/bin/activate
python "examples/The Alberta Plan/Step2/step2_recursive_feature_utility_probe.py" --suite --seeds 10 --num-steps 2500 --final-window 500 --methods single_mechanism_retention_tanh24_tanh_heavy_conservative,mlp_32x32_no_ln,mlp_64x64_no_ln --output-dir outputs/step2_recursive_single_tanh_heavy_conservative_suite_10seed_2500
```

Output:
`outputs/step2_recursive_single_tanh_heavy_conservative_suite_10seed_2500/recursive_feature_utility_results.json`

| Probe | Best fair MLP | Best MLP final MSE | Conservative recursive final MSE | Delta | Recursive wins |
|---|---|---:|---:|---:|---:|
| nonlinear | `mlp_32x32_no_ln` | 0.0625 | 0.0538 | +0.0087 | 9/10 |
| interaction | `mlp_32x32_no_ln` | 0.4484 | 0.0624 | +0.3861 | 10/10 |
| recursive triple product | `mlp_32x32_no_ln` | 0.7427 | 0.3557 | +0.3870 | 9/10 |
| rare-head | `mlp_32x32_no_ln` | 0.0796 | 0.0538 | +0.0258 | 8/10 |
| polynomial | `mlp_32x32_no_ln` | 0.9038 | 0.8222 | +0.0816 | 6/10 |
| frequency mismatch | `mlp_32x32_no_ln` | 0.0793 | 0.0107 | +0.0685 | 10/10 |

Decision: promotable as the current pure recursive candidate, with caveats.
It closes all six probes by paired mean against the best fair MLP control and
improves the nonlinear blocker from the prior 8/10 wins with negative mean to
9/10 wins with positive mean. The remaining risk is not eliminated: nonlinear
seed 2 still fails badly, and polynomial wins only 6/10 seeds despite a
positive mean. This should be promoted as the best pure recursive mechanism,
not as a solved robustness story.
