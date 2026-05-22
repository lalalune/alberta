# Step 2 Generator-Internal Meta-Resource Closure

Date: 2026-05-05

## Audit

`GeneratorMetaResourceManager` is now a real generator-internal controller, not
only a learner-policy selector.  Its sampled policy is chosen before candidate
construction and controls operation choice, parent-selection mode, replacement
rate, promotion margin, candidate minimum age, and residual-imprint scale.
Provenance is recorded on active and candidate slots, so credit is delayed:
policies are updated from later utility assigned to the features or candidates
they constructed.

The positive causal properties are:

- The policy decision precedes candidate refresh, promotion, or active-slot
  replacement.
- The update signal is feature/candidate utility by construction provenance,
  with an optional delayed promotion bonus for candidates that survive testing.
- The manager can use contextual Hedge over all finite provenance scores, or an
  EXP3-style sampled-policy update with importance weighting and explicit
  exploration.
- Operation and parent-selection priors are represented as initial policy
  preferences; residual-imprint, replacement-rate, promotion-margin, and
  candidate-age aggressiveness are represented as policy costs.

The remaining limitations are:

- Credit is still coarse at the bundled-policy level. It can say which bundle
  worked, not which subchoice inside the bundle caused the gain.
- Candidate utility is a proxy for future learner value. Promotion credit helps,
  but there is no full counterfactual replay of rejected candidates.
- EXP3 is high variance on these short online probes; it is useful as a stress
  check, not yet a better default.

## Commands

```bash
source .venv/bin/activate
pytest tests/test_resource_manager.py -q
python "examples/The Alberta Plan/Step2/step2_generator_meta_resource_closure.py" \
  --smoke \
  --output-dir outputs/step2_generator_meta_smoke
python "examples/The Alberta Plan/Step2/step2_generator_meta_resource_closure.py" \
  --seeds 3 \
  --num-steps 800 \
  --final-window 200 \
  --n-features 18 \
  --candidate-count 12 \
  --replacement-interval 25 \
  --min-feature-age 30 \
  --candidate-min-age 15 \
  --output-dir outputs/step2_generator_meta_closure_3seed
```

## Variants

| Variant | Mechanism |
|---|---|
| `fixed_residual` | Fixed residual-imprint compositional generator. |
| `hedge` | Contextual Hedge over generator policies. |
| `exp3` | Sampled-policy EXP3-style update with explicit exploration. |
| `hedge_priors_credit_budget` | Operation/parent priors, delayed promotion credit, and residual-imprint/replacement/promotion costs. |

## 3-Seed Results

Final-window MSE, lower is better.

| Probe | Variant | Final MSE | Fixed - variant | Wins |
|---|---|---:|---:|---:|
| nonlinear | `fixed_residual` | 0.529674 +/- 0.189745 | n/a | n/a |
| nonlinear | `hedge` | 0.684313 +/- 0.020729 | -0.154640 | 1/3 |
| nonlinear | `exp3` | 0.588993 +/- 0.076598 | -0.059319 | 1/3 |
| nonlinear | `hedge_priors_credit_budget` | 0.425808 +/- 0.039275 | +0.103865 | 1/3 |
| interaction | `fixed_residual` | 1.928413 +/- 0.280386 | n/a | n/a |
| interaction | `hedge` | 1.793916 +/- 0.170612 | +0.134498 | 1/3 |
| interaction | `exp3` | 2.267415 +/- 0.287589 | -0.339001 | 0/3 |
| interaction | `hedge_priors_credit_budget` | 1.749662 +/- 0.246676 | +0.178752 | 2/3 |
| recursive | `fixed_residual` | 1.168465 +/- 0.062255 | n/a | n/a |
| recursive | `hedge` | 1.226716 +/- 0.173518 | -0.058251 | 1/3 |
| recursive | `exp3` | 1.206710 +/- 0.152880 | -0.038245 | 1/3 |
| recursive | `hedge_priors_credit_budget` | 1.330762 +/- 0.073870 | -0.162297 | 0/3 |
| rare_feature | `fixed_residual` | 1.096957 +/- 0.289101 | n/a | n/a |
| rare_feature | `hedge` | 1.376266 +/- 0.178203 | -0.279309 | 2/3 |
| rare_feature | `exp3` | 0.988751 +/- 0.347152 | +0.108207 | 2/3 |
| rare_feature | `hedge_priors_credit_budget` | 1.199673 +/- 0.236917 | -0.102715 | 2/3 |

All training metrics were finite.  The best learned variant depends on the
probe: priors/credit/budget helped nonlinear and interaction, EXP3 helped the
rare-feature active MSE, and no learned variant beat the fixed generator on the
recursive triple-product probe.

## Interpretation

This is a well-characterized partial result.  The mechanism is now causal and
internal to generator choices, and it improves some probe families, but it is
not robust enough to close Step 2 as the canonical default.  The fixed
residual-imprint generator remains the safer default because the recursive
probe regressed and the seed-level win counts are inconsistent.

## Recommendation

Keep `learn_generator_resources=False` as the canonical default.  Keep
generator meta-resource management as an opt-in Step 2 research mechanism, with
`hedge_priors_credit_budget` as the most useful variant to study next and
`exp3` as an exploration stress test for sparse/rare-feature settings.
