# Step 2 Paper And Deployment Quality Plan

Date: 2026-05-06.

## Current Status

The promoted Step 2 learner is:

```python
UPGDLearner.step2_default(n_heads)
```

It is a single learner, not a portfolio: width-32 target-structure UPGD with
ObGD bounding, low-noise Rademacher perturbations every 16 steps, and lean
state disabled for optional unit-recycling and meta-gradient mechanisms.

The evidence package now supports an internal research-paper-quality claim, but
not an unconditional universality theorem or a general replacement for all
large production MLP/FFN deployments.

## Claim Discipline

Defensible paper claim:

> In continual supervised multi-task streams, a resource-efficient
> target-structure UPGD learner can outperform fair MLP baselines while
> maintaining explicit online utility estimates for hidden features and
> low-utility exploratory perturbations under a fixed resource budget.

Defensible deployment claim:

> For the current Step 2 online vector-target workloads, the promoted width-32
> UPGD branch is a drop-in replacement candidate for a width-64
> `MultiHeadMLPLearner`: same `init`, `predict`, and `update` style; fewer
> trainable parameters; faster JAX scan throughput in the recorded benchmark;
> better aggregate synthetic and sklearn-digits performance.

Do not claim:

- universal representation learning in theorem form;
- superiority to modern transformer FFNs under ordinary offline training;
- same-width UPGD is cheaper than MLP;
- class-blocked retention is fully solved;
- production readiness outside online continual supervised workloads.

## Evidence Gates

The reproducible gate is:

```bash
python benchmarks/step2_upgd_evidence_gate.py
```

It reads the recorded 30-seed synthetic results, 30-seed digits results, and
throughput benchmark, then fails if the promoted claim no longer clears the
paper/deployment thresholds.

Required gate conditions:

| Area | Gate |
|---|---|
| Factory | `step2_default` config is frozen to the promoted branch. |
| Lean state | disabled unit-recycling/meta-history buffers are absent at init. |
| Synthetic | positive paired diff and all-seed wins on all three out-of-class streams. |
| Digits MSE | final-window MSE diff at least `+0.005`, with at least `140/150` wins. |
| Digits accuracy | held-out test accuracy diff at least `+0.02`, with at least `130/150` wins. |
| Class-blocked | positive final-window MSE and test accuracy; final-window accuracy no worse than `-0.005`. |
| Throughput | width-32 UPGD faster than MLP64 on one-hot and dense targets. |
| Resource use | width-32 UPGD uses no more trainable parameters or float state than MLP64 in the recorded benchmark. |

## Paper-Quality Package

Minimum paper artifact set:

| Artifact | Path | Status |
|---|---|---|
| Readiness audit | `docs/research/step2_presentation_readiness_audit.md` | ready |
| Slide draft | `docs/research/step2_presentation_slide_draft.md` | ready |
| Current-best synthesis | `docs/research/step2_current_best.md` | ready |
| Compute note | `docs/research/step2_compute_efficient_upgd.md` | ready |
| Target-structure stress | `docs/research/step2_target_structure_upgd_stress.md` | ready |
| Synthetic 30-seed JSON/summary | `output/subagents/compute_efficiency/small_rademacher_synthetic_30seed_6000/` | ready |
| Digits 30-seed JSON/summary | `output/subagents/compute_efficiency/small_rademacher_digits_30seed_h64baseline/` | ready |
| Throughput benchmark | `output/benchmarks/step2_upgd_efficiency_fused_heads_4096/` | ready |
| Evidence gate | `benchmarks/step2_upgd_evidence_gate.py` | ready |

Minimum paper sections:

1. Problem: Step 2 asks for supervised feature finding under a resource budget.
2. Baseline: fair MLPs are strong but do not expose utility/discard decisions.
3. Method: target-structure UPGD, utility, perturbation, ObGD bounds, lean state.
4. Target normalization: why target-structure replaces target-density.
5. Synthetic experiments: out-of-class polynomial, frequency, compositional.
6. External online digits: iid, class-blocked, permuted pixels, mask noise, label drift.
7. Compute and state: scan throughput, parameters, float state, same-width caveat.
8. Transformer-facing demo: Tiny Shakespeare as integration only.
9. Threats to validity: small external dataset breadth, class-blocked retention, JAX/hardware specificity, no theorem.
10. Step 3 implication: supervised feature utility becomes predictive/GVF feature utility.

## Production Deployment Bar

A deployment that replaces an MLP with `UPGDLearner.step2_default` should satisfy
these gates before being treated as production quality:

| Gate | Requirement |
|---|---|
| API compatibility | Uses `init(feature_dim, key)`, `predict(state, obs)`, `update(state, obs, targets)`. |
| Target contract | MSE targets use dense/sparse vector targets with NaN inactive heads; CE targets are non-negative one-hot/simplex vectors. |
| Shape validation | invalid `n_heads`, `hidden_sizes`, `feature_dim`, and core rates fail before JIT compilation. |
| Serialization | `to_config`/`from_config` roundtrip preserves the exact promoted settings. |
| JIT stability | focused UPGD unit tests pass under `pytest tests/test_upgd.py`. |
| Resource budget | width and perturbation interval are fixed in config, not routed dynamically by dataset. |
| Monitoring | log loss, mean utility, min utility, max perturbation magnitude from `UPGDUpdateResult.metrics`. |
| Rollback | keep an MLP64 comparator available during deployment evaluation, but do not blend it into the learner claim. |
| Drift policy | class-blocked or rare-class workloads need a separate retention monitor because this remains the known stress row. |
| Acceptance | run `benchmarks/step2_upgd_evidence_gate.py` and a domain-specific shadow evaluation before replacing an MLP. |

## Remaining Research Gaps

These are real gaps, not presentation polish:

1. **Class-blocked retention**: final-window MSE and held-out accuracy are
   positive, but class-blocked final-window accuracy remains slightly negative.
2. **External breadth**: sklearn digits is useful but too small to support broad
   deployment claims. Add larger online tabular/image streams before public
   benchmark claims.
3. **Same-width efficiency**: the production win comes from a smaller successful
   UPGD branch, not cheaper same-width updates.
4. **Theory**: the plausible theorem is conditional utility-preserving
   exploration under bounded online updates, not universal feature discovery.
5. **Transformer FFN replacement**: Tiny Shakespeare currently demonstrates
   integration, not a strong language-model win.

## Next Experiments Worth Running

Priority order:

1. Rerun `benchmarks/step2_upgd_efficiency.py` after the lean-state fix to
   update state-size and throughput numbers. **Done for the 4,096-step
   artifact: width-32 UPGD is now `1.45x` faster on one-hot and `1.57x` faster
   on dense targets, with float state `4,476` vs MLP64 `9,706`.**
2. Add one larger external stream with online labels and no offline batching:
   OpenML tabular classification/regression or true MNIST/OPMNIST.
3. Add a class-blocked retention ablation with monitoring only, not deployment
   blending: retained-head norm, bias drift, utility collapse, perturbation
   energy by layer.
4. Run Tiny Shakespeare with a stronger fair baseline and enough seeds to decide
   whether the CE branch is only compatible or actually competitive.
5. Convert the Step 2 evidence gate into CI once the long-running artifacts are
   stable enough for the repository.
