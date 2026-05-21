# Step 2 Current Best

This is the current cleaned interpretation of Step 2 evidence in the repo.

## 2026-05-07 Update: Associative Core Promotion

The sparse-KV sequence-memory probe has been promoted into the package as a
fixed-budget JAX Step 2 kernel:

`alberta_framework.core.associative_memory.AssociativeMemoryLearner`

It is exposed through `Step2AssociativeConfig`,
`make_step2_associative_learner()`, `run_step2_associative_smoke()`, and the
end-to-end pipeline mode `step2="associative"`. The learner constructs
context-derived feature keys, predicts before writing the current target,
updates all matched feature rows every step, credits row utility by one-step
loss advantage, and replaces low-utility rows when the configured budget is
full.

This closes the implementation gap where the most promising sparse discrete
memory mechanism lived only as an experiment script. It does not close the
stronger philosophical gap: the core now has optional learned soft gates for
feature family, suffix window, and effective budget, but the outer operation
set and maximum resource budget are still declared finite choices. No formal
theorem yet establishes arbitrary recursive feature discovery. The next
evidence step is to run the core learner, not the older probe, through the
external sequence and OPMNIST protocols.

## 2026-05-07 Update: Strict Readout Consistency Closed

The remaining one-branch digit readout conflict is now closed by a heavier
64-64 target-structure UPGD branch with a separately bounded two-timescale
simplex readout:

`upgd64_64_structure_sigma1e_4_adaptk035_065_lr06_repx075_meta001_twotime_fastx2_trunk2_slow0_sepbound_notrunk_tight`

It is exposed in code as:

`UPGDLearner.step2_strict_digit_readout_default(n_heads)`

This is still one learner state, not a portfolio or dataset router. The slow
readout is the original linear-MSE head. The fast readout is a softmax/CE head
that receives its own ObGD-bounded readout budget and sends gated CE pressure
into the shared trunk only on recent simplex/repeated-target streams. The slow
MSE branch stops consuming shared update budget when the simplex gate is fully
open.

The 30-seed five-regime digit confirmation at
`output/subagents/class_blocked_retention/fasttrunk_sepbound_x2_strict_30seed_6000/digits_ablation_SUMMARY.md`
beats the best same-run fair MLP on every final-window MSE row:

| Digit regime | Best fair MLP | Final-window MSE diff | Wins |
|---|---|---:|---:|
| IID | `mlp64_64` | `+0.006172` | `30/30` |
| Class-blocked | `mlp64_64` | `+0.000189` | `26/30` |
| Label drift | `mlp64_64` | `+0.005352` | `30/30` |
| Mask noise | `mlp64_64` | `+0.009692` | `30/30` |
| Permuted pixels | `mlp64` | `+0.007323` | `30/30` |

Retained accuracy is also positive on the retained-test regimes where that
metric is meaningful: IID `+0.010761` (`26/30`), class-blocked `+0.424985`
(`30/30`), mask-noise `+0.020037` (`28/30`), and permuted `+0.019295`
(`21/30`). Label-drift retained accuracy is a numerical tie and is not a
promotion criterion because the target map intentionally changes over time.

Updated interpretation: `UPGDLearner.step2_default` remains the simple
resource-efficient target-structure default, while the two-timescale 64-64 row
is now the strict online digit/readout default. The class-blocked vs
label-drift split is no longer the main blocker.

The synthetic regression check at
`output/subagents/class_blocked_retention/fasttrunk_sepbound_x2_synthetic_30seed_6000/synthetic_ablation_SUMMARY.md`
prevents a broader promotion. The strict digit row beats the best MLP on
polynomial (`+0.017758`, `29/30`) and frequency (`+0.053843`, `27/30`), but
loses compositional regression (`-0.079795`, `0/30`). Therefore the strict
two-timescale row is not the global Step 2 default. The remaining scientific
pressure is broader: prove only conditional theorems, keep scalar/temporal
passthrough as a causal target-structure readout problem, and learn or derive
the strict readout controls before claiming broad superiority.

## 2026-05-06 Update: External Completion Reruns

The remaining external-domain and adaptive-structure reruns are now in the
record. They strengthen the promoted UPGD family, but they also reject a
universal "UPGD beats every fair MLP" claim.

The 10-seed external suite at
`output/subagents/external_breadth_sklearn/promoted_step2_default_10seed_all/external_suite_SUMMARY.md`
shows `UPGDLearner.step2_default` beating MLP64 final-window MSE on 7 of 10
domains: shuffled digits, permuted digits, masked/noisy digits, wine,
breast-cancer, dense exact-zero regression, and sparse multilabel. It loses
class-blocked online MSE while winning class-blocked retained test accuracy by
`+0.6881` (`10/10` seeds), and it loses the scalar regression rows:
diabetes regression and delayed/history temporal regression.

The deeper-MLP control at
`output/subagents/external_breadth_sklearn/promoted_step2_default_10seed_all_mlpdeep/external_suite_SUMMARY.md`
keeps the same qualitative result. UPGD remains strongest on the image-like,
tabular classification, dense-zero, and multilabel rows, but `mlp_deep` wins
diabetes regression and class-blocked final-window MSE, while MLP64 wins the
temporal delayed-history row.

The tuning probe at
`output/subagents/external_breadth_sklearn/regression_multilabel_tune_10seed/external_suite_SUMMARY.md`
improves but does not close scalar regression. `upgd_fast` and `upgd_mean` beat
MLP64 on diabetes final-window MSE, but they still lose to `mlp_deep`.
`upgd_fast` nearly ties MLP64 on delayed-history temporal regression, but does
not beat it. The same tuning confirms that sparse multilabel is a strong UPGD
row: all UPGD variants beat both MLP baselines on final-window MSE, and
`upgd_mean` gives the best retained test accuracy.

The native target-structure adaptive digit rerun at
`output/subagents/structure_adaptive_retention/allregime_10seed_6000/digits_ablation_SUMMARY.md`
partially closes the five-regime digit pressure test. The 64-64
structure/repetition branch beats the best MLP on IID (`+0.0058`, `10/10`),
label-drift (`+0.0035`, `10/10`), mask-noise (`+0.0095`, `10/10`), and
permuted digits (`+0.0037`, `10/10`) final-window MSE. It still loses
class-blocked final-window MSE. The 64-wide `structure_meta003_notrunk` branch
gives the best class-blocked retained accuracy (`+0.0640`, `10/10`) but also
loses class-blocked online MSE. No single branch currently dominates the best
same-run MLP on all online and retained digit metrics.

Updated interpretation: target-structure UPGD remains the promoted Step 2
family for online vector-output continual learning, and the default is
especially strong for dense/vector targets, one-hot/multiclass streams,
multilabel targets, image-like pixel perturbations, and retained
class-blocked memory. The remaining blockers are scalar tabular regression,
delayed/history temporal regression, and the class-blocked online-tracking vs
held-out-retention tradeoff against the wider MLP baseline. The 2026-05-07
strict readout closeout above supersedes the class-blocked part of this
historical blocker list.

## 2026-05-06 Update: Scalar Regression Rescue

The scalar regression blockers are no longer hard blockers for the UPGD family,
but the fix is not yet the default. The new passthrough/deep scalar UPGD rows
in `step2_external_suite.py` show that the failure was a readout bottleneck:

| Domain | Rescue row | Result |
|---|---|---|
| `diabetes_regression` | `upgd_reg_passthrough_deep` | final-window MSE `0.3461 +/- 0.0113`, beating `MLP(64)` `10/10` and `MLP(64,64)` by a large mean margin |
| `temporal_delayed_history` | `upgd_temporal_fast_passthrough` / no-mutation control | final-window MSE about `0.431`, beating `MLP(64)` `10/10` |

Full external rerun:
`output/subagents/external_breadth_sklearn/passthrough_rescue_full_10seed_all/external_suite_SUMMARY.md`.

The important caveat is that passthrough scalar rows damage digit and
multilabel streams where the existing `UPGDLearner.step2_default` is strongest.
The promoted default therefore remains the hidden-readout target-structure
branch. The next canonical improvement is not a portfolio, but a causal
target-structure readout rule: keep hidden readout for simplex/vector
classification and allow normalized `hidden_plus_input` readout for scalar
regression.

At this point in the record, the class-blocked online-MSE hole remained open.
The bias/repetition rescue grid at
`output/subagents/class_blocked_retention/bias_repetition_rescue_10seed_6000/digits_ablation_SUMMARY.md`
did not beat `MLP(64,64)` on final-window MSE. It did preserve the existing
retention advantage: `meta003_notrunk_tight` retained the best test accuracy
(`+0.0705`, `10/10` vs best MLP), and several bias-damped rows stayed positive
on retained accuracy, but strict online tracking still belongs to the wider
MLP in this protocol. This is superseded by the later two-timescale readout
closeout.

## 2026-05-06 Update: Class-Blocked Softmax Closeout

The strict class-blocked online-MSE hole is now closed by a narrow 64-64
softmax UPGD branch:

`upgd64_64_structure_sigma1e_4_adaptk035_065_lr07_repx075_meta001_softmax_notrunk_tight`

In
`output/subagents/class_blocked_retention/softmax_closeout_10seed_6000/digits_ablation_SUMMARY.md`,
this branch beats the same-run best fair MLP on mean class-blocked
final-window MSE and wins a majority of paired seeds:

| Metric | Result |
|---|---:|
| UPGD final-window MSE | `0.0019926` |
| `MLP(64,64)` final-window MSE | `0.0019993` |
| Paired wins | `6/10` |
| UPGD retained test accuracy | `0.8692 +/- 0.0080` |
| `MLP(64,64)` retained test accuracy | `0.1007 +/- 0.0006` |

The five-regime check at
`output/subagents/class_blocked_retention/softmax_lr07_allregime_10seed_6000/digits_ablation_SUMMARY.md`
shows that this is not a universal digit replacement. The same branch wins
IID, class-blocked, mask-noise, and permuted final-window MSE, but loses
label-drift final-window MSE to `MLP(64,64)`. Earlier linear-MSE 64-64
structure branches won label drift but lost class blocked.

Updated issue at this stage: class-blocked was no longer the blocker by
itself. The open problem was a single causal readout rule that gets both
softmax-style class-blocked retention/tracking and linear-MSE-style label-drift
tracking without dataset routing. The 2026-05-07 strict two-timescale branch
now closes that local digit readout problem.

2026-05-07 readout review:
`docs/research/step2_readout_consistency_review.md` records the latest
diagnosis and two adaptive-readout probes. A target-repeat-gated
`adaptive_simplex` readout preserves the label-drift side in a 3-seed probe and
improves class-blocked relative to linear MSE, but it still does not close
class-blocked online MSE. A CE-loss floor is rejected because it collapses
label-drift. The next mechanism should decouple readout loss from prediction
transform and test a robust simplex loss continuum, not tune kappa, sigma, or
repetition harder.

The decoupling and robust-loss pass has now also been run. Explicit
`readout_loss_mode` and `readout_prediction_mode` controls were added, along
with GCE and clipped-logit prediction probes. The critical 3-seed
class-blocked/label-drift run rejects all simple bridges:
linear-loss/softmax-prediction, CE-loss/identity-prediction,
CE-loss/clipped-prediction, GCE, and adaptive GCE all fail at least one of the
two rows badly. The remaining plausible mechanism is no longer a scalar loss
tweak; it is a genuinely two-timescale readout inside one learner state, with
both heads updated every step and sharing the same UPGD trunk.

The factorized label-map pass has now also been run. `factorized_simplex` adds
a row-stochastic causal adapter `A_t` and predicts `softmax(logits) @ A_t`.
Plain factorization is rejected: it remains below `mlp64_64` on class-blocked
and loses label-drift badly. `adaptive_factorized_simplex` gates the adapter by
target-repeat EMA and preserves label-drift wins, but it does not improve
class-blocked beyond the earlier `adaptive_simplex` branch. This makes the next
readout direction sharper: the missing ingredient is not a single target-map
adapter, but separate readout timescales or better causal readout diagnostics.

2026-05-07 two-timescale update:
`two_timescale_simplex` is now implemented with a slow linear-MSE head and a
fast simplex CE head in the same UPGD state. The head-only version preserves
label-drift but does not close class-blocked. Adding gated fast-head CE pressure
to the shared trunk gives the first positive 3-seed bridge:
`upgd64_64_structure_sigma1e_4_adaptk035_065_lr06_repx075_meta001_twotime_fastx2_trunk05_notrunk_tight`
beats the best same-run MLP by `+0.000028` on class-blocked final-window MSE
(`2/3` wins) and by `+0.00398` on label-drift final-window MSE (`3/3` wins).
That first bridge is now superseded by the fair strict row documented in the
2026-05-07 closeout above. The important implementation correction is that the
fast head must be ObGD-bounded; the promoted row uses a separate bounded
readout budget plus slow simplex MSE suppression.

Diagnostics were also added to the ablation JSON for future UPGD runs:
utility distributions, expected low-utility perturbation-energy share, hidden
activation stable/effective rank, learned control multipliers, and unit
replacement counts/ages. See `docs/research/step2_two_timescale_readout.md`.

## 2026-05-06 Update: Production Memory Path

The OPMNIST retained-view result is now served by a JAX fixed-budget prototype
memory, not by D18. The production-facing Step 2 split is:

- `UPGDLearner.step2_default(n_heads)` for the differentiable plastic supervised
  learner.
- `PrototypeMemoryLearner(PrototypeMemoryConfig(...))` for retained
  class-view memory when OPMNIST-style permutation retention is the target.

The compact 3-seed OPMNIST evidence:

| Method | Final MSE | Final Acc | Test MSE | Test Acc | Runtime s |
|---|---:|---:|---:|---:|---:|
| `upgd_h64_softmax_ce` | `0.073840` | `0.465000` | `0.062649` | `0.722500` | `1.061956` |
| `proto_s20` | `0.026361` | `0.825000` | `0.013433` | `0.907333` | `0.618043` |
| `hybrid_h64_s20_uncertain` | `0.026303` | `0.826667` | `0.014230` | `0.910500` | `1.553537` |

Decision: D18 is retired from the production Step 2 path. Keep D18 only as a
historical synthetic all-14 research artifact. The active OPMNIST closure is
JAX prototype memory, optionally paired with UPGD when the deployed learner
needs both hidden-feature plasticity and view-retention memory.

The later full 800-task OpenML MNIST run used the packaged UPGD-memory path
against raw and sharpened MLP baselines. That run closes the scale-completion
question for one seed, but it changes the interpretation: UPGD-memory is a
strong online-tracking learner at published task count, while retained
all-permutation test performance remains below the best fair MLP.

Sources: `docs/research/step2_new_directions/d22_upgd_prototype_hybrid_local3.md`
and
`docs/research/step2_new_directions/step2_upgd_d18_d20_efficiency_decision.md`.

## 2026-05-06 Update: Compute-Efficient Promotion

The promoted Step 2 default is now the resource-efficient target-structure
UPGD branch:

`UPGDLearner.step2_default(n_heads)`

Core settings: hidden size 32, `loss_normalization="target_structure"`,
`ObGDBounding(kappa=0.5)`, `perturbation_sigma=1e-4`,
`perturbation_noise="rademacher"`, `perturbation_interval=16`, and lean
bookkeeping with unit-utility traces and previous-gradient traces disabled
unless their corresponding mechanisms are active.

Why this supersedes the slower meta branch as the advertised default:

- It remains a single learner, not a portfolio or router.
- It still assigns online utility to hidden weights and explores low-utility
  features.
- It removes readout meta-plasticity, adaptive kappa, and recycling traces from
  the default.
- It beats the 64-hidden-unit MLP baseline while running faster in the JAX scan
  throughput benchmark.

Key compute/quality evidence:

| Benchmark | Variant | Result |
|---|---|---|
| Efficiency one-hot | width-32 lean Rademacher UPGD | `38,610.7` steps/s vs MLP64 `26,551.5` |
| Efficiency dense | width-32 lean Rademacher UPGD | `38,322.6` steps/s vs MLP64 `24,432.1` |
| Synthetic polynomial | width-32 lean Rademacher UPGD | `+0.5634 +/- 0.0320`, `30/30` wins vs best MLP |
| Synthetic frequency | width-32 lean Rademacher UPGD | `+0.6107 +/- 0.0410`, `30/30` wins vs best MLP |
| Synthetic compositional | width-32 lean Rademacher UPGD | `+0.0781 +/- 0.0036`, `30/30` wins vs best MLP |
| Digits final-window MSE | width-32 lean Rademacher UPGD | `+0.0078 +/- 0.0004`, `147/150` wins vs MLP64 |
| Digits test accuracy | width-32 lean Rademacher UPGD | `+0.0269 +/- 0.0018`, `135/150` wins vs MLP64 |

Width 16 is the max-speed branch (`77,275.4` one-hot steps/s and `47,926.4`
dense steps/s in the same benchmark), but width 32 is the conservative default
because it is more robust on class-blocked and permuted-pixel digit rows.

Sources: `docs/research/step2_compute_efficient_upgd.md`,
`output/subagents/compute_efficiency/small_rademacher_synthetic_30seed_6000/out_of_class_SUMMARY.md`,
and `output/subagents/compute_efficiency/small_rademacher_digits_30seed_h64baseline/SUMMARY.md`.

## 2026-05-06 Update: Promoted Step 2 Learner

The promoted single-learner Step 2 answer is now **target-structure UPGD**:

`UPGDLearner(loss_normalization="target_structure", perturbation_sigma=1e-4, bounder_kappa=0.5, ...)`

This supersedes the earlier split in which mean-loss UPGD won the dense
synthetic vector-target streams while sum-loss UPGD won sparse one-hot digits.
It also supersedes the interim `target_density` bridge. `target_density`
divides by the number of nonzero supervised targets, which preserves one-hot
digit pressure but over-boosts dense exact-zero rows and sparse multilabel
targets. `target_structure` uses sum-style loss only for non-negative simplex
targets with total mass 1, and otherwise uses mean-style loss over active
heads. It is the smallest robust bridge found so far: no portfolio, router, MLP
fallback, or dataset-specific switch.

Why this is more Alberta-Plan-friendly than the earlier D18 closure:

- It is a single continually updated learner with nonlinear hidden features.
- Hidden feature utility is tracked online from weight magnitude and gradient
  demand.
- Low-utility hidden weights are perturbed, and optional low-utility hidden
  units can be recycled.
- Output heads support vector targets directly, so the same representation is
  shared across supervised tasks.
- The useful controls are causal and temporally uniform: bounded ObGD, optional
  fast/slow loss adaptive `kappa`, and optional gradient-alignment readout
  plasticity.

The practical promoted defaults are:

| Use case | Variant | Reason |
|---|---|---|
| Simple normalization check | `structure_sigma1e4_kappa05` | Smallest target-structure bridge in the canonical synthetic harness; 30/30 wins on all dense synthetic out-of-class streams and no target-density ambiguity. |
| Best no-meta digit branch | `upgd_density_sigma1e_4_adaptk035_065_lr05_e1` | Same loss behavior as `target_structure` on one-hot digits; matches learned-meta aggregate final-window MSE and has the best held-out digit accuracy in the 30-seed simplification rerun. |
| Best average/test branch | `structure_meta003_notrunk` on synthetic, density-equivalent `meta003_notrunk_tight` on one-hot digits | Readout-only meta-plasticity remains useful, but trunk meta is not needed. |
| Strict online-MSE branch | fixed-kappa or repetition-gated density-equivalent branches | Repetition helps strict class-blocked online MSE but gives back aggregate held-out accuracy; it is not the promoted average/test default. |
| Strict digit/readout branch | `twotime_fastx2_trunk2_slow0_sepbound` 64-64 target-structure UPGD | Separately bounded two-timescale readout; beats the best same-run fair MLP on all five 30-seed digit final-window MSE rows while preserving retained accuracy where meaningful. |

Key 30-seed evidence after the simplification/structure reruns:

| Benchmark | Variant | Result vs fair MLP |
|---|---|---|
| Dense-zero stress | `target_structure` | `+0.000232 +/- 0.000311`, `5/8` wins; matches mean and fixes target-density's negative row |
| Sparse multilabel stress | `target_structure` | `+0.026938 +/- 0.000865`, `8/8` wins; matches mean and avoids target-density's weaker margin |
| Synthetic polynomial | `structure_sigma1e4_kappa05` | `+0.5481 +/- 0.0317`, `30/30` wins |
| Synthetic frequency | `structure_sigma1e4_kappa05` | `+0.5790 +/- 0.0379`, `30/30` wins |
| Synthetic compositional | `structure_sigma1e4_kappa05` | `+0.0925 +/- 0.0039`, `30/30` wins |
| Digits final-window MSE | density-equivalent `lr05_e1` | `+0.0062 +/- 0.0003`, `120/150` wins |
| Digits test accuracy | density-equivalent `lr05_e1` | `+0.0299 +/- 0.0019`, `138/150` wins |
| Digits fixed-kappa control | density-equivalent `kappa05` | final-window MSE `+0.0036 +/- 0.0002`, `147/150` wins |

Sources:
`docs/research/step2_target_structure_upgd_stress.md`,
`output/subagents/target_structure_canonical_out_of_class_30seed/out_of_class_SUMMARY.md`,
`output/subagents/upgd_simplification_scale_digits_lr05e1_30seed/SUMMARY.md`, and
`output/subagents/upgd_simplification_structure_digits_30seed/digits_ablation_SUMMARY.md`.

The full promotion reruns close the target-structure hole but also narrow the
claim. The alternate `step2_upgd_ablation.py` synthetic protocol and the
five-regime digits promotion matrix do not support saying that the simple
fixed-kappa row is universally superior to the strongest same-run fair MLP. In
particular, `output/subagents/worker_a_upgd_promotion_digits_30seed/digits_ablation_SUMMARY.md`
shows `structure_sigma1e_4_kappa05` losing final-window MSE to the best MLP on
IID, class-blocked, label-drift, and mask-noise digits, while winning only the
permuted regime. The promoted claim is therefore normalization-level plus the
adaptive/readout UPGD digit branches, not unconditional fixed-kappa dominance.

Simplification ablation verdict:

| Component | Verdict | Evidence |
|---|---|---|
| `loss_normalization="target_structure"` | Essential default | It preserves one-hot sum pressure, preserves dense mean pressure, and fixes the exact-zero/multilabel ambiguity that `target_density` leaves. |
| `perturbation_sigma=1e-4` | Keep as conservative default | Quick digits and synthetic sweeps found sigma `0` close to sigma `1e-4`, but low-noise perturbation is the causal feature-exploration mechanism and has no observed 30-seed penalty. |
| ObGD bounds | Essential | The successful variants all use bounded updates; unbounded UPGD was rejected in earlier bounder/kappa sweeps. |
| Adaptive `kappa` | Required for the serious digit branch | It is not needed to demonstrate the normalization bridge on canonical synthetic streams, but the strict five-regime digit matrix rejects the simple fixed-kappa row as a universal default. |
| Group meta-plasticity | Optional | Readout-only meta matches the best digit aggregate, but the no-meta `lr05_e1` branch matches final-window MSE and wins held-out accuracy. |
| Trunk meta | Removable | `meta003_notrunk_tight` matched or beat trunk-meta branches; no current promotion needs learned trunk plasticity. |
| Head-bias meta | Removable for the average/test default | Helpful only inside learned readout ablations, not needed by the no-meta digit branch. |
| Repeated-target plasticity | Removable from the default | It improves strict class-blocked online MSE but lowers aggregate held-out behavior. |
| Hidden-unit replacement | Removable from the default | Low-utility perturbation is enough in the promoted evidence; recycle/replacement variants are not top rows. |
| Utility definition | Keep current `|w * grad|` family | The current causal utility signal underlies all positive UPGD runs; no alternate utility beat it in this ablation batch. |

Critical limitation: this is an empirical Step 2 closure, not a theorem. It
does not prove universal recursive representation learning, and the new strict
promotion runs show that fixed-kappa UPGD is not enough for an unconditional
"beats every fair MLP" claim. The remaining scientific pressure points are the
hand-set two-timescale readout controls, scalar tabular regression, and
delayed/history temporal regression.
The strongest current no-portfolio statement is narrower: target-structure is
the correct promoted loss rule, and adaptive/readout UPGD remains the best
single-learner Step 2 family because it combines dense vector regression wins,
one-hot digit wins by simplex-equivalent evidence, multilabel wins, explicit
feature utility, and resource-managed perturbation.

See `docs/research/step2_upgd_critical_paper.md` for the current critical
paper draft, including the conditional theorem that is actually defensible, the
known-vs-novel split, required ablations, and the promotion bar for replacing
target-density with target-structure UPGD. The important correction is that the
repo should not claim unconditional universality: the supported claim is
scale-consistent online adaptation over the Step 2 target families, with
utility-guided feature exploration under bounded updates.

The remaining-hole work is now wired as runnable experiments rather than prose
only:

- Full target-structure promotion commands and five-regime digits coverage:
  `docs/research/step2_upgd_promotion_matrix.md`.
- Class-blocked stability/plasticity ablations and the current conservative
  retained-test compromise:
  `docs/research/step2_class_blocked_upgd_ablation.md`.
- External/domain-matrix expansion for tabular regression, exact-zero dense
  regression, sparse multilabel targets, masked/noisy image-like streams, and
  delayed temporal targets:
  `docs/research/step2_external_benchmarks.md`.
- Reusable diagnostics for utility distributions, perturbation-energy
  allocation, hidden activation ranks, learned multipliers, and unit
  replacement state:
  `alberta_framework.core.diagnostics`.

## Superseded Simple Non-Router Learner: D18 Persistent Trace

Historical note: before target-structure UPGD closed the mean/sum loss conflict,
the best candidate under the stricter "one learner, no output
portfolio, no router, no selected expert" constraint is the D18 additive
resource-basis learner with a causal persistence-gated target trace:

`d18_step2_gain_l2_0p1 + simplex_output + target_trace_persistence_gate`

This learner has one prediction path and one residual update. It combines a
resource-managed RKHS core, tanh/Fourier basis, tiny polynomial/unified residual
blocks, learned block gains, online-discovered one-hot simplex projection, and a
causal target trace that is enabled only when previous targets show persistent
one-hot dynamics. It is not a deployment portfolio and does not route among
learners or MLP experts.

Historical canonical command:

```bash
python "examples/The Alberta Plan/Step2/new_directions/d18_simple_universal_resource_basis.py" \
  --datasets all \
  --steps 1200 \
  --n-seeds 10 \
  --final-window 300 \
  --configs step2_gain_l2_0p1 \
  --simplex-output \
  --target-trace-scale 4 \
  --target-trace-decay 0.95 \
  --target-trace-clip 1.0 \
  --target-trace-persistence-gate \
  --target-persistence-decay 0.95 \
  --target-persistence-power 6 \
  --output-dir outputs/step2_main_d18_persistent_trace_p6_all_10seed \
  --note-path docs/research/step2_main_d18_persistent_trace_p6_all_10seed.md
```

The shorter named config is equivalent for future runs, except that the method
label changes to `d18_step2_persistent_trace`:

```bash
python "examples/The Alberta Plan/Step2/new_directions/d18_simple_universal_resource_basis.py" \
  --datasets all \
  --steps 1200 \
  --n-seeds 10 \
  --final-window 300 \
  --configs step2_persistent_trace \
  --output-dir outputs/step2_main_d18_persistent_trace_p6_all_10seed \
  --note-path docs/research/step2_main_d18_persistent_trace_p6_all_10seed.md
```

On the 14-regime, 10-seed matrix, it is positive on every aggregate
final-window MSE comparison against the same-run best fair MLP width. Total
seed-level final-window MSE wins/losses/ties are `138/2/0`.

| Weakest row | Final-window MSE diff | Wins/losses/ties |
|---|---:|---:|
| Digits class-blocked | +0.00141 +/- 0.00012 | 10/0/0 |
| Digits mask noise | +0.00915 +/- 0.00203 | 10/0/0 |
| Synthetic compositional | +0.04220 +/- 0.01785 | 9/1/0 |
| Synthetic polynomial | +0.09408 +/- 0.03062 | 9/1/0 |

Because D18 uses a one-hot simplex projection on digit targets, the fair audit
also compares against a projected-MLP MSE derived from the MLP's final-window
accuracy: `0.2 * (1 - accuracy)` for 10 classes. The D18 candidate remains
positive by mean on all digit rows under that stricter check. The 30-seed risk
confirmation preserves the two hardest digit rows:

| Risk row | Raw MSE diff | Raw wins/losses/ties | Projected-MLP diff | Projected wins/losses/ties |
|---|---:|---:|---:|---:|
| Digits class-blocked | +0.00147 | 30/0/0 | +0.00024 | 11/2/17 |
| Digits mask noise | +0.00999 | 29/1/0 | +0.00147 | 18/11/1 |

Source:
`outputs/step2_canonical/simple_d18_persistent_trace_all_10seed_results.json`,
`outputs/step2_canonical/simple_d18_persistent_trace_all_10seed_SUMMARY.md`,
`outputs/step2_canonical/simple_d18_persistent_trace_risk_digits_30seed_results.json`,
and `docs/research/step2_main_d18_persistent_trace_p6_all_10seed.md`.

Critical limitation: this is the best simple non-router learner found so far,
but it is not yet an elegant single feature-construction principle. It is still
a hand-assembled additive resource-basis learner. The clean alternatives tested
in parallel failed different blockers: pure residual-birth failed
algebraic/periodic/compositional rows; pure kernel failed the six-blocker
matrix; wide random features failed synthetic polynomial/frequency; and
calibration-only simplex mechanisms failed class-blocked digits. The evidence
therefore supports "non-router learner beats fair MLP on the current matrix,"
not an unqualified claim that Step 2 is theoretically solved.

## Best Standalone Learner: Tuned UPGD

`UPGDLearner` is the strongest single learner on the canonical synthetic
out-of-class Step 2 suite, and the tuned single-learner digits variant now also
beats the fair `mlp64` baseline on the external sklearn-digits universality
matrix.

On the canonical synthetic out-of-class suite, UPGD wins all three streams
against the best fair MLP width with 30 seeds:

| Stream | Best MLP final MSE | UPGD final MSE | Paired diff | Wins |
|---|---:|---:|---:|---:|
| Out-of-class polynomial | 1.1458 | 0.5767 | +0.569 +/- 0.032 | 30/30 |
| Frequency mismatch | 1.1689 | 0.6335 | +0.535 +/- 0.037 | 30/30 |
| Compositional 2-layer oracle | 0.1908 | 0.1634 | +0.027 +/- 0.003 | 29/30 |

Source: `outputs/step2_canonical/out_of_class_results.json` and
`docs/research/step2_results.md`.

On external sklearn-digits, the best current single-learner family is tuned
UPGD with low-noise utility perturbations and aggressive ObGD bounding. The
minimal already-scaled baseline before learned meta-control is:

`upgd_sum_sigma1e_4_adaptk035_065_lr06`

That is sum multi-task MSE, low-utility perturbation `sigma=1e-4`, base ObGD
`kappa=0.5`, hidden-feature step-size multiplier `0.6`, and an online
fast/slow-loss controller that clips effective `kappa` to `[0.35, 0.65]`.
The learned group-plasticity section below supersedes this as the current
average-MSE/test default, but this simpler baseline remains the important
control.

Corrected 30 seeds x 5 regimes = 150 paired comparisons against the same-run
fair `mlp64`:

| Metric | MLP | Tuned UPGD | Paired diff | Wins |
|---|---:|---:|---:|---:|
| Final-window MSE | 0.0402 +/- 0.0017 | 0.0344 +/- 0.0015 | +0.0059 +/- 0.0003 | 127/150 |
| Final-window accuracy | 0.8304 +/- 0.0088 | 0.8562 +/- 0.0078 | +0.0258 +/- 0.0018 | 123/150 |
| Test MSE | 0.0578 +/- 0.0032 | 0.0517 +/- 0.0033 | +0.0061 +/- 0.0003 | 138/150 |
| Test accuracy | 0.7184 +/- 0.0244 | 0.7408 +/- 0.0253 | +0.0224 +/- 0.0020 | 119/150 |

Source: `output/subagents/combined_upgd/kappa_lr_adaptive_top_30seed/SUMMARY.md`
and `docs/research/step2_single_upgd_kappa_featurefinding_results.md`.

Latest density-control follow-up:
`output/subagents/combined_upgd/digits_density_control_30seed/SUMMARY.md`
improves the standalone UPGD digit control further. The best final-window MSE
variant, `upgd_density_sigma1e_4_adaptk035_065_lr06_meta003_notrunk_tight`,
beats `mlp64` by `+0.0061 +/- 0.0003` over 150 paired digit comparisons and
improves test accuracy by `+0.0296 +/- 0.0020`. Combined with the same
target-structure mechanism's 30-seed synthetic wins and the one-hot equivalence
between density and structure on digits, this replaces D18 as the more
Alberta-Plan-friendly current closure.

The repetition-compromise confirmation at
`output/subagents/combined_upgd/digits_density_repetition_compromise_30seed/SUMMARY.md`
does not change that conclusion. The same non-repetition
`meta003_notrunk_tight` variant remains the strongest aggregate digit control:
`+0.0063 +/- 0.0003` final-window MSE and `+0.0298 +/- 0.0017` test accuracy
versus `mlp64`. Repetition-boosted variants improve some class-blocked MSE
rows but give back aggregate test accuracy.

The class-blocked follow-up adds target-repetition-gated output-head plasticity
without changing the sum-MSE objective, trunk, or learner family. The key strict
variant is:

`upgd_sum_sigma1e_4_adaptk035_065_lr06_repx075`

On 30 fresh seeds x 5 regimes = 150 paired comparisons, it was the first
single-learner UPGD variant here to beat `mlp64` on final-window MSE in every
seed/regime case:

| Variant | Final-window MSE diff | Wins | Test accuracy diff |
|---|---:|---:|---:|
| `adaptk035_065_lr06` | +0.0058 +/- 0.0003 | 127/150 | +0.0236 +/- 0.0020 |
| `repx025` | +0.0057 +/- 0.0003 | 147/150 | +0.0209 +/- 0.0021 |
| `repx075` | +0.0055 +/- 0.0002 | 150/150 | +0.0181 +/- 0.0023 |

Source:
`output/subagents/combined_upgd/repetition_top_30seed/SUMMARY.md`.

Interpretation at that stage: `repx075` became the strict online
universality/no-loss final-window MSE anchor, while `repx025` was the
conservative class-blocked compromise. Later learned group-plasticity runs
improved held-out behavior further.

The newest single-learner follow-up adds online meta-control to UPGD.
Consecutive gradient alignment now learns bounded log multipliers for trunk
updates, output-head weights, output-head biases, and repeated-target head
plasticity. A second controller can also learn a bounded log multiplier on
effective ObGD `kappa`.

The 30-seed validation is now decisive enough to separate the two ideas. Learned
group plasticity is useful; learned kappa is not yet a default. The next
ablation showed that trunk meta-plasticity is not needed for the best
average/test branch. On 30 fresh seeds x 5 regimes = 150 paired comparisons
against the same-run fair `mlp64`:

| Variant | Final-window MSE diff | Wins | Test MSE diff | Test accuracy diff |
|---|---:|---:|---:|---:|
| `meta003_notrunk_tight` | +0.0061 +/- 0.0003 | 120/150 | +0.0078 +/- 0.0003 | +0.0293 +/- 0.0018 |
| `meta003_tight` | +0.0060 +/- 0.0003 | 120/150 | +0.0076 +/- 0.0003 | +0.0280 +/- 0.0018 |
| `meta003_noheadb_tight` | +0.0059 +/- 0.0003 | 120/150 | +0.0074 +/- 0.0003 | +0.0271 +/- 0.0017 |
| `repx075_meta001_notrunk_tight` | +0.0057 +/- 0.0002 | 149/150 | +0.0047 +/- 0.0004 | +0.0212 +/- 0.0016 |
| `repx075` | +0.0056 +/- 0.0002 | 150/150 | +0.0037 +/- 0.0006 | +0.0213 +/- 0.0020 |

Source:
`output/subagents/combined_upgd/meta_group_ablation_top_30seed/SUMMARY.md`.

Interpretation: use `meta003_notrunk_tight` as the promoted average-MSE/test
default within the single UPGD family. This is simpler than the previous
learn-every-group setting: it learns readout plasticity but leaves trunk
plasticity fixed. It is not strict no-loss because it still loses the
`class_blocked` final-window MSE row. At that stage fixed `repx075` was the
strict online-MSE no-loss anchor. The later target-density repetition
compromise promoted `repx075_meta001_notrunk_tight` as the better strict branch
because it reached `150/150` online-MSE wins with better aggregate held-out
accuracy than fixed `repx075`.

Learned kappa remains in the implementation as a clean ablation and possible
future stability controller, but the 30-seed evidence says that the current
mechanism to promote is learned readout plasticity, not learned trunk
plasticity or learned kappa.

The LR follow-up confirms the frontier rather than producing one winner. On a
separate 30-seed block, `upgd_sum_sigma1e_4_adaptk035_065_lr05_e1` matches the
best final-window MSE diff at `+0.0059`, while fixed
`upgd_sum_sigma1e_4_kappa05_lr045` is best for held-out test MSE
(`+0.0071`) and test accuracy (`+0.0265`). The online-MSE default remains
adaptive `lr=0.6` because it is stronger on final-window accuracy and less bad
on class-blocked final-window MSE.

The 2026-05-06 simplification rerun changes the default loss rule from
`target_density` to `target_structure`. On dense synthetic targets, existing
structure variants match the density variants within seed noise while keeping
all three streams at `30/30` wins against the same-run best fair MLP. On
one-hot digits, the density-control `lr05_e1` and `meta003_notrunk_tight`
results are structurally equivalent to target-structure loss because every
active target is a non-negative simplex target. The no-meta `lr05_e1` branch
matches the learned-meta final-window MSE margin (`+0.0062 +/- 0.0003`) and has
the best test-accuracy margin (`+0.0299 +/- 0.0019`) in that scaled rerun.
Therefore the simple promoted learner is target-structure UPGD with fixed
ObGD/low-noise perturbation, with adaptive-kappa/readout-meta branches retained
as evidence-backed variants rather than required machinery.

Important caveat: repetition-gated head plasticity solves class-blocked
online-MSE against MLP, but class-blocked held-out retention is still the main
unresolved Step 2 pressure point.

## Best Legacy Overall Step 2 System: Strict Universal Portfolio

The strongest legacy Step 2 system is `step2_universal_portfolio.py`: a
temporally-uniform live portfolio over fair MLP widths, low-noise UPGD, and
dynamic sparse rewiring.  It uses discounted Hedge and causal retention guards.
This remains useful as a reference result. Tuned UPGD is still the best
single-learner branch, while the conclusive telemetry-gated portfolio below is
the current broad all-benchmark winner.

Promoted 10-seed result against the best fair MLP width:

| Regime | Final-window MSE vs best MLP | Test accuracy vs best MLP |
|---|---:|---:|
| Synthetic polynomial | +0.0414 | N/A |
| Synthetic frequency | +0.0168 | N/A |
| Synthetic compositional | +0.0018 | N/A |
| Digits IID | +0.0071 | +0.0128 |
| Digits class-blocked | +0.0000 | +0.0970 |
| Digits permuted pixels | +0.0075 | +0.0156 |
| Digits mask noise | +0.0094 | +0.0158 |
| Digits label drift | +0.0055 | +0.0128 |

30-seed risk checks preserve the main conclusion on synthetic compositional,
synthetic frequency, class-blocked retention, and non-blocked digits.

Source: `outputs/step2_canonical/universal_portfolio_strict_SUMMARY.md`.

## Best Broad All-Benchmark Candidate: Conclusive Telemetry-Gated Portfolio

The broadest current Step 2 candidate is `step2_conclusive_learner.py`, which
routes over recursive, polynomial, Fourier, random-tanh, and fair MLP experts.
The promoted 10-seed run disables UPGD/dynamic-sparse for deployment routing,
uses discounted Hedge with eta `0.5`, adds a causal telemetry gate that
recovers the stronger Worker-B route policy when recent routing telemetry
matches its failure signature, and blends in the live MLP selector as a `0.5`
floor. This is the first broad all-suite run with no seed-level final-window
MSE losses against the same-run best fair MLP:

```bash
python "examples/The Alberta Plan/Step2/step2_conclusive_learner.py" \
  --benchmarks all \
  --steps 1200 \
  --n-seeds 10 \
  --final-window 300 \
  --warmup-steps 250 \
  --weighting-scheme discounted_hedge \
  --hedge-eta 0.5 \
  --hedge-discount 0.995 \
  --selector-window 0 \
  --stacker-step-size 0.006 \
  --safe-route-sources recursive_features,polynomial_features \
  --digits-deployment-objective all_h128_blend \
  --h128-blend-weight 0.5 \
  --route-policy-mode telemetry_worker_b \
  --route-telemetry-window 300 \
  --worker-b-switch-margin 0.010 \
  --mlp-floor-blend-weight 0.5 \
  --mlp-floor-source selector \
  --disable-experts upgd_low_noise,dynamic_sparse \
  --output-dir outputs/step2_main_eta05_telemetry_worker_b_floor05_all_10seed \
  --note-path docs/research/step2_main_eta05_telemetry_worker_b_floor05_all_10seed.md
```

On the current 10-seed all-benchmark validation matrix, it is nonnegative on
14/14 aggregate final-window MSE comparisons against the same-run best fair MLP
and has no seed-level final-window MSE losses. Total seed-level final-window
MSE wins/losses/ties are `130/0/10`; the 10 ties are class-blocked digits
online MSE. That class-blocked row still beats the best fair MLP on held-out
accuracy by `+0.3492 +/- 0.0080` at `10/0/0`.

The previous weak rows are now positive at seed level:

| Row | Final-window MSE diff | Wins/losses/ties |
|---|---:|---:|
| Controlled rare | +0.0115 +/- 0.0025 | 10/0/0 |
| Synthetic compositional | +0.0413 +/- 0.0099 | 10/0/0 |
| Synthetic polynomial | +0.0489 +/- 0.0286 | 10/0/0 |

Source:
`outputs/step2_canonical/conclusive_telemetry_worker_b_floor05_results.json`,
`outputs/step2_canonical/conclusive_telemetry_worker_b_floor05_SUMMARY.md`, and
`docs/research/step2_main_eta05_telemetry_worker_b_floor05_all_10seed.md`.

This is the best broad current-matrix learner found so far. It is still a
portfolio/resource-allocation result, not a single universal feature
construction algorithm.

## Best Learned Allocation Result: Resource Manager

`step2_resource_manager_stateful_external.py` is the strongest learned resource
allocation result.  It causally allocates among static MLP, low/high UPGD, and
CBP-like replacement experts on stateful external digits regimes.

Key 10-seed paired results versus `mlp_static`:

| Regime | Final-window MSE diff | Test accuracy diff |
|---|---:|---:|
| Recurrent permutation | +0.0108, 10/10 | +0.0199, 9/10 |
| Recurrent mask noise | +0.0109, 10/10 | +0.0456, 10/10 |
| Class-blocked retention | +0.0010, 10/10 | -0.0007 tracking; +0.1121 retention |

This is the best evidence that learned resource allocation is useful, not just
a hand-picked fixed portfolio.

Source: `outputs/step2_canonical/resource_manager_stateful_external_SUMMARY.md`.

## Best Feature-Economy Branches

The market/replicator work is not the best universal Step 2 solution, but it is
the best sparse feature-discovery evidence:

- `exp01_replicator_mutator.py`: strong on polynomial and frequency when the
  grammar contains useful atoms.
- `exp03_market_auction.py`: strong market selection with rent/bankruptcy,
  meta-constructor priors, and persistent candidate nursery support.
- `exp05_fast_slow_memory.py`: narrow positive on polynomial only.

Current reading: these should feed the next UPGD/resource-manager hybrid, not
replace the portfolio.

## Cleanup Decision

Failed moonshot scripts and raw outputs were removed from the active tree.  The
kept runnable Step 2 moonshot files are:

- `exp01_replicator_mutator.py`
- `exp03_market_auction.py`
- `exp05_fast_slow_memory.py`
- `m01_centered_targets.py`
- `m04_hashed_quadratic_lift.py`

The individual failed branches remain summarized in synthesis docs, but their
raw scripts/results are no longer active working candidates.

## RLS Calibration Transfer Check

The Step 8 separate RLS reward model transfers safely to Step 2 only as a
tiny prediction-space calibrator, not as a new default. In the OPMNIST-style
sklearn-digits 28x28 stream, `step2_hybrid_memory_trace_rls_cal` uses previous
RLS weights and the previous deployment gate for the current prediction, then
updates both from the current label for the next step.

The fully prequential 5-seed/5000-step check shows small online-MSE benefit
over `step2_hybrid_memory_trace` (`+0.000260 +/- 0.000060`, `5/5` wins), but
accuracy and retained-test metrics are mixed: final-window accuracy is
`+0.003000 +/- 0.002280`, held-out test MSE is `-0.000058 +/- 0.000092`, and
held-out test accuracy is `+0.000040 +/- 0.000818`. Keep the branch available
via `--include-rls-calibrated`, but do not promote it.

## Dream-Replay Transfer Check

Causal dream replay is now implemented in the OPMNIST runner as
`DreamReplayLearner`. It stores already-seen examples after their current
prediction is scored, prioritizes by surprise or learning progress, performs
extra updates after the real step, then lowers priority as dream loss falls.

The mechanism is positive for the bare `upgd_structure_softmax_h64` branch. In
the 5-seed/5000-step sklearn-digits 28x28 OPMNIST-style run,
`upgd_structure_softmax_h64_dream_surprise2` beats the bare h64 softmax UPGD on
all paired metrics: online MSE `+0.017353`, online accuracy `+0.105320`,
final-window MSE `+0.013490`, final-window accuracy `+0.102400`, held-out test
MSE `+0.003607`, and held-out test accuracy `+0.022920`, all `5/5` wins.

It is not a default promotion. Surprise dream replay around the stronger
`step2_hybrid_memory_trace` was harmful in the 3-seed/2000-step check, and the
best single-UPGD dream variants still lose every metric to the primary
UPGD-memory hybrid in the 5-seed/5000-step run. Current interpretation:
dreaming is a useful plasticity amplifier for weak single UPGD, not a
replacement for retained-view memory.

## Research Direction

Lean into three things:

1. Treat target-structure UPGD as the promoted supervised Step 2 learner: one
   prediction path, no deployment portfolio, no output router, explicit hidden
   feature utility, low-utility perturbation/recycling, and positive 30-seed
   evidence on dense synthetic vector targets and sparse online digits.
2. Keep the JAX prototype memory as the active retained-view memory path.
   D18, the strict universal portfolio, recursive-router, and native deep
   lifecycle work are reference baselines or rejected alternate paths, not the
   active closure claim.
3. Continue scale replication separately. The current full-source/full-task
   OpenML MNIST latest-best comparison is now complete at 800/800 blocks
   (48,000,000 online updates, one seed). The UPGD-memory variants beat the
   best same-run fair MLP on online MSE, online accuracy, and final-window MSE,
   but they still lose final-window accuracy and all-permutation held-out test
   metrics. Treat this as completed scale evidence with a retained-view
   generalization caveat, not as a universal OPMNIST victory.

The next serious work is no longer another local digit blocker sweep. It is
external scale replication plus causal readout generalization for scalar
regression, delayed/history temporal regression, sparse multilabel, and
dense/vector targets under the same strict two-timescale implementation.
