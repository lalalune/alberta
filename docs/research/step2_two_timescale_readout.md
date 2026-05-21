# Step 2 Two-Timescale Readout

Date: 2026-05-07.

## Status

Promoted as the strict one-branch digit readout after the 30-seed
five-regime confirmation passed.

The mechanism is `readout_mode="two_timescale_simplex"` with an optional
`readout_fast_trunk_gradient_multiplier`.

## Mechanism

The learner has one shared UPGD trunk and two readout heads in the same learner
state:

- slow head: linear-MSE head, trained on target-structure MSE;
- fast head: simplex CE head, trained on non-negative unit-mass targets.

The emitted prediction is a causal target-repeat/simplex blend:

`pred = gate * softmax(fast_logits) + (1 - gate) * slow_logits`

The first version trained only the fast head. The current version can also send
the fast CE head's hidden cotangent into the shared trunk, scaled by
`readout_fast_trunk_gradient_multiplier * gate`. This keeps label-drift mostly
linear because label-drift has low target-repeat pressure, while class-blocked
streams train probability-shaped features once labels persist.

Two fairness fixes are now part of the candidate:

- fast readout-head updates are ObGD-bounded, not unbounded;
- target-free `predict()` requires recent simplex targets as well as repeated
  targets before it blends toward the fast probability head.

The current strict candidate also uses
`readout_fast_head_bounder_mode="separate"` and
`readout_slow_simplex_gradient_multiplier=0.0`. The fast readout head gets its
own bounded readout budget, while the slow linear-MSE branch stops consuming
the shared bounded budget when the simplex gate is fully open.

## Evidence

| Run | Result |
|---|---|
| `output/subagents/class_blocked_retention/twotime_probe_3seed_6000/digits_ablation_SUMMARY.md` | Head-only fast readout preserves label drift but misses class-blocked. |
| `output/subagents/class_blocked_retention/fasttrunk_probe_3seed_6000/digits_ablation_SUMMARY.md` | `fastx2_trunk05` is positive on both rows: class-blocked `+0.000028`, `2/3`; label drift `+0.00398`, `3/3`. |
| `output/subagents/class_blocked_retention/diagnostics_smoke_1seed_300/digits_ablation_results.json` | Confirms diagnostics now include utility distributions, low-utility perturbation-energy share, activation ranks, learned controls, and unit replacement counts/ages. |
| `output/subagents/class_blocked_retention/fasttrunk_bound_patched_probe_5seed_6000/digits_ablation_SUMMARY.md` | Bounding the fast head inside the shared ObGD budget rejects the earlier unbounded bridge: class-blocked falls below MLP again. |
| `output/subagents/class_blocked_retention/fasttrunk_sepbound_x2_probe_5seed_6000/digits_ablation_SUMMARY.md` | Separately bounded fast readout plus slow-branch suppression clears the two-row risk check: class-blocked `+0.000348`, `5/5`; label drift `+0.005517`, `5/5`. |
| `output/subagents/class_blocked_retention/fasttrunk_sepbound_x2_allregime_10seed_6000/digits_ablation_SUMMARY.md` | `fastx2_trunk2_slow0_sepbound` beats the best same-run MLP on all five digit final-window MSE rows: IID `+0.006491` `10/10`, class-blocked `+0.000237` `9/10`, label drift `+0.004997` `10/10`, mask-noise `+0.009731` `10/10`, permuted `+0.006885` `10/10`. |
| `output/subagents/class_blocked_retention/fasttrunk_sepbound_x2_strict_30seed_6000/digits_ablation_SUMMARY.md` | The strict candidate repeats at 30 seeds against the best same-run fair MLP: IID `+0.006172` `30/30`, class-blocked `+0.000189` `26/30`, label drift `+0.005352` `30/30`, mask-noise `+0.009692` `30/30`, permuted `+0.007323` `30/30`. Retained accuracy is positive on IID, class-blocked, mask-noise, and permuted digits; label-drift retained accuracy is a numerical tie. |
| `output/subagents/class_blocked_retention/fasttrunk_sepbound_x2_synthetic_30seed_6000/synthetic_ablation_SUMMARY.md` | The strict digit row is not a global synthetic default: polynomial `+0.017758` `29/30` and frequency `+0.053843` `27/30` are positive, but compositional regression loses `-0.079795` with `0/30` wins. |

The promoted strict digit/readout row is:

`upgd64_64_structure_sigma1e_4_adaptk035_065_lr06_repx075_meta001_twotime_fastx2_trunk2_slow0_sepbound_notrunk_tight`

The corresponding code factory is:

`UPGDLearner.step2_strict_digit_readout_default(n_heads)`

## Promotion Gate

The branch is promoted for the strict digit readout because it satisfied the
digit gate:

- it beats the best same-run fair MLP on all five digit final-window MSE rows
  at 10 seeds;
- it preserves nonnegative retained-test accuracy on IID, class-blocked,
  mask-noise, and permuted digits;
- it repeats at 30 seeds.

The synthetic dense/vector regression matrix rejects making this row broader
than the strict digit readout. The fast simplex branch is causally gated off on
non-simplex targets, but the surrounding digit-tuned adaptive/repetition/meta
controls are not synthetic-optimal. Keep `UPGDLearner.step2_default` as the
global lightweight target-structure default.

## Theorem Shape

The honest theorem is conditional.

Assume a recurrent stream alternates between locally persistent simplex
segments and bounded-variation target-remapping segments. Assume the UPGD trunk
class contains a representation whose slow linear head has low squared error
under remapping and whose fast simplex head has low CE error on persistent
segments. Then the two-timescale readout has tracking error bounded by:

- slow-head online regret on remapped/nonpersistent segments;
- fast-head CE regret on persistent simplex segments;
- gate error from misclassifying persistence;
- trunk approximation error;
- variation of the target-remapping path.

This is not a theorem of universal feature construction. The universality claim
still requires assumptions about recurrence, finite feature budgets,
approximation class, and nonstationarity.

## Known Weaknesses

- The current gate is target-repeat/simplex EMA, not a learned uncertainty
  model.
- Fast head, fast trunk, and slow simplex budget multipliers are still hand-set
  in the candidate row.
- The digit-tuned adaptive/repetition/meta controls lose synthetic
  compositional regression, so this is not a broad universal default.
- The mechanism adds output parameters, so the paper must frame it as a single
  composite readout inside one learner, not as an expert portfolio.
