# Step 2 Current-Best Weakness And Direction Probe

Date: 2026-05-06

## Scope

This note audits the current strongest Step 2 research approaches and runs
small targeted probes on their weak spots.

The two general approaches are:

1. **D18 `step2_canonical`**: the strongest current all-14-regime non-router
   empirical learner. It combines a resource-managed RKHS core, random
   tanh/Fourier basis, strict degree-3 polynomial RLS residual, unified residual
   basis, learned additive gains, target persistence, and prototype retention.
2. **Lean/target-structure UPGD**: the simpler differentiable learner path and
   the best production-lean candidate. It is much closer to a core JAX learner
   and has strong compact OPMNIST evidence, but weaker all-purpose mechanism
   coverage.

D20 multi-prototype memory is treated here as a **component direction** rather
than a full general learner: it is exceptionally strong on OPMNIST retention but
does not yet cover the full Step 2 suite.

## New Probe Commands

```bash
source .venv/bin/activate

python "examples/The Alberta Plan/Step2/step2_theory_falsification.py" \
  --scenarios rotating_relevant_subspace \
  --steps 600 \
  --n-seeds 3 \
  --methods upgd,mlp,mlp_deep \
  --upgd-width 64 \
  --mlp-width 64 \
  --output-dir outputs/step2_weakness_probe/rotation_upgd64 \
  --note-path docs/research/step2_weakness_probe_rotation_upgd64.md

python "examples/The Alberta Plan/Step2/step2_theory_falsification.py" \
  --scenarios rotating_relevant_subspace \
  --steps 600 \
  --n-seeds 3 \
  --methods upgd,mlp,mlp_deep \
  --upgd-width 128 \
  --mlp-width 64 \
  --output-dir outputs/step2_weakness_probe/rotation_upgd128 \
  --note-path docs/research/step2_weakness_probe_rotation_upgd128.md

python "examples/The Alberta Plan/Step2/new_directions/d20_multiprototype_opmnist.py" \
  --steps 1000 \
  --n-seeds 3 \
  --final-window 200 \
  --mnist-source sklearn_digits_28x28 \
  --n-permutations 5 \
  --task-block-size 200 \
  --task-sampling sequential_epoch \
  --slots-per-class 5 \
  --prototype-update-rate 0.3 \
  --prototype-novelty-threshold 0.08 \
  --prototype-bandwidth 0.01 \
  --output-dir outputs/step2_weakness_probe/d20_slots5 \
  --result-prefix d20_slots5 \
  --note-path docs/research/step2_weakness_probe_d20_slots5.md

python "examples/The Alberta Plan/Step2/new_directions/d20_multiprototype_opmnist.py" \
  --steps 1000 \
  --n-seeds 3 \
  --final-window 200 \
  --mnist-source sklearn_digits_28x28 \
  --n-permutations 5 \
  --task-block-size 200 \
  --task-sampling sequential_epoch \
  --slots-per-class 10 \
  --prototype-update-rate 0.3 \
  --prototype-novelty-threshold 0.08 \
  --prototype-bandwidth 0.01 \
  --output-dir outputs/step2_weakness_probe/d20_slots10 \
  --result-prefix d20_slots10 \
  --note-path docs/research/step2_weakness_probe_d20_slots10.md

python "examples/The Alberta Plan/Step2/new_directions/d23_upgd_lean_sweep_opmnist.py" \
  --steps 1000 \
  --n-seeds 3 \
  --final-window 200 \
  --mnist-source sklearn_digits_28x28 \
  --n-permutations 5 \
  --task-block-size 200 \
  --task-sampling sequential_epoch \
  --output-dir outputs/step2_weakness_probe/d23_lean_opmnist_3seed \
  --result-prefix d23_lean_opmnist_3seed \
  --note-path docs/research/step2_weakness_probe_d23_lean_opmnist_3seed.md

python "examples/The Alberta Plan/Step2/new_directions/d18_simple_universal_resource_basis.py" \
  --datasets controlled_nonlinear,synthetic_compositional,digits_mask_noise,digits_class_blocked \
  --steps 600 \
  --n-seeds 2 \
  --final-window 150 \
  --configs step2_canonical,step2_no_poly,step2_no_unified,basis_only \
  --output-dir outputs/step2_weakness_probe/d18_component_stress \
  --note-path docs/research/step2_weakness_probe_d18_component_stress.md
```

## UPGD Weaknesses

### 1. Rotating Subspace Is Not A Capacity Problem

The previously observed rotating-subspace failure gets worse when UPGD is made
wider:

| Variant | Diff vs best MLP | Wins/losses/ties |
|---|---:|---:|
| UPGD width 64 | `-0.169757` | `0/3/0` |
| UPGD width 128 | `-0.393721` | `0/3/0` |

Positive diff favors UPGD. Both wider variants lose every seed, so the failure
is not explained by too few hidden units. The missing ingredient is adaptation
geometry: the learner needs observable time/context, faster subspace tracking,
explicit recurrent/history features, or an update rule that can follow a moving
projection without treating the motion as ordinary feature utility noise.

Most promising directions:

- Add a small learned temporal/context basis to the UPGD input path and rerun
  rotating-subspace stress. This directly tests the theory assumption that time
  or context must be observable.
- Try a fast-slow UPGD split: fast output/readout adaptation over a slower
  perturbed trunk. The rotating task is a drift-rate mismatch.
- Try a covariance/natural-gradient or orthogonalized hidden update for the
  first layer. The relevant subspace rotates, so update geometry matters more
  than feature count.

### 2. Compact OPMNIST Retention Is Strong, But Online Tracking Accuracy Is Weak

The 3-seed D23 compact OPMNIST sweep confirms that lean UPGD beats the best fair
MLP on held-out retention metrics:

| Metric | Best UPGD vs best MLP | Wins/losses/ties |
|---|---:|---:|
| Final-window MSE | `+0.006103` | `3/0/0` |
| Final-window accuracy | `-0.020000` | `1/2/0` |
| Held-out test MSE | `+0.017187` | `3/0/0` |
| Held-out test accuracy | `+0.153833` | `3/0/0` |

This is a good production signal: lean UPGD can beat MLP on compact retained
permuted-pixel generalization while keeping the learner simple. But the negative
final-window accuracy means it is not yet uniformly better on classification
tracking. The weakness is the same tradeoff seen elsewhere: retained
representation versus immediate class tracking.

Most promising directions:

- Use CE/softmax readout as the primary OPMNIST UPGD path and track accuracy as
  the acceptance metric, not MSE alone.
- Add a tiny class-view memory head to UPGD rather than scaling width. D20
  results below show memory geometry is the missing part.
- Keep lean tracking: full utility/history tracking did not improve retained
  metrics enough to justify the extra state in this compact run.

### 3. Published-Scale OPMNIST Retention Is Still Open

The 800-task OPMNIST run is now complete for one latest-best seed. It is not a
clean closure because held-out all-permutation metrics still trail the best
fair MLP, even though UPGD-memory wins online MSE, online accuracy, and
final-window MSE. This is no longer a durable-run infrastructure problem; it is
a learner/objective boundary.

Most promising directions:

- Treat full 800-task OPMNIST as an external-scale confirmation queue, not a
  fast design loop.
- Run the accuracy-primary full command only after the learner path is fixed on
  compact accuracy/tracking tradeoffs.
- Keep checkpoint/status artifacts mandatory so negative partial evidence is
  visible.

## D20 Memory Direction

D20 is not a general Step 2 learner, but it is the clearest answer to the
OPMNIST retention geometry problem. The new budget sweep shows that even small
prototype budgets beat fair MLPs:

| Slots/class | Prototypes | Final MSE diff | Final accuracy diff | Test MSE diff | Test accuracy diff |
|---:|---:|---:|---:|---:|---:|
| 5 | `50.0` | `+0.022128` | `+0.145000` | `+0.039321` | `+0.186500` |
| 10 | `98.3` | `+0.036389` | `+0.225000` | `+0.062806` | `+0.328833` |
| 20, prior canonical | `178.3` | `+0.049980` | `+0.305000` | `+0.070844` | `+0.381000` |

The useful shape is monotone: more prototypes improve retained accuracy, but
even 5 slots/class is positive on all metrics. Ten slots/class looks like the
best near-term compromise: it recovers most of the 20-slot advantage with about
half the memory.

Most promising directions:

- Port D20 to a fixed-shape JAX learner and make it a memory component, not a
  separate NumPy runner.
- Attach D20 memory to UPGD as an auxiliary readout only when online uncertainty
  or persistence says retained class views matter.
- Learn the slot budget or novelty threshold from utility/uncertainty rather
  than hand setting `slots_per_class`.

## D18 Weaknesses

### 1. D18 Is Empirically Strong But Architecturally Heavy

The all-14 evidence says D18 is the strongest current empirical Step 2 learner,
but it is not production-grade in its current form:

- It is a Python/NumPy research runner, not a fused core JAX learner.
- The prior assessment measured a mean runtime ratio around `7.64x` versus best
  MLP on the all-14 run.
- The mechanism has several static choices: bank families, budgets, random tanh
  width, polynomial degree cap, Fourier frequencies, gain anchors, clipping,
  readout decays, and persistence threshold.

The engineering direction is straightforward: do not keep expanding D18. Distill
it.

### 2. D18’s Class-Blocked Retention Depends On The Canonical Memory Path

The component stress test shows class-blocked held-out accuracy collapses when
canonical D18 memory/retention machinery is removed:

| Method | Final MSE | Held-out test accuracy |
|---|---:|---:|
| `d18_step2_canonical` | `0.0027` | `0.4249` |
| `d18_step2_no_poly` | `0.0038` | `0.1197` |
| `d18_step2_no_unified` | `0.0038` | `0.1113` |
| `d18_basis_only` | `0.0035` | `0.1002` |
| best MLP | `0.0041` | `0.1271` |

The online final-window MSE still looks fine without pieces of D18, but retained
test accuracy does not. This is the same retention/tracking split seen in UPGD.

Most promising directions:

- Separate online tracking and deployment/retention heads explicitly in a core
  learner.
- Replace the current class-blocked retention path with a D20-style
  multi-prototype memory; D20 gives a simpler geometry than D18's current
  mixture of target trace plus prototype retention.
- Learn the persistence gate and memory timescale as resource-manager actions.

### 3. D18 Mask-Noise Held-Out Accuracy Is Still Fragile In Short Runs

On the short two-seed stress test, best-D18 final-window MSE beats best MLP, but
held-out test accuracy is slightly negative:

| Metric | Best D18 vs best MLP |
|---|---:|
| Final-window MSE | `+0.0029 +/- 0.0011`, `2/0/0` |
| Held-out accuracy | `-0.0306 +/- 0.0380`, `1/1/0` |

This is not enough to overturn the 10-seed all-14 result, but it identifies the
fragile row. Mask noise stresses whether the retained memory and basis readouts
are robust when many input features are intermittently irrelevant.

Most promising directions:

- Add online feature dropout/feature reliability estimates before D18's memory
  and basis blocks.
- Make prototype and target-trace updates confidence-weighted under mask noise.
- Test whether D20-style memory should operate on denoised/UPGD hidden features
  instead of raw pixels.

### 4. D18 OPMNIST Is Not Competitive With UPGD/D20

Existing efficiency evidence says D18 has poor OPMNIST deployment:

- D18 bridge held-out accuracy: `0.475667`.
- Lean UPGD held-out accuracy: around `0.70` on compact OPMNIST.
- D20 held-out accuracy: `0.907333` in the 20-slot canonical compact run.
- D18 was about `59x` slower than `upgd_h64_softmax_ce` in the recorded compact
  OPMNIST comparison.

D18 should not be the OPMNIST production path. Its useful ideas are the learned
resource allocation and target-persistence split, not the full architecture.

## Portfolio/Router Line

The older universal portfolio remains important evidence, but it is not a
candidate for the desired simple learner claim. Its weaknesses are conceptual:

- It wins by routing among learners rather than discovering one representation.
- The class-imbalance retention guard is hand-specified.
- It is heavier than either UPGD or D20 memory.
- It does not produce a theorem of universal feature construction.

The right use of the portfolio line is distillation: every place it routes
between MLP, UPGD, dynamic sparse, and retention should become a learned
internal resource allocation or memory-timescale decision inside one learner.

## Direction Ranking

### Highest Value

1. **UPGD + D20-style memory head**. The evidence is strongest: UPGD is simple
   and good on compact OPMNIST; D20 solves the retained class-view geometry.
   Start with 10 slots/class, fixed budget, JAX scan, and a learned novelty or
   uncertainty gate.
2. **Temporal/context features for rotating subspace**. Wider UPGD fails, so
   the next experiment must change observability or adaptation geometry.
3. **D18 distillation into learned modules**. Keep the pieces that are validated
   by ablation: learned additive gains, target persistence split, and memory;
   avoid preserving the full heavy RKHS/poly/unified stack as production code.

### Medium Value

4. **Learn persistence thresholds and memory timescales**. Static thresholds are
   working but philosophically weak and likely brittle under label drift/mask
   noise.
5. **Feature reliability under mask noise**. This directly targets the weakest
   short-run D18 row.
6. **Accuracy-primary UPGD OPMNIST runs**. MSE can look good while final-window
   accuracy is negative; promotion should be accuracy-first for classification.

### Lower Value

7. **More UPGD width**. The rotating-subspace sweep falsifies raw capacity as a
   fix.
8. **Full D18 OPMNIST scaling**. Current evidence says it is slow and worse than
   UPGD/D20.
9. **More hand-built portfolio guards**. Useful as diagnostic probes, but
   contrary to the simple-learner goal.

## Proposed Next Experiments

1. Implement a fixed-shape JAX D20 memory component with `slots_per_class=10`.
   Compare raw-input memory, UPGD-hidden memory, and hybrid logits on compact
   OPMNIST.
2. Add time/context features to `step2_theory_falsification.py` rotating
   subspace and test whether UPGD recovers when the hidden assumption is made
   true.
3. Add a fast-slow UPGD variant: slow trunk, fast readout, optional low-rank
   first-layer rotation adapter.
4. Run D18 component ablations at 10 seeds only for `digits_mask_noise` and
   `digits_class_blocked`, because those rows expose the retention/calibration
   tradeoff.
5. Move D18's target-persistence threshold into a learned two-action manager:
   retain-memory versus current-tracking mode.
6. If the UPGD+D20 hybrid is positive on compact OPMNIST and rotating-context
   probes, run the full 800-task OPMNIST checkpoint command with accuracy as the
   primary metric.
