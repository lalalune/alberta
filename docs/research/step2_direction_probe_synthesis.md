# Step 2 Direction Probe Synthesis

This note records the implementation and evidence pass requested after the
current-best weakness audit.  The goal was not to add another portfolio; it was
to test simple mechanisms that could become one learner:

1. UPGD plus fixed-shape D20-style memory.
2. Temporal/context features for rotating-subspace failures.
3. Fast/slow UPGD timescale controls.
4. A distilled D18 with fewer static components.
5. Online memory novelty/persistence adaptation.
6. Reliability/confidence handling for noisy classification streams.
7. Accuracy-primary compact OPMNIST once the UPGD-memory path was positive.

## Implemented

- Added `UPGDMemoryLearner` in `src/alberta_framework/core/upgd_memory.py`.
  It keeps one UPGD component and one fixed-budget prototype memory component.
  Both update every time step.  A learned scalar blend logit is updated online
  by the blended prediction loss, and confidence/reliability terms modulate the
  blend without route selection.
- Added runtime novelty threshold support to `PrototypeMemoryLearner` via
  `update_with_novelty_threshold`.
- Added `TemporalContextFeaturizer` in
  `src/alberta_framework/core/temporal_context.py`.  It exposes raw input,
  EMA context, innovation, sinusoidal phase, and optional phase-gated input
  products.
- Added packaged Step 2 hybrid factory `make_step2_hybrid_learner`.
- Added D18 distillation configs:
  `step2_distilled_memory` and `step2_distilled_memory_nogates`.
- Updated the D22 OPMNIST runner to evaluate the packaged UPGD-memory learner,
  not only the older experiment-local hybrid.

## Rotating Relevant Subspace

Command:

```bash
python "examples/The Alberta Plan/Step2/step2_theory_falsification.py" \
  --scenarios rotating_relevant_subspace \
  --steps 600 \
  --n-seeds 3 \
  --methods upgd,upgd_context,upgd_fastslow,mlp,mlp_deep \
  --upgd-width 64 \
  --mlp-width 64 \
  --output-dir outputs/step2_direction_probe/rotation_phase_context_fastslow \
  --note-path docs/research/step2_direction_probe_rotation_phase_context_fastslow.md
```

Final-window MSE by seed:

| Method | Seed 0 | Seed 1 | Seed 2 |
|---|---:|---:|---:|
| UPGD | 0.618420 | 0.582346 | 0.506219 |
| UPGD + phase context | 0.422332 | 0.474122 | 0.512810 |
| UPGD fast/slow | 0.754053 | 0.657167 | 0.694936 |
| MLP h64 | 0.513547 | 0.504466 | 0.502921 |
| MLP h64/h64 | 0.403138 | 0.399738 | 0.394839 |

Interpretation:

- Plain width was the wrong fix.  UPGD h128 without context had already made
  this failure worse.
- Phase-gated context is a real improvement: it beats plain UPGD on 2/3 seeds
  and beats the shallow MLP on 2/3 seeds.
- It still does not beat the deep MLP.  The remaining gap is not solved by this
  simple context wrapper.
- The current fast/slow UPGD control is a reject for this stressor; it was worse
  than plain UPGD on all three seeds.

Follow-up command:

```bash
python "examples/The Alberta Plan/Step2/step2_theory_falsification.py" \
  --scenarios rotating_relevant_subspace \
  --steps 1000 \
  --n-seeds 10 \
  --methods upgd,upgd_context_dense,upgd_context_phase_only,mlp,mlp_deep \
  --upgd-width 64 \
  --mlp-width 64 \
  --output-dir outputs/step2_direction_probe/rotation_dense_context_10seed_1000 \
  --note-path docs/research/step2_direction_probe_rotation_dense_context_10seed_1000.md
```

The phase-only dense context variant closes this stressor:

| Method | Final-Window MSE |
|---|---:|
| UPGD | 0.583041 +/- 0.019532 |
| MLP h64 | 0.522663 +/- 0.012168 |
| MLP h64/h64 | 0.390348 +/- 0.008226 |
| UPGD + dense context | 0.248597 +/- 0.008516 |
| UPGD + dense phase-only context | 0.241974 +/- 0.006165 |

Dense phase-only context beats the deep MLP on all 10 paired seeds.  The paired
MSE improvement versus deep MLP is +0.148374 +/- 0.010516.  This is now
productionized as `make_step2_temporal_context()` plus
`make_step2_temporal_learner()`.

## Compact OPMNIST, Accuracy Primary

Command:

```bash
python "examples/The Alberta Plan/Step2/new_directions/d22_upgd_prototype_hybrid_opmnist.py" \
  --steps 1000 \
  --n-seeds 3 \
  --final-window 200 \
  --mnist-source sklearn_digits_28x28 \
  --n-permutations 5 \
  --task-block-size 200 \
  --task-sampling sequential_epoch \
  --output-dir outputs/step2_direction_probe/d22_core_upgdmem_alloc18 \
  --result-prefix d22_core_upgdmem_alloc18 \
  --note-path docs/research/step2_direction_probe_d22_core_upgdmem_alloc18.md
```

Key rows:

| Method | Final Acc | Test Acc | Final MSE | Test MSE | Active Prototypes | Eval Gate |
|---|---:|---:|---:|---:|---:|---:|
| Best fair MLP (`mlp_h128`) | 0.545000 | 0.542667 | 0.079304 | 0.085347 |  |  |
| UPGD h64 softmax | 0.465000 | 0.722500 | 0.073840 | 0.062649 |  |  |
| D20 prototype s20 | 0.825000 | 0.907333 | 0.026361 | 0.013433 | 178.33 |  |
| Packaged UPGD-memory s20 alloc18 | 0.821667 | 0.911333 | 0.029733 | 0.018325 | 171.00 | 0.663 |
| Packaged UPGD-memory s20 alloc18 mem0 | 0.818333 | 0.910333 | 0.026871 | 0.014452 | 171.00 | 0.841 |

Interpretation:

- The packaged UPGD-memory learner decisively beats the best fair MLP on both
  final-window accuracy and held-out accuracy.
- The adaptive novelty target was the important knob.  The earlier 0.02 target
  under-allocated memory, leaving only about 73-82 active prototypes.  The 0.18
  target produced about 171 active prototypes, close to D20's 178.
- The promoted packaged default is now the s20/alloc18 setting.  It slightly
  beats D20 on held-out accuracy in this compact run, while retaining a
  differentiable UPGD path.
- This is still compact OPMNIST evidence, not the full 800-task result.

## D18 Distillation

Command:

```bash
python "examples/The Alberta Plan/Step2/new_directions/d18_simple_universal_resource_basis.py" \
  --datasets controlled_nonlinear,synthetic_compositional,digits_mask_noise,digits_class_blocked \
  --steps 600 \
  --n-seeds 2 \
  --final-window 150 \
  --configs step2_canonical,step2_distilled_memory,step2_distilled_memory_nogates,step2_no_poly \
  --output-dir outputs/step2_direction_probe/d18_distilled \
  --note-path docs/research/step2_direction_probe_d18_distilled.md
```

Main outcomes:

- `step2_distilled_memory` nearly matches canonical on controlled nonlinear and
  class-blocked retention while removing the polynomial block.
- `step2_distilled_memory_nogates` improves the short mask-noise row in this
  2-seed probe: final MSE 0.0520 and test accuracy 0.7894 versus best MLP test
  accuracy 0.7421.
- Removing persistence gates is not broadly safe: the no-gates variant collapses
  on controlled nonlinear and synthetic compositional.
- The D18 lesson is therefore selective.  The polynomial block is removable in
  some regimes, but the static persistence gates cannot be globally removed yet.

## Current Decisions

- Promote the packaged UPGD-memory default with 20 prototype slots per class,
  `target_allocation_rate=0.18`, `initial_memory_logit=0.0`, and the
  update-time target-trace prior
  `target_trace_blend_scale=0.8, target_trace_pressure_threshold=0.5`.
- Promote dense phase-only temporal context for observable rotating-subspace
  stressors.  It beats the deep MLP in the 10-seed stress probe.
- Reject the current fast/slow UPGD setting for rotating subspace.
- Keep D18 distillation as an analysis tool, not the production path.  Its
  no-polynomial memory variant is promising; no-gates is not robust.

## D26 Digit Stress for Packaged UPGD-Memory

Command:

```bash
python "examples/The Alberta Plan/Step2/new_directions/d26_upgd_memory_digit_stress.py" \
  --datasets digits_mask_noise,digits_class_blocked \
  --steps 600 \
  --n-seeds 5 \
  --final-window 150 \
  --variant-set trace \
  --output-dir outputs/step2_direction_probe/d26_upgd_memory_trace_updateonly \
  --result-prefix d26_upgd_memory_trace_updateonly \
  --note-path docs/research/step2_direction_probe_d26_upgd_memory_trace_updateonly.md
```

Mask-noise results:

| Method | Final MSE | Final Acc | Test MSE | Test Acc |
|---|---:|---:|---:|---:|
| Best fair MLP | 0.057344 | 0.726667 | 0.051908 | 0.764750 |
| UPGD-memory s20 alloc18 | 0.025773 | 0.825333 | 0.023315 | 0.848609 |
| Promoted trace hybrid | 0.026678 | 0.808000 | 0.023242 | 0.840074 |

Class-blocked results:

| Method | Final MSE | Final Acc | Test MSE | Test Acc |
|---|---:|---:|---:|---:|
| Best fair MLP | 0.004264 | 0.988000 | 0.137827 | 0.129128 |
| UPGD-memory s20 alloc18 | 0.012261 | 0.912000 | 0.085987 | 0.431540 |
| UPGD-memory s20 alloc18 mem0 | 0.009233 | 0.946667 | 0.089056 | 0.486456 |
| Promoted trace hybrid | 0.001810 | 0.993333 | 0.088671 | 0.486085 |

Interpretation:

- UPGD-memory wins every mask-noise metric over best fair MLP across 5 seeds.
- On class-blocked streams, plain memory-retention variants still lose online
  final-window tracking to the best MLP.  The update-time target-trace prior
  closes that specific persistence gap while ordinary held-out `predict()`
  remains observation-based.
- The single promoted trace hybrid beats the best same-run fair MLP on all four
  tracked metrics for both digit stressors in this 5-seed D26 run.  It is not
  always the best UPGD-memory variant within each stressor, but it is the
  simplest one-learner setting that clears both rows.

## Remaining Gaps

- The packaged UPGD-memory learner now has true OpenML MNIST OPMNIST evidence
  through 10 full published-size 60,000-example blocks and real CIFAR-10
  evidence.  See `step2_upgd_memory_image_scale_assessment.md`.  Full
  800-task OPMNIST completion is still open: the durable single-learner run is
  10/800 blocks, not 800/800.
- Real CIFAR-10 is positive on held-out metrics and IID online metrics, but
  class-blocked final-window MSE still favors the best fair MLP while
  final-window accuracy and held-out retention favor UPGD-memory.
- Rotating-subspace stress is closed for observable temporal drift with dense
  phase-only context, but this assumes time/context features are admissible.
- D18 static gate removal is not solved.
- The current fast/slow UPGD control needs a new design or should be dropped
  from the candidate set.
