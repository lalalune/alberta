# Step 2 UPGD, D18, D20 Efficiency Decision

Date: 2026-05-06.

## Question

Assess computational efficiency and empirical value of fair MLP, D18, D20, and
UPGD on the current OPMNIST Step 2 benchmark. Decide whether D18 or D20 should
be dropped in favor of UPGD.

## Evidence Artifacts

- D21 same-protocol UPGD run:
  `outputs/step2_new_directions/d21_upgd_opmnist_efficiency_local3/d21_upgd_opmnist_efficiency_local3_results.json`
- D21 note:
  `docs/research/step2_new_directions/d21_upgd_opmnist_efficiency_local3.md`
- Raw UPGD/MLP throughput run:
  `outputs/step2_new_directions/upgd_vs_mlp_throughput_784/efficiency_results.json`
- D18 OPMNIST bridge:
  `outputs/step2_new_directions/d18_opmnist_bridge_softmax_local3/d18_opmnist_bridge_softmax_local3_results.json`
- D20 local OPMNIST:
  `outputs/step2_new_directions/d20_multiprototype_opmnist_local3/d20_multiprototype_opmnist_local3_results.json`
- D22 JAX prototype memory / UPGD+memory hybrid:
  `outputs/step2_new_directions/d22_upgd_prototype_hybrid_local3/d22_upgd_prototype_hybrid_local3_results.json`
- D23 UPGD lean compute-knob sweep:
  `outputs/step2_new_directions/d23_upgd_lean_sweep_local3/d23_upgd_lean_sweep_local3_results.json`

All OPMNIST results below use 3 paired seeds, 1000 online steps, final window
200, 5 permutation tasks, sklearn digits resized to 28x28, except the noted
OpenML smoke result.

## OPMNIST Accuracy And Wall-Clock

| Method | Final MSE | Final Acc | Test MSE | Test Acc | Runtime s | Steps/s | State / capacity |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Best fair MLP by held-out accuracy (`mlp_h128`, D21) | 0.081985 | 0.483333 | 0.080672 | 0.534333 | 1.635565 | 713.5 | 101770 trainable params class |
| Best UPGD by held-out accuracy (`upgd_h64_softmax_ce`) | 0.072523 | 0.510000 | 0.062853 | 0.708500 | 1.061659 | 942.5 | 50890 params, 101084 float state |
| Best UPGD by held-out MSE (`upgd_h64_linear_mse`) | 0.074260 | 0.483333 | 0.062275 | 0.653000 | 0.879462 | 1200.9 | 50890 params, 101084 float state |
| JAX prototype memory (`proto_s20`, D22) | 0.026361 | 0.825000 | 0.013433 | 0.907333 | 0.618043 | 1640.5 | 157000 float prototype state |
| UPGD+memory, best test MSE (`hybrid_h64_s20_advantage`, D22) | 0.026292 | 0.825000 | 0.013407 | 0.907333 | 1.528355 | 655.1 | 258084 float state |
| UPGD+memory, best test accuracy (`hybrid_h64_s20_uncertain`, D22) | 0.026303 | 0.826667 | 0.014230 | 0.910500 | 1.553537 | 645.7 | 258084 float state |
| D20 multi-prototype | 0.026361 | 0.825000 | 0.013433 | 0.907333 | 5.057055 | 197.7 | about 157000 float prototype slots, 178 active prototypes |
| D18 canonical bridge | 0.069065 | 0.580000 | 0.085167 deployment MSE | 0.475667 | 62.901070 | 15.9 | large multi-bank static basis, Python-heavy runner |

Interpretation:

- UPGD is objectively better than the fair MLP on the compact OPMNIST criterion:
  D21 wins over the per-seed best MLP on final-window MSE, held-out test MSE,
  and held-out test accuracy in all 3 seeds. Final-window accuracy is close:
  2 UPGD wins, 1 MLP win.
- UPGD is far better than D18 for OPMNIST deployment. D18 has slightly stronger
  online final-window MSE/accuracy than UPGD, but its held-out retention is worse
  than UPGD and it is about 59x slower than `upgd_h64_softmax_ce` in these
  recorded runners.
- D20 is not beaten by UPGD. D20 has much stronger OPMNIST retention:
  +0.198833 held-out accuracy versus `upgd_h64_softmax_ce`, and 0.013433 test
  MSE versus 0.062853. It costs about 4.8x more wall-clock than that UPGD run,
  but the accuracy gap is too large to drop D20.
- The JAX prototype memory removes the old D20 implementation caveat. `proto_s20`
  exactly preserves the D20 metric profile on the compact OPMNIST protocol while
  reducing runtime from 5.057s to 0.618s. In this runner it is faster than the
  standalone UPGD path and far faster than D18.
- The UPGD+memory hybrids work, but the retained OPMNIST advantage comes from
  the memory head. The best hybrid test accuracy is 0.910500, only slightly
  above `proto_s20` at 0.907333, while costing about 2.5x the prototype-only
  runtime. Use the hybrid when a differentiable UPGD path must coexist with
  memory; use `proto_s20` when this benchmark is purely retained class-view
  recognition.

## Raw Update Throughput

This run excludes compilation after one warmup pass and uses random one-hot
targets with 784 input dimensions and 10 heads.

| Method | Hidden | Steps/s | Relative to same-width MLP | Trainable params | Float state |
| --- | ---: | ---: | ---: | ---: | ---: |
| `mlp16` | 16 | 12158.1 | 1.00x | 12730 | 25498 |
| `upgd16_structure_rademacher_lean_interval16` | 16 | 5105.7 | 0.42x | 12730 | 25292 |
| `upgd16_structure_simple` | 16 | 1024.2 | 0.08x | 12730 | 38071 |
| `mlp32` | 32 | 7607.3 | 1.00x | 25450 | 50954 |
| `upgd32_structure_rademacher_lean_interval16` | 32 | 3016.4 | 0.40x | 25450 | 50556 |
| `upgd32_structure_simple` | 32 | 550.4 | 0.07x | 25450 | 76103 |
| `mlp64` | 64 | 4685.4 | 1.00x | 50890 | 101866 |
| `upgd64_structure_sigma0` | 64 | 1721.2 | 0.37x | 50890 | 152167 |
| `upgd64_structure_simple_interval16` | 64 | 1306.8 | 0.28x | 50890 | 152167 |
| `upgd64_structure_lean_interval16` | 64 | 1163.7 | 0.25x | 50890 | 101084 |
| `upgd64_structure_simple` | 64 | 354.2 | 0.08x | 50890 | 152167 |
| `upgd64_structure_meta` | 64 | 214.1 | 0.05x | 50890 | 152167 |

This means UPGD is not intrinsically cheaper than a same-width MLP update.
Its advantage is performance per useful retained representation, not raw
floating-point throughput. The knobs that matter computationally are clear:

- Full per-step perturbation and utility tracking are expensive.
- Lean tracking and less frequent perturbations recover a large fraction of MLP
  throughput.
- Same-width UPGD has the same trainable parameter count as MLP but generally
  more state when full utility/history tracking is enabled.

## Complexity Accounting

For input dimension `D = 784`, classes `C = 10`, hidden units `H`, and D20
prototype slots per class `S = 20`:

- One-hidden-layer MLP/UPGD trainable parameters:
  `H * D + H + H * C + C`.
  For `H = 64`, this is `50890`.
- UPGD uses the same trainable parameter count as the comparable MLP, but the
  learner state also stores utility, age, perturbation, and tracking arrays.
  In D21 `upgd_h64_softmax_ce` stores `101084` floating scalar state slots.
- D20 stores `C * S * D = 156800` prototype mean floats plus `C * S = 200`
  count floats and `200` integer timestamps. Prediction computes distances to
  up to `C * S = 200` prototypes, so the dominant prediction cost is
  `O(C * S * D)`. With the current settings, that is about `156800` coordinate
  comparisons per prediction before softmax.

## Decision

Drop D18 as a production Step 2 contender for OPMNIST and for the main
efficient learner path. Keep it only as a research artifact for the all-14
synthetic portfolio history and for ideas that may be distilled later. The
current D18 implementation is too slow and loses the key held-out OPMNIST
retention comparison.

Do not drop D20's mechanism. Replace the old Python D20 runner with the JAX
`PrototypeMemoryLearner` for production-facing Step 2 memory experiments. UPGD
does not match prototype memory on OPMNIST retention, and the JAX memory is no
longer computationally disqualified.

Make UPGD the default Step 2 differentiable learner candidate. The D21 result is
enough to promote UPGD over fair MLP on the compact OPMNIST criterion, and it is
far simpler than D18. The efficiency target should now be a lean UPGD variant
with infrequent perturbations and no unnecessary tracking, plus a D20-style
memory head that is budgeted, learned, and eventually fused in JAX.

## D23 Lean UPGD Sweep

The D23 sweep tested width, perturbation interval, perturbation noise, sigma-zero
removal, and full tracking on the same 3-seed compact OPMNIST protocol.

Main findings:

- Full tracking is pruned. `upgd_h64_rademacher_i16_fulltrack` matches the lean
  variant's metrics but increases float state from 101084 to 152167 and is
  slower in this runner.
- Interval 4 and interval 32 do not materially improve accuracy over interval
  16. Keep interval 16 as the balanced default.
- Normal noise gives a tiny held-out accuracy gain in D23
  (`0.702667` vs `0.696833` for rademacher interval 16) but is not a decisive
  enough improvement to replace bounded Rademacher noise in the default.
- Sigma-zero is surprisingly competitive on OPMNIST (`0.701667` test accuracy)
  and fastest among the D23 h64 rows. This benchmark alone does not require
  perturbation, but synthetic feature-discovery evidence still justifies keeping
  low-noise perturbation in the universal Step 2 default.
- Width 32 remains the better compact fallback when state is constrained:
  `50556` float state and `0.687000` held-out accuracy, versus h64's `101084`
  float state and roughly `0.697-0.703` accuracy.

## Completed Work

- Port D20-style prototype memory to a JAX learner with fixed budgets and JIT
  scan support so wall-clock comparisons are implementation-fair. Done in
  `src/alberta_framework/core/prototype_memory.py`.
- Test UPGD plus a small D20-style memory head, using the D20 memory only where
  UPGD prediction uncertainty says retention support is needed. Done in D22.
- Sweep lean UPGD perturbation interval, noise family, and tracking removal on
  OPMNIST. Done in D23.
- Replace D18 from future production summaries with UPGD/D20 unless the question
  is explicitly about the older synthetic all-14 result. Done for the current
  Step 2 decision and current-best summaries.

Residual research, not an immediate blocker:

- Learn the prototype-memory budget, novelty threshold, and bandwidth instead of
  fixing them.
- Fuse the hybrid path more tightly if future tasks require both differentiable
  hidden-feature learning and retained view memory in one deployed learner.
