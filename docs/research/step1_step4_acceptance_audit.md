# Step 1-4 Acceptance Audit

Date: 2026-05-07

This audit separates the local package acceptance boundary from the broader
Alberta Plan research boundary.

## Literature Alignment

- Alberta Plan: Step 1 is fixed-feature continual supervised learning; Step 2
  introduces supervised feature finding; Step 3 moves to GVF prediction in
  sequential streams; Step 4 moves to continual control with the critic
  presumably built from Steps 1-3. Source:
  <https://arxiv.org/abs/2208.11173>.
- Step 1 mechanisms are known algorithms: LMS, IDBD, Autostep, online
  normalization, and TD extensions. IDBD traces to Sutton 1992
  (<https://m.aaai.org/Library/AAAI/1992/aaai92-027.php>); Autostep traces to
  Mahmood, Sutton, Degris, and Pilarski 2012
  (<https://armahmood.github.io/files/MSDP-Autostep-ICASSP-2012.pdf>).
- Step 2 nonlinear streaming ingredients align with the stream-x/ObGD family
  (<https://arxiv.org/abs/2410.14606>), UPGD
  (<https://arxiv.org/abs/2302.03281>), and the continual-backprop/loss-of-
  plasticity line (<https://www.nature.com/articles/s41586-024-07711-7>).
  The local target-structure loss, strict two-timescale digit readout, and
  package-level modular composition are local experimental extensions.
- Step 3 aligns with GVF/Horde and true-online TD foundations. Horde is the
  right reference for many parallel prediction demons
  (<https://josephmodayil.com/papers/horde-final.pdf>); true-online TD(lambda)
  is the reference for exact online forward-view equivalence in the linear TD
  setting (<https://proceedings.mlr.press/v32/seijen14.html>).
- Step 4 local SARSA is standard on-policy TD control; the Horde actor-critic
  path is a framework adapter, not yet a promoted research result.

## Local Acceptance Status

| Step | Local package status | Remaining research boundary |
|---|---|---|
| Step 1 | Accepted for public fixed-feature continual supervised learners and smoke/evidence gates. | Claim wording must stay scoped: IDBD wins Sutton-style rows; Autostep is the robustness default; Adam/RMSprop can win some broader rows. |
| Step 2 | Accepted for the current supervised UPGD/UPGD-family empirical matrix. Production exposes broad target-structure UPGD, strict digit/readout UPGD, retained-view memory, associative sequence memory, and temporal context helpers. The OPMNIST task-count protocol is now complete for one 800-task/48M-update OpenML MNIST seed. | No theorem of universal representation learning. TD/GVF feature construction, hidden-state discovery, multi-seed OPMNIST performance closure, and the held-out all-permutation retention gap remain separate boundaries. |
| Step 3 | Accepted for given-feature GVF/Horde-style prediction with shared, independent, and mixed routings plus linear off-policy TD plumbing. | Full callable GVF question functions, nonlinear off-policy Horde/GQ/GTD, and robust learned GVF feature discovery remain open. |
| Step 4 | SARSA accepted as the canonical local Step 4a control path. Horde actor-critic is implemented, tested, and pipeline-wired. | Horde actor-critic is not canonical until it beats Q/SARSA on predefined control evidence; average-reward control is Step 5/6. |

## Pipeline Fixes Landed

- `Step2UPGDConfig` now forwards `sparsity`, `use_layer_norm`,
  `loss_normalization`, and `readout_mode` instead of silently dropping them.
- The strict digit/readout preset is exposed through `Step2UPGDConfig` and the
  production Step 2 facade.
- SARSA now reports TD error against `Q(s_t, a_t)` rather than `Q(s_{t+1},
  a_t)`.
- SARSA pipeline control mirrors Step 3 GVFs as prediction demons inside the
  SARSA Horde, giving a real Step 3-to-Step 4 knowledge bridge.
- Horde actor-critic pipeline mode now keeps a single synchronized critic state
  and skips the redundant discarded Step 3 update.
- Linear multi-head diagnostics now use head traces and head optimizer states,
  so the default linear Horde path reports nonzero feature activity after
  updates.
- Documented CLI commands are now declared package scripts.

## Acceptance Commands

```bash
source .venv/bin/activate
pytest tests/ -q
ruff check .
mypy
```

Historical full-suite local result:

- `pytest tests/ -q`: 1360 passed, 36 third-party deprecation warnings.
- `ruff check .`: passed.
- `mypy`: passed.

Focused verification from this audit turn:

```bash
.venv/bin/python -m pytest \
  tests/test_normalizers.py \
  tests/test_alberta_plan_step1_streams.py \
  tests/test_step1_replication.py \
  tests/test_step2_completion_criteria.py \
  tests/test_step2_canonical.py \
  tests/test_associative_memory.py \
  tests/test_prototype_memory.py \
  tests/test_upgd.py \
  tests/test_upgd_memory.py \
  tests/test_step2_cifar_stream.py \
  tests/test_step2_upgd_memory_opmnist.py \
  tests/test_step3_production.py \
  tests/test_horde.py \
  tests/test_independent_demon_horde.py \
  tests/test_mixed_horde.py \
  tests/test_sarsa.py \
  tests/test_actor_critic.py \
  tests/test_horde_actor_critic.py \
  tests/test_pipeline.py -q
```

Result: `333 passed in 234.72s`.

Additional artifact-gate verification after the published-scale OPMNIST
completion:

```bash
pytest tests/test_step2_completion_criteria.py tests/test_step2_opmnist_protocol.py -q
```

Result: `11 passed in 0.55s`.

Additional focused Step 1/2/3/security verification:

```bash
pytest \
  tests/test_step1_replication.py \
  tests/test_step2_cifar_stream.py \
  tests/test_step2_upgd_memory_opmnist.py \
  tests/test_step3_production.py \
  tests/test_throughput_smoke.py \
  tests/test_security_integration.py \
  tests/test_production_steps.py -q
```

Result: `76 passed in 37.33s`.

Security-gym sibling contract verification:

```bash
pytest \
  ../security-gym/tests/test_env.py \
  ../security-gym/tests/test_env_hybrid.py \
  ../security-gym/tests/test_scan_stream.py -q
```

Result: `80 passed, 1 skipped in 0.49s`.

Scoped static checks from this audit turn:

- `ruff check src/alberta_framework/steps/step1.py src/alberta_framework/steps/step2.py src/alberta_framework/steps/step3.py src/alberta_framework/steps/step4.py src/alberta_framework/pipeline.py tests/test_pipeline.py tests/test_step3_production.py tests/test_step2_cifar_stream.py tests/test_step2_upgd_memory_opmnist.py`: passed.
- `mypy --no-incremental src/alberta_framework/steps src/alberta_framework/pipeline.py`: passed.

Public facade smoke matrix from this audit turn:

| Surface | Result |
|---|---|
| Step 1 | finite, metrics shape `(64, 4)`, final-window MSE `6.2857` |
| Step 2 UPGD | finite, metrics shape `(64, 4)`, final-window MSE `0.3691` |
| Step 2 associative | finite, metrics shape `(64, 8)`, NLL `1.9822 -> 0.0000045` |
| Step 3 Horde | finite, per-demon metrics shape `(64, 3, 3)`, TD errors shape `(64, 3)` |
| Step 4 SARSA | finite, Q shape `(16, 2)`, TD error shape `(16,)`, action shape `(16,)` |
| Step 1-4 pipeline | finite, feature shape `(16, 16)`, Horde prediction shape `(16, 3)`, Q shape `(16, 2)` |

Step 2 image-stream evidence added after the original audit:

- CIFAR-10 focused real-data confirmation:
  `docs/research/step2_image_block_promotion_cifar_hybrid_focused_5seed_2000.md`.
  `step2_hybrid_memory_trace_adaptive_sharp` clears iid and class-blocked
  CIFAR against the expanded MLP comparator set at 5/5 paired seeds for
  final-window MSE and held-out accuracy in the class-blocked regime, and
  behaves like the raw hybrid winner on iid.
- OPMNIST published-scale protocol confirmation:
  `outputs/step2_canonical/upgd_memory_opmnist_latest_best_800block_1seed_results.json`
  and
  `outputs/step2_canonical/upgd_memory_opmnist_latest_best_800block_1seed_SUMMARY.md`.
  This closes the task-count/update-budget gate for one seed: true OpenML
  MNIST, canonical 60,000/10,000 split, 800 random pixel permutations,
  60,000-example task blocks, 48,000,000 online updates, no task id, and all
  800 held-out permutation views. It is not an unqualified performance win:
  UPGD-memory wins online MSE, online accuracy, and final-window MSE, while
  fair MLP comparators still win final-window accuracy and held-out
  all-permutation test MSE/accuracy.
- Single-UPGD H128 full-scale follow-up:
  `outputs/step2_canonical/upgd_memory_opmnist_single_upgd_h128_800block_1seed_results.json`
  and
  `outputs/step2_canonical/upgd_memory_opmnist_single_upgd_h128_800block_1seed_SUMMARY.md`.
  The softmax-H128 UPGD candidate also completes the full 800-task protocol.
  Versus `mlp_h128`, it wins online MSE, online accuracy, final-window MSE,
  and all-permutation held-out test accuracy, while losing final-window
  accuracy and held-out test MSE.

## Not Accepted Yet

- A theorem proving universal recursive representation learning.
- Integrated nonlinear off-policy Horde with GQ/GTD/TDC-style guarantees.
- Average-reward GVF/control.
- Horde actor-critic promotion over Q/SARSA.
- Multi-seed OPMNIST performance closure and cross-entropy parity.
- External rlsecd/chronos active-defense closure; `security-gym` is present and
  tested, but `rlsecd` and `chronos-sec` remain external.
