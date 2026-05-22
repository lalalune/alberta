# Step 4 Completion Plan

## Status

Step 4 local framework scope is complete for the current Alberta Framework
repository.

The completed scope includes:

- `SARSAAgent` as the on-policy TD-control baseline over Horde action heads.
- `ActorCriticAgent` as the narrow discrete softmax actor with a linear value
  critic, immutable JAX state, traces, scan-array loop, config roundtrip, and
  ObGD-compatible bounder hook.
- `HordeActorCriticAgent` as the Step 1-3 critic path: the first Horde head
  supplies the actor-critic TD error/advantage, auxiliary prediction demons
  update on the same transition, per-head trace decay is supported, optional
  history-feature/CBP variants are wired, explicit transition discounts are
  accepted, and actor bounding is available.
- Bounded 10-seed bsuite reports for SARSA-vs-Q, Step 4
  Q/SARSA/actor-critic, and primary-family SARSA-vs-Q probes.
- The 10-seed Step 3 DoD-9 critic-for-control capstone.
- An explicit external-blocker record for unavailable `rlsecd`/`chronos-sec`
  daemon integration.

This document is now a completion record and command index, not an open work
plan. The bounded bsuite reports are intentionally narrower than a full bsuite
family expansion: they use selected `bsuite_id` values and 2000 continuing
steps per seed so the evidence is reproducible in this repository.

## Acceptance Evidence

| Gate | Requirement | Evidence artifact |
|------|-------------|-------------------|
| AC-1 | Actor-critic learner with immutable state, scan-compatible loops, config serialization, and focused tests | `src/alberta_framework/core/actor_critic.py`, `tests/test_actor_critic.py` |
| AC-2 | Critic consumes Step 1-3 machinery: GVF/Horde predictions, TD targets, traces, optional history features and CBP | `src/alberta_framework/core/horde_actor_critic.py`, `tests/test_horde_actor_critic.py`, `output/step3_dod9/summary.json` |
| AC-3 | Actor update has a documented policy-gradient objective and stability guard such as ObGD-style bounding | `docs/guide/actor-critic-control.md`, actor bounder tests |
| AC-4 | SARSA remains the baseline, not the completion claim | `TODO.md`, `ROADMAP.md`, `docs/research/step4_results.md` |
| AC-5 | 10-seed SARSA-vs-Q bsuite catch/cartpole report and broader primary sweep | `outputs/bsuite/sarsa_vs_q_catch_cartpole_10seed/`, `outputs/bsuite/sarsa_vs_q_primary_10seed/` |
| AC-6 | Step 3 DoD-9 capstone compares SARSA-only, SARSA+prediction-Horde, and SARSA+Horde+CBP+history | `output/step3_dod9/results.csv`, `output/step3_dod9/summary.json`, `docs/research/step3_dod9_capstone.md` |
| AC-7 | rlsecd/security-gym integration blockers are either closed in sibling repos or explicitly listed as external blockers | `docs/research/step4_results.md`, `docs/research/rlsecd_external_blockers.md` |

## Reproduction Commands

Prepare the environment:

```bash
source .venv/bin/activate
pip install -e '.[bsuite]'
python -m pip install 'git+https://github.com/google-deepmind/bsuite.git'
```

Run SARSA-vs-Q catch/cartpole:

```bash
python benchmarks/bsuite/run_sweep.py \
  --sarsa-vs-q \
  --bsuite-ids catch/0 cartpole/0 \
  --num_steps 2000 \
  --seeds 0 1 2 3 4 5 6 7 8 9 \
  --save_path outputs/bsuite/sarsa_vs_q_catch_cartpole_10seed \
  --comparison-report outputs/bsuite/sarsa_vs_q_catch_cartpole_10seed/sarsa_vs_q.md \
  --overwrite
```

Run the broader bounded primary bsuite sweep:

```bash
python benchmarks/bsuite/run_sweep.py \
  --sarsa-vs-q \
  --all-primary \
  --max-ids-per-experiment 1 \
  --num_steps 2000 \
  --seeds 0 1 2 3 4 5 6 7 8 9 \
  --save_path outputs/bsuite/sarsa_vs_q_primary_10seed \
  --comparison-report outputs/bsuite/sarsa_vs_q_primary_10seed/sarsa_vs_q.md \
  --overwrite
```

Run Step 4 Q/SARSA/actor-critic catch/cartpole:

```bash
python benchmarks/bsuite/run_sweep.py \
  --step4-comparison \
  --bsuite-ids catch/0 cartpole/0 \
  --num_steps 2000 \
  --seeds 0 1 2 3 4 5 6 7 8 9 \
  --save_path outputs/bsuite/step4_catch_cartpole_10seed \
  --comparison-report outputs/bsuite/step4_catch_cartpole_10seed/step4.md \
  --overwrite
```

Run Step 3 DoD-9:

```bash
python "examples/The Alberta Plan/Step3/dod9_capstone_sweep.py" \
  --n-seeds 10 \
  --steps 30000 \
  --last-window 5000 \
  --output-dir output/step3_dod9
```

Run throughput gates:

```bash
python benchmarks/horde_throughput.py --n-steps 10000 --output-dir output/step3_throughput
python benchmarks/sarsa_throughput.py --n-steps 10000 --output-dir output/step3_throughput
```

## External Integration Boundary

`security-gym` was available locally and its action/reward/termination contract
is mirrored by `alberta_framework.security`. Direct daemon closure for
`rlsecd` and `chronos-sec` remains outside this repository because the expected
public URLs were unavailable. The candidate tracks and required daemon evidence
are recorded in `docs/research/rlsecd_external_blockers.md`.
