# Actor-Critic Control (Step 4 Completion)

Step 4 of the Alberta Plan is not just "any control algorithm." The plan names
**Control I: Continual actor-critic control** and states that the critic should
presumably be the result of Steps 1-3. In this repository, that means Step 4 is
complete only when the Step 3 GVF/Horde prediction machinery is used as the
critic inside a continual actor-critic control agent.

The current worktree includes a narrow `ActorCriticAgent` with a discrete
softmax actor, linear value critic, eligibility traces, scan-array loop, config
roundtrip, and ObGD bounder hook. It also includes `HordeActorCriticAgent`,
which uses the Step 3 `HordeLearner` as the critic: the configured value head
supplies the actor's TD-error/advantage and remaining heads act as auxiliary
prediction demons.

## Acceptance Criteria

Step 4 completion requires all of the following:

1. A continual actor-critic learner with immutable JAX state, explicit key
   handling, scan-compatible update loops, and the same optimizer/bounder/
   normalizer composition used by the rest of the framework. The current
   `ActorCriticAgent` covers the narrow discrete linear-critic version.
2. A critic backed by Step 1-3 machinery: GVF/Horde TD targets, optional
   prediction demons, per-head trace decay, and compatibility with history
   features and Continual Backprop where enabled.
3. An actor with policy-gradient updates, eligibility traces or an explicitly
   documented trace-free baseline, and ObGD-style update bounding or an
   equivalent stability guard.
4. Discrete-action support sufficient to match the current SARSA and bsuite
   control surfaces; continuous actions are a stretch goal unless Step 4 scope
   is widened.
5. Full empirical evidence: SARSA baseline, actor-critic, and actor-critic plus
   Step 3 critic variants compared on the same control matrix with multi-seed
   statistics.
6. Daemon integration readiness: single-step `predict`/`act`/`update` APIs,
   checkpoint/config roundtrip, throughput evidence, and explicit security-gym
   action/reward mapping.

The current SARSA implementation satisfies the first control baseline for
discrete actions. The local discrete-control Step 4 gate is now closed by the
Horde-backed actor-critic path, 10-seed bsuite reports, and the Step 3 DoD-9
critic-for-control capstone. Active-defense deployment remains external to this
repository because it requires sibling `rlsecd` and `security-gym` contracts.

## SARSA Baseline Boundary

`SARSAAgent` is the Step 4a baseline:

- It maps each discrete action to a Horde control demon.
- It computes the SARSA target externally as `r + gamma * Q(s', a')`.
- It keeps control-demon `gamma=0` internally because each action head predicts
  the externally computed target.
- It supports optional prediction demons in the same Horde.

This is valuable evidence that Horde can support on-policy TD control, but it
is not the Alberta Plan Step 4 endpoint. Step 4 completion requires an actor
that learns a policy directly from a Step 3 critic.

## Required Empirical Commands

Activate the project environment first:

```bash
source .venv/bin/activate
```

Run the full-family SARSA-vs-Q catch/cartpole comparison with at least
10 seeds. This is the bounded 200-step horizon report currently recorded under
`outputs/bsuite/sarsa_vs_q_catch_cartpole_10seed/`.

```bash
python benchmarks/bsuite/run_sweep.py \
  --sarsa-vs-q \
  --experiments catch cartpole \
  --num_steps 200 \
  --seeds 0 1 2 3 4 5 6 7 8 9 \
  --save_path outputs/bsuite/sarsa_vs_q_catch_cartpole_10seed \
  --comparison-report outputs/bsuite/sarsa_vs_q_catch_cartpole_10seed/sarsa_vs_q.md \
  --overwrite
```

Run broader bsuite coverage before claiming general control evidence:

```bash
python benchmarks/bsuite/run_sweep.py \
  --sarsa-vs-q \
  --all-primary \
  --num_steps 200 \
  --seeds 0 1 2 3 4 5 6 7 8 9 \
  --save_path outputs/bsuite/sarsa_vs_q_primary_10seed \
  --comparison-report outputs/bsuite/sarsa_vs_q_primary_10seed/sarsa_vs_q.md \
  --overwrite
```

Run the Step 4 Q/SARSA/actor-critic bsuite comparison represented by the
current local artifacts. The generated `step4.md` uses `metric=auto`, so catch
uses total regret and cartpole uses episode return.

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

Run the Step 3 DoD-9 critic-for-control capstone:

```bash
python "examples/The Alberta Plan/Step3/dod9_capstone_sweep.py" \
  --n-seeds 10 \
  --steps 30000 \
  --last-window 5000 \
  --output-dir output/step3_dod9
```

Run local throughput gates for the core control path:

```bash
python benchmarks/horde_throughput.py --n-steps 10000 --output-dir output/step3_throughput
python benchmarks/sarsa_throughput.py --n-steps 10000 --output-dir output/step3_throughput
```

The generated reports are summarized in `docs/research/step4_results.md`.

## Implemented Scope and Follow-Up Tracks

- **Track A: discrete softmax actor-critic.** Implemented for the local
  discrete Step 4 gate with policy-gradient updates and bounded actor steps.
- **Track B: actor plus auxiliary prediction Horde.** Implemented as the
  Horde-backed actor-critic path; further evidence should compare critic-only,
  critic plus prediction demons, and history-feature/CBP variants on broader
  control matrices.
- **Track C: average-reward actor-critic preview.** Useful for Steps 5-6, but
  should not block discounted Step 4 completion unless the roadmap scope is
  explicitly changed.
- **Track D: security-gym active-defense track.** Treat security actions as the
  discrete policy, use rlsecd features and prediction demons as the critic, and
  evaluate active defense under continuing rollouts. This remains external to
  this repository until the sibling rlsecd interfaces and rollout logs are
  available.

## Sibling Integration Blockers

The framework provides the learner APIs and local evidence gates, but
rlsecd/security-gym active-defense completion requires sibling work:

- `security-gym` must expose a stable discrete action enum and reward contract
  for pass, alert, throttle, block, unblock, and isolate or the current sibling
  equivalent.
- `rlsecd` must provide a `--gym-control` event loop that calls single-step
  `act` and `update` without buffering away temporal order.
- The sibling feature schema must be pinned so the Step 3 prediction demons and
  Step 4 critic consume the same observation vector at train and deploy time.
- Rollout logs must include `(state, action, reward, next_state, termination,
  policy_metadata)` so bsuite-like reports and oracle review can reproduce the
  control evidence.
- End-to-end throughput must be measured in rlsecd, including parsing, feature
  extraction, learner update, checkpoint/reporting overhead, and action
  dispatch. Core framework throughput alone is insufficient evidence.
