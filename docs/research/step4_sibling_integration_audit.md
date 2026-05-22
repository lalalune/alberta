# Step 4 Sibling Integration Audit

Date: 2026-05-06

Workspace inspected: `/Users/shawwalters/Desktop/nca_fun`.

## Repository Availability

`security-gym` was pulled into
`/Users/shawwalters/Desktop/nca_fun/security-gym` from
`https://github.com/j-klawson/security-gym.git` at HEAD
`b397670311f12603df5d6a5c35125abdb5ec94b7`.

`rlsecd` and `chronos-sec` were not reachable at the expected URLs:

- `https://github.com/j-klawson/rlsecd.git`
- `https://github.com/j-klawson/chronos-sec.git`

Both returned `Repository not found` from `git ls-remote`; web search did not
surface an alternate public URL. Direct daemon edits remain blocked until the
correct repository URLs or credentials are available.

## security-gym API Verified

The pulled `security-gym` repo exposes:

- Text env: `SecurityLogStream-Text-v0`
- Hybrid env: `SecurityLogStream-Hybrid-v0`
- Action space: Gymnasium `Dict({"action": Discrete(6), "risk_score":
  Box(0, 10, shape=(1,))})`
- Action ids: `0=pass`, `1=alert`, `2=throttle`, `3=block_source`,
  `4=unblock`, `5=isolate`
- Stream semantics: `terminated` is always `False`; `truncated` marks stream
  exhaustion.
- Reward components: asymmetric action reward, risk-score MSE penalty, and
  ongoing consequence feedback from dropped blocked/throttled events.

## Framework-Side Contract Added

`alberta_framework.security` now defines the stable integration surface that
downstream active-defense code can consume without importing JAX learners:

- `SecurityAction`: stable six-action enum with SARSA/Horde-compatible integer
  head indices:
  `pass`, `alert`, `throttle`, `block`, `unblock`, `isolate`.
- `SECURITY_GYM_ACTION_NAMES`, `security_gym_action_name()`, and
  `to_security_gym_action()`: bridge the framework's `block` name to
  `security-gym`'s concrete `block_source` action dict.
- `security_gym_action_reward()`: reproduces the verified v0.4.x immediate
  action-reward table for contract tests.
- `SecurityRewardWeights` and `security_reward()`: named reward component
  mapping for reproducible scalar reward construction.
- `SecurityFeatureSchema`: versioned flat feature schema with observation-length
  validation and JSON-compatible serialization.
- `SecurityRolloutStep`: rollout transition record containing
  `state`, `action`, `reward`, `next_state`, `terminated`, `truncated`, and
  `policy_metadata`.
- `ThroughputMeter` and `ThroughputMeasurement`: lightweight wall-clock
  event-throughput hooks for daemon smoke tests.

## Remaining External Blockers

- `rlsecd --gym-control` cannot be implemented here because the `rlsecd`
  command-line entry point, event loop, checkpoint path, and feature extractor
  APIs were unavailable.
- End-to-end throughput for the actual rlsecd event path cannot be measured
  here. The framework hook now exists, but the measurement must wrap the real
  event receive -> feature extraction -> policy -> environment action ->
  learner update path.
- Feature schema compatibility can only be validated against the contract above
  until `rlsecd` exposes its concrete observation fields.
- Rollout logs can now use `SecurityRolloutStep`, but existing sibling log
  writers could not be audited or migrated without the `rlsecd` repo.

## Expected Downstream Wiring

The intended `rlsecd --gym-control` loop is:

1. Import `SecurityAction`, `N_SECURITY_ACTIONS`, `SecurityFeatureSchema`,
   `SecurityRolloutStep`, and `ThroughputMeter`.
2. Initialize `SARSAAgent(SARSAConfig(n_actions=N_SECURITY_ACTIONS, ...))` or
   `ActorCriticAgent(ActorCriticConfig(n_actions=N_SECURITY_ACTIONS, ...))`
   using the schema's `feature_dim`.
3. At each event, produce a flat feature vector matching
   `SecurityFeatureSchema`.
4. Select an action, translate `SecurityAction(action_id)` to the local
   environment actuation command, and log `policy_metadata` such as epsilon,
   Q-values or policy probabilities, TD error, and learner step.
5. Build the scalar reward with `security_reward()` from named environment
   components and append a `SecurityRolloutStep`.
6. Call `ThroughputMeter.tick()` after each complete transition update.
