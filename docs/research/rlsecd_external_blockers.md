# rlsecd External Integration Blockers

This repository has completed the framework-side learner APIs and local
evidence gates for the current Step 1-4 scope. The sibling `security-gym`
checkout is present under `/Users/shawwalters/Desktop/nca_fun/security-gym`,
was fast-forwarded to `4b4c7b6e322f7b18817949990dfb583aa5686056`, installed
into the Alberta Framework virtualenv, and validated against its environment
contract tests. The remaining rlsecd items are external because no sibling
`rlsecd` or `chronos-sec` checkout, production daemon loop, or security event
logs were available under `/Users/shawwalters/Desktop/nca_fun` during the
audit.

Recheck on 2026-05-08: `find /Users/shawwalters/Desktop/nca_fun -maxdepth 3`
still found no `rlsecd` or `chronos-sec` checkout. `gh search repos rlsecd`
returned no accessible repositories, and exact `git ls-remote` probes for
`shawwalters/rlsecd` and `shawwalters/chronos-sec` returned "Repository not
found." The local `security-gym` checkout is
`https://github.com/j-klawson/security-gym.git` at
`4b4c7b6e322f7b18817949990dfb583aa5686056`.

Recheck on 2026-05-21: `find /Users/shawwalters/Desktop/nca_fun -maxdepth 3`
again found no `rlsecd` or `chronos-sec` checkout. `gh repo view
shawwalters/rlsecd` and `gh repo view shawwalters/chronos-sec` both returned
"Could not resolve to a Repository"; `gh search repos rlsecd --limit 10`
returned `[]`.

Recheck on 2026-05-21 after the latest Step 4 probes: the reproducible audit
script `benchmarks/rlsecd_external_audit.py` wrote
`outputs/rlsecd_external_audit/status.json`. It found the local
`security-gym` checkout but no local `rlsecd` or `chronos-sec` checkout under
the configured workspace roots, and the GitHub probes still did not expose
accessible `shawwalters/rlsecd` or `shawwalters/chronos-sec` repositories.
This artifact is now referenced directly by the Step 3 solution gate.

## Step 4 Active-Defense Control

External dependency:

- rlsecd checkout with the production event loop and security event schema.
- security-gym rollout logs or equivalent active-defense environment logs.
- Stable feature schema shared by Step 3 prediction demons and the Step 4
  control critic.

Framework readiness:

- `SARSAAgent` exposes single-step `act`/`update` control through Horde.
- `ActorCriticAgent` and `HordeActorCriticAgent` cover the local discrete
  actor-critic control path.
- `alberta_framework.security` mirrors the verified local `security-gym`
  action ids and reward/termination contract.
- `benchmarks/security_gym_counterfactual_rollout.py` now generates a local
  `security-gym` counterfactual rollout artifact at
  `outputs/security_gym_counterfactual_rollout/results.json`. The checked
  result passed with pass-only reward `-72.0`, oracle-block reward `-4.2`,
  reward lift `67.8`, recall lift `1.0`, and zero oracle false positives.
  This proves the local environment/action/rollout contract, but still does
  not prove the unavailable `rlsecd` daemon loop.
- Local `security-gym` tests passed:
  `tests/test_env.py`, `tests/test_env_hybrid.py`, and
  `tests/test_scan_stream.py`.
- Local bsuite reports and local Horde/SARSA throughput evidence are recorded
  in `docs/research/step4_results.md`.

Acceptance criteria in rlsecd:

- A `--gym-control` or equivalent mode preserves temporal order and calls
  single-step `act`/`update` without batching away transitions.
- Rollout logs include `(state, action, reward, next_state, termination,
  policy_metadata)`.
- End-to-end throughput includes parsing, feature extraction, learner update,
  checkpoint/reporting, and action dispatch.
- The rollout report separates policy quality from safety fallback behavior and
  environment termination events.

Candidate command shape:

```bash
python -m rlsecd.daemon \
  --gym-control \
  --control-agent horde_actor_critic \
  --rollout-log output/rlsecd_control_rollouts.jsonl \
  --metrics-out output/rlsecd_control_metrics.json
```

## AF-2: IDBD-MLP 100k-Event Replay

External dependency:

- rlsecd replay harness with stable event ordering.
- A 100k-event log slice with feature schema metadata.
- A wrapper that maps rlsecd events into `MLPLearner` or `MultiHeadMLPLearner`
  single-step `predict`/`update` calls.

Framework readiness:

- IDBD-MLP is implemented with `IDBDParamState`.
- MLP and multi-head learners expose single-step JIT-compatible update APIs.
- Config serialization and checkpoint utilities are available.

Acceptance criteria in rlsecd:

- Run 100k ordered events without NaN/Inf in predictions, parameters, traces,
  or per-parameter step-size state.
- Persist checkpoint and config at the end of replay.
- Emit final-window prediction metrics and step-size summaries.

Candidate command shape:

```bash
python -m rlsecd.replay \
  --learner idbd_mlp \
  --events data/replay_100k.jsonl \
  --checkpoint-out output/idbd_mlp_100k \
  --metrics-out output/idbd_mlp_100k/metrics.json
```

## AF-2: IDBD-MLP Full 1.6M Log Stability

External dependency:

- Full 1.6M-event rlsecd log.
- Long-run replay harness with resumable checkpoints.
- Memory/throughput telemetry in the rlsecd process.

Acceptance criteria in rlsecd:

- Complete the full log without numerical divergence.
- Resume correctly from at least one midpoint checkpoint.
- Report throughput, max memory, final-window loss, and step-size statistics.

Candidate command shape:

```bash
python -m rlsecd.replay \
  --learner idbd_mlp \
  --events data/full_1_6m.jsonl \
  --resume-dir output/idbd_mlp_1_6m/checkpoints \
  --metrics-out output/idbd_mlp_1_6m/metrics.json
```

## Orbax Checkpoint Migration

External dependency:

- rlsecd `SecurityAgent` checkpoint save/load call sites.
- Existing production checkpoint compatibility expectations.

Framework readiness:

- `save_checkpoint`, `load_checkpoint`, `load_checkpoint_metadata`, and
  `checkpoint_exists` are implemented in Alberta Framework.
- Learners expose `to_config()`/`from_config()` roundtrips.

Acceptance criteria in rlsecd:

- `SecurityAgent` writes framework Orbax checkpoints.
- Restart loads state and config without changing predictions on a fixed
  validation batch.
- Legacy checkpoint migration behavior is explicitly documented or retired.

## Framework Config Serialization in rlsecd

External dependency:

- rlsecd configuration layer and deployment config files.

Acceptance criteria in rlsecd:

- SecurityAgent constructs learners from framework config dictionaries.
- Saved checkpoint metadata includes learner, optimizer, normalizer, bounder,
  and feature-schema config.
- A config roundtrip test verifies equivalent predictions before and after
  serialization.

## Periodic Feature Relevance Reporting

External dependency:

- rlsecd reporting loop and metrics sink.
- Stable feature names aligned with the deployed feature vector.

Framework readiness:

- `compute_feature_relevance` and `compute_feature_sensitivity` are available.

Acceptance criteria in rlsecd:

- Every 60 seconds, rlsecd emits top feature relevance values with feature
  names and timestamp.
- Reporting runs without blocking the learner update loop.
- A smoke test verifies the report schema on a deterministic fixture.

Candidate command shape:

```bash
python -m rlsecd.daemon \
  --feature-relevance-interval-s 60 \
  --metrics-out output/rlsecd_metrics.jsonl
```
