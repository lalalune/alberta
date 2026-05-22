# rlsecd External Acceptance Spec

Status: pending external `rlsecd` / `chronos-sec` repositories.

## rlsecd_gym_control_horde_sarsa_daemon

TODO: External: rlsecd `--gym-control` mode: existing 5 prediction demons + SARSA control demon

Required repositories: `rlsecd`, `chronos-sec`, `security-gym`

Command:

```bash
python -m rlsecd.daemon --gym-control --prediction-demons 5 --control-agent sarsa --action-heads 6 --rollout-log outputs/rlsecd_gym_control/rollouts.jsonl --metrics-out outputs/rlsecd_gym_control/metrics.json
```

Required artifacts:
- `outputs/rlsecd_gym_control/rollouts.jsonl`
- `outputs/rlsecd_gym_control/metrics.json`
- `outputs/rlsecd_gym_control/config.json`

Pass conditions:
- n_prediction_demons == 5
- n_control_actions == 6
- n_transitions > 0 and equals the rollout JSONL row count
- config.json declares SARSA control, 5 prediction demons, 6 actions, temporal uniformity, framework SARSAAgent/HordeLearner usage, and the security-gym action vocabulary
- all rollout records preserve temporal order with unique step ids
- all rollout records include state, action, reward, next_state, termination, and policy metadata with finite numeric state/reward values
- all action ids are valid security-gym action heads
- mean_reward and sarsa_td_error_final_window are finite scalar metrics

## rlsecd_end_to_end_daemon_throughput

TODO: External: rlsecd end-to-end throughput must include parsing, feature extraction, learner update, checkpoint/reporting, and action dispatch

Required repositories: `rlsecd`, `chronos-sec`

Command:

```bash
python -m rlsecd.benchmarks.throughput --include-parsing --include-feature-extraction --include-learner-update --include-checkpoint-reporting --include-action-dispatch --metrics-out outputs/rlsecd_throughput/metrics.json
```

Required artifacts:
- `outputs/rlsecd_throughput/metrics.json`

Pass conditions:
- n_events > 0
- events_per_second > 0
- wall_clock_s > 0
- events_per_second agrees with n_events / wall_clock_s within 5%
- all five pipeline stage timings are present
- stage_event_counts records all measured events for parsing, feature extraction, learner update, checkpoint/reporting, and action dispatch
- measurement wraps the real daemon event path

## rlsecd_oracle_experience_export

TODO: External: generate `(state, action, reward, outcome)` experience for autoresearch LLM oracle pipeline from rlsecd/security-gym rollouts

Required repositories: `rlsecd`, `security-gym`

Command:

```bash
python -m rlsecd.exports.oracle_experience --rollout-log outputs/rlsecd_gym_control/rollouts.jsonl --records-out outputs/rlsecd_oracle_experience/records.jsonl --manifest-out outputs/rlsecd_oracle_experience/manifest.json
```

Required artifacts:
- `outputs/rlsecd_oracle_experience/records.jsonl`
- `outputs/rlsecd_oracle_experience/manifest.json`

Pass conditions:
- n_records > 0
- schema == rlsecd.oracle_experience.v1
- each record includes state, action, reward, outcome, source_rollout_step, and policy metadata
- oracle records are ordered by unique source rollout step ids
- manifest names an existing source rlsecd/security-gym rollout log
- manifest proves records were exported from a production rollout
- manifest source rollout record count matches the source JSONL

## rlsecd_idbd_mlp_100k_replay

TODO: External: AF-2 IDBD-MLP 100k-event replay test in rlsecd

Required repositories: `rlsecd`

Command:

```bash
python -m rlsecd.replay --learner idbd_mlp --events data/replay_100k.jsonl --checkpoint-out outputs/idbd_mlp_100k/checkpoint --metrics-out outputs/idbd_mlp_100k/metrics.json
```

Required artifacts:
- `outputs/idbd_mlp_100k/checkpoint`
- `outputs/idbd_mlp_100k/metrics.json`

Pass conditions:
- n_events >= 100000
- all_finite is true
- finite_components marks predictions, parameters, traces, and step sizes as finite
- validation_batch_size > 0
- checkpoint roundtrip preserves predictions on a fixed validation batch

## rlsecd_idbd_mlp_full_log_stability

TODO: External: AF-2 IDBD-MLP full 1.6M log stability test

Required repositories: `rlsecd`

Command:

```bash
python -m rlsecd.replay --learner idbd_mlp --events data/full_1_6m.jsonl --resume-dir outputs/idbd_mlp_1_6m/checkpoints --metrics-out outputs/idbd_mlp_1_6m/metrics.json
```

Required artifacts:
- `outputs/idbd_mlp_1_6m/checkpoints`
- `outputs/idbd_mlp_1_6m/metrics.json`

Pass conditions:
- n_events >= 1600000
- all_finite is true through the final event
- finite_components marks predictions, parameters, traces, and step sizes as finite
- checkpoint_count >= 2
- midpoint checkpoint resume completes with equivalent final metrics

## rlsecd_security_agent_orbax_checkpoint_v2

TODO: External: simplify rlsecd SecurityAgent to use Orbax checkpoint utilities (format v2)

Required repositories: `rlsecd`

Command:

```bash
python -m rlsecd.tests.checkpoint_roundtrip --format v2 --checkpoint-dir outputs/rlsecd_checkpoint_v2 --metrics-out outputs/rlsecd_checkpoint_v2/metrics.json
```

Required artifacts:
- `outputs/rlsecd_checkpoint_v2`
- `outputs/rlsecd_checkpoint_v2/metrics.json`
- `outputs/rlsecd_checkpoint_v2/metadata.json`

Pass conditions:
- format_version == 2
- metadata.json declares alberta.rlsecd.security_agent_checkpoint.v2
- metadata includes framework checkpoint schema fields
- restored learner, optimizer, normalizer, and step_count match the saved SecurityAgent state
- prediction_roundtrip_max_abs_diff <= 1e-6

## rlsecd_security_agent_framework_config_serialization

TODO: External: simplify rlsecd SecurityAgent to use framework config serialization

Required repositories: `rlsecd`

Command:

```bash
python -m rlsecd.tests.config_roundtrip --metrics-out outputs/rlsecd_config_roundtrip/metrics.json
```

Required artifacts:
- `outputs/rlsecd_config_roundtrip/metrics.json`

Pass conditions:
- all framework config dictionaries roundtrip without dropped fields
- serialized component type names are present for learner, optimizer, normalizer, and feature schema
- unknown_config_keys and dropped_config_keys are empty
- restored config schema version matches the saved schema version
- SecurityAgent reconstructed from config produces equivalent predictions

## rlsecd_feature_relevance_periodic_reporting

TODO: External: integrate `compute_feature_relevance` into rlsecd periodic reporting (60s interval)

Required repositories: `rlsecd`

Command:

```bash
python -m rlsecd.daemon --feature-relevance-interval-s 60 --metrics-out outputs/rlsecd_feature_relevance/metrics.jsonl
```

Required artifacts:
- `outputs/rlsecd_feature_relevance/metrics.jsonl`

Pass conditions:
- feature_relevance_interval_s == 60
- at least two timestamped reports are emitted
- report timestamps are ordered with approximately 60 seconds between reports
- every report contains non-empty feature names and matching finite relevance values
- reporting uses alberta_framework.core.diagnostics.compute_feature_relevance
- reporting does not block or skip learner updates
