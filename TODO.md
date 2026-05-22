# TODO

- [ ] External: rlsecd `--gym-control` mode: existing 5 prediction demons + SARSA control demon
- [ ] External: rlsecd end-to-end throughput must include parsing, feature extraction, learner update, checkpoint/reporting, and action dispatch
- [ ] External: generate `(state, action, reward, outcome)` experience for autoresearch LLM oracle pipeline from rlsecd/security-gym rollouts
- [ ] External: AF-2 IDBD-MLP 100k-event replay test in rlsecd
- [ ] External: AF-2 IDBD-MLP full 1.6M log stability test
- [ ] External: simplify rlsecd SecurityAgent to use Orbax checkpoint utilities (format v2)
- [ ] External: simplify rlsecd SecurityAgent to use framework config serialization
- [ ] External: integrate `compute_feature_relevance` into rlsecd periodic reporting (60s interval)
- [x] Sim-to-real surrogate demonstration: `benchmarks/prototype_sim_to_real_transfer.py`;
      `outputs/prototype_sim_to_real_transfer/results.json`. Physical robot
      deployment remains outside this local surrogate.
