# Step 3 Off-Policy Horde Backend

Status: implemented as a stable first nonlinear backend, not a full
Gradient-TD/GQ/TDC closure.

## What Changed

`src/alberta_framework/core/off_policy_horde.py` adds
`OffPolicyHordeLearner`, a shared-trunk multi-head nonlinear Horde backend
with one importance-sampling ratio per demon per transition.

The public update APIs are:

```python
result = learner.update_with_ratios(
    state,
    observation,
    cumulants,
    next_observation,
    rhos,
)

result = learner.update_with_probabilities(
    state,
    observation,
    cumulants,
    next_observation,
    target_probabilities,
    behavior_probabilities,
)
```

The scan helper is:

```python
result = run_off_policy_horde_learning_loop(
    learner,
    state,
    observations,
    cumulants,
    next_observations,
    rhos,
)
```

## Implemented Update

For each demon `i`, the backend computes:

`delta_i = c_i + gamma_i V_i(s') - V_i(s)`

`rho_i = pi_i(a_t | s_t) / b(a_t | s_t)`

`rho_update_i = min(max(rho_i, 0), ratio_clip)`

`rho_trace_i = min(max(rho_i, 0), trace_ratio_clip)`

`effective_error_i = rho_update_i * delta_i`

The shared nonlinear trunk receives the summed current-step cotangent:

`sum_i effective_error_i grad_h V_i(s)`

The output head for demon `i` uses a per-transition trace coefficient:

`gamma_i * lambda_i * rho_trace_i`

Inactive demons are represented by `NaN` cumulants and preserve their
parameters, traces, and optimizer states.

## Relation To Existing `off_policy_td.py`

`core/off_policy_td.py` remains the linear off-policy TD module. It implements
linear per-decision importance sampling, optional Retrace-style clipping, and
linear ETD(lambda). It is the right place for simple feature-vector
experiments, Baird-style probes, and exact linear off-policy checks.

`core/off_policy_horde.py` is the nonlinear shared-trunk Horde backend. It
supports vector GVF demons, shared nonlinear features, per-demon ratios, JAX
scan, and the same optimizer/bounder/normalizer style as the rest of the Step 3
stack.

The two modules are complementary:

| Capability | `off_policy_td.py` | `off_policy_horde.py` |
|---|---:|---:|
| Linear value function | Yes | Yes, via `hidden_sizes=()` |
| Nonlinear shared trunk | No | Yes |
| Multiple GVF demons | No | Yes |
| Per-demon ratios | No | Yes |
| ETD(lambda) | Yes, linear | No |
| Full GTD/GQ/TDC MSPBE correction | No | No |

## What This Does Not Solve

This is not a full GQ(lambda), GTD2, or TDC implementation. The implemented
backend is a clipped weighted semi-gradient TD backend. It does not maintain the
secondary weight vector required by Gradient-TD algorithms, and it does not
optimize the MSPBE objective.

The current nonlinear trunk uses current-step gradients only. This avoids the
known shared-trunk trace coupling issue in nonlinear multi-head learning. Output
heads can use ratio-aware traces, but full per-demon nonlinear trunk traces
would require either independent demon trunks or `O(n_demons * trunk_params)`
trace storage.

## Literature Position

The conservative baseline follows the off-policy importance-ratio and clipped
trace lineage from Precup, Sutton, and Singh's off-policy traces and the later
Retrace(lambda) safety framing by Munos et al. The repo's existing linear module
also includes ETD(lambda), following Sutton, Mahmood, and White's emphatic TD
work.

Gradient-TD/GQ/TDC is a different family. Sutton et al.'s GTD/TDC work and
Maei and Sutton's GQ(lambda) line add secondary weights to descend on projected
Bellman-error objectives under off-policy sampling. Those are the next backend,
not a claim made by this module.

Primary references:

- [Fast gradient-descent methods for temporal-difference learning with linear function approximation](https://www.davidsilver.uk/wp-content/uploads/2020/03/gtd.pdf)
- [GQ(lambda): a general gradient algorithm for temporal-difference prediction learning with eligibility traces](https://www.researchgate.net/publication/215990384_GQlambda_A_general_gradient_algorithm_for_temporal-difference_prediction_learning_with_eligibility_traces)
- [An emphatic approach to the problem of off-policy temporal-difference learning](https://jmlr.csail.mit.edu/papers/v17/14-488.html)
- [Safe and efficient off-policy reinforcement learning](https://papers.nips.cc/paper/6538-safe-and-efficient-off-policy-reinforcement-learning)

## Acceptance Evidence

The focused test suite is `tests/test_off_policy_horde.py`. It covers:

- initialization, prediction, config roundtrip;
- finite nonlinear updates and shape invariants;
- clipping and per-demon ratio effects;
- target/behavior probability API equivalence;
- scan-loop compatibility;
- a seeded positive-control off-policy bandit-style prediction where the target
  policy value beats the behavior-distribution estimate.

## Remaining Blockers

The next backend should add one of these, explicitly:

1. Linear multi-demon GTD2/TDC/GQ(lambda) with secondary weights and exact
   equation-level tests.
2. Independent nonlinear demon trunks with per-demon secondary weights.
3. A shared-trunk approximation to nonlinear GTD with documented bias and
   ablations against the stable backend added here.

Until then, repo claims should say: nonlinear off-policy Horde has a clipped
importance-weighted semi-gradient backend, while full Gradient-TD/GQ/TDC Horde
remains open.
