# D05 Homeostatic Trunk Plasticity Regularizers

## Core Hypothesis

Step 2 MLPs lose useful online plasticity because hidden units become
functionally dormant: their activations may be nonzero, but their effective
prediction sensitivity is tiny, so future errors cannot move the trunk through
those units. A small homeostatic regularizer on the trunk can preserve a broad
distribution of active, sensitive hidden units and improve nonstationary
tracking without changing targets, adding frozen features, resetting units, or
ensembling learners.

The candidate method is `dormant_activation_jacfloor`: augment the online
prediction loss with a trunk-only dormant-unit activation penalty, a soft
positive/negative occupancy penalty, and a low-weight Jacobian floor.

## Why Different

This is not centered targets, ECOC, RFF, hashed quadratic features, residual
imprinting, fast/slow gating, surprise reset, prototype context,
orthogonalization, or EMA prediction. Those mechanisms alter the target code,
input representation, learner population, explicit feature basis, parameter
state, or evaluation weights. D05 instead leaves the fair MLP architecture and
online data stream intact, and changes only the local training objective for
the shared trunk. It is a smooth regularization test of the plasticity
bottleneck, not a feature-construction or replacement mechanism.

## Mathematical Grounding

The current Step 2 MLP trunk is
`u_l = LN(W_l h_{l-1} + b_l)`, `h_l = phi_a(u_l)`, where
`phi_a(u) = max(u, a u)` and `a = 0.01` by default. For a scalar prediction
`y_hat`, the trunk update through hidden unit `j` in layer `l` scales with
`|d y_hat / d u_{l,j}|`. A unit can therefore be alive by activation magnitude
but dormant for learning if its downstream sensitivity is near zero. LayerNorm
controls per-sample scale, but it does not enforce per-unit temporal occupancy
or preserve output sensitivity.

Maintain script-local exponential statistics for each hidden unit:

```text
m_{l,j,t} = rho m_{l,j,t-1} + (1-rho) |h_{l,j,t}|
g_{l,j,t} = rho g_{l,j,t-1} + (1-rho) |d y_hat_t / d u_{l,j,t}|
```

Use stop-gradient dormant gates
`c^m_{l,j} = sigmoid((tau_m - m_{l,j}) / T_m)` and
`c^g_{l,j} = sigmoid((tau_g - g_{l,j}) / T_g)`. The per-step objective is:

```text
L_t = 0.5 (target_t - y_hat_t)^2
    + lambda_m sum_l mean_j c^m_{l,j} softplus(tau_m - |h_{l,j,t}|)^2
    + lambda_g sum_l mean_j c^g_{l,j} softplus(tau_g - |J_{l,j,t}|)^2
    + lambda_p sum_l (mean_j sigmoid(k u_{l,j,t}) - 0.5)^2
```

where `J_{l,j,t} = d y_hat_t / d u_{l,j,t}`. The activation term prevents
long-term unused units from staying quiet, the Jacobian floor prevents hidden
features from becoming disconnected from the head, and the occupancy term
discourages all units from living on the same LeakyReLU branch. The
regularizer should be weak: its role is to keep gradient pathways available,
not to dominate prediction learning.

## Why It Could Beat UPGD/CBP And Fair MLP

Fair MLP training optimizes current prediction error only, so it can spend
hidden capacity on the current regime and leave little gradient-carrying
capacity for later regimes. CBP and surprise reset restore capacity by
replacing units; UPGD restores plasticity by perturbing parameters. Both can
damage useful features or trade away current-block fit. D05 is less
destructive: it continuously nudges marginal units back toward usable activation
and sensitivity before hard replacement is needed. If Step 2 failures are
partly caused by gradual trunk saturation rather than missing feature classes,
this should improve final-window loss on drift streams while retaining the fair
MLP's current-regime accuracy.

## Minimal Implementation Sketch

Implement only a standalone experiment script, not core learner changes:
`examples/The Alberta Plan/Step2/step2_plasticity_regularizers.py`.

Reuse the fair MLP comparator settings from the moonshot and plasticity-hybrid
runs: `hidden_sizes=(64,)`, `sparsity=0.5`, `use_layer_norm=True`, ObGD
bounding with `kappa=2.0`, same seeds, same streams, and paired initial
parameters. In the script-local learner, copy the MLP forward pass so it can
return `(prediction, preactivations, activations)`. Compute prediction-loss
gradients and regularizer gradients with `jax.value_and_grad`; add regularizer
gradients only to hidden-layer weights and biases, leaving the output head
trained by prediction error alone. Keep the EMA statistics in script-local
state updated every time step.

Start with three methods: `mlp`, `dormant_activation` with `lambda_m` and
`lambda_p`, and `dormant_activation_jacfloor` adding `lambda_g`. Suggested
initial grid: `rho=0.99`, `tau_m=0.05`, `tau_g=1e-3`, `lambda_m in {1e-4,3e-4}`,
`lambda_p=1e-4`, `lambda_g in {1e-6,1e-5}`.

## Metrics And Success Bar

Report paired seed differences versus fair MLP on final-window MSE, final-window
accuracy for digits, held-out final-phase accuracy, mean dormant fraction
`mean(m < tau_m)`, mean low-sensitivity fraction `mean(g < tau_g)`, activation
positive-branch occupancy, effective rank of first hidden activations, and
NaN/finite checks.

Promote only if `dormant_activation_jacfloor` beats fair MLP on paired
final-window MSE in at least two of three nonstationary regimes, does not reduce
digits final-window accuracy by more than `0.010`, and improves at least one
plasticity diagnostic without increasing NaN failures. Treat a pure diagnostic
win with worse loss as negative, matching the M09 orthogonalization lesson.

## Risks And Negative Controls

The main risk is over-regularization: the loss may keep units active but reduce
specialization. Include `lambda=0` as an implementation identity check, a
`shuffled_regularizer_gate` control that permutes dormant gates across units,
and an `activation_only` ablation to test whether the Jacobian floor adds value
or just moves head weights. Also run a stationary IID digits control; a method
that only helps by injecting noise should lose there. If all regularized
variants improve dormant metrics while losing MSE, the anti-saturation
hypothesis is not the Step 2 bottleneck at this horizon.

## Exact First Command

```bash
source .venv/bin/activate && python "examples/The Alberta Plan/Step2/step2_plasticity_regularizers.py" --suite both --methods mlp dormant_activation dormant_activation_jacfloor --synthetic-streams out_of_class_polynomial frequency_mismatch compositional --digits-variants iid class_blocked permuted_blocks --synthetic-steps 1500 --digits-steps 1200 --n-seeds 3 --final-window 300 --output-dir output/step2_new_directions/d05_plasticity_regularizers
```
