# D03 Dynamic Sparse Connectivity

## Core Hypothesis

Step 2 failures may be less about missing nonlinear capacity and more about
where a fixed parameter budget is spent over time. A sparse MLP whose active
connections are rewired online can track changing feature relevance better than
a fair dense or statically sparse MLP, because it reallocates edge budget toward
currently high-gradient input-hidden and hidden-hidden interactions without
adding units, heads, target codes, memory banks, or ensembles.

The experiment is a RigL/SET-style `DynamicSparseMLP`: keep each weight matrix
at fixed density `rho`, update active weights every step with the existing
MLP/ObGD path, and every `K` steps replace a fraction `zeta` of active edges.
Prune active edges with small saliency `|w_ij|`; grow inactive edges with large
instantaneous gradient saliency `|dL/dw_ij|`, initialized at zero or LeCun-scale
noise. The active parameter count and nominal FLOP budget are constant.

## Why Different

This is not another target encoding, feature lift, reset, context sidecar,
orthogonalizer, EMA predictor, or learner ensemble. It changes the graph on
which the ordinary MLP learns. Unlike CBP/surprise reset, it does not replace
whole hidden units. Unlike UPGD, it does not perturb all parameters to estimate
utility. Unlike hashed quadratic features, it does not bake in a candidate
hypothesis class. The mechanism is structural credit assignment at the
connection level.

## Mathematical Grounding

For layer `l`, maintain weights `W_l` and binary mask `M_l in {0,1}^{shape(W_l)}`
with `||M_l||_0 = m_l = floor(rho * |W_l|)`. Prediction uses
`W_l^eff = M_l * W_l`. Ordinary online loss is

`L_t(theta, M) = 0.5 * ||f(x_t; M * theta) - y_t||^2`.

On every time step, update only active weights:

`W_{l,t+1} = M_l * (W_{l,t} - alpha * B_t * grad_{W_l} L_t)`,

where `B_t` is the existing ObGD bounder scale if enabled. Every `K` steps,
rewire `r_l = floor(zeta * m_l)` edges per layer:

- prune `P_l = bottom_r({|W_l,ij| : M_l,ij = 1})`;
- grow `G_l = top_r({|grad_{W_l,ij} L_t| : M_l,ij = 0})`;
- set `M_l <- (M_l \ P_l) union G_l`;
- set grown weights to `0` for a pure RigL probe, with a LeCun-noise ablation.

The pruning rule is the first-order/small-weight approximation behind magnitude
pruning: small active weights have low immediate functional contribution. The
growth rule uses the first-order loss decrease available to an inactive edge:
if an edge currently has `w_ij = 0`, then opening it with a small step in
`-grad_ij L_t` improves loss by approximately `eta * |grad_ij L_t|^2`.
RigL uses this magnitude-prune/gradient-grow idea under fixed sparse training;
SET supplies the complementary evolutionary baseline: magnitude pruning plus
random growth from an initially sparse Erdos-Renyi graph. Sources:
[Mocanu et al. 2018](https://www.nature.com/articles/s41467-018-04316-3) and
[Evci et al. 2020](https://proceedings.mlr.press/v119/evci20a.html).

## Why It Could Beat Previous Iteration And Fair MLP

The previous Step 2 positives either matched a known hypothesis class
(`hashed_quadratic`) or won synthetically while losing external ordinary digits
(`UPGD`). Dynamic sparsity attacks a different bottleneck: fair MLPs spend
capacity on a fixed random sparse initialization, and dense fair MLPs can keep
many weak, stale connections alive after context shifts. Rewiring can turn a
single width-64 MLP into a sequence of locally specialized sparse subnetworks
while preserving temporal uniformity and parameter budget. It should help most
on class-blocked digits, permuted-pixel digits, and out-of-class synthetic
streams where the relevant input-hidden edges change faster than hidden units
need to be born or reset.

## Minimal Implementation Sketch

Prototype outside shared core first in
`examples/The Alberta Plan/Step2/new_directions/d03_dynamic_sparse_connectivity.py`.
Define a local `DynamicSparseMLPState` wrapping `MLPLearnerState` plus
`masks`, `rewire_count`, and `key`. Reuse `MLPLearner._forward` and the same
loss/gradient calculation, but multiply each weight by its mask in forward,
trace, optimizer, and update application. Biases stay dense. Run variants:

- `fair_mlp_64_64`: existing MLP comparator, same hidden sizes and normalizer.
- `static_sparse_mlp`: same initial masks and density, no rewiring.
- `set_sparse_mlp`: magnitude prune, random inactive-edge growth.
- `rigl_sparse_mlp`: magnitude prune, gradient inactive-edge growth.
- `rigl_zero_init` versus `rigl_lecun_init` for grown edges.

Use `rho in {0.05, 0.10, 0.20}`, `K in {50, 100, 250}`, and
`zeta in {0.02, 0.05}` for the pilot only. Keep hidden sizes fixed at `(64, 64)`
unless parameter matching requires a smaller dense comparator; report active
edge count and total stored dense-array count separately.

## Metrics / Success Bar

Primary metric: paired final-window MSE versus best fair MLP on the same seed.
Secondary metrics: final-window accuracy for digits, held-out test accuracy for
digits, adaptation half-life after stream/context shift, active-edge churn rate,
dead-unit fraction, mask Jaccard distance across phases, and wall-clock step
time.

Promote only if `rigl_sparse_mlp` beats both `fair_mlp_64_64` and
`static_sparse_mlp` on at least two stream families, one of them external
digits, with at least `8/10` paired pilot wins and no more than `0.01` held-out
accuracy regression where MSE improves. A synthetic-only win is not enough.

## Risks / Negative Controls

Gradient growth may be noisy in one-sample online learning; include SET random
growth to test whether gradients matter. Rewiring may erase slowly useful rare
features; include class-blocked and recurring-context streams and track mask
reuse. Sparse arrays may not yield real JAX speedups; treat speed as diagnostic,
not success. Negative controls: no-rewire static sparse MLP, random-rewire at
same churn rate, dense fair MLP with identical optimizer/bounder/normalizer,
and UPGD with the published best pilot settings.

## Exact First Command

```bash
source .venv/bin/activate && python "examples/The Alberta Plan/Step2/new_directions/d03_dynamic_sparse_connectivity.py" --steps 6000 --n-seeds 10 --final-window 500 --streams polynomial,frequency,compositional,digits_class_blocked,digits_permuted --methods fair_mlp_64_64,static_sparse_mlp,set_sparse_mlp,rigl_sparse_mlp,upgd --densities 0.05,0.10,0.20 --rewire-intervals 50,100,250 --rewire-fractions 0.02,0.05 --output-dir outputs/step2_new_directions/d03_dynamic_sparse_connectivity
```
