# D18 Simple Universal Learner Assessment

## Question

Can Step 2 be closed with a simple universal learner rather than a router or
mixture of MLP experts?

## Current Best Simple Candidate

The strongest simple candidate is D18: one additive predictor with every
component updated every timestep:

- resource-managed RKHS core,
- fixed tanh/Fourier basis readout,
- optional finite polynomial residual blocks for ablation only.

There is no MLP expert, no output router, and no expert selection in the
canonical candidate.

Two fixed variants matter:

- `d18_core_0p5_basis_0p4`: strongest digits retention.
- `d18_core_0p25_basis_0p6`: stronger compositional/frequency behavior with
  smaller digit MSE margins.

## Broad 3-Seed Evidence

### `d18_core_0p5_basis_0p4`

Source: `outputs/step2_new_directions/d18_core0p5_blockers_3seed/results.json`.

This variant beats best fair MLP on most blockers:

| Dataset | D18 final MSE | Best MLP final MSE | Paired diff | Wins |
|---|---:|---:|---:|---:|
| controlled_frequency | 0.0329 | 0.1569 | +0.1240 | 3/0/0 |
| controlled_interaction | 0.0203 | 0.4283 | +0.4079 | 3/0/0 |
| digits_label_drift | 0.0320 | 0.0383 | +0.0061 | 3/0/0 |
| digits_mask_noise | 0.0451 | 0.0478 | +0.0028 | 3/0/0 |
| digits_permuted_pixels | 0.0381 | 0.0493 | +0.0110 | 3/0/0 |
| synthetic_compositional | 0.2411 | 0.2758 | +0.0306 | 2/1/0 |
| synthetic_frequency | 0.9093 | 1.1493 | +0.2392 | 3/0/0 |
| synthetic_polynomial | 1.0385 | 0.9475 | -0.0910 | 1/2/0 |

Digits held-out accuracy is strongly positive:

- label drift: +0.0519 test accuracy, 3/0/0.
- mask noise: +0.0266 test accuracy, 3/0/0.
- permuted pixels: +0.0451 test accuracy, 3/0/0.

### `d18_core_0p25_basis_0p6`

Source:
`outputs/step2_new_directions/d18_core0p25_basis0p6_blockers_3seed/results.json`.

This variant strengthens compositional/frequency but weakens digit MSE margins:

| Dataset | D18 final MSE | Best MLP final MSE | Paired diff | Wins |
|---|---:|---:|---:|---:|
| controlled_frequency | 0.0364 | 0.1569 | +0.1204 | 3/0/0 |
| controlled_interaction | 0.0510 | 0.4283 | +0.3773 | 3/0/0 |
| digits_label_drift | 0.0378 | 0.0383 | +0.0003 | 2/1/0 |
| digits_mask_noise | 0.0471 | 0.0478 | +0.0007 | 2/1/0 |
| digits_permuted_pixels | 0.0412 | 0.0493 | +0.0080 | 3/0/0 |
| synthetic_compositional | 0.2277 | 0.2758 | +0.0440 | 3/0/0 |
| synthetic_frequency | 0.8410 | 1.1493 | +0.3074 | 3/0/0 |
| synthetic_polynomial | 1.0380 | 0.9475 | -0.0906 | 0/3/0 |

Held-out digit accuracy remains positive:

- label drift: +0.0377 test accuracy, 3/0/0.
- mask noise: +0.0204 test accuracy, 3/0/0.
- permuted pixels: +0.0445 test accuracy, 3/0/0.

## What Improved

D18 is a real advance over the previous mixture direction:

- It beats fair MLP on external stateful digits without routing to an MLP.
- It beats fair MLP on synthetic compositional without using an MLP expert.
- It beats fair MLP on frequency and controlled interaction benchmarks.
- It is temporally uniform: all included blocks update every timestep.
- It is simple enough to reason about: additive core plus fixed basis.

The best canonical choice depends on priority:

- choose `core_0p5_basis_0p4` if digit MSE robustness matters most;
- choose `core_0p25_basis_0p6` if the Step 2 compositional criterion matters
  most.

## Remaining Gap

The full Step 2 claim is not closed because synthetic polynomial still loses to
the best fair MLP under the current benchmark protocol.

This is not an ambiguous gap:

- `core_0p5_basis_0p4`: -0.0910 paired MSE diff, 1/2/0.
- `core_0p25_basis_0p6`: -0.0906 paired MSE diff, 0/3/0.

The failing stream is a changing degree-3 polynomial oracle. It is the one
place where the simple additive learner still lacks a fixed mechanism that
beats MLP without hurting the other regimes.

## Failed Fixes

### Finite Polynomial LMS

Source: `outputs/step2_new_directions/d18_poly_focus_3seed/results.json`.

Adding a finite degree-3 polynomial LMS block did not close the gap:

- best polynomial result stayed worse than MLP;
- stronger polynomial scales hurt digits;
- faster LMS and higher residual gain made polynomial worse.

Conclusion: slow normalized LMS over explicit products is not enough.

### Polynomial RLS

Sources:

- `outputs/step2_new_directions/d18_poly_rls_focus_3seed/results.json`
- `outputs/step2_new_directions/d18_poly_rls_small_scale_3seed/results.json`
- `outputs/step2_new_directions/d18_lowcore_poly_rls_3seed/results.json`
- `outputs/step2_new_directions/d18_poly_rls_forget97_3seed/results.json`

RLS was the first polynomial mechanism with a positive mean signal:

- `core_0p5_basis_0p4_poly_rls_0p6` reached 0.8073 mean MSE on synthetic
  polynomial versus best MLP 0.9475.

But it was not robust:

- it won only the hard seed and lost the two easy seeds;
- it degraded digits sharply at useful scales;
- stronger forgetting was numerically unstable.

Conclusion: polynomial RLS is useful evidence, but not canonical.

### D14 Unified Basis Fusion

Sources:

- `outputs/step2_new_directions/d14_basis_digits_probe_3seed/results.json`
- `outputs/step2_new_directions/d18_unified_poly_3seed/results.json`

D14 unified basis wins synthetic polynomial as a standalone learner:

- best D14 basis: 0.8571 MSE versus best MLP 0.9475, 3/0/0.

But it loses digits and compositional as a standalone learner, and when fused
as a small D18 residual block it does not close polynomial:

- best fused D18 unified variant: 1.0344 MSE, still worse than best MLP.

Conclusion: unified fixed bases are useful diagnostic components, but the safe
additive residual version is too weak.

## Compute Cost

D18 is more expensive than the MLP baselines.

The expensive part is the RKHS core. With default budgets 64/128/128, the RLS
matrix work is roughly:

`64^2 + 128^2 + 128^2 = 36,864`

per step before kernel costs. The tanh/Fourier basis is cheaper than the RKHS
core but still nontrivial at width 512.

Pruning priorities:

1. reduce or throttle arccosine first;
2. test algebraic budget 96 before touching raw;
3. test tanh width 256;
4. simplify the learned manager into static growth only if early allocation is
   not material.

Do not prune the algebraic bank entirely; it carries much of the digit
retention behavior.

## Assessment

Step 2 is substantially stronger but not solved beyond doubt.

The correct statement is:

> D18 provides a simple non-router learner that beats fair MLP across the
> external digits suite, controlled interaction/frequency, synthetic frequency,
> and synthetic compositional. It does not yet beat fair MLP on synthetic
> polynomial, and every tested polynomial fix either fails the seed-win bar or
> damages digits.

The remaining research gap is not a mixture problem. It is a representation and
adaptation problem: the learner needs a simple, temporally uniform way to expose
context-changing degree-3 algebraic structure without letting that block
interfere with digits and easier polynomial seeds.

## Recommendation

Do not claim 100% Step 2 closure yet.

Promote `d18_core_0p25_basis_0p6` as the current simple compositional candidate
and `d18_core_0p5_basis_0p4` as the current safer digits candidate. Keep
polynomial RLS and unified residual variants as documented negative/diagnostic
ablations, not canonical defaults.

The next mathematically clean direction should be a non-router algebraic block
with internal shrinkage or orthogonalization, not another output router:

- orthogonalize polynomial products against the tanh/Fourier/core predictions;
- update polynomial products on residuals but project out digit-like directions;
- use stable square-root RLS or diagonalized natural-gradient updates instead
  of dense covariance RLS with unstable forgetting;
- expose strict degree-3 products only, since the oracle uses strict triples.

