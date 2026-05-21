# DiffEML Pure EML Experiment Matrix

This is the evidence plan for showing that DiffEML is a learnable, hard
deployable EML-derived circuit family rather than a soft neural proxy. The
claim metric is always the hardened or packed circuit, never the relaxed soft
model by itself.

## Anti-LARP Contract

Promoted runs must satisfy all of these:

- `gate_mode="eml_template"` so selected gates have executable EML-threshold
  expressions.
- `head_mode` is one of `group_sum`, `class_vote`, or `signed_class_vote`.
  `linear` is allowed only as a baseline or negative-control artifact.
- `packed_eval=True` for selector-based circuits.
- Report `test_soft_accuracy`, `test_hard_accuracy`,
  `packed_hard_test_accuracy`, `soft_vs_hard_gap`, `packed_vs_hard_gap`, and
  `compiled_packed_bytes`.
- Report at least a majority baseline. Image runs also report a same-feature
  MLP baseline column when the helper can run it.
- Reject any promoted run where packed hard accuracy differs from JAX hard
  accuracy.

## Matrix Tiers

| Tier | Purpose | Default smoke coverage | Full coverage | Claim if it passes |
| --- | --- | --- | --- | --- |
| Boolean gates | Prove the selector can learn hard EML-derived gates | XOR, AND, XNOR, OR | all 16 masks | EML templates span and learn every two-input Boolean function |
| Continuous threshold | Show a small continuous function becomes learnable through threshold bits and hard EML circuits | XOR quadrants, diagonal halfspace | plus 4x4 checkerboard | DiffEML can approximate thresholded continuous functions through finite partitions |
| Image smoke | Check real dataset plumbing with pure readouts only | digits, CIFAR smoke | digits, MNIST, CIFAR | Accuracy and compression survive hard packed deployment |

The runner is:

```bash
python "examples/The Alberta Plan/Step2/step2_diffeml_pure_eml_scale_suite.py" \
  --scale smoke \
  --run matrix \
  --output outputs/diffeml_image_demo/pure_eml_scale_suite_smoke.json
```

Runnable cheap evidence:

```bash
python "examples/The Alberta Plan/Step2/step2_diffeml_pure_eml_scale_suite.py" \
  --scale smoke \
  --run boolean
```

The continuous and image modes use the existing DiffEML image-circuit helpers:

```bash
python "examples/The Alberta Plan/Step2/step2_diffeml_pure_eml_scale_suite.py" \
  --scale smoke \
  --run continuous

python "examples/The Alberta Plan/Step2/step2_diffeml_pure_eml_scale_suite.py" \
  --scale smoke \
  --run images
```

`--run images` may require cached or downloadable CIFAR/MNIST data. Dataset
errors are recorded as result rows rather than promoted claims.

## What This Removes

The suite does not promote the current strongest `linear`-head CIFAR result.
That result is useful engineering evidence, but it is not pure EML deployment.
This matrix asks the harder question: after relaxation and hardening, can the
deployed object be just Boolean wiring, selected EML-derived gate masks, and
Boolean/count readout metadata?

The suite also does not treat soft accuracy as success. A large
`soft_vs_hard_gap` means the relaxation is doing work that the deployed circuit
does not retain. That run can guide debugging, but it is not evidence for the
paper claim.

## Scaling Questions

The matrix is designed to expose the real scaling bottleneck:

- If Boolean gates pass but continuous thresholds fail, the issue is topology or
  threshold feature construction.
- If continuous thresholds pass but image smokes fail, the issue is image
  topology, readout allocation, or feature budget.
- If image hard accuracy is good but `class_vote` trails `linear` badly, the
  linear head is still doing non-EML class-evidence mixing.
- If packed bytes are dominated by wiring indices, structured topology and
  implicit wiring are the compression path.

## Acceptance Bar

For a credible "DiffEML works as a learnable EML thing" claim, the first paper
quality artifact should show:

- 16/16 Boolean gates recovered over multiple seeds.
- At least two synthetic continuous-threshold tasks with hard and packed
  accuracy near the Bayes target and small soft-hard gaps.
- Digits and CIFAR smoke results with pure readouts, exact packed/JAX hard
  agreement, byte counts, and same-feature baselines.
- A clear negative-control table showing the gap to `linear` head and MLP
  baselines rather than hiding it.

This does not yet prove competitiveness with DiffLogic. It creates the evidence
discipline required before that comparison is meaningful.
