# Worker E Step 2 External Online Benchmark

Dataset: `sklearn.datasets.load_digits` (bundled 8x8 handwritten digits; no network).
Protocol: 5 seeds, 3000 online training steps, last-500 final window, class_blocked=False.

Methods are architecture-matched: same hidden sizes, step size, sparsity, layer norm, and ObGD bounding. UPGD adds utility-scaled perturbations.

## Aggregate

| Metric | MLP | UPGD | Paired diff | UPGD wins | MLP wins |
|---|---:|---:|---:|---:|---:|
| final_window_mse | 0.0204 +/- 0.0002 | 0.0237 +/- 0.0003 | -0.0033 | 0/5 | 5/5 |
| test_mse | 0.0226 +/- 0.0008 | 0.0254 +/- 0.0002 | -0.0028 | 0/5 | 5/5 |
| final_window_accuracy | 0.9668 +/- 0.0040 | 0.9496 +/- 0.0050 | -0.0172 | 1/5 | 4/5 |
| test_accuracy | 0.9477 +/- 0.0055 | 0.9354 +/- 0.0034 | -0.0122 | 0/5 | 5/5 |

Interpretation: this is real external-dataset evidence for the online MLP-vs-UPGD comparison, not merely a scaffold. It is still small-scale: the dataset is bundled, low-dimensional, and the protocol is a prequential supervised stream rather than a full continual RL setting.
