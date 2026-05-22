# Step 2 No-Gate UPGD OPMNIST Follow-Up

This note tracks the response to the stricter Step 2 constraint: the promoted
learner should be a simple learner, not a portfolio, output router, or
deployment gate. The UPGD-memory path remains useful evidence, but its explicit
UPGD-vs-memory blend is selector-like enough that it should not be treated as
the clean answer under this criterion.

## Candidate Set

The OPMNIST runner now supports fixed-readout UPGD candidates:

- `upgd_structure_linear_h64`
- `upgd_structure_softmax_h64`
- `upgd_structure_linear_h128`
- `upgd_structure_softmax_h128`

These are plain `UPGDLearner.step2_default` instances with
`loss_normalization="target_structure"` and no memory blend, readout gate,
prototype fallback, or MLP fallback.

The runner also includes two rejected calibration probes:

- `upgd_structure_softmax_h{64,128}_smooth{30,40}`: fixed uniform simplex floor.
- `upgd_structure_brier_h{64,128}`: softmax predictions trained with Brier/MSE
  loss through the new `readout_mode="softmax_mse"`.

Both are still single fixed-rule learners; neither is a route or gate.

## 1% True-MNIST OPMNIST Screen

Protocol:

- true OpenML MNIST, canonical split;
- 8/800 completed 60,000-example permutation blocks;
- 480,000 online updates;
- task id hidden from the learner;
- all 800 permutation views evaluated on the full MNIST test split.

Artifact:
`outputs/step2_upgd_memory_opmnist_single_upgd_screen/single_upgd_1pct_eval800views_results.json`.

Best no-gate row from the raw fixed-readout screen:

| Method | Online MSE | Online Acc | Final MSE | Final Acc | All-View Test MSE | All-View Test Acc |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | `0.022595` | `0.902610` | `0.017911` | `0.931200` | `0.106706` | `0.106325` |
| `mlp_h128` | `0.022563` | `0.906850` | `0.016612` | `0.937200` | `0.109681` | `0.105344` |
| `upgd_structure_linear_h128` | `0.020060` | `0.907071` | `0.015402` | `0.935800` | `0.098785` | `0.106841` |
| `upgd_structure_softmax_h128` | `0.012348` | `0.921777` | `0.008232` | `0.949000` | `0.153436` | `0.110784` |

Interpretation:

- `upgd_structure_softmax_h128` is the best online/final no-gate learner and
  beats the fair MLPs on all online/final metrics plus all-view test accuracy.
- `upgd_structure_linear_h128` is the retained-MSE probe: it beats the fair MLPs
  on online MSE, final-window MSE, all-view test MSE, and all-view test
  accuracy, but it narrowly loses final-window accuracy.
- The raw screen therefore finds a real no-gate UPGD frontier, not a decisive
  universal row.

## Rejected Fixed Calibration Probes

Fixed uniform smoothing does not close the retained-MSE gap.

Artifact:
`outputs/step2_upgd_memory_opmnist_single_upgd_screen/smoothed_single_upgd_1pct_eval800views_results.json`.

Best smoothed row:

| Method | Online MSE Diff | Final MSE Diff | Test MSE Diff | Test Acc Diff |
|---|---:|---:|---:|---:|
| `upgd_structure_softmax_h128_smooth30` | `+0.004642` | `+0.002211` | `-0.022988` | `+0.002344` |

Positive MSE diffs favor UPGD. Smoothing `0.30` preserves online/final MSE but
still loses all-view test MSE. Smoothing `0.40` further reduces overconfidence
but makes online/final MSE negative.

Softmax-Brier also does not close retained MSE.

Artifact:
`outputs/step2_upgd_memory_opmnist_single_upgd_screen/brier_single_upgd_1pct_eval800views_results.json`.

| Method | Online MSE Diff | Final MSE Diff | Test MSE Diff | Test Acc Diff |
|---|---:|---:|---:|---:|
| `upgd_structure_brier_h128` | `+0.010506` | `+0.009019` | `-0.043516` | `+0.000977` |

The Brier learner is excellent for online/final tracking but remains
overconfident enough on unseen permutation views to lose all-view test MSE.

## Fixed-Temperature Readout Probe

The next no-gate calibration probe keeps the underlying learner exactly fixed:
one `UPGDLearner.step2_default(..., readout_mode="softmax_ce")` is updated
online, and fixed probability-temperature readouts are scored from the same
raw predictions. This is not a route or deployment gate because every
temperature sees the same learner state and the same predictions; it is a
constant readout transform.

The H128 full-scale temperature sweep ran under:

`outputs/step2_upgd_memory_opmnist_single_upgd_full/softmax_h128_temperature_sweep_800task_*`.

It did not close the all-metric bar.

Artifact:
`outputs/step2_upgd_memory_opmnist_single_upgd_full/softmax_h128_temperature_sweep_800task_results.json`.

| Method | Online MSE | Online Acc | Final MSE | Final Acc | All-View Test MSE | All-View Test Acc |
|---|---:|---:|---:|---:|---:|---:|
| `upgd_structure_softmax_h128_temp1p0` | `0.018764` | `0.883596` | `0.013905` | `0.915400` | `0.149146` | `0.134131` |
| `upgd_structure_softmax_h128_temp2p0` | `0.017057` | `0.883596` | `0.012948` | `0.915400` | `0.129353` | `0.134131` |
| `upgd_structure_softmax_h128_temp3p0` | `0.017729` | `0.883596` | `0.013962` | `0.915400` | `0.115265` | `0.134131` |
| `upgd_structure_softmax_h128_temp4p0` | `0.020741` | `0.883596` | `0.017177` | `0.915400` | `0.105994` | `0.134131` |
| `upgd_structure_softmax_h128_temp6p0` | `0.032297` | `0.883596` | `0.029314` | `0.915400` | `0.096564` | `0.134131` |
| `upgd_structure_softmax_h128_temp8p0` | `0.044718` | `0.883596` | `0.042398` | `0.915400` | `0.092880` | `0.134131` |
| `upgd_structure_softmax_h128_temp12p0` | `0.060907` | `0.883596` | `0.059458` | `0.915400` | `0.090518` | `0.134131` |

Compared with the same-run best fair MLP from
`single_upgd_h128_800task_eval800views_results.json`, H128 `temp4p0` beats
online MSE, online accuracy, final-window MSE, final-window accuracy, and
all-view test accuracy, but still loses all-view test MSE by `0.005362`.
H128 `temp6p0` closes retained all-view test MSE but loses online and
final-window MSE. H128 therefore confirms the mechanism but is not the
promoted no-gate OPMNIST solution.

The first H256 1% screen is positive enough to justify a full H256 scale run.

Artifact:
`outputs/step2_upgd_memory_opmnist_single_upgd_full/softmax_h256_temperature_sweep_1pct_eval800views_results.json`.

Protocol:

- true OpenML MNIST, canonical split;
- 8/800 completed 60,000-example permutation blocks;
- 480,000 online updates;
- task id hidden from the learner;
- all 800 permutation views evaluated on the full MNIST test split.

H256 temperature rows:

| Method | Online MSE | Online Acc | Final MSE | Final Acc | All-View Test MSE | All-View Test Acc |
|---|---:|---:|---:|---:|---:|---:|
| `upgd_structure_softmax_h256_temp1p0` | `0.012387` | `0.921562` | `0.008089` | `0.949400` | `0.152250` | `0.111978` |
| `upgd_structure_softmax_h256_temp2p0` | `0.011616` | `0.921562` | `0.007464` | `0.949400` | `0.130928` | `0.111978` |
| `upgd_structure_softmax_h256_temp4p0` | `0.018001` | `0.921562` | `0.011528` | `0.949400` | `0.106252` | `0.111978` |
| `upgd_structure_softmax_h256_temp6p0` | `0.032436` | `0.921562` | `0.024718` | `0.949400` | `0.097042` | `0.111978` |
| `upgd_structure_softmax_h256_temp8p0` | `0.045937` | `0.921562` | `0.039153` | `0.949400` | `0.093543` | `0.111978` |
| `upgd_structure_softmax_h256_temp12p0` | `0.062242` | `0.921562` | `0.057778` | `0.949400` | `0.091227` | `0.111978` |

Against the same 1% fair MLP baselines above, `temp4p0` is the first row in
this no-gate pass that beats the best MLP on all six reported metrics. The
margin on retained all-view test MSE is small, so it is only a scale candidate,
not a promoted result.

The full H256 run is now the main successor experiment:

```bash
python "examples/The Alberta Plan/Step2/step2_upgd_temperature_sweep_opmnist.py" \
  --allow-openml-download \
  --hidden-size 256 \
  --temperatures 1 2 3 4 5 6 8 \
  --result-prefix softmax_h256_temperature_sweep_800task_eval800views
```

It completed and produced the first full-scale all-metric no-gate OPMNIST win.

Artifacts:

- `outputs/step2_upgd_memory_opmnist_single_upgd_full/softmax_h256_temperature_sweep_800task_eval800views_results.json`
- `outputs/step2_canonical/opmnist_softmax_h256_temp_sweep_800task_eval800views_results.json`

Final H256 metrics:

| Method | Online MSE | Online Acc | Final MSE | Final Acc | All-View Test MSE | All-View Test Acc |
|---|---:|---:|---:|---:|---:|---:|
| `upgd_structure_softmax_h256_temp1p0` | `0.018379` | `0.887991` | `0.013598` | `0.918200` | `0.145980` | `0.160753` |
| `upgd_structure_softmax_h256_temp2p0` | `0.016554` | `0.887991` | `0.012458` | `0.918200` | `0.127756` | `0.160753` |
| `upgd_structure_softmax_h256_temp3p0` | `0.016648` | `0.887991` | `0.012763` | `0.918200` | `0.114332` | `0.160753` |
| `upgd_structure_softmax_h256_temp4p0` | `0.018839` | `0.887991` | `0.014898` | `0.918200` | `0.105102` | `0.160753` |
| `upgd_structure_softmax_h256_temp5p0` | `0.023330` | `0.887991` | `0.019375` | `0.918200` | `0.099118` | `0.160753` |
| `upgd_structure_softmax_h256_temp6p0` | `0.029410` | `0.887991` | `0.025626` | `0.918200` | `0.095392` | `0.160753` |
| `upgd_structure_softmax_h256_temp8p0` | `0.042020` | `0.887991` | `0.038897` | `0.918200` | `0.091661` | `0.160753` |

Best same-run fair MLP reference from
`single_upgd_h128_800task_eval800views_results.json`:

| Metric | Best MLP | Value |
|---|---|---:|
| Online MSE | `mlp_h128` | `0.028855` |
| Online Acc | `mlp_h128` | `0.864847` |
| Final MSE | `mlp_h128` | `0.021230` |
| Final Acc | `mlp_h128` | `0.909800` |
| All-View Test MSE | `mlp_h128` | `0.100632` |
| All-View Test Acc | `mlp_h128` | `0.130353` |

The promoted fixed-temperature row is `upgd_structure_softmax_h256_temp5p0`.
Differences below are signed so positive favors UPGD:

| Metric | Difference |
|---|---:|
| Online MSE | `+0.005525` |
| Online Acc | `+0.023143` |
| Final MSE | `+0.001856` |
| Final Acc | `+0.008400` |
| All-View Test MSE | `+0.001514` |
| All-View Test Acc | `+0.030400` |

Interpretation:

- The retained-MSE margin is small but positive at the full 800-task,
  48,000,000-update OpenML MNIST scale.
- The learner is still simple under the stricter constraint: one UPGD network,
  one fixed readout temperature, no route, no deployment gate, no prototype
  fallback, and no MLP fallback.
- This closes the current full-scale OPMNIST empirical gap for the no-gate
  Step 2 path. It remains empirical evidence, not a mathematical universality
  theorem.

## Full 800-Task Run

The full-scale clean run completed:

```bash
python "examples/The Alberta Plan/Step2/step2_upgd_memory_opmnist.py" \
  --mnist-published-scale \
  --allow-openml-download \
  --n-seeds 1 \
  --final-window 5000 \
  --chunk-size 60000 \
  --include-single-upgd \
  --evaluate-all-permutation-views \
  --max-test-permutation-views 800 \
  --only-methods mlp_h64,mlp_h128,upgd_structure_linear_h128,upgd_structure_softmax_h128 \
  --output-dir outputs/step2_upgd_memory_opmnist_single_upgd_full \
  --result-prefix single_upgd_h128_800task_eval800views \
  --note-path docs/research/step2_upgd_memory_opmnist_single_upgd_h128_800task_eval800views.md \
  --status-path outputs/step2_upgd_memory_opmnist_single_upgd_full/single_upgd_h128_800task_eval800views_status.json \
  --force-restart
```

Artifact:
`outputs/step2_upgd_memory_opmnist_single_upgd_full/single_upgd_h128_800task_eval800views_results.json`.

Protocol:

- true OpenML MNIST, canonical split;
- 800/800 completed 60,000-example permutation blocks;
- 48,000,000 online updates;
- task id hidden from the learner;
- all 800 permutation views evaluated on the full MNIST test split;
- `matches_dohare_opmnist_core_protocol=true`;
- `matches_dohare_opmnist_published_task_count=true`.

Final metrics:

| Method | Online MSE | Online Acc | Final MSE | Final Acc | All-View Test MSE | All-View Test Acc |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | `0.033825` | `0.839150` | `0.026268` | `0.892600` | `0.103178` | `0.127530` |
| `mlp_h128` | `0.028855` | `0.864847` | `0.021230` | `0.909800` | `0.100632` | `0.130353` |
| `upgd_structure_linear_h128` | `0.029401` | `0.857807` | `0.021811` | `0.906200` | `0.100694` | `0.130291` |
| `upgd_structure_softmax_h128` | `0.018753` | `0.883623` | `0.014868` | `0.908600` | `0.148280` | `0.134712` |

Differences versus the same-run best fair MLP:

| Candidate | Online MSE | Online Acc | Final MSE | Final Acc | Test MSE | Test Acc |
|---|---:|---:|---:|---:|---:|---:|
| `upgd_structure_linear_h128` | `-0.000546` | `-0.007040` | `-0.000581` | `-0.003600` | `-0.000062` | `-0.000062` |
| `upgd_structure_softmax_h128` | `+0.010102` | `+0.018775` | `+0.006362` | `-0.001200` | `-0.047648` | `+0.004359` |

Positive values favor UPGD.

Conclusion:

- `upgd_structure_softmax_h128` is a strong simple no-gate online learner. It
  closes the full-scale OPMNIST final-window MSE gap and improves all-view test
  accuracy, but it is not an all-metric solution because its probability
  distribution is badly overconfident under all-view test MSE and it narrowly
  misses final-window accuracy.
- `upgd_structure_linear_h128` is not promotable. At full scale it is a slight
  loss/tie against `mlp_h128` across every reported metric.
- The honest current statement is therefore: full published-scale OPMNIST gives
  positive no-gate evidence for softmax UPGD on online tracking and accuracy,
  but no simple no-gate learner in this pass universally beats the best fair
  MLP on all retained-view metrics.

Promotion bar for any successor:

- `upgd_structure_softmax_h128` is promotable only if the full 800-task result
  retains its online/final advantage and no longer loses the retained all-view
  test metrics after all 800 permutation views have been observed.
- `upgd_structure_linear_h128` is promotable only if it keeps the retained-MSE
  advantage and closes the final-window accuracy miss at full scale.

If neither row satisfies those conditions, the honest conclusion is that simple
target-structure UPGD is the best no-gate Step 2 family found so far, but
OPMNIST retained-view generalization remains only partially closed.
