# Direction 9 Expert Mixture Scaled

Protocol: 10 seeds, 1200 steps, final window 300, Hedge eta=8.0, discount=0.995.

Methods: architecture-matched MLP and UPGD experts plus a discounted Hedge convex mixture. Expert metrics are the same experts used inside the mixture, updated on the same stream.

## digits_class_blocked

Primary metric: `final_window_accuracy`.

| Method | final_window_mse | online_mean_mse | final_window_accuracy | test_accuracy | final weights |
|---|---:|---:|---:|---:|---:|
| mixture | 0.0052 +/- 0.0001 | 0.0089 +/- 0.0001 | 0.9847 +/- 0.0015 | 0.1195 +/- 0.0079 | MLP 1.000, UPGD 0.000 |
| mlp | 0.0052 +/- 0.0001 | 0.0091 +/- 0.0001 | 0.9847 +/- 0.0015 | 0.1195 +/- 0.0079 |  |
| upgd | 0.0208 +/- 0.0007 | 0.0270 +/- 0.0003 | 0.9300 +/- 0.0047 | 0.2111 +/- 0.0177 |  |

Best-expert regret: 0.0000 +/- 0.0000; failures 0/10.

## digits_iid

Primary metric: `final_window_accuracy`.

| Method | final_window_mse | online_mean_mse | final_window_accuracy | test_accuracy | final weights |
|---|---:|---:|---:|---:|---:|
| mixture | 0.0295 +/- 0.0007 | 0.0441 +/- 0.0004 | 0.9237 +/- 0.0055 | 0.9306 +/- 0.0032 | MLP 0.710, UPGD 0.290 |
| mlp | 0.0315 +/- 0.0008 | 0.0449 +/- 0.0005 | 0.9150 +/- 0.0060 | 0.9200 +/- 0.0035 |  |
| upgd | 0.0318 +/- 0.0005 | 0.0514 +/- 0.0004 | 0.9093 +/- 0.0054 | 0.9135 +/- 0.0045 |  |

Best-expert regret: -0.0053 +/- 0.0039; failures 2/10.

## digits_label_drift

Primary metric: `final_window_accuracy`.

| Method | final_window_mse | online_mean_mse | final_window_accuracy | test_accuracy | final weights |
|---|---:|---:|---:|---:|---:|
| mixture | 0.0396 +/- 0.0008 | 0.0566 +/- 0.0006 | 0.8623 +/- 0.0075 | 0.8946 +/- 0.0029 | MLP 1.000, UPGD 0.000 |
| mlp | 0.0396 +/- 0.0008 | 0.0568 +/- 0.0006 | 0.8623 +/- 0.0075 | 0.8946 +/- 0.0029 |  |
| upgd | 0.0569 +/- 0.0008 | 0.0743 +/- 0.0006 | 0.6893 +/- 0.0080 | 0.8345 +/- 0.0073 |  |

Best-expert regret: 0.0000 +/- 0.0000; failures 0/10.

## digits_mask_noise

Primary metric: `final_window_accuracy`.

| Method | final_window_mse | online_mean_mse | final_window_accuracy | test_accuracy | final weights |
|---|---:|---:|---:|---:|---:|
| mixture | 0.0459 +/- 0.0014 | 0.0580 +/- 0.0009 | 0.8003 +/- 0.0117 | 0.8282 +/- 0.0131 | MLP 0.353, UPGD 0.647 |
| mlp | 0.0493 +/- 0.0016 | 0.0606 +/- 0.0008 | 0.7873 +/- 0.0140 | 0.8174 +/- 0.0130 |  |
| upgd | 0.0485 +/- 0.0014 | 0.0627 +/- 0.0010 | 0.7870 +/- 0.0130 | 0.8258 +/- 0.0136 |  |

Best-expert regret: -0.0020 +/- 0.0024; failures 4/10.

## digits_permuted_pixels

Primary metric: `final_window_accuracy`.

| Method | final_window_mse | online_mean_mse | final_window_accuracy | test_accuracy | final weights |
|---|---:|---:|---:|---:|---:|
| mixture | 0.0506 +/- 0.0009 | 0.0613 +/- 0.0005 | 0.7993 +/- 0.0082 | 0.8525 +/- 0.0061 | MLP 0.999, UPGD 0.001 |
| mlp | 0.0506 +/- 0.0009 | 0.0615 +/- 0.0005 | 0.7993 +/- 0.0082 | 0.8525 +/- 0.0061 |  |
| upgd | 0.0579 +/- 0.0012 | 0.0734 +/- 0.0006 | 0.7073 +/- 0.0141 | 0.8104 +/- 0.0116 |  |

Best-expert regret: 0.0000 +/- 0.0000; failures 0/10.

## synthetic_compositional

Primary metric: `final_window_mse`.

| Method | final_window_mse | online_mean_mse | final_window_accuracy | test_accuracy | final weights |
|---|---:|---:|---:|---:|---:|
| mixture | 0.2780 +/- 0.0456 | 0.2207 +/- 0.0199 |  |  | MLP 1.000, UPGD 0.000 |
| mlp | 0.2780 +/- 0.0456 | 0.2200 +/- 0.0199 |  |  |  |
| upgd | 0.5048 +/- 0.1195 | 0.3615 +/- 0.0505 |  |  |  |

Best-expert regret: 0.0000 +/- 0.0000; failures 0/10.

## synthetic_frequency

Primary metric: `final_window_mse`.

| Method | final_window_mse | online_mean_mse | final_window_accuracy | test_accuracy | final weights |
|---|---:|---:|---:|---:|---:|
| mixture | 1.5700 +/- 0.3558 | 1.3909 +/- 0.1414 |  |  | MLP 0.293, UPGD 0.707 |
| mlp | 1.6235 +/- 0.3676 | 1.4328 +/- 0.1476 |  |  |  |
| upgd | 1.6326 +/- 0.3532 | 1.4133 +/- 0.1434 |  |  |  |

Best-expert regret: 0.0125 +/- 0.0041; failures 7/10.

## synthetic_polynomial

Primary metric: `final_window_mse`.

| Method | final_window_mse | online_mean_mse | final_window_accuracy | test_accuracy | final weights |
|---|---:|---:|---:|---:|---:|
| mixture | 0.9583 +/- 0.2066 | 0.9426 +/- 0.0917 |  |  | MLP 0.000, UPGD 1.000 |
| mlp | 1.0311 +/- 0.2152 | 1.0141 +/- 0.0964 |  |  |  |
| upgd | 0.9583 +/- 0.2066 | 0.9419 +/- 0.0916 |  |  |  |

Best-expert regret: 0.0000 +/- 0.0000; failures 0/10.

## Critical Interpretation

This is a small universality probe, not a broad universality result. A useful outcome is a mixture that is close to the stronger expert on each stream without knowing the stream identity. Any gain here should be treated as routing/adaptation evidence until it survives more seeds, larger streams, and additional out-of-class generators.

Failure cases are regimes where the mixture's primary metric trails the better of MLP and UPGD. Positive best-expert regret favors the better expert; zero or negative means the mixture tied or beat both experts on that paired seed.
