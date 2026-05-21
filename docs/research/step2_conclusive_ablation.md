# Step 2 Conclusive Learner Ablations

Protocol: 1 seed(s), 500 steps, final window 150, benchmarks `controlled_polynomial,controlled_frequency,synthetic_frequency,digits_class_blocked,digits_label_drift`. Positive differences favor the conclusive learner over the same-run best fair MLP.

Important cost note: the canonical runner still executes disabled experts so diagnostics remain paired. The `pruned cost` column is the estimated per-step active-expert fraction for an implementation that physically removes those experts.

## Variant Summary

| Variant | Mean final MSE diff | Worst final MSE diff | Nonnegative datasets | Mean test acc diff | Wall-clock s | Pruned cost |
|---|---:|---:|---:|---:|---:|---:|
| `full` | +0.279349 | +0.000000 | 5/5 | -0.056586 | 37.3 | 1.00x |
| `no_recursive` | +0.257826 | +0.000000 | 5/5 | -0.056586 | 49.8 | 0.91x |
| `no_polynomial` | +0.265672 | +0.000000 | 5/5 | -0.056586 | 124.0 | 0.96x |
| `no_fourier` | +0.188674 | -0.008726 | 4/5 | -0.056586 | 67.7 | 0.97x |
| `no_tanh_random` | +0.279162 | +0.000000 | 5/5 | -0.056586 | 59.2 | 0.95x |
| `no_basis_features` | +0.172482 | -0.017794 | 4/5 | -0.056586 | 273.1 | 0.89x |
| `no_safe_routes` | +0.279416 | +0.000000 | 5/5 | -0.056586 | 174.8 | 1.00x |
| `no_all_convex` | +0.273486 | +0.000000 | 5/5 | -0.056586 | 182.6 | 1.00x |
| `no_all_selector` | +0.273658 | +0.000000 | 5/5 | -0.056586 | 188.1 | 1.00x |
| `no_upgd_dynamic` | +0.279099 | +0.000000 | 5/5 | -0.056586 | 107.3 | 0.81x |
| `no_upgd` | +0.278466 | +0.000000 | 5/5 | -0.056586 | 51.9 | 0.89x |
| `no_dynamic_sparse` | +0.278622 | +0.000000 | 5/5 | -0.056586 | 59.4 | 0.92x |
| `no_class_guard` | +0.279581 | +0.001162 | 5/5 | -0.056586 | 87.5 | 1.00x |
| `no_retention_override` | +0.279349 | +0.000000 | 5/5 | -0.056586 | 57.7 | 1.00x |
| `accuracy_deploy` | +0.279349 | +0.000000 | 5/5 | -0.064007 | 49.7 | 1.00x |
| `no_gate_learning` | +0.279416 | +0.000000 | 5/5 | -0.056586 | 37.3 | 1.00x |
| `short_selector` | +0.277062 | +0.000000 | 5/5 | -0.056586 | 71.5 | 1.00x |
| `long_selector` | +0.285580 | +0.000000 | 5/5 | -0.056586 | 62.9 | 1.00x |
| `low_hedge_eta` | +0.272704 | +0.000000 | 5/5 | -0.057514 | 90.9 | 1.00x |
| `high_hedge_eta` | +0.279634 | +0.000000 | 5/5 | -0.054731 | 80.0 | 1.00x |

## Critical Assessment

Coverage: this matrix ablates every current conclusive-routing expert family, the main route families, the safe-gate update, the digits deployment/retention guards, selector window length, and hedge sharpness. It does not ablate every MLP architecture inside the fair comparator grid because those MLPs are also the baseline bar; disabling them would change the question from component ablation to baseline redefinition.

Overall accuracy: the full learner is nonnegative on 5/5 compact final-window MSE tasks, with mean final-window MSE advantage +0.279349 over the same-run best fair MLP. That is the main positive result. The main negative result is held-out digits accuracy: mean test-accuracy delta is -0.056586, so this compact ablation does not prove held-out classification accuracy superiority.

Most important accuracy component: Fourier features. Removing Fourier drops synthetic frequency from +0.444745 to -0.008726; that is the only individual expert ablation that flips a compact task from win to loss. Removing all explicit basis features is worse still on the same task (-0.017794).

Second-order accuracy component: polynomial features. Removing polynomial features leaves controlled polynomial positive but reduces its margin from +0.496828 to +0.428542. So polynomial features improve the intended algebraic regime, but the recursive/MLP routes still keep the benchmark above MLP.

Recursive features help but are not solely responsible here. Removing recursive features reduces mean final-window advantage from +0.279349 to +0.257826, mainly through controlled frequency and polynomial margin loss. The compact matrix does not show recursive features as essential for every task, but it does show they add useful nonlinear tracking margin.

Simplification candidates: UPGD low-noise and dynamic sparse routing are almost neutral in this matrix. Removing both changes mean final-window advantage from +0.279349 to +0.279099 while reducing estimated pruned expert compute to 0.81x. Fixed random tanh is also nearly neutral here (+0.279162), making a minimal deployment candidate: recursive + polynomial + Fourier + MLP grid, without UPGD, dynamic sparse, or random tanh.

Route simplification: safe recursive interpolation and safe-gate learning are neutral in this compact matrix. Removing safe routes changes mean final-window advantage to +0.279416; freezing gates changes it to +0.279416. This supports keeping safe routes as an optional research path, not as mandatory minimal compute for the current benchmark mix.

Route choices that matter mildly: removing all-convex or all-selector routes lowers mean advantage to +0.273486 and +0.273658. Neither breaks the compact suite, but both reduce margin, so the router benefits from having both soft and hard all-expert choices.

Hyperparameter sensitivity: the long selector window is best in this one-seed compact matrix (+0.285580); short selector and low hedge eta are worse (+0.277062, +0.272704). High hedge eta is slightly better than full on mean final MSE (+0.279634) and less negative on mean test accuracy (-0.054731). These are tuning leads, not settled canonical choices, because they are compact one-seed results.

Deployment/guard outcome: accuracy-based deployment worsens mean held-out digits accuracy from -0.056586 to -0.064007. Disabling the class guard improves class-blocked online MSE but does not improve held-out accuracy in this compact run. The retention override is neutral here. Therefore the remaining unsolved issue is not the online MSE benchmark bar; it is a stronger held-out classifier deployment rule or feature learner that also beats fair MLP on digits test accuracy.

Cost conclusion: diagnostic wall-clock is not a clean compute measure because every variant still executes all experts for paired traces and each subprocess pays JAX compile costs. The useful cost column is estimated pruned expert compute. Removing UPGD + dynamic sparse gives the cleanest savings at 0.81x with negligible accuracy loss; removing all basis features reaches 0.89x but breaks synthetic frequency; removing recursive reaches 0.91x but loses nonlinear margin. A physically pruned runner is still needed for true wall-clock claims.

What remains missing: multi-seed replication of this ablation table; a physical pruned-compute implementation; stronger digits held-out accuracy deployment; and broader harder stateful external benchmarks. The compact evidence is enough to identify which components matter, but not enough to claim every ablated choice is fully optimized.

## Component Deltas

- `no_recursive`: mean final-MSE-diff delta vs full -0.021523; worst-dataset delta +0.000000; mean test-accuracy delta +0.000000. Remove recursive feature expert and all safe recursive routes from routing.
- `no_polynomial`: mean final-MSE-diff delta vs full -0.013677; worst-dataset delta +0.000000; mean test-accuracy delta +0.000000. Remove explicit polynomial basis expert from routing.
- `no_fourier`: mean final-MSE-diff delta vs full -0.090675; worst-dataset delta -0.008726; mean test-accuracy delta +0.000000. Remove explicit Fourier basis expert from routing.
- `no_tanh_random`: mean final-MSE-diff delta vs full -0.000187; worst-dataset delta +0.000000; mean test-accuracy delta +0.000000. Remove fixed random tanh feature expert from routing.
- `no_basis_features`: mean final-MSE-diff delta vs full -0.106867; worst-dataset delta -0.017794; mean test-accuracy delta +0.000000. Remove polynomial, Fourier, and fixed random tanh basis experts.
- `no_safe_routes`: mean final-MSE-diff delta vs full +0.000067; worst-dataset delta +0.000000; mean test-accuracy delta +0.000000. Keep recursive expert available but remove safe recursive interpolation routes.
- `no_all_convex`: mean final-MSE-diff delta vs full -0.005863; worst-dataset delta +0.000000; mean test-accuracy delta +0.000000. Remove the all-expert convex hedge route.
- `no_all_selector`: mean final-MSE-diff delta vs full -0.005691; worst-dataset delta +0.000000; mean test-accuracy delta +0.000000. Remove the all-expert hard selector route.
- `no_upgd_dynamic`: mean final-MSE-diff delta vs full -0.000250; worst-dataset delta +0.000000; mean test-accuracy delta +0.000000. Remove UPGD and dynamic sparse experts from conclusive routing.
- `no_upgd`: mean final-MSE-diff delta vs full -0.000883; worst-dataset delta +0.000000; mean test-accuracy delta +0.000000. Remove UPGD low-noise expert from conclusive routing.
- `no_dynamic_sparse`: mean final-MSE-diff delta vs full -0.000727; worst-dataset delta +0.000000; mean test-accuracy delta +0.000000. Remove dynamic sparse expert from conclusive routing.
- `no_class_guard`: mean final-MSE-diff delta vs full +0.000232; worst-dataset delta +0.001162; mean test-accuracy delta +0.000000. Disable the digits recent-class MLP guard by making its window unreachable.
- `no_retention_override`: mean final-MSE-diff delta vs full +0.000000; worst-dataset delta +0.000000; mean test-accuracy delta +0.000000. Disable held-out class-imbalance recursive retention override.
- `accuracy_deploy`: mean final-MSE-diff delta vs full +0.000000; worst-dataset delta +0.000000; mean test-accuracy delta -0.007421. Use online final-window accuracy, not MSE route, for digits deployment.
- `no_gate_learning`: mean final-MSE-diff delta vs full +0.000067; worst-dataset delta +0.000000; mean test-accuracy delta +0.000000. Freeze safe recursive gates at their zero initialization.
- `short_selector`: mean final-MSE-diff delta vs full -0.002287; worst-dataset delta +0.000000; mean test-accuracy delta +0.000000. Use a short route-loss selector window.
- `long_selector`: mean final-MSE-diff delta vs full +0.006231; worst-dataset delta +0.000000; mean test-accuracy delta +0.000000. Use a long route-loss selector window.
- `low_hedge_eta`: mean final-MSE-diff delta vs full -0.006645; worst-dataset delta +0.000000; mean test-accuracy delta -0.000928. Make convex hedge routes less winner-take-all.
- `high_hedge_eta`: mean final-MSE-diff delta vs full +0.000285; worst-dataset delta +0.000000; mean test-accuracy delta +0.001855. Make convex hedge routes more winner-take-all.

## Dataset Detail

### controlled_frequency

| Variant | Final MSE diff | W/L/T | Test acc diff | Safe route frac | MLP-protected frac |
|---|---:|---:|---:|---:|---:|
| `full` | +0.443427 | 1/0/0 | +nan | 0.144 | 0.300 |
| `no_recursive` | +0.344524 | 1/0/0 | +nan | 0.000 | 0.300 |
| `no_polynomial` | +0.443427 | 1/0/0 | +nan | 0.144 | 0.300 |
| `no_fourier` | +0.443474 | 1/0/0 | +nan | 0.036 | 0.300 |
| `no_tanh_random` | +0.443427 | 1/0/0 | +nan | 0.144 | 0.300 |
| `no_basis_features` | +0.443474 | 1/0/0 | +nan | 0.026 | 0.300 |
| `no_safe_routes` | +0.443474 | 1/0/0 | +nan | 0.000 | 0.300 |
| `no_all_convex` | +0.443427 | 1/0/0 | +nan | 0.144 | 0.300 |
| `no_all_selector` | +0.443032 | 1/0/0 | +nan | 0.700 | 0.300 |
| `no_upgd_dynamic` | +0.443427 | 1/0/0 | +nan | 0.144 | 0.300 |
| `no_upgd` | +0.443427 | 1/0/0 | +nan | 0.144 | 0.300 |
| `no_dynamic_sparse` | +0.443427 | 1/0/0 | +nan | 0.144 | 0.300 |
| `no_class_guard` | +0.443427 | 1/0/0 | +nan | 0.144 | 0.300 |
| `no_retention_override` | +0.443427 | 1/0/0 | +nan | 0.144 | 0.300 |
| `accuracy_deploy` | +0.443427 | 1/0/0 | +nan | 0.144 | 0.300 |
| `no_gate_learning` | +0.443474 | 1/0/0 | +nan | 0.000 | 0.300 |
| `short_selector` | +0.443355 | 1/0/0 | +nan | 0.156 | 0.300 |
| `long_selector` | +0.443474 | 1/0/0 | +nan | 0.000 | 0.300 |
| `low_hedge_eta` | +0.443427 | 1/0/0 | +nan | 0.144 | 0.300 |
| `high_hedge_eta` | +0.443427 | 1/0/0 | +nan | 0.144 | 0.300 |

### controlled_polynomial

| Variant | Final MSE diff | W/L/T | Test acc diff | Safe route frac | MLP-protected frac |
|---|---:|---:|---:|---:|---:|
| `full` | +0.496828 | 1/0/0 | +nan | 0.260 | 0.300 |
| `no_recursive` | +0.487830 | 1/0/0 | +nan | 0.000 | 0.300 |
| `no_polynomial` | +0.428542 | 1/0/0 | +nan | 0.116 | 0.300 |
| `no_fourier` | +0.497047 | 1/0/0 | +nan | 0.260 | 0.300 |
| `no_tanh_random` | +0.496905 | 1/0/0 | +nan | 0.260 | 0.300 |
| `no_basis_features` | +0.425052 | 1/0/0 | +nan | 0.146 | 0.300 |
| `no_safe_routes` | +0.496828 | 1/0/0 | +nan | 0.000 | 0.300 |
| `no_all_convex` | +0.487830 | 1/0/0 | +nan | 0.288 | 0.300 |
| `no_all_selector` | +0.496828 | 1/0/0 | +nan | 0.260 | 0.300 |
| `no_upgd_dynamic` | +0.496965 | 1/0/0 | +nan | 0.258 | 0.300 |
| `no_upgd` | +0.496893 | 1/0/0 | +nan | 0.260 | 0.300 |
| `no_dynamic_sparse` | +0.496901 | 1/0/0 | +nan | 0.260 | 0.300 |
| `no_class_guard` | +0.496828 | 1/0/0 | +nan | 0.260 | 0.300 |
| `no_retention_override` | +0.496828 | 1/0/0 | +nan | 0.260 | 0.300 |
| `accuracy_deploy` | +0.496828 | 1/0/0 | +nan | 0.260 | 0.300 |
| `no_gate_learning` | +0.496828 | 1/0/0 | +nan | 0.000 | 0.300 |
| `short_selector` | +0.495942 | 1/0/0 | +nan | 0.160 | 0.300 |
| `long_selector` | +0.487830 | 1/0/0 | +nan | 0.318 | 0.300 |
| `low_hedge_eta` | +0.487830 | 1/0/0 | +nan | 0.288 | 0.300 |
| `high_hedge_eta` | +0.499900 | 1/0/0 | +nan | 0.164 | 0.300 |

### digits_class_blocked

| Variant | Final MSE diff | W/L/T | Test acc diff | Safe route frac | MLP-protected frac |
|---|---:|---:|---:|---:|---:|
| `full` | +0.000000 | 0/0/1 | -0.087199 | 0.000 | 1.000 |
| `no_recursive` | +0.000000 | 0/0/1 | -0.087199 | 0.000 | 1.000 |
| `no_polynomial` | +0.000000 | 0/0/1 | -0.087199 | 0.000 | 1.000 |
| `no_fourier` | +0.000000 | 0/0/1 | -0.087199 | 0.000 | 1.000 |
| `no_tanh_random` | +0.000000 | 0/0/1 | -0.087199 | 0.000 | 1.000 |
| `no_basis_features` | +0.000000 | 0/0/1 | -0.087199 | 0.000 | 1.000 |
| `no_safe_routes` | +0.000000 | 0/0/1 | -0.087199 | 0.000 | 1.000 |
| `no_all_convex` | +0.000000 | 0/0/1 | -0.087199 | 0.000 | 1.000 |
| `no_all_selector` | +0.000000 | 0/0/1 | -0.087199 | 0.000 | 1.000 |
| `no_upgd_dynamic` | +0.000000 | 0/0/1 | -0.087199 | 0.000 | 1.000 |
| `no_upgd` | +0.000000 | 0/0/1 | -0.087199 | 0.000 | 1.000 |
| `no_dynamic_sparse` | +0.000000 | 0/0/1 | -0.087199 | 0.000 | 1.000 |
| `no_class_guard` | +0.001162 | 1/0/0 | -0.087199 | 0.004 | 0.748 |
| `no_retention_override` | +0.000000 | 0/0/1 | -0.087199 | 0.000 | 1.000 |
| `accuracy_deploy` | +0.000000 | 0/0/1 | -0.087199 | 0.000 | 1.000 |
| `no_gate_learning` | +0.000000 | 0/0/1 | -0.087199 | 0.000 | 1.000 |
| `short_selector` | +0.000000 | 0/0/1 | -0.087199 | 0.000 | 1.000 |
| `long_selector` | +0.000000 | 0/0/1 | -0.087199 | 0.000 | 1.000 |
| `low_hedge_eta` | +0.000000 | 0/0/1 | -0.087199 | 0.000 | 1.000 |
| `high_hedge_eta` | +0.000000 | 0/0/1 | -0.087199 | 0.000 | 1.000 |

### digits_label_drift

| Variant | Final MSE diff | W/L/T | Test acc diff | Safe route frac | MLP-protected frac |
|---|---:|---:|---:|---:|---:|
| `full` | +0.011745 | 1/0/0 | -0.025974 | 0.270 | 0.570 |
| `no_recursive` | +0.012031 | 1/0/0 | -0.025974 | 0.000 | 0.758 |
| `no_polynomial` | +0.011648 | 1/0/0 | -0.025974 | 0.228 | 0.534 |
| `no_fourier` | +0.011574 | 1/0/0 | -0.025974 | 0.088 | 0.426 |
| `no_tanh_random` | +0.011745 | 1/0/0 | -0.025974 | 0.280 | 0.578 |
| `no_basis_features` | +0.011676 | 1/0/0 | -0.025974 | 0.044 | 0.406 |
| `no_safe_routes` | +0.012031 | 1/0/0 | -0.025974 | 0.000 | 0.748 |
| `no_all_convex` | +0.011745 | 1/0/0 | -0.025974 | 0.270 | 0.584 |
| `no_all_selector` | +0.011745 | 1/0/0 | -0.025974 | 0.314 | 0.570 |
| `no_upgd_dynamic` | +0.011745 | 1/0/0 | -0.025974 | 0.270 | 0.570 |
| `no_upgd` | +0.011745 | 1/0/0 | -0.025974 | 0.270 | 0.570 |
| `no_dynamic_sparse` | +0.011745 | 1/0/0 | -0.025974 | 0.270 | 0.570 |
| `no_class_guard` | +0.011745 | 1/0/0 | -0.025974 | 0.270 | 0.570 |
| `no_retention_override` | +0.011745 | 1/0/0 | -0.025974 | 0.270 | 0.570 |
| `accuracy_deploy` | +0.011745 | 1/0/0 | -0.040816 | 0.270 | 0.570 |
| `no_gate_learning` | +0.012031 | 1/0/0 | -0.025974 | 0.000 | 0.748 |
| `short_selector` | +0.011423 | 1/0/0 | -0.025974 | 0.172 | 0.616 |
| `long_selector` | +0.011689 | 1/0/0 | -0.025974 | 0.000 | 0.566 |
| `low_hedge_eta` | +0.011452 | 1/0/0 | -0.027829 | 0.292 | 0.548 |
| `high_hedge_eta` | +0.012121 | 1/0/0 | -0.022263 | 0.168 | 0.540 |

### synthetic_frequency

| Variant | Final MSE diff | W/L/T | Test acc diff | Safe route frac | MLP-protected frac |
|---|---:|---:|---:|---:|---:|
| `full` | +0.444745 | 1/0/0 | +nan | 0.000 | 0.312 |
| `no_recursive` | +0.444748 | 1/0/0 | +nan | 0.000 | 0.312 |
| `no_polynomial` | +0.444745 | 1/0/0 | +nan | 0.000 | 0.312 |
| `no_fourier` | -0.008726 | 0/1/0 | +nan | 0.228 | 0.614 |
| `no_tanh_random` | +0.443734 | 1/0/0 | +nan | 0.000 | 0.310 |
| `no_basis_features` | -0.017794 | 0/1/0 | +nan | 0.226 | 0.770 |
| `no_safe_routes` | +0.444745 | 1/0/0 | +nan | 0.000 | 0.312 |
| `no_all_convex` | +0.424427 | 1/0/0 | +nan | 0.000 | 0.324 |
| `no_all_selector` | +0.416683 | 1/0/0 | +nan | 0.000 | 0.312 |
| `no_upgd_dynamic` | +0.443360 | 1/0/0 | +nan | 0.000 | 0.308 |
| `no_upgd` | +0.440264 | 1/0/0 | +nan | 0.000 | 0.310 |
| `no_dynamic_sparse` | +0.441036 | 1/0/0 | +nan | 0.000 | 0.310 |
| `no_class_guard` | +0.444745 | 1/0/0 | +nan | 0.000 | 0.312 |
| `no_retention_override` | +0.444745 | 1/0/0 | +nan | 0.000 | 0.312 |
| `accuracy_deploy` | +0.444745 | 1/0/0 | +nan | 0.000 | 0.312 |
| `no_gate_learning` | +0.444745 | 1/0/0 | +nan | 0.012 | 0.300 |
| `short_selector` | +0.434592 | 1/0/0 | +nan | 0.000 | 0.356 |
| `long_selector` | +0.484909 | 1/0/0 | +nan | 0.000 | 0.300 |
| `low_hedge_eta` | +0.420810 | 1/0/0 | +nan | 0.000 | 0.316 |
| `high_hedge_eta` | +0.442721 | 1/0/0 | +nan | 0.000 | 0.320 |

## Assessment Template

- Accuracy improvers are variants whose removal causes the final-MSE diff or held-out accuracy diff to fall relative to `full`.
- Accuracy degraders are variants whose removal improves those deltas; they are candidates for simplification or conditional routing.
- Tuned choices in this run are the explicit basis step sizes, tanh width, route switch margin, and class guard max recent classes inherited from the canonical tuned candidate. The remaining ablation knobs are sensitivity checks, not fully optimized hyperparameter sweeps.
- A component can be valuable for Step 2 even when it does not help every single benchmark if it is selected only under regimes where its causal route-loss record supports it.
