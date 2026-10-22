# Fully Nested Score-Level Policy Benchmark

- Command: `scripts/grouped_inner_splits.py --config configs/score_level_tabrepo_nested.yaml --repeats 6 --jobs 12 --resume --output-dir analysis-output/v2/score_level_tabrepo_grouped_inner`
- Completed artifact/split rows: 24
- Protocol: selector rows are rebuilt from outer-training datasets only; held-out datasets provide validation/runtime artifacts for target-side selection and test metrics for final evaluation only.

## Summary

| artifact | method | mean_rank | tied_best_rate | mean_total_seconds | mean_rank_regret | mean_normalized_rank_regret | n_splits |
| --- | --- | --- | --- | --- | --- | --- | --- |
| TabRepo | oracle | 12.1175 | 1 | 190.376 | 11.1175 | 0.0073245 | 24 |
| TabRepo | meta_selector_tabrepo_ensemble_k3 | 222.439 | 0.0233179 | 703.23 | 221.439 | 0.145403 | 24 |
| TabRepo | nested_policy_selected | 223.426 | 0.0233179 | 703.23 | 222.426 | 0.146052 | 24 |
| TabRepo | meta_selector_tabrepo_ensemble_k5 | 223.675 | 0.0233179 | 703.23 | 222.675 | 0.146239 | 24 |
| TabRepo | meta_selector_tabrepo_ensemble_k10 | 223.781 | 0.0233179 | 703.23 | 222.781 | 0.146307 | 24 |
| TabRepo | meta_selector_meta_feature_guard_002_k8_rt0 | 253.747 | 0.0416358 | 244.854 | 252.747 | 0.165947 | 24 |
| TabRepo | meta_selector_meta_feature_guard_005_k4_rt0p03 | 254.584 | 0.0402412 | 235.477 | 253.584 | 0.166499 | 24 |
| TabRepo | meta_selector_meta_feature_guard_005 | 254.849 | 0.040519 | 239.935 | 253.849 | 0.166668 | 24 |
| TabRepo | meta_selector_meta_feature_guard_005_k4_rt0 | 255.288 | 0.0407968 | 245.311 | 254.288 | 0.166957 | 24 |
| TabRepo | meta_selector_meta_feature_guard_002_k4_rt0 | 255.481 | 0.0435802 | 248.817 | 254.481 | 0.167095 | 24 |
| TabRepo | meta_selector_meta_feature_guard_010_k4_rt0p1 | 255.756 | 0.0407968 | 218.601 | 254.756 | 0.167278 | 24 |
| TabRepo | meta_selector_meta_feature_guard_005_k12_rt0p03 | 256.273 | 0.0407913 | 246.7 | 255.273 | 0.167602 | 24 |
| TabRepo | meta_selector_meta_feature_guard_002_k12_rt0 | 256.319 | 0.0416301 | 246.043 | 255.319 | 0.167628 | 24 |
| TabRepo | meta_selector_meta_feature_guard_005_k12_rt0 | 256.398 | 0.0405135 | 249.069 | 255.398 | 0.167684 | 24 |
| TabRepo | meta_selector_meta_feature_guard_005_k8_rt0 | 256.56 | 0.0407913 | 250.308 | 255.56 | 0.16779 | 24 |
| TabRepo | meta_selector_meta_feature_guard_010_k12_rt0p03 | 256.688 | 0.0407913 | 251.315 | 255.688 | 0.167875 | 24 |
| TabRepo | meta_selector_meta_feature_guard_010 | 256.701 | 0.0405135 | 251.343 | 255.701 | 0.167882 | 24 |
| TabRepo | meta_selector_meta_feature_guard_020_k4_rt0p1 | 256.741 | 0.0405135 | 251.342 | 255.741 | 0.167909 | 24 |
| TabRepo | meta_selector_meta_feature_guard_010_k8_rt0p1 | 256.793 | 0.038324 | 223.691 | 255.793 | 0.167944 | 24 |
| TabRepo | meta_selector_meta_feature_guard_010_k4_rt0 | 256.965 | 0.0405135 | 250.681 | 255.965 | 0.168055 | 24 |
| TabRepo | meta_selector_meta_feature_guard_002_k4_rt0p03 | 256.941 | 0.0424637 | 184.035 | 255.941 | 0.168055 | 24 |
| TabRepo | meta_selector_meta_feature_guard_010_k12_rt0 | 257.086 | 0.0407913 | 251.405 | 256.086 | 0.168135 | 24 |
| TabRepo | meta_selector_meta_feature_guard_020 | 257.086 | 0.0407913 | 251.405 | 256.086 | 0.168135 | 24 |
| TabRepo | meta_selector_meta_feature_guard_020_k12_rt0 | 257.086 | 0.0407913 | 251.405 | 256.086 | 0.168135 | 24 |
| TabRepo | meta_selector_meta_feature_guard_020_k12_rt0p03 | 257.086 | 0.0407913 | 251.405 | 256.086 | 0.168135 | 24 |
| TabRepo | meta_selector_meta_feature_guard_020_k12_rt0p1 | 257.086 | 0.0407913 | 251.405 | 256.086 | 0.168135 | 24 |
| TabRepo | meta_selector_meta_feature_guard_020_k4_rt0 | 257.086 | 0.0407913 | 251.405 | 256.086 | 0.168135 | 24 |
| TabRepo | meta_selector_meta_feature_guard_020_k4_rt0p03 | 257.086 | 0.0407913 | 251.405 | 256.086 | 0.168135 | 24 |
| TabRepo | meta_selector_meta_feature_guard_020_k8_rt0 | 257.086 | 0.0407913 | 251.405 | 256.086 | 0.168135 | 24 |
| TabRepo | meta_selector_meta_feature_guard_020_k8_rt0p1 | 257.086 | 0.0407913 | 251.405 | 256.086 | 0.168135 | 24 |
| TabRepo | meta_selector_task_adaptive_budgeted | 257.086 | 0.0407913 | 251.405 | 256.086 | 0.168135 | 24 |
| TabRepo | meta_selector_meta_feature_guard_010_k4_rt0p03 | 257.101 | 0.0407913 | 250.66 | 256.101 | 0.168147 | 24 |
| TabRepo | meta_selector_meta_feature_guard_010_k8_rt0 | 257.111 | 0.0407913 | 251.404 | 256.111 | 0.168151 | 24 |
| TabRepo | meta_selector_meta_feature_guard_010_k12_rt0p1 | 257.181 | 0.0396965 | 229.137 | 256.181 | 0.168196 | 24 |
| TabRepo | meta_selector_meta_feature_guard_002 | 257.936 | 0.0347072 | 181.048 | 256.936 | 0.168685 | 24 |
| TabRepo | meta_selector_meta_feature_guard_002_k12_rt0p03 | 258.196 | 0.0360961 | 188.685 | 257.196 | 0.168858 | 24 |
| TabRepo | meta_selector_family_prior_budgeted | 258.344 | 0.0410634 | 357.231 | 257.344 | 0.168964 | 24 |
| TabRepo | meta_selector_family_prior_budgeted_5000s | 258.356 | 0.0410634 | 357.231 | 257.356 | 0.168972 | 24 |
| TabRepo | meta_selector_family_prior_budgeted_1000s | 258.624 | 0.0380244 | 164.212 | 257.624 | 0.169147 | 24 |
| TabRepo | meta_selector_meta_feature_guard_005_k4_rt0p1 | 264.008 | 0.0327578 | 120.435 | 263.008 | 0.172691 | 24 |
| TabRepo | meta_selector_meta_feature_guard_005_k8_rt0p1 | 264.078 | 0.0302628 | 112 | 263.078 | 0.172734 | 24 |
| TabRepo | meta_selector_family_prior_budgeted_500s | 264.702 | 0.032763 | 111.506 | 263.702 | 0.173132 | 24 |
| TabRepo | meta_selector_family_prior_budgeted_100s | 265.296 | 0.0272072 | 32.2899 | 264.296 | 0.173529 | 24 |
| TabRepo | meta_selector_meta_feature_guard_002_k4_rt0p1 | 265.826 | 0.0297018 | 65.3366 | 264.826 | 0.173879 | 24 |
| TabRepo | meta_selector_meta_feature_guard_002_k8_rt0p1 | 266.614 | 0.0286016 | 49.1033 | 265.614 | 0.174399 | 24 |
| TabRepo | meta_selector_meta_feature_guard_005_k12_rt0p1 | 266.779 | 0.0311074 | 99.4241 | 265.779 | 0.174504 | 24 |
| TabRepo | meta_selector_meta_feature_guard_002_k12_rt0p1 | 266.796 | 0.0274904 | 41.8688 | 265.796 | 0.174522 | 24 |
| TabRepo | meta_selector_single | 266.788 | 0.0749534 | 264.408 | 265.788 | 0.174595 | 24 |
| TabRepo | meta_selector_ablate_validation_only | 267.882 | 0.0766258 | 265.109 | 266.882 | 0.17531 | 24 |
| TabRepo | meta_selector_calibrated_budget_10000s | 274.542 | 0.0543809 | 319.698 | 273.542 | 0.179628 | 24 |
| TabRepo | reference_optimism_corrected_task | 283.965 | 0.0352682 | 145.673 | 282.965 | 0.185934 | 24 |
| TabRepo | reference_metafeature_knn_validation | 285.42 | 0.0302798 | 319.713 | 284.42 | 0.186723 | 24 |
| TabRepo | reference_optimism_corrected | 288.294 | 0.0355238 | 127.085 | 287.294 | 0.188769 | 24 |
| TabRepo | reference_optimism_corrected_budget | 288.294 | 0.0355238 | 127.085 | 287.294 | 0.188769 | 24 |
| TabRepo | reference_single_best_global | 379.221 | 0.0210959 | 363.202 | 378.221 | 0.248622 | 24 |
| TabRepo | reference_single_best_task | 382.288 | 0.0202569 | 460.196 | 381.288 | 0.250523 | 24 |
| TabRepo | reference_priors_only | 427.576 | 0.0186122 | 187.776 | 426.576 | 0.280412 | 24 |
| TabRepo | reference_metafeature_knn | 501.339 | 0.0152732 | 272.037 | 500.339 | 0.328838 | 24 |

## Selected Policies

| artifact | selected_policy | n_splits |
| --- | --- | --- |
| TabRepo | meta_selector_tabrepo_ensemble_k3 | 21 |
| TabRepo | meta_selector_tabrepo_ensemble_k10 | 2 |
| TabRepo | meta_selector_tabrepo_ensemble_k5 | 1 |

## Paired Tests

| artifact | method | reference_method | n_splits | mean_rank_difference | mean_normalized_regret_difference | mean_runtime_difference_seconds | rank_wilcoxon_p_value | rank_holm_p_value |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| TabRepo | nested_policy_selected | meta_selector_tabrepo_ensemble_k3 | 24 | 0.98777 | 0.000648758 | 0 | 0.25 | 0.5 |
| TabRepo | nested_policy_selected | meta_selector_task_adaptive_budgeted | 24 | -33.6601 | -0.0220832 | 451.825 | 2.98023e-06 | 1.19209e-05 |
| TabRepo | nested_policy_selected | meta_selector_meta_feature_guard_005 | 24 | -31.4225 | -0.0206153 | 463.296 | 1.3113e-05 | 3.93391e-05 |
| TabRepo | nested_policy_selected | meta_selector_family_prior_budgeted | 24 | -34.9181 | -0.0229122 | 345.999 | 1.66893e-06 | 8.34465e-06 |
| TabRepo | nested_policy_selected | meta_selector_ablate_validation_only | 24 | -44.4557 | -0.029258 | 438.121 | 1.19209e-07 | 8.34465e-07 |
| TabRepo | nested_policy_selected | meta_selector_single | 24 | -43.3614 | -0.0285423 | 438.822 | 1.19209e-07 | 8.34465e-07 |
| TabRepo | nested_policy_selected | meta_selector_tabrepo_ensemble_k5 | 24 | -0.248484 | -0.000187092 | 0 | 0.4452 | 0.5 |
