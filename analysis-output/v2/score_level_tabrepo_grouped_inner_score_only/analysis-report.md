# Fully Nested Score-Level Policy Benchmark

- Command: `scripts/score_level_policy_pool_ablation.py --nested-output analysis-output/v2/score_level_tabrepo_grouped_inner --output-dir analysis-output/v2/score_level_tabrepo_grouped_inner_score_only`
- Completed artifact/split rows: 24
- Protocol: selector rows are rebuilt from outer-training datasets only; held-out datasets provide validation/runtime artifacts for target-side selection and test metrics for final evaluation only.

## Summary

| artifact | method | mean_rank | tied_best_rate | mean_total_seconds | mean_rank_regret | mean_normalized_rank_regret | n_splits |
| --- | --- | --- | --- | --- | --- | --- | --- |
| TabRepo | oracle | 12.1175 | 1 | 190.376 | 11.1175 | 0.0073245 | 24 |
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
| TabRepo | nested_policy_selected | 260.417 | 0.039163 | 229.741 | 259.417 | 0.170324 | 24 |
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
| TabRepo | meta_selector_family_prior_budgeted_5000s | 4 |
| TabRepo | meta_selector_family_prior_budgeted_1000s | 3 |
| TabRepo | meta_selector_meta_feature_guard_002_k4_rt0 | 3 |
| TabRepo | meta_selector_meta_feature_guard_002_k12_rt0p03 | 2 |
| TabRepo | meta_selector_meta_feature_guard_002_k4_rt0p03 | 2 |
| TabRepo | meta_selector_meta_feature_guard_005_k4_rt0 | 2 |
| TabRepo | meta_selector_meta_feature_guard_005_k4_rt0p03 | 2 |
| TabRepo | meta_selector_meta_feature_guard_002_k8_rt0 | 1 |
| TabRepo | meta_selector_meta_feature_guard_005 | 1 |
| TabRepo | meta_selector_meta_feature_guard_005_k12_rt0p03 | 1 |
| TabRepo | meta_selector_meta_feature_guard_005_k4_rt0p1 | 1 |
| TabRepo | meta_selector_meta_feature_guard_010_k8_rt0p1 | 1 |
| TabRepo | meta_selector_single | 1 |

## Paired Tests

| artifact | method | reference_method | n_splits | mean_rank_difference | mean_normalized_regret_difference | mean_runtime_difference_seconds | rank_wilcoxon_p_value | rank_holm_p_value |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| TabRepo | nested_policy_selected | meta_selector_task_adaptive_budgeted | 24 | 3.33089 | 0.00218828 | -21.6644 | 0.329983 | 0.989948 |
| TabRepo | nested_policy_selected | meta_selector_meta_feature_guard_005 | 24 | 5.56849 | 0.00365623 | -10.194 | 0.167807 | 0.839033 |
| TabRepo | nested_policy_selected | meta_selector_family_prior_budgeted | 24 | 2.07296 | 0.00135929 | -127.49 | 0.329983 | 0.989948 |
| TabRepo | nested_policy_selected | meta_selector_ablate_validation_only | 24 | -7.46472 | -0.00498648 | -35.3686 | 0.22919 | 0.916759 |
| TabRepo | nested_policy_selected | meta_selector_single | 24 | -6.37039 | -0.00427077 | -34.6674 | 0.360391 | 0.989948 |
