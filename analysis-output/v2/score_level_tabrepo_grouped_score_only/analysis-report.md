# Fully Nested Score-Level Policy Benchmark

- Command: `scripts/score_level_policy_pool_ablation.py --nested-output analysis-output/v2/score_level_tabrepo_grouped --output-dir analysis-output/v2/score_level_tabrepo_grouped_score_only`
- Completed artifact/split rows: 24
- Protocol: selector rows are rebuilt from outer-training datasets only; held-out datasets provide validation/runtime artifacts for target-side selection and test metrics for final evaluation only.

## Summary

| artifact | method | mean_rank | tied_best_rate | mean_total_seconds | mean_rank_regret | mean_normalized_rank_regret | n_splits |
| --- | --- | --- | --- | --- | --- | --- | --- |
| TabRepo | oracle | 12.1175 | 1 | 190.376 | 11.1175 | 0.0073245 | 24 |
| TabRepo | meta_selector_meta_feature_guard_002_k8_rt0 | 251.723 | 0.0429968 | 239.932 | 250.723 | 0.164638 | 24 |
| TabRepo | meta_selector_meta_feature_guard_005_k4_rt0p03 | 252.611 | 0.0424358 | 226.426 | 251.611 | 0.165218 | 24 |
| TabRepo | meta_selector_meta_feature_guard_005_k4_rt0 | 252.933 | 0.0435469 | 236.271 | 251.933 | 0.165418 | 24 |
| TabRepo | meta_selector_meta_feature_guard_005 | 253.121 | 0.0418802 | 226.248 | 252.121 | 0.165547 | 24 |
| TabRepo | nested_policy_selected | 253.127 | 0.0410469 | 212.941 | 252.127 | 0.165553 | 24 |
| TabRepo | meta_selector_meta_feature_guard_002_k4_rt0 | 253.211 | 0.0446639 | 245.606 | 252.211 | 0.165604 | 24 |
| TabRepo | meta_selector_meta_feature_guard_010_k4_rt0 | 253.604 | 0.0432691 | 231.919 | 252.604 | 0.16586 | 24 |
| TabRepo | meta_selector_meta_feature_guard_005_k12_rt0 | 253.652 | 0.0432691 | 230.521 | 252.652 | 0.16589 | 24 |
| TabRepo | meta_selector_meta_feature_guard_010_k4_rt0p1 | 253.648 | 0.0418911 | 209.344 | 252.648 | 0.165899 | 24 |
| TabRepo | meta_selector_meta_feature_guard_010_k4_rt0p03 | 253.692 | 0.0432691 | 231.869 | 252.692 | 0.165921 | 24 |
| TabRepo | meta_selector_meta_feature_guard_010 | 253.944 | 0.0432691 | 232.533 | 252.944 | 0.166085 | 24 |
| TabRepo | meta_selector_meta_feature_guard_010_k12_rt0p03 | 253.944 | 0.0432691 | 232.533 | 252.944 | 0.166085 | 24 |
| TabRepo | meta_selector_meta_feature_guard_020_k4_rt0p1 | 253.944 | 0.0432691 | 232.533 | 252.944 | 0.166085 | 24 |
| TabRepo | meta_selector_meta_feature_guard_010_k12_rt0 | 254.028 | 0.0432691 | 232.62 | 253.028 | 0.166141 | 24 |
| TabRepo | meta_selector_meta_feature_guard_020 | 254.028 | 0.0432691 | 232.62 | 253.028 | 0.166141 | 24 |
| TabRepo | meta_selector_meta_feature_guard_020_k12_rt0 | 254.028 | 0.0432691 | 232.62 | 253.028 | 0.166141 | 24 |
| TabRepo | meta_selector_meta_feature_guard_020_k12_rt0p03 | 254.028 | 0.0432691 | 232.62 | 253.028 | 0.166141 | 24 |
| TabRepo | meta_selector_meta_feature_guard_020_k12_rt0p1 | 254.028 | 0.0432691 | 232.62 | 253.028 | 0.166141 | 24 |
| TabRepo | meta_selector_meta_feature_guard_020_k4_rt0 | 254.028 | 0.0432691 | 232.62 | 253.028 | 0.166141 | 24 |
| TabRepo | meta_selector_meta_feature_guard_020_k4_rt0p03 | 254.028 | 0.0432691 | 232.62 | 253.028 | 0.166141 | 24 |
| TabRepo | meta_selector_meta_feature_guard_020_k8_rt0 | 254.028 | 0.0432691 | 232.62 | 253.028 | 0.166141 | 24 |
| TabRepo | meta_selector_meta_feature_guard_020_k8_rt0p1 | 254.028 | 0.0432691 | 232.62 | 253.028 | 0.166141 | 24 |
| TabRepo | meta_selector_task_adaptive_budgeted | 254.028 | 0.0432691 | 232.62 | 253.028 | 0.166141 | 24 |
| TabRepo | meta_selector_meta_feature_guard_010_k8_rt0 | 254.052 | 0.0432691 | 232.619 | 253.052 | 0.166156 | 24 |
| TabRepo | meta_selector_meta_feature_guard_005_k12_rt0p03 | 254.157 | 0.042997 | 227.55 | 253.157 | 0.166222 | 24 |
| TabRepo | meta_selector_meta_feature_guard_005_k8_rt0 | 254.298 | 0.0432691 | 232.659 | 253.298 | 0.166319 | 24 |
| TabRepo | meta_selector_meta_feature_guard_002_k4_rt0p03 | 254.565 | 0.0424466 | 178.445 | 253.565 | 0.166511 | 24 |
| TabRepo | meta_selector_meta_feature_guard_010_k8_rt0p1 | 254.83 | 0.0418802 | 219.908 | 253.83 | 0.166669 | 24 |
| TabRepo | meta_selector_meta_feature_guard_002_k12_rt0 | 255.031 | 0.0429968 | 240.799 | 254.031 | 0.16679 | 24 |
| TabRepo | meta_selector_meta_feature_guard_010_k12_rt0p1 | 255.116 | 0.0421636 | 219.899 | 254.116 | 0.166852 | 24 |
| TabRepo | meta_selector_family_prior_budgeted | 255.313 | 0.0405079 | 363.407 | 254.313 | 0.166982 | 24 |
| TabRepo | meta_selector_family_prior_budgeted_5000s | 255.323 | 0.0405079 | 363.403 | 254.323 | 0.166989 | 24 |
| TabRepo | meta_selector_meta_feature_guard_002 | 257.28 | 0.0385523 | 176.888 | 256.28 | 0.168286 | 24 |
| TabRepo | meta_selector_family_prior_budgeted_1000s | 258.056 | 0.0377467 | 163.741 | 257.056 | 0.168775 | 24 |
| TabRepo | meta_selector_meta_feature_guard_002_k12_rt0p03 | 258.092 | 0.0391135 | 184.101 | 257.092 | 0.168808 | 24 |
| TabRepo | meta_selector_meta_feature_guard_005_k4_rt0p1 | 261.8 | 0.0322018 | 111.858 | 260.8 | 0.171254 | 24 |
| TabRepo | meta_selector_meta_feature_guard_005_k8_rt0p1 | 263.63 | 0.031363 | 105.057 | 262.63 | 0.17246 | 24 |
| TabRepo | meta_selector_meta_feature_guard_002_k4_rt0p1 | 264.121 | 0.029424 | 62.8895 | 263.121 | 0.172777 | 24 |
| TabRepo | meta_selector_family_prior_budgeted_500s | 264.999 | 0.0324852 | 110.92 | 263.999 | 0.173326 | 24 |
| TabRepo | meta_selector_family_prior_budgeted_100s | 265.94 | 0.0272072 | 32.2157 | 264.94 | 0.173963 | 24 |
| TabRepo | meta_selector_single | 266.788 | 0.0749534 | 264.408 | 265.788 | 0.174595 | 24 |
| TabRepo | meta_selector_ablate_validation_only | 267.882 | 0.0766258 | 265.109 | 266.882 | 0.17531 | 24 |
| TabRepo | meta_selector_meta_feature_guard_002_k12_rt0p1 | 268.153 | 0.0274904 | 42.7387 | 267.153 | 0.175422 | 24 |
| TabRepo | meta_selector_meta_feature_guard_002_k8_rt0p1 | 268.182 | 0.0283238 | 44.3649 | 267.182 | 0.175439 | 24 |
| TabRepo | meta_selector_meta_feature_guard_005_k12_rt0p1 | 268.34 | 0.0308183 | 89.1204 | 267.34 | 0.175543 | 24 |
| TabRepo | meta_selector_calibrated_budget_10000s | 273.969 | 0.0516031 | 314.949 | 272.969 | 0.179247 | 24 |
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
| TabRepo | meta_selector_meta_feature_guard_002_k4_rt0 | 4 |
| TabRepo | meta_selector_meta_feature_guard_010_k8_rt0p1 | 4 |
| TabRepo | meta_selector_family_prior_budgeted_5000s | 3 |
| TabRepo | meta_selector_family_prior_budgeted_1000s | 2 |
| TabRepo | meta_selector_family_prior_budgeted_500s | 2 |
| TabRepo | meta_selector_meta_feature_guard_005_k4_rt0 | 2 |
| TabRepo | meta_selector_meta_feature_guard_005_k4_rt0p03 | 2 |
| TabRepo | meta_selector_meta_feature_guard_002_k8_rt0 | 1 |
| TabRepo | meta_selector_meta_feature_guard_005_k12_rt0 | 1 |
| TabRepo | meta_selector_meta_feature_guard_010_k12_rt0p1 | 1 |
| TabRepo | meta_selector_meta_feature_guard_010_k4_rt0p03 | 1 |
| TabRepo | meta_selector_task_adaptive_budgeted | 1 |

## Paired Tests

| artifact | method | reference_method | n_splits | mean_rank_difference | mean_normalized_regret_difference | mean_runtime_difference_seconds | rank_wilcoxon_p_value | rank_holm_p_value |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| TabRepo | nested_policy_selected | meta_selector_task_adaptive_budgeted | 24 | -0.901353 | -0.000587037 | -19.6788 | 0.430433 | 1 |
| TabRepo | nested_policy_selected | meta_selector_meta_feature_guard_005 | 24 | 0.00558812 | 6.35922e-06 | -13.3071 | 0.94927 | 1 |
| TabRepo | nested_policy_selected | meta_selector_family_prior_budgeted | 24 | -2.1862 | -0.00142879 | -150.466 | 0.516761 | 1 |
| TabRepo | nested_policy_selected | meta_selector_ablate_validation_only | 24 | -14.7555 | -0.00975676 | -52.168 | 0.0137793 | 0.0688964 |
| TabRepo | nested_policy_selected | meta_selector_single | 24 | -13.6612 | -0.00904104 | -51.4668 | 0.0269277 | 0.107711 |
