# Fully Nested Score-Level Policy Benchmark

- Command: `scripts/score_level_policy_benchmark.py --config configs/score_level_tabarena_extended_pool.yaml --jobs 10`
- Completed artifact/split rows: 100
- Protocol: selector rows are rebuilt from outer-training datasets only; held-out datasets provide validation/runtime artifacts for target-side selection and test metrics for final evaluation only.

## Summary

| artifact | method | mean_rank | tied_best_rate | mean_total_seconds | mean_rank_regret | mean_normalized_rank_regret | n_splits |
| --- | --- | --- | --- | --- | --- | --- | --- |
| TabArena | oracle | 1.01684 | 1 | 46563.3 | 0.0168373 | 0.000391565 | 100 |
| TabArena | meta_selector_family_prior_budgeted | 8.72219 | 0.134403 | 4385.25 | 7.72219 | 0.179586 | 100 |
| TabArena | meta_selector_meta_feature_guard_002 | 8.80611 | 0.136595 | 1979.09 | 7.80611 | 0.181537 | 100 |
| TabArena | meta_selector_meta_feature_guard_002_k8_rt0 | 8.82542 | 0.136014 | 2496.94 | 7.82542 | 0.181986 | 100 |
| TabArena | meta_selector_meta_feature_guard_002_k12_rt0p03 | 8.85224 | 0.136944 | 1937.07 | 7.85224 | 0.18261 | 100 |
| TabArena | extended_descriptor_guard_002 | 8.85436 | 0.137833 | 1919.92 | 7.85436 | 0.182659 | 100 |
| TabArena | meta_selector_meta_feature_guard_002_k12_rt0 | 8.87148 | 0.136043 | 2440.06 | 7.87148 | 0.183058 | 100 |
| TabArena | meta_selector_meta_feature_guard_005 | 8.88965 | 0.135715 | 2169.6 | 7.88965 | 0.18348 | 100 |
| TabArena | meta_selector_meta_feature_guard_005_k12_rt0p03 | 8.94328 | 0.136421 | 2137.87 | 7.94328 | 0.184727 | 100 |
| TabArena | meta_selector_meta_feature_guard_005_k8_rt0 | 8.95012 | 0.132452 | 2597.99 | 7.95012 | 0.184886 | 100 |
| TabArena | meta_selector_meta_feature_guard_010_k8_rt0p1 | 8.95408 | 0.135969 | 1930.36 | 7.95408 | 0.184979 | 100 |
| TabArena | extended_descriptor_guard_005 | 8.97325 | 0.13538 | 2118.31 | 7.97325 | 0.185424 | 100 |
| TabArena | meta_selector_meta_feature_guard_005_k4_rt0 | 8.97521 | 0.132412 | 2641.13 | 7.97521 | 0.18547 | 100 |
| TabArena | meta_selector_meta_feature_guard_005_k4_rt0p03 | 8.98371 | 0.133085 | 2275.86 | 7.98371 | 0.185668 | 100 |
| TabArena | meta_selector_meta_feature_guard_002_k4_rt0 | 8.98438 | 0.130551 | 2491.71 | 7.98438 | 0.185683 | 100 |
| TabArena | meta_selector_meta_feature_guard_010_k12_rt0p1 | 8.98883 | 0.136262 | 1928.41 | 7.98883 | 0.185787 | 100 |
| TabArena | meta_selector_meta_feature_guard_005_k12_rt0p1 | 9.0096 | 0.137078 | 1472.46 | 8.0096 | 0.18627 | 100 |
| TabArena | meta_selector_meta_feature_guard_010_k4_rt0p1 | 9.01469 | 0.133823 | 1997.76 | 8.01469 | 0.186388 | 100 |
| TabArena | meta_selector_meta_feature_guard_005_k12_rt0 | 9.04206 | 0.131359 | 2603.89 | 8.04206 | 0.187025 | 100 |
| TabArena | meta_selector_meta_feature_guard_002_k4_rt0p03 | 9.05034 | 0.130506 | 1911.43 | 8.05034 | 0.187217 | 100 |
| TabArena | meta_selector_meta_feature_guard_005_k8_rt0p1 | 9.05922 | 0.136085 | 1403.1 | 8.05922 | 0.187424 | 100 |
| TabArena | meta_selector_calibrated_budget_10000s | 9.09374 | 0.125814 | 4433.2 | 8.09374 | 0.188227 | 100 |
| TabArena | meta_selector_meta_feature_guard_010_k4_rt0p03 | 9.11217 | 0.129676 | 2636.73 | 8.11217 | 0.188655 | 100 |
| TabArena | meta_selector_meta_feature_guard_010_k4_rt0 | 9.11773 | 0.129242 | 2697.99 | 8.11773 | 0.188784 | 100 |
| TabArena | meta_selector_meta_feature_guard_010_k8_rt0 | 9.14252 | 0.129222 | 2717.64 | 8.14252 | 0.189361 | 100 |
| TabArena | meta_selector_meta_feature_guard_010 | 9.14417 | 0.129385 | 2666.62 | 8.14417 | 0.189399 | 100 |
| TabArena | meta_selector_meta_feature_guard_010_k12_rt0p03 | 9.14708 | 0.129395 | 2716.37 | 8.14708 | 0.189467 | 100 |
| TabArena | meta_selector_meta_feature_guard_020 | 9.15195 | 0.129261 | 2738.01 | 8.15195 | 0.18958 | 100 |
| TabArena | meta_selector_meta_feature_guard_020_k12_rt0 | 9.15195 | 0.129261 | 2738.01 | 8.15195 | 0.18958 | 100 |
| TabArena | meta_selector_meta_feature_guard_020_k12_rt0p03 | 9.15195 | 0.129261 | 2738.01 | 8.15195 | 0.18958 | 100 |
| TabArena | meta_selector_meta_feature_guard_020_k4_rt0 | 9.15195 | 0.129261 | 2738.01 | 8.15195 | 0.18958 | 100 |
| TabArena | meta_selector_meta_feature_guard_020_k4_rt0p03 | 9.15195 | 0.129261 | 2738.01 | 8.15195 | 0.18958 | 100 |
| TabArena | meta_selector_meta_feature_guard_020_k8_rt0 | 9.15195 | 0.129261 | 2738.01 | 8.15195 | 0.18958 | 100 |
| TabArena | meta_selector_task_adaptive_budgeted | 9.15195 | 0.129261 | 2738.01 | 8.15195 | 0.18958 | 100 |
| TabArena | meta_selector_meta_feature_guard_010_k12_rt0 | 9.15427 | 0.129205 | 2722.31 | 8.15427 | 0.189634 | 100 |
| TabArena | meta_selector_meta_feature_guard_020_k12_rt0p1 | 9.15427 | 0.129205 | 2722.31 | 8.15427 | 0.189634 | 100 |
| TabArena | meta_selector_meta_feature_guard_020_k8_rt0p1 | 9.15427 | 0.129205 | 2722.31 | 8.15427 | 0.189634 | 100 |
| TabArena | meta_selector_meta_feature_guard_020_k4_rt0p1 | 9.15586 | 0.129211 | 2712.09 | 8.15586 | 0.189671 | 100 |
| TabArena | extended_metafeature_knn_validation | 9.16142 | 0.101091 | 76262.6 | 8.16142 | 0.1898 | 100 |
| TabArena | nested_policy_selected | 9.16522 | 0.123119 | 10338.3 | 8.16522 | 0.189889 | 100 |
| TabArena | meta_selector_meta_feature_guard_005_k4_rt0p1 | 9.42013 | 0.124986 | 1230.08 | 8.42013 | 0.195817 | 100 |
| TabArena | meta_selector_meta_feature_guard_002_k12_rt0p1 | 9.43672 | 0.133797 | 969.764 | 8.43672 | 0.196203 | 100 |
| TabArena | meta_selector_meta_feature_guard_002_k8_rt0p1 | 9.5534 | 0.132706 | 738.803 | 8.5534 | 0.198916 | 100 |
| TabArena | extended_metafeature_knn | 9.81721 | 0.0957331 | 54136.8 | 8.81721 | 0.205051 | 100 |
| TabArena | meta_selector_family_prior_budgeted_500s | 9.83813 | 0.150494 | 74.724 | 8.83813 | 0.205538 | 100 |
| TabArena | meta_selector_meta_feature_guard_002_k4_rt0p1 | 9.83978 | 0.129673 | 616.196 | 8.83978 | 0.205576 | 100 |
| TabArena | meta_selector_family_prior_budgeted_1000s | 9.99849 | 0.136353 | 237.802 | 8.99849 | 0.209267 | 100 |
| TabArena | meta_selector_family_prior_budgeted_5000s | 10.4208 | 0.129506 | 1733.97 | 9.42076 | 0.219087 | 100 |
| TabArena | meta_selector_family_prior_budgeted_100s | 10.931 | 0.106931 | 31.3983 | 9.93098 | 0.230953 | 100 |
| TabArena | meta_selector_ablate_validation_only | 13.1123 | 0.0980355 | 78032.4 | 12.1123 | 0.281681 | 100 |
| TabArena | meta_selector_single | 13.1123 | 0.0980355 | 78032.4 | 12.1123 | 0.281681 | 100 |

## Selected Policies

| artifact | selected_policy | n_splits |
| --- | --- | --- |
| TabArena | meta_selector_family_prior_budgeted | 29 |
| TabArena | meta_selector_meta_feature_guard_002_k8_rt0 | 12 |
| TabArena | extended_metafeature_knn | 8 |
| TabArena | meta_selector_meta_feature_guard_002 | 8 |
| TabArena | meta_selector_meta_feature_guard_002_k4_rt0 | 7 |
| TabArena | extended_descriptor_guard_002 | 5 |
| TabArena | extended_metafeature_knn_validation | 5 |
| TabArena | meta_selector_meta_feature_guard_002_k12_rt0 | 3 |
| TabArena | meta_selector_meta_feature_guard_002_k4_rt0p03 | 3 |
| TabArena | meta_selector_meta_feature_guard_005_k8_rt0 | 3 |
| TabArena | meta_selector_meta_feature_guard_002_k12_rt0p03 | 2 |
| TabArena | meta_selector_meta_feature_guard_005_k4_rt0 | 2 |
| TabArena | meta_selector_meta_feature_guard_010_k4_rt0p1 | 2 |
| TabArena | extended_descriptor_guard_005 | 1 |
| TabArena | meta_selector_calibrated_budget_10000s | 1 |
| TabArena | meta_selector_meta_feature_guard_002_k12_rt0p1 | 1 |
| TabArena | meta_selector_meta_feature_guard_002_k4_rt0p1 | 1 |
| TabArena | meta_selector_meta_feature_guard_002_k8_rt0p1 | 1 |
| TabArena | meta_selector_meta_feature_guard_005_k4_rt0p03 | 1 |
| TabArena | meta_selector_meta_feature_guard_005_k8_rt0p1 | 1 |
| TabArena | meta_selector_meta_feature_guard_010 | 1 |
| TabArena | meta_selector_meta_feature_guard_010_k12_rt0p1 | 1 |
| TabArena | meta_selector_meta_feature_guard_010_k4_rt0p03 | 1 |
| TabArena | meta_selector_task_adaptive_budgeted | 1 |

## Paired Tests

| artifact | method | reference_method | n_splits | mean_rank_difference | mean_normalized_regret_difference | mean_runtime_difference_seconds | rank_wilcoxon_p_value | rank_holm_p_value |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| TabArena | nested_policy_selected | meta_selector_task_adaptive_budgeted | 100 | 0.0132697 | 0.000308597 | 7600.32 | 0.107301 | 0.107301 |
| TabArena | nested_policy_selected | meta_selector_meta_feature_guard_005 | 100 | 0.275565 | 0.00640848 | 8168.73 | 0.0414681 | 0.0829361 |
| TabArena | nested_policy_selected | meta_selector_family_prior_budgeted | 100 | 0.443032 | 0.0103031 | 5953.07 | 3.7928e-05 | 0.000113784 |
| TabArena | nested_policy_selected | meta_selector_ablate_validation_only | 100 | -3.94706 | -0.091792 | -67694 | 6.31161e-18 | 3.1558e-17 |
| TabArena | nested_policy_selected | meta_selector_single | 100 | -3.94706 | -0.091792 | -67694 | 6.31161e-18 | 3.1558e-17 |
