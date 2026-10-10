# Fully Nested Score-Level Policy Benchmark

- Command: `scripts/score_level_policy_benchmark.py --config configs/score_level_tabarena_attribution_inner_shared.yaml --jobs 12`
- Completed artifact/split rows: 100
- Protocol: selector rows are rebuilt from outer-training datasets only; held-out datasets provide validation/runtime artifacts for target-side selection and test metrics for final evaluation only.

## Summary

| artifact | method | mean_rank | tied_best_rate | mean_total_seconds | mean_rank_regret | mean_normalized_rank_regret | n_splits |
| --- | --- | --- | --- | --- | --- | --- | --- |
| TabArena | oracle | 1.01684 | 1 | 46563.3 | 0.0168373 | 0.000391565 | 100 |
| TabArena | reference_optimism_corrected_task | 8.52642 | 0.177498 | 9454.33 | 7.52642 | 0.175033 | 100 |
| TabArena | meta_selector_family_prior_budgeted | 8.72219 | 0.134403 | 4385.25 | 7.72219 | 0.179586 | 100 |
| TabArena | meta_selector_meta_feature_guard_002 | 8.80611 | 0.136595 | 1979.09 | 7.80611 | 0.181537 | 100 |
| TabArena | meta_selector_meta_feature_guard_002_k8_rt0 | 8.82542 | 0.136014 | 2496.94 | 7.82542 | 0.181986 | 100 |
| TabArena | meta_selector_meta_feature_guard_002_k12_rt0p03 | 8.85224 | 0.136944 | 1937.07 | 7.85224 | 0.18261 | 100 |
| TabArena | meta_selector_meta_feature_guard_002_k12_rt0 | 8.87148 | 0.136043 | 2440.06 | 7.87148 | 0.183058 | 100 |
| TabArena | reference_optimism_corrected | 8.87647 | 0.17745 | 9878.38 | 7.87647 | 0.183174 | 100 |
| TabArena | meta_selector_meta_feature_guard_005 | 8.88965 | 0.135715 | 2169.6 | 7.88965 | 0.18348 | 100 |
| TabArena | meta_selector_meta_feature_guard_005_k12_rt0p03 | 8.94328 | 0.136421 | 2137.87 | 7.94328 | 0.184727 | 100 |
| TabArena | meta_selector_meta_feature_guard_005_k8_rt0 | 8.95012 | 0.132452 | 2597.99 | 7.95012 | 0.184886 | 100 |
| TabArena | meta_selector_meta_feature_guard_010_k8_rt0p1 | 8.95408 | 0.135969 | 1930.36 | 7.95408 | 0.184979 | 100 |
| TabArena | meta_selector_meta_feature_guard_005_k4_rt0 | 8.97521 | 0.132412 | 2641.13 | 7.97521 | 0.18547 | 100 |
| TabArena | meta_selector_meta_feature_guard_005_k4_rt0p03 | 8.98371 | 0.133085 | 2275.86 | 7.98371 | 0.185668 | 100 |
| TabArena | meta_selector_meta_feature_guard_002_k4_rt0 | 8.98438 | 0.130551 | 2491.71 | 7.98438 | 0.185683 | 100 |
| TabArena | meta_selector_meta_feature_guard_010_k12_rt0p1 | 8.98883 | 0.136262 | 1928.41 | 7.98883 | 0.185787 | 100 |
| TabArena | nested_policy_selected | 8.99792 | 0.132554 | 2709.08 | 7.99792 | 0.185998 | 100 |
| TabArena | meta_selector_meta_feature_guard_005_k12_rt0p1 | 9.0096 | 0.137078 | 1472.46 | 8.0096 | 0.18627 | 100 |
| TabArena | meta_selector_meta_feature_guard_010_k4_rt0p1 | 9.01469 | 0.133823 | 1997.76 | 8.01469 | 0.186388 | 100 |
| TabArena | meta_selector_meta_feature_guard_005_k12_rt0 | 9.04206 | 0.131359 | 2603.89 | 8.04206 | 0.187025 | 100 |
| TabArena | meta_selector_meta_feature_guard_002_k4_rt0p03 | 9.05034 | 0.130506 | 1911.43 | 8.05034 | 0.187217 | 100 |
| TabArena | meta_selector_meta_feature_guard_005_k8_rt0p1 | 9.05922 | 0.136085 | 1403.1 | 8.05922 | 0.187424 | 100 |
| TabArena | meta_selector_calibrated_budget_10000s | 9.09374 | 0.125814 | 4433.2 | 8.09374 | 0.188227 | 100 |
| TabArena | meta_selector_meta_feature_guard_010_k4_rt0p03 | 9.11217 | 0.129676 | 2636.73 | 8.11217 | 0.188655 | 100 |
| TabArena | meta_selector_meta_feature_guard_010_k4_rt0 | 9.11773 | 0.129242 | 2697.99 | 8.11773 | 0.188784 | 100 |
| TabArena | reference_optimism_corrected_budget | 9.11851 | 0.145825 | 1664.32 | 8.11851 | 0.188803 | 100 |
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
| TabArena | reference_metafeature_knn_validation | 9.16142 | 0.101091 | 76262.6 | 8.16142 | 0.1898 | 100 |
| TabArena | meta_selector_meta_feature_guard_005_k4_rt0p1 | 9.42013 | 0.124986 | 1230.08 | 8.42013 | 0.195817 | 100 |
| TabArena | meta_selector_meta_feature_guard_002_k12_rt0p1 | 9.43672 | 0.133797 | 969.764 | 8.43672 | 0.196203 | 100 |
| TabArena | meta_selector_meta_feature_guard_002_k8_rt0p1 | 9.5534 | 0.132706 | 738.803 | 8.5534 | 0.198916 | 100 |
| TabArena | reference_metafeature_knn | 9.81721 | 0.0957331 | 54136.8 | 8.81721 | 0.205051 | 100 |
| TabArena | meta_selector_family_prior_budgeted_500s | 9.83813 | 0.150494 | 74.724 | 8.83813 | 0.205538 | 100 |
| TabArena | meta_selector_meta_feature_guard_002_k4_rt0p1 | 9.83978 | 0.129673 | 616.196 | 8.83978 | 0.205576 | 100 |
| TabArena | meta_selector_family_prior_budgeted_1000s | 9.99849 | 0.136353 | 237.802 | 8.99849 | 0.209267 | 100 |
| TabArena | reference_single_best_global | 10.101 | 0.0436988 | 52637.6 | 9.10101 | 0.211651 | 100 |
| TabArena | reference_validation_best_default | 10.2008 | 0.127912 | 200.982 | 9.20077 | 0.213972 | 100 |
| TabArena | meta_selector_family_prior_budgeted_5000s | 10.4208 | 0.129506 | 1733.97 | 9.42076 | 0.219087 | 100 |
| TabArena | reference_single_best_task | 10.4463 | 0.0793466 | 61371.2 | 9.44631 | 0.219682 | 100 |
| TabArena | meta_selector_family_prior_budgeted_100s | 10.931 | 0.106931 | 31.3983 | 9.93098 | 0.230953 | 100 |
| TabArena | reference_priors_only | 10.9477 | 0.147021 | 91.5613 | 9.94767 | 0.231341 | 100 |
| TabArena | reference_validation_best_tuned | 12.2684 | 0.0867765 | 78114.6 | 11.2684 | 0.262057 | 100 |
| TabArena | meta_selector_ablate_validation_only | 13.1123 | 0.0980355 | 78032.4 | 12.1123 | 0.281681 | 100 |
| TabArena | meta_selector_single | 13.1123 | 0.0980355 | 78032.4 | 12.1123 | 0.281681 | 100 |
| TabArena | reference_validation_best_tuned_ensemble | 13.1219 | 0.0883644 | 84340.6 | 12.1219 | 0.281906 | 100 |

## Selected Policies

| artifact | selected_policy | n_splits |
| --- | --- | --- |
| TabArena | meta_selector_family_prior_budgeted | 26 |
| TabArena | meta_selector_meta_feature_guard_002_k8_rt0 | 8 |
| TabArena | meta_selector_meta_feature_guard_005_k8_rt0p1 | 8 |
| TabArena | meta_selector_meta_feature_guard_002_k4_rt0 | 7 |
| TabArena | meta_selector_meta_feature_guard_005_k12_rt0p1 | 6 |
| TabArena | meta_selector_task_adaptive_budgeted | 5 |
| TabArena | meta_selector_meta_feature_guard_002 | 4 |
| TabArena | meta_selector_meta_feature_guard_005_k4_rt0 | 4 |
| TabArena | meta_selector_meta_feature_guard_005_k4_rt0p03 | 4 |
| TabArena | meta_selector_meta_feature_guard_005_k4_rt0p1 | 4 |
| TabArena | meta_selector_meta_feature_guard_010_k4_rt0p1 | 4 |
| TabArena | meta_selector_meta_feature_guard_010_k8_rt0p1 | 4 |
| TabArena | meta_selector_meta_feature_guard_002_k12_rt0p03 | 3 |
| TabArena | meta_selector_meta_feature_guard_005_k8_rt0 | 3 |
| TabArena | meta_selector_meta_feature_guard_002_k4_rt0p03 | 2 |
| TabArena | meta_selector_meta_feature_guard_002_k8_rt0p1 | 2 |
| TabArena | meta_selector_meta_feature_guard_005 | 2 |
| TabArena | meta_selector_calibrated_budget_10000s | 1 |
| TabArena | meta_selector_meta_feature_guard_002_k12_rt0 | 1 |
| TabArena | meta_selector_meta_feature_guard_002_k12_rt0p1 | 1 |
| TabArena | meta_selector_meta_feature_guard_005_k12_rt0p03 | 1 |

## Paired Tests

| artifact | method | reference_method | n_splits | mean_rank_difference | mean_normalized_regret_difference | mean_runtime_difference_seconds | rank_wilcoxon_p_value | rank_holm_p_value |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| TabArena | nested_policy_selected | meta_selector_task_adaptive_budgeted | 100 | -0.154032 | -0.00358215 | -28.9225 | 0.0202423 | 0.0404846 |
| TabArena | nested_policy_selected | meta_selector_meta_feature_guard_005 | 100 | 0.108263 | 0.00251774 | 539.484 | 0.0632515 | 0.0632515 |
| TabArena | nested_policy_selected | meta_selector_family_prior_budgeted | 100 | 0.27573 | 0.00641233 | -1676.17 | 0.00102519 | 0.00307558 |
| TabArena | nested_policy_selected | meta_selector_ablate_validation_only | 100 | -4.11436 | -0.0956828 | -75323.3 | 5.7672e-18 | 2.8836e-17 |
| TabArena | nested_policy_selected | meta_selector_single | 100 | -4.11436 | -0.0956828 | -75323.3 | 5.7672e-18 | 2.8836e-17 |
