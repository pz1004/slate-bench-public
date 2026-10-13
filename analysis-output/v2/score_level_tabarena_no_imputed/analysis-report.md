# Fully Nested Score-Level Policy Benchmark

- Command: `scripts/score_level_policy_benchmark.py --config configs/score_level_tabarena_no_imputed.yaml --jobs 12`
- Completed artifact/split rows: 100
- Protocol: selector rows are rebuilt from outer-training datasets only; held-out datasets provide validation/runtime artifacts for target-side selection and test metrics for final evaluation only.

## Summary

| artifact | method | mean_rank | tied_best_rate | mean_total_seconds | mean_rank_regret | mean_normalized_rank_regret | n_splits |
| --- | --- | --- | --- | --- | --- | --- | --- |
| TabArena | oracle | 1.00996 | 1 | 46563.3 | 0.00996355 | 0.000243824 | 100 |
| TabArena | reference_single_best_global | 8.10932 | 0.149538 | 48332 | 7.10932 | 0.169864 | 100 |
| TabArena | reference_optimism_corrected_task | 8.47294 | 0.166436 | 10759.3 | 7.47294 | 0.177315 | 100 |
| TabArena | meta_selector_calibrated_budget_10000s | 8.57738 | 0.14395 | 3225.42 | 7.57738 | 0.180035 | 100 |
| TabArena | meta_selector_family_prior_budgeted | 8.64621 | 0.158736 | 3498.76 | 7.64621 | 0.181172 | 100 |
| TabArena | reference_optimism_corrected | 8.79275 | 0.16263 | 8960.17 | 7.79275 | 0.184676 | 100 |
| TabArena | meta_selector_meta_feature_guard_002_k8_rt0 | 8.80592 | 0.147999 | 2119.3 | 7.80592 | 0.185438 | 100 |
| TabArena | meta_selector_meta_feature_guard_005 | 8.82234 | 0.146722 | 1978.69 | 7.82234 | 0.186015 | 100 |
| TabArena | meta_selector_meta_feature_guard_002_k4_rt0 | 8.8361 | 0.149637 | 2121.67 | 7.8361 | 0.18613 | 100 |
| TabArena | meta_selector_meta_feature_guard_010_k8_rt0p1 | 8.8556 | 0.146417 | 1790.93 | 7.8556 | 0.186959 | 100 |
| TabArena | meta_selector_meta_feature_guard_010_k12_rt0p1 | 8.85962 | 0.14743 | 1816.66 | 7.85962 | 0.187004 | 100 |
| TabArena | meta_selector_meta_feature_guard_005_k4_rt0p03 | 8.86817 | 0.145652 | 1922.01 | 7.86817 | 0.187092 | 100 |
| TabArena | meta_selector_meta_feature_guard_005_k4_rt0 | 8.8714 | 0.144657 | 2157.35 | 7.8714 | 0.187125 | 100 |
| TabArena | reference_metafeature_knn_validation | 8.90364 | 0.115018 | 71433.5 | 7.90364 | 0.18716 | 100 |
| TabArena | reference_optimism_corrected_budget | 8.91333 | 0.14735 | 1573.31 | 7.91333 | 0.187513 | 100 |
| TabArena | meta_selector_meta_feature_guard_002_k12_rt0 | 8.90333 | 0.146233 | 2088.17 | 7.90333 | 0.187813 | 100 |
| TabArena | meta_selector_meta_feature_guard_005_k8_rt0 | 8.90771 | 0.144328 | 2157.99 | 7.90771 | 0.187919 | 100 |
| TabArena | meta_selector_meta_feature_guard_005_k12_rt0p03 | 8.91862 | 0.145582 | 1971.5 | 7.91862 | 0.188344 | 100 |
| TabArena | meta_selector_meta_feature_guard_005_k12_rt0 | 8.95036 | 0.144203 | 2136.64 | 7.95036 | 0.188981 | 100 |
| TabArena | reference_single_best_task | 8.9041 | 0.113732 | 50115.3 | 7.9041 | 0.189935 | 100 |
| TabArena | meta_selector_meta_feature_guard_002 | 8.99548 | 0.144704 | 1724.48 | 7.99548 | 0.190072 | 100 |
| TabArena | meta_selector_meta_feature_guard_010_k4_rt0p03 | 9.00151 | 0.143406 | 2145.49 | 8.00151 | 0.190235 | 100 |
| TabArena | meta_selector_meta_feature_guard_002_k12_rt0p03 | 9.00586 | 0.144187 | 1719.34 | 8.00586 | 0.190396 | 100 |
| TabArena | meta_selector_meta_feature_guard_010 | 9.01674 | 0.142743 | 2159.14 | 8.01674 | 0.190584 | 100 |
| TabArena | nested_policy_selected | 9.03426 | 0.140392 | 2508.27 | 8.03426 | 0.190851 | 100 |
| TabArena | meta_selector_meta_feature_guard_010_k12_rt0p03 | 9.03274 | 0.142761 | 2159.84 | 8.03274 | 0.190968 | 100 |
| TabArena | meta_selector_meta_feature_guard_010_k4_rt0 | 9.05476 | 0.141179 | 2228.53 | 8.05476 | 0.19145 | 100 |
| TabArena | meta_selector_meta_feature_guard_020_k4_rt0p1 | 9.06054 | 0.14185 | 2189.25 | 8.06054 | 0.191621 | 100 |
| TabArena | meta_selector_meta_feature_guard_010_k8_rt0 | 9.07387 | 0.141043 | 2220.58 | 8.07387 | 0.191894 | 100 |
| TabArena | meta_selector_meta_feature_guard_020_k4_rt0 | 9.07621 | 0.141194 | 2206.89 | 8.07621 | 0.191967 | 100 |
| TabArena | meta_selector_meta_feature_guard_020_k4_rt0p03 | 9.07633 | 0.141235 | 2206.06 | 8.07633 | 0.19197 | 100 |
| TabArena | meta_selector_meta_feature_guard_010_k12_rt0 | 9.08409 | 0.141219 | 2209.55 | 8.08409 | 0.192158 | 100 |
| TabArena | meta_selector_meta_feature_guard_020_k8_rt0p1 | 9.08403 | 0.141194 | 2204.62 | 8.08403 | 0.192169 | 100 |
| TabArena | meta_selector_meta_feature_guard_020 | 9.08678 | 0.141194 | 2204.62 | 8.08678 | 0.192233 | 100 |
| TabArena | meta_selector_meta_feature_guard_020_k12_rt0 | 9.08678 | 0.141194 | 2204.62 | 8.08678 | 0.192233 | 100 |
| TabArena | meta_selector_meta_feature_guard_020_k12_rt0p03 | 9.08678 | 0.141194 | 2204.62 | 8.08678 | 0.192233 | 100 |
| TabArena | meta_selector_meta_feature_guard_020_k12_rt0p1 | 9.08678 | 0.141194 | 2204.62 | 8.08678 | 0.192233 | 100 |
| TabArena | meta_selector_meta_feature_guard_020_k8_rt0 | 9.08678 | 0.141194 | 2204.62 | 8.08678 | 0.192233 | 100 |
| TabArena | meta_selector_task_adaptive_budgeted | 9.08678 | 0.141194 | 2204.62 | 8.08678 | 0.192233 | 100 |
| TabArena | meta_selector_meta_feature_guard_010_k4_rt0p1 | 9.19502 | 0.141679 | 1639.98 | 8.19502 | 0.194917 | 100 |
| TabArena | meta_selector_meta_feature_guard_002_k4_rt0p03 | 9.24761 | 0.142341 | 1625.35 | 8.24761 | 0.195884 | 100 |
| TabArena | meta_selector_family_prior_budgeted_5000s | 9.28841 | 0.150654 | 1677.77 | 8.28841 | 0.19713 | 100 |
| TabArena | reference_metafeature_knn | 9.31567 | 0.115424 | 58941.9 | 8.31567 | 0.198583 | 100 |
| TabArena | meta_selector_meta_feature_guard_005_k12_rt0p1 | 9.38081 | 0.137803 | 1291.41 | 8.38081 | 0.199589 | 100 |
| TabArena | meta_selector_meta_feature_guard_005_k8_rt0p1 | 9.50178 | 0.13736 | 1113.52 | 8.50178 | 0.202538 | 100 |
| TabArena | meta_selector_meta_feature_guard_005_k4_rt0p1 | 9.73021 | 0.13244 | 1006.29 | 8.73021 | 0.208002 | 100 |
| TabArena | meta_selector_family_prior_budgeted_1000s | 9.87401 | 0.144796 | 192.712 | 8.87401 | 0.211632 | 100 |
| TabArena | meta_selector_meta_feature_guard_002_k12_rt0p1 | 9.88301 | 0.133738 | 830.806 | 8.88301 | 0.21212 | 100 |
| TabArena | meta_selector_family_prior_budgeted_500s | 9.88444 | 0.152425 | 76.0116 | 8.88444 | 0.212184 | 100 |
| TabArena | meta_selector_meta_feature_guard_002_k8_rt0p1 | 9.90316 | 0.133747 | 588.953 | 8.90316 | 0.21256 | 100 |
| TabArena | meta_selector_meta_feature_guard_002_k4_rt0p1 | 9.97472 | 0.134825 | 548.65 | 8.97472 | 0.214214 | 100 |
| TabArena | reference_validation_best_default | 10.1525 | 0.127912 | 200.982 | 9.15248 | 0.216632 | 100 |
| TabArena | meta_selector_family_prior_budgeted_100s | 10.7006 | 0.116841 | 30.2013 | 9.70063 | 0.232913 | 100 |
| TabArena | reference_validation_best_tuned | 12.2138 | 0.0867765 | 77309.2 | 11.2138 | 0.264569 | 100 |
| TabArena | reference_priors_only | 12.8487 | 0.101046 | 96.5839 | 11.8487 | 0.283403 | 100 |
| TabArena | meta_selector_ablate_validation_only | 13.0923 | 0.0980355 | 78032.4 | 12.0923 | 0.284481 | 100 |
| TabArena | meta_selector_single | 13.0923 | 0.0980355 | 78032.4 | 12.0923 | 0.284481 | 100 |
| TabArena | reference_validation_best_tuned_ensemble | 13.102 | 0.0883644 | 84340.6 | 12.102 | 0.284729 | 100 |

## Selected Policies

| artifact | selected_policy | n_splits |
| --- | --- | --- |
| TabArena | meta_selector_calibrated_budget_10000s | 30 |
| TabArena | meta_selector_family_prior_budgeted | 17 |
| TabArena | meta_selector_meta_feature_guard_002_k8_rt0 | 12 |
| TabArena | meta_selector_meta_feature_guard_002 | 7 |
| TabArena | meta_selector_meta_feature_guard_002_k12_rt0p03 | 4 |
| TabArena | meta_selector_meta_feature_guard_002_k4_rt0p03 | 4 |
| TabArena | meta_selector_meta_feature_guard_005_k4_rt0 | 4 |
| TabArena | meta_selector_meta_feature_guard_002_k4_rt0 | 3 |
| TabArena | meta_selector_meta_feature_guard_005_k12_rt0 | 3 |
| TabArena | meta_selector_family_prior_budgeted_5000s | 2 |
| TabArena | meta_selector_meta_feature_guard_005 | 2 |
| TabArena | meta_selector_meta_feature_guard_005_k8_rt0 | 2 |
| TabArena | meta_selector_meta_feature_guard_010_k4_rt0 | 2 |
| TabArena | meta_selector_meta_feature_guard_010_k8_rt0p1 | 2 |
| TabArena | meta_selector_family_prior_budgeted_1000s | 1 |
| TabArena | meta_selector_meta_feature_guard_002_k12_rt0 | 1 |
| TabArena | meta_selector_meta_feature_guard_002_k12_rt0p1 | 1 |
| TabArena | meta_selector_meta_feature_guard_005_k12_rt0p1 | 1 |
| TabArena | meta_selector_meta_feature_guard_005_k4_rt0p1 | 1 |
| TabArena | meta_selector_meta_feature_guard_010_k12_rt0p1 | 1 |

## Paired Tests

| artifact | method | reference_method | n_splits | mean_rank_difference | mean_normalized_regret_difference | mean_runtime_difference_seconds | rank_wilcoxon_p_value | rank_holm_p_value |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| TabArena | nested_policy_selected | meta_selector_task_adaptive_budgeted | 100 | -0.0525194 | -0.0013817 | 303.653 | 0.161287 | 0.161287 |
| TabArena | nested_policy_selected | meta_selector_meta_feature_guard_005 | 100 | 0.211927 | 0.00483625 | 529.573 | 0.0534727 | 0.106945 |
| TabArena | nested_policy_selected | meta_selector_family_prior_budgeted | 100 | 0.388055 | 0.00967953 | -990.488 | 5.98304e-05 | 0.000179491 |
| TabArena | nested_policy_selected | meta_selector_ablate_validation_only | 100 | -4.05807 | -0.0936294 | -75524.1 | 8.24818e-17 | 4.12409e-16 |
| TabArena | nested_policy_selected | meta_selector_single | 100 | -4.05807 | -0.0936294 | -75524.1 | 8.24818e-17 | 4.12409e-16 |
