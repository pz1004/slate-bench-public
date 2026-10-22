# Fully Nested Score-Level Policy Benchmark

- Command: `scripts/score_level_policy_benchmark.py --config configs/score_level_tabarena_no_flagged.yaml --jobs 10 --resume`
- Completed artifact/split rows: 100
- Protocol: selector rows are rebuilt from outer-training datasets only; held-out datasets provide validation/runtime artifacts for target-side selection and test metrics for final evaluation only.

## Summary

| artifact | method | mean_rank | tied_best_rate | mean_total_seconds | mean_rank_regret | mean_normalized_rank_regret | n_splits |
| --- | --- | --- | --- | --- | --- | --- | --- |
| TabArena | oracle | 1.01684 | 1 | 46563.3 | 0.0168373 | 0.000414792 | 100 |
| TabArena | reference_single_best_global | 8.13166 | 0.149538 | 48332 | 7.13166 | 0.169335 | 100 |
| TabArena | reference_optimism_corrected_task | 8.50306 | 0.165999 | 10679 | 7.50306 | 0.177307 | 100 |
| TabArena | meta_selector_calibrated_budget_10000s | 8.60979 | 0.143658 | 3196.25 | 7.60979 | 0.179977 | 100 |
| TabArena | meta_selector_family_prior_budgeted | 8.67125 | 0.158566 | 3521.62 | 7.67125 | 0.181137 | 100 |
| TabArena | reference_optimism_corrected | 8.80304 | 0.162801 | 8862.81 | 7.80304 | 0.184209 | 100 |
| TabArena | meta_selector_meta_feature_guard_002_k8_rt0 | 8.79396 | 0.145897 | 2098.16 | 7.79396 | 0.184437 | 100 |
| TabArena | meta_selector_meta_feature_guard_002_k12_rt0 | 8.85656 | 0.147511 | 2048.56 | 7.85656 | 0.185953 | 100 |
| TabArena | meta_selector_meta_feature_guard_002_k4_rt0 | 8.86256 | 0.148063 | 2050.36 | 7.86256 | 0.185956 | 100 |
| TabArena | meta_selector_meta_feature_guard_005 | 8.8729 | 0.14605 | 1941.96 | 7.8729 | 0.186487 | 100 |
| TabArena | reference_metafeature_knn_validation | 8.91727 | 0.115339 | 71480.3 | 7.91727 | 0.186729 | 100 |
| TabArena | reference_optimism_corrected_budget | 8.93101 | 0.147607 | 1567.23 | 7.93101 | 0.187254 | 100 |
| TabArena | meta_selector_meta_feature_guard_005_k4_rt0p03 | 8.91861 | 0.144772 | 1817.83 | 7.91861 | 0.187502 | 100 |
| TabArena | meta_selector_meta_feature_guard_010_k8_rt0p1 | 8.92257 | 0.1451 | 1756.19 | 7.92257 | 0.187735 | 100 |
| TabArena | meta_selector_meta_feature_guard_005_k8_rt0 | 8.93723 | 0.143477 | 2134.38 | 7.93723 | 0.187892 | 100 |
| TabArena | meta_selector_meta_feature_guard_010_k12_rt0p1 | 8.93243 | 0.146996 | 1787.13 | 7.93243 | 0.187932 | 100 |
| TabArena | meta_selector_meta_feature_guard_005_k12_rt0p03 | 8.93752 | 0.14686 | 1956.46 | 7.93752 | 0.187995 | 100 |
| TabArena | meta_selector_meta_feature_guard_005_k4_rt0 | 8.9455 | 0.145038 | 2111.9 | 7.9455 | 0.188007 | 100 |
| TabArena | reference_single_best_task | 8.92241 | 0.114065 | 50500.2 | 7.92241 | 0.188926 | 100 |
| TabArena | meta_selector_meta_feature_guard_005_k12_rt0 | 8.98084 | 0.143173 | 2104.59 | 7.98084 | 0.188944 | 100 |
| TabArena | meta_selector_meta_feature_guard_002 | 8.98722 | 0.144455 | 1697.33 | 7.98722 | 0.189108 | 100 |
| TabArena | meta_selector_meta_feature_guard_002_k12_rt0p03 | 8.98964 | 0.145386 | 1700.61 | 7.98964 | 0.189209 | 100 |
| TabArena | meta_selector_meta_feature_guard_010_k4_rt0p03 | 9.03613 | 0.141533 | 2115.82 | 8.03613 | 0.190289 | 100 |
| TabArena | meta_selector_meta_feature_guard_010 | 9.05283 | 0.142342 | 2107.63 | 8.05283 | 0.190709 | 100 |
| TabArena | nested_policy_selected | 9.06661 | 0.140049 | 2555.33 | 8.06661 | 0.190771 | 100 |
| TabArena | meta_selector_meta_feature_guard_010_k4_rt0 | 9.08412 | 0.140597 | 2157.07 | 8.08412 | 0.191392 | 100 |
| TabArena | meta_selector_meta_feature_guard_010_k12_rt0p03 | 9.09512 | 0.141715 | 2120.9 | 8.09512 | 0.191699 | 100 |
| TabArena | meta_selector_meta_feature_guard_020_k4_rt0p1 | 9.10021 | 0.14107 | 2117.07 | 8.10021 | 0.191818 | 100 |
| TabArena | meta_selector_meta_feature_guard_020_k4_rt0p03 | 9.10212 | 0.140379 | 2146.28 | 8.10212 | 0.191849 | 100 |
| TabArena | meta_selector_meta_feature_guard_020_k4_rt0 | 9.10212 | 0.140338 | 2147.12 | 8.10212 | 0.191849 | 100 |
| TabArena | meta_selector_meta_feature_guard_020_k12_rt0p1 | 9.10669 | 0.140549 | 2133.44 | 8.10669 | 0.191971 | 100 |
| TabArena | meta_selector_meta_feature_guard_020 | 9.10937 | 0.140449 | 2143.99 | 8.10937 | 0.192031 | 100 |
| TabArena | meta_selector_meta_feature_guard_020_k12_rt0 | 9.10937 | 0.140449 | 2143.99 | 8.10937 | 0.192031 | 100 |
| TabArena | meta_selector_meta_feature_guard_020_k8_rt0 | 9.10937 | 0.140449 | 2143.99 | 8.10937 | 0.192031 | 100 |
| TabArena | meta_selector_meta_feature_guard_010_k12_rt0 | 9.11048 | 0.140823 | 2153.14 | 8.11048 | 0.192045 | 100 |
| TabArena | meta_selector_meta_feature_guard_020_k12_rt0p03 | 9.11065 | 0.140449 | 2141.37 | 8.11065 | 0.192063 | 100 |
| TabArena | meta_selector_task_adaptive_budgeted | 9.11065 | 0.140449 | 2141.37 | 8.11065 | 0.192063 | 100 |
| TabArena | meta_selector_meta_feature_guard_010_k8_rt0 | 9.1119 | 0.14054 | 2154.99 | 8.1119 | 0.192081 | 100 |
| TabArena | meta_selector_meta_feature_guard_020_k8_rt0p1 | 9.11381 | 0.140508 | 2131.01 | 8.11381 | 0.192136 | 100 |
| TabArena | meta_selector_meta_feature_guard_010_k4_rt0p1 | 9.18521 | 0.142432 | 1557.21 | 8.18521 | 0.193921 | 100 |
| TabArena | meta_selector_meta_feature_guard_002_k4_rt0p03 | 9.26438 | 0.140455 | 1580.85 | 8.26438 | 0.195506 | 100 |
| TabArena | meta_selector_family_prior_budgeted_5000s | 9.2764 | 0.151273 | 1697.59 | 8.2764 | 0.195873 | 100 |
| TabArena | reference_metafeature_knn | 9.33565 | 0.115424 | 58937.4 | 8.33565 | 0.197894 | 100 |
| TabArena | meta_selector_meta_feature_guard_005_k12_rt0p1 | 9.39277 | 0.139677 | 1277.52 | 8.39277 | 0.199047 | 100 |
| TabArena | meta_selector_meta_feature_guard_005_k8_rt0p1 | 9.51816 | 0.139396 | 1064.85 | 8.51816 | 0.20199 | 100 |
| TabArena | meta_selector_meta_feature_guard_005_k4_rt0p1 | 9.68529 | 0.135276 | 966.378 | 8.68529 | 0.206035 | 100 |
| TabArena | meta_selector_family_prior_budgeted_1000s | 9.88764 | 0.14427 | 194.86 | 8.88764 | 0.210834 | 100 |
| TabArena | meta_selector_meta_feature_guard_002_k12_rt0p1 | 9.87942 | 0.135967 | 727.721 | 8.87942 | 0.210927 | 100 |
| TabArena | meta_selector_family_prior_budgeted_500s | 9.88475 | 0.15346 | 76.0057 | 8.88475 | 0.211027 | 100 |
| TabArena | meta_selector_meta_feature_guard_002_k8_rt0p1 | 9.88454 | 0.137074 | 527.471 | 8.88454 | 0.211056 | 100 |
| TabArena | meta_selector_meta_feature_guard_002_k4_rt0p1 | 9.95772 | 0.138343 | 497.424 | 8.95772 | 0.212754 | 100 |
| TabArena | reference_validation_best_default | 10.1668 | 0.127912 | 200.982 | 9.16682 | 0.216242 | 100 |
| TabArena | meta_selector_family_prior_budgeted_100s | 10.7316 | 0.118319 | 29.9267 | 9.73157 | 0.232387 | 100 |
| TabArena | reference_validation_best_tuned | 12.232 | 0.0867765 | 78114.6 | 11.232 | 0.264217 | 100 |
| TabArena | reference_priors_only | 12.8371 | 0.101717 | 96.6337 | 11.8371 | 0.281339 | 100 |
| TabArena | meta_selector_ablate_validation_only | 13.1061 | 0.0980355 | 78032.4 | 12.1061 | 0.284135 | 100 |
| TabArena | meta_selector_single | 13.1061 | 0.0980355 | 78032.4 | 12.1061 | 0.284135 | 100 |
| TabArena | reference_validation_best_tuned_ensemble | 13.1158 | 0.0883644 | 84340.6 | 12.1158 | 0.284383 | 100 |

## Selected Policies

| artifact | selected_policy | n_splits |
| --- | --- | --- |
| TabArena | meta_selector_calibrated_budget_10000s | 30 |
| TabArena | meta_selector_family_prior_budgeted | 20 |
| TabArena | meta_selector_meta_feature_guard_002_k8_rt0 | 11 |
| TabArena | meta_selector_meta_feature_guard_002_k12_rt0 | 6 |
| TabArena | meta_selector_meta_feature_guard_002_k4_rt0 | 4 |
| TabArena | meta_selector_meta_feature_guard_002_k4_rt0p03 | 4 |
| TabArena | meta_selector_task_adaptive_budgeted | 4 |
| TabArena | meta_selector_meta_feature_guard_002 | 2 |
| TabArena | meta_selector_meta_feature_guard_002_k12_rt0p03 | 2 |
| TabArena | meta_selector_meta_feature_guard_005_k12_rt0 | 2 |
| TabArena | meta_selector_meta_feature_guard_005_k12_rt0p1 | 2 |
| TabArena | meta_selector_meta_feature_guard_005_k4_rt0p03 | 2 |
| TabArena | meta_selector_meta_feature_guard_010_k4_rt0 | 2 |
| TabArena | meta_selector_meta_feature_guard_010_k8_rt0p1 | 2 |
| TabArena | meta_selector_family_prior_budgeted_1000s | 1 |
| TabArena | meta_selector_family_prior_budgeted_5000s | 1 |
| TabArena | meta_selector_meta_feature_guard_002_k12_rt0p1 | 1 |
| TabArena | meta_selector_meta_feature_guard_005_k4_rt0 | 1 |
| TabArena | meta_selector_meta_feature_guard_005_k8_rt0 | 1 |
| TabArena | meta_selector_meta_feature_guard_010_k12_rt0p1 | 1 |
| TabArena | meta_selector_meta_feature_guard_010_k4_rt0p03 | 1 |

## Paired Tests

| artifact | method | reference_method | n_splits | mean_rank_difference | mean_normalized_regret_difference | mean_runtime_difference_seconds | rank_wilcoxon_p_value | rank_holm_p_value |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| TabArena | nested_policy_selected | meta_selector_task_adaptive_budgeted | 100 | -0.0440373 | -0.00129152 | 413.963 | 0.475529 | 0.475529 |
| TabArena | nested_policy_selected | meta_selector_meta_feature_guard_005 | 100 | 0.193714 | 0.00428396 | 613.372 | 0.145134 | 0.290268 |
| TabArena | nested_policy_selected | meta_selector_family_prior_budgeted | 100 | 0.395361 | 0.00963397 | -966.295 | 8.78611e-05 | 0.000263583 |
| TabArena | nested_policy_selected | meta_selector_ablate_validation_only | 100 | -4.03947 | -0.0933643 | -75477 | 6.16607e-17 | 3.08303e-16 |
| TabArena | nested_policy_selected | meta_selector_single | 100 | -4.03947 | -0.0933643 | -75477 | 6.16607e-17 | 3.08303e-16 |
