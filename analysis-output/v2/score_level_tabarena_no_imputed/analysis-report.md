# Fully Nested Score-Level Policy Benchmark

- Command: `scripts/score_level_policy_benchmark.py --config configs/score_level_tabarena_no_imputed.yaml --jobs 20`
- Completed artifact/split rows: 100
- Protocol: selector rows are rebuilt from outer-training datasets only; held-out datasets provide validation/runtime artifacts for target-side selection and test metrics for final evaluation only.

## Summary

| artifact | method | mean_rank | tied_best_rate | mean_total_seconds | mean_rank_regret | mean_normalized_rank_regret | n_splits |
| --- | --- | --- | --- | --- | --- | --- | --- |
| TabArena | oracle | 1.01684 | 1 | 46563.3 | 0.0168373 | 0.000435599 | 100 |
| TabArena | reference_single_best_global | 8.12307 | 0.149538 | 48332 | 7.12307 | 0.170688 | 100 |
| TabArena | reference_optimism_corrected_task | 8.5065 | 0.166191 | 10822.5 | 7.5065 | 0.178457 | 100 |
| TabArena | meta_selector_calibrated_budget_10000s | 8.61816 | 0.142662 | 3200.79 | 7.61816 | 0.181462 | 100 |
| TabArena | meta_selector_family_prior_budgeted | 8.67197 | 0.1579 | 3517.25 | 7.67197 | 0.182119 | 100 |
| TabArena | meta_selector_meta_feature_guard_002_k8_rt0 | 8.78371 | 0.147097 | 2168.15 | 7.78371 | 0.185229 | 100 |
| TabArena | reference_optimism_corrected | 8.8023 | 0.162785 | 8949.48 | 7.8023 | 0.185269 | 100 |
| TabArena | meta_selector_meta_feature_guard_002_k4_rt0 | 8.83459 | 0.147969 | 2104.1 | 7.83459 | 0.186417 | 100 |
| TabArena | meta_selector_meta_feature_guard_005 | 8.82816 | 0.147032 | 2004.12 | 7.82816 | 0.186507 | 100 |
| TabArena | meta_selector_meta_feature_guard_010_k12_rt0p1 | 8.87664 | 0.14744 | 1861.85 | 7.87664 | 0.187776 | 100 |
| TabArena | reference_metafeature_knn_validation | 8.91011 | 0.115841 | 71411.6 | 7.91011 | 0.187823 | 100 |
| TabArena | meta_selector_meta_feature_guard_005_k4_rt0 | 8.8929 | 0.144196 | 2150.33 | 7.8929 | 0.187942 | 100 |
| TabArena | meta_selector_meta_feature_guard_005_k4_rt0p03 | 8.90025 | 0.143723 | 1924.77 | 7.90025 | 0.188174 | 100 |
| TabArena | meta_selector_meta_feature_guard_010_k8_rt0p1 | 8.89164 | 0.145688 | 1849.04 | 7.89164 | 0.188184 | 100 |
| TabArena | reference_optimism_corrected_budget | 8.92551 | 0.147551 | 1576.24 | 7.92551 | 0.188201 | 100 |
| TabArena | meta_selector_meta_feature_guard_002_k12_rt0 | 8.91394 | 0.14489 | 2126.04 | 7.91394 | 0.188422 | 100 |
| TabArena | meta_selector_meta_feature_guard_005_k12_rt0p03 | 8.92465 | 0.145824 | 2004.46 | 7.92465 | 0.188852 | 100 |
| TabArena | meta_selector_meta_feature_guard_005_k8_rt0 | 8.94638 | 0.143381 | 2197.62 | 7.94638 | 0.189178 | 100 |
| TabArena | meta_selector_meta_feature_guard_005_k12_rt0 | 9.00065 | 0.142785 | 2164.83 | 8.00065 | 0.190489 | 100 |
| TabArena | reference_single_best_task | 8.91785 | 0.113732 | 50115.3 | 7.91785 | 0.190789 | 100 |
| TabArena | meta_selector_meta_feature_guard_010 | 9.0434 | 0.142104 | 2174.36 | 8.0434 | 0.191615 | 100 |
| TabArena | meta_selector_meta_feature_guard_010_k4_rt0p03 | 9.04547 | 0.141835 | 2166.33 | 8.04547 | 0.191622 | 100 |
| TabArena | meta_selector_meta_feature_guard_010_k4_rt0 | 9.05491 | 0.141041 | 2221.54 | 8.05491 | 0.191793 | 100 |
| TabArena | meta_selector_meta_feature_guard_002 | 9.05289 | 0.142031 | 1731.85 | 8.05289 | 0.191832 | 100 |
| TabArena | nested_policy_selected | 9.06586 | 0.140615 | 2542.14 | 8.06586 | 0.191954 | 100 |
| TabArena | meta_selector_meta_feature_guard_010_k12_rt0p03 | 9.08161 | 0.141838 | 2169.09 | 8.08161 | 0.192554 | 100 |
| TabArena | meta_selector_meta_feature_guard_002_k12_rt0p03 | 9.08077 | 0.142432 | 1769.3 | 8.08077 | 0.192564 | 100 |
| TabArena | meta_selector_meta_feature_guard_020_k4_rt0p1 | 9.09415 | 0.140366 | 2214.32 | 8.09415 | 0.192812 | 100 |
| TabArena | meta_selector_meta_feature_guard_010_k8_rt0 | 9.10394 | 0.140472 | 2240.71 | 8.10394 | 0.192994 | 100 |
| TabArena | meta_selector_meta_feature_guard_020_k4_rt0 | 9.11962 | 0.139248 | 2237 | 8.11962 | 0.193381 | 100 |
| TabArena | meta_selector_meta_feature_guard_020_k4_rt0p03 | 9.1229 | 0.139203 | 2236.5 | 8.1229 | 0.193465 | 100 |
| TabArena | meta_selector_meta_feature_guard_020 | 9.12711 | 0.139159 | 2236.54 | 8.12711 | 0.193571 | 100 |
| TabArena | meta_selector_meta_feature_guard_020_k12_rt0 | 9.12711 | 0.139159 | 2236.54 | 8.12711 | 0.193571 | 100 |
| TabArena | meta_selector_meta_feature_guard_020_k12_rt0p03 | 9.12711 | 0.139159 | 2236.54 | 8.12711 | 0.193571 | 100 |
| TabArena | meta_selector_meta_feature_guard_020_k12_rt0p1 | 9.12711 | 0.139159 | 2236.54 | 8.12711 | 0.193571 | 100 |
| TabArena | meta_selector_task_adaptive_budgeted | 9.12711 | 0.139159 | 2236.54 | 8.12711 | 0.193571 | 100 |
| TabArena | meta_selector_meta_feature_guard_020_k8_rt0 | 9.12956 | 0.139034 | 2236.43 | 8.12956 | 0.193635 | 100 |
| TabArena | meta_selector_meta_feature_guard_020_k8_rt0p1 | 9.12996 | 0.139118 | 2235.84 | 8.12996 | 0.193643 | 100 |
| TabArena | meta_selector_meta_feature_guard_010_k12_rt0 | 9.13524 | 0.139309 | 2240.07 | 8.13524 | 0.193769 | 100 |
| TabArena | meta_selector_meta_feature_guard_010_k4_rt0p1 | 9.14746 | 0.141164 | 1737.28 | 8.14746 | 0.194152 | 100 |
| TabArena | meta_selector_meta_feature_guard_002_k4_rt0p03 | 9.26878 | 0.139697 | 1622.06 | 8.26878 | 0.196683 | 100 |
| TabArena | meta_selector_family_prior_budgeted_5000s | 9.28701 | 0.150155 | 1705.16 | 8.28701 | 0.197327 | 100 |
| TabArena | reference_metafeature_knn | 9.32668 | 0.115424 | 58699.1 | 8.32668 | 0.199228 | 100 |
| TabArena | meta_selector_meta_feature_guard_005_k12_rt0p1 | 9.38918 | 0.136802 | 1353.1 | 8.38918 | 0.200176 | 100 |
| TabArena | meta_selector_meta_feature_guard_005_k8_rt0p1 | 9.4959 | 0.137154 | 1229.14 | 8.4959 | 0.202816 | 100 |
| TabArena | meta_selector_meta_feature_guard_005_k4_rt0p1 | 9.7872 | 0.13265 | 998.556 | 8.7872 | 0.20979 | 100 |
| TabArena | meta_selector_meta_feature_guard_002_k12_rt0p1 | 9.84589 | 0.133416 | 843.565 | 8.84589 | 0.21163 | 100 |
| TabArena | meta_selector_meta_feature_guard_002_k8_rt0p1 | 9.84781 | 0.135771 | 572.233 | 8.84781 | 0.211701 | 100 |
| TabArena | meta_selector_family_prior_budgeted_1000s | 9.90608 | 0.143606 | 197.758 | 8.90608 | 0.212842 | 100 |
| TabArena | meta_selector_family_prior_budgeted_500s | 9.89851 | 0.152715 | 75.8589 | 8.89851 | 0.212983 | 100 |
| TabArena | meta_selector_meta_feature_guard_002_k4_rt0p1 | 9.96096 | 0.134927 | 539.256 | 8.96096 | 0.21426 | 100 |
| TabArena | reference_validation_best_default | 10.1662 | 0.127912 | 200.982 | 9.16623 | 0.217513 | 100 |
| TabArena | meta_selector_family_prior_budgeted_100s | 10.7134 | 0.116782 | 30.2031 | 9.7134 | 0.233618 | 100 |
| TabArena | reference_validation_best_tuned | 12.2275 | 0.0867765 | 77309.2 | 11.2275 | 0.265669 | 100 |
| TabArena | reference_priors_only | 12.8691 | 0.100746 | 96.823 | 11.8691 | 0.284625 | 100 |
| TabArena | meta_selector_ablate_validation_only | 13.1061 | 0.0980355 | 78032.4 | 12.1061 | 0.28538 | 100 |
| TabArena | meta_selector_single | 13.1061 | 0.0980355 | 78032.4 | 12.1061 | 0.28538 | 100 |
| TabArena | reference_validation_best_tuned_ensemble | 13.1158 | 0.0883644 | 84340.6 | 12.1158 | 0.285628 | 100 |

## Selected Policies

| artifact | selected_policy | n_splits |
| --- | --- | --- |
| TabArena | meta_selector_calibrated_budget_10000s | 29 |
| TabArena | meta_selector_family_prior_budgeted | 20 |
| TabArena | meta_selector_meta_feature_guard_002_k8_rt0 | 15 |
| TabArena | meta_selector_meta_feature_guard_002 | 7 |
| TabArena | meta_selector_meta_feature_guard_002_k4_rt0 | 5 |
| TabArena | meta_selector_meta_feature_guard_005_k12_rt0 | 3 |
| TabArena | meta_selector_family_prior_budgeted_5000s | 2 |
| TabArena | meta_selector_meta_feature_guard_002_k12_rt0p03 | 2 |
| TabArena | meta_selector_meta_feature_guard_002_k4_rt0p03 | 2 |
| TabArena | meta_selector_meta_feature_guard_005_k4_rt0 | 2 |
| TabArena | meta_selector_meta_feature_guard_005_k8_rt0 | 2 |
| TabArena | meta_selector_meta_feature_guard_010_k4_rt0 | 2 |
| TabArena | meta_selector_meta_feature_guard_010_k8_rt0p1 | 2 |
| TabArena | meta_selector_family_prior_budgeted_1000s | 1 |
| TabArena | meta_selector_meta_feature_guard_002_k12_rt0p1 | 1 |
| TabArena | meta_selector_meta_feature_guard_005_k12_rt0p03 | 1 |
| TabArena | meta_selector_meta_feature_guard_005_k4_rt0p03 | 1 |
| TabArena | meta_selector_meta_feature_guard_005_k8_rt0p1 | 1 |
| TabArena | meta_selector_meta_feature_guard_010_k12_rt0p1 | 1 |
| TabArena | meta_selector_meta_feature_guard_010_k4_rt0p03 | 1 |

## Paired Tests

| artifact | method | reference_method | n_splits | mean_rank_difference | mean_normalized_regret_difference | mean_runtime_difference_seconds | rank_wilcoxon_p_value | rank_holm_p_value |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| TabArena | nested_policy_selected | meta_selector_task_adaptive_budgeted | 100 | -0.0612549 | -0.00161721 | 305.601 | 0.0424378 | 0.0848756 |
| TabArena | nested_policy_selected | meta_selector_meta_feature_guard_005 | 100 | 0.237695 | 0.00544696 | 538.024 | 0.0633233 | 0.0848756 |
| TabArena | nested_policy_selected | meta_selector_family_prior_budgeted | 100 | 0.393887 | 0.009835 | -975.111 | 2.18872e-05 | 6.56617e-05 |
| TabArena | nested_policy_selected | meta_selector_ablate_validation_only | 100 | -4.04023 | -0.0934263 | -75490.2 | 2.34131e-17 | 1.17065e-16 |
| TabArena | nested_policy_selected | meta_selector_single | 100 | -4.04023 | -0.0934263 | -75490.2 | 2.34131e-17 | 1.17065e-16 |
