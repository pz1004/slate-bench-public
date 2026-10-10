# Fully Nested Score-Level Policy Benchmark

- Command: `scripts/score_level_policy_benchmark.py --config configs/score_level_tabarena_attribution_rank_fix.yaml --jobs 12`
- Completed artifact/split rows: 100
- Protocol: selector rows are rebuilt from outer-training datasets only; held-out datasets provide validation/runtime artifacts for target-side selection and test metrics for final evaluation only.

## Summary

| artifact | method | mean_rank | tied_best_rate | mean_total_seconds | mean_rank_regret | mean_normalized_rank_regret | n_splits |
| --- | --- | --- | --- | --- | --- | --- | --- |
| TabArena | oracle | 1.01684 | 1 | 46563.3 | 0.0168373 | 0.000391565 | 100 |
| TabArena | meta_selector_family_prior_budgeted | 8.78275 | 0.134191 | 4282.73 | 7.78275 | 0.180994 | 100 |
| TabArena | meta_selector_meta_feature_guard_002 | 8.91649 | 0.138729 | 2121.25 | 7.91649 | 0.184104 | 100 |
| TabArena | meta_selector_meta_feature_guard_002_k12_rt0 | 8.94404 | 0.136347 | 2630.28 | 7.94404 | 0.184745 | 100 |
| TabArena | meta_selector_meta_feature_guard_002_k8_rt0 | 8.94758 | 0.136195 | 2637.87 | 7.94758 | 0.184828 | 100 |
| TabArena | meta_selector_meta_feature_guard_002_k12_rt0p03 | 8.96301 | 0.139528 | 2067.51 | 7.96301 | 0.185186 | 100 |
| TabArena | meta_selector_meta_feature_guard_010_k8_rt0p1 | 8.99188 | 0.136485 | 2081.17 | 7.99188 | 0.185858 | 100 |
| TabArena | meta_selector_meta_feature_guard_010_k12_rt0p1 | 9.00011 | 0.136954 | 2088.21 | 8.00011 | 0.186049 | 100 |
| TabArena | meta_selector_meta_feature_guard_005 | 9.00121 | 0.136385 | 2310.34 | 8.00121 | 0.186075 | 100 |
| TabArena | meta_selector_meta_feature_guard_005_k12_rt0p03 | 9.03015 | 0.135292 | 2377.53 | 8.03015 | 0.186748 | 100 |
| TabArena | meta_selector_meta_feature_guard_005_k8_rt0 | 9.08567 | 0.131721 | 2804.53 | 8.08567 | 0.188039 | 100 |
| TabArena | meta_selector_meta_feature_guard_002_k4_rt0 | 9.08698 | 0.131565 | 2610.2 | 8.08698 | 0.188069 | 100 |
| TabArena | meta_selector_meta_feature_guard_005_k4_rt0p03 | 9.10965 | 0.131087 | 2450.86 | 8.10965 | 0.188597 | 100 |
| TabArena | meta_selector_meta_feature_guard_005_k4_rt0 | 9.11356 | 0.129712 | 2807.9 | 8.11356 | 0.188687 | 100 |
| TabArena | meta_selector_meta_feature_guard_005_k12_rt0 | 9.11799 | 0.131352 | 2797.48 | 8.11799 | 0.188791 | 100 |
| TabArena | nested_policy_selected | 9.12007 | 0.131081 | 3137.22 | 8.12007 | 0.188839 | 100 |
| TabArena | meta_selector_meta_feature_guard_010_k4_rt0p1 | 9.14369 | 0.13337 | 2184.44 | 8.14369 | 0.189388 | 100 |
| TabArena | meta_selector_meta_feature_guard_010_k4_rt0p03 | 9.18524 | 0.12945 | 2859.53 | 8.18524 | 0.190354 | 100 |
| TabArena | meta_selector_meta_feature_guard_005_k12_rt0p1 | 9.19116 | 0.135937 | 1535.43 | 8.19116 | 0.190492 | 100 |
| TabArena | meta_selector_meta_feature_guard_010_k4_rt0 | 9.2124 | 0.129256 | 2936.82 | 8.2124 | 0.190986 | 100 |
| TabArena | meta_selector_meta_feature_guard_010 | 9.21242 | 0.129525 | 2882.4 | 8.21242 | 0.190987 | 100 |
| TabArena | meta_selector_calibrated_budget_10000s | 9.23099 | 0.125373 | 4272.11 | 8.23099 | 0.191418 | 100 |
| TabArena | meta_selector_meta_feature_guard_020_k4_rt0p1 | 9.24466 | 0.128839 | 2939.98 | 8.24466 | 0.191736 | 100 |
| TabArena | meta_selector_meta_feature_guard_010_k12_rt0p03 | 9.26224 | 0.129485 | 2932.64 | 8.26224 | 0.192145 | 100 |
| TabArena | meta_selector_meta_feature_guard_010_k8_rt0 | 9.27203 | 0.128757 | 2988.38 | 8.27203 | 0.192373 | 100 |
| TabArena | meta_selector_meta_feature_guard_020 | 9.27334 | 0.128738 | 2987.15 | 8.27334 | 0.192403 | 100 |
| TabArena | meta_selector_meta_feature_guard_020_k12_rt0 | 9.27334 | 0.128738 | 2987.15 | 8.27334 | 0.192403 | 100 |
| TabArena | meta_selector_meta_feature_guard_020_k12_rt0p03 | 9.27334 | 0.128738 | 2987.15 | 8.27334 | 0.192403 | 100 |
| TabArena | meta_selector_meta_feature_guard_020_k12_rt0p1 | 9.27334 | 0.128738 | 2987.15 | 8.27334 | 0.192403 | 100 |
| TabArena | meta_selector_meta_feature_guard_020_k8_rt0 | 9.27334 | 0.128738 | 2987.15 | 8.27334 | 0.192403 | 100 |
| TabArena | meta_selector_task_adaptive_budgeted | 9.27334 | 0.128738 | 2987.15 | 8.27334 | 0.192403 | 100 |
| TabArena | meta_selector_meta_feature_guard_020_k8_rt0p1 | 9.27374 | 0.128738 | 2986.38 | 8.27374 | 0.192413 | 100 |
| TabArena | meta_selector_meta_feature_guard_010_k12_rt0 | 9.27387 | 0.128738 | 2986.3 | 8.27387 | 0.192416 | 100 |
| TabArena | meta_selector_meta_feature_guard_020_k4_rt0 | 9.27433 | 0.128697 | 2987.51 | 8.27433 | 0.192426 | 100 |
| TabArena | meta_selector_meta_feature_guard_020_k4_rt0p03 | 9.27433 | 0.128697 | 2987.51 | 8.27433 | 0.192426 | 100 |
| TabArena | meta_selector_meta_feature_guard_002_k4_rt0p03 | 9.2774 | 0.131697 | 2029.46 | 8.2774 | 0.192498 | 100 |
| TabArena | meta_selector_meta_feature_guard_005_k8_rt0p1 | 9.32683 | 0.135379 | 1358.94 | 8.32683 | 0.193647 | 100 |
| TabArena | meta_selector_meta_feature_guard_005_k4_rt0p1 | 9.53646 | 0.136389 | 1152.77 | 8.53646 | 0.198522 | 100 |
| TabArena | meta_selector_meta_feature_guard_002_k12_rt0p1 | 9.69253 | 0.135963 | 940.135 | 8.69253 | 0.202152 | 100 |
| TabArena | meta_selector_meta_feature_guard_002_k8_rt0p1 | 9.76678 | 0.139547 | 706.854 | 8.76678 | 0.203879 | 100 |
| TabArena | meta_selector_meta_feature_guard_002_k4_rt0p1 | 9.78196 | 0.141361 | 641.859 | 8.78196 | 0.204232 | 100 |
| TabArena | meta_selector_family_prior_budgeted_1000s | 9.95366 | 0.153241 | 170.126 | 8.95366 | 0.208225 | 100 |
| TabArena | meta_selector_family_prior_budgeted_500s | 9.97359 | 0.153188 | 76.5634 | 8.97359 | 0.208688 | 100 |
| TabArena | meta_selector_family_prior_budgeted_5000s | 10.2857 | 0.134521 | 1654.25 | 9.28571 | 0.215947 | 100 |
| TabArena | meta_selector_family_prior_budgeted_100s | 11.1202 | 0.112369 | 31.0628 | 10.1202 | 0.235354 | 100 |
| TabArena | meta_selector_ablate_validation_only | 13.1123 | 0.0980355 | 78032.4 | 12.1123 | 0.281681 | 100 |
| TabArena | meta_selector_single | 13.1123 | 0.0980355 | 78032.4 | 12.1123 | 0.281681 | 100 |

## Selected Policies

| artifact | selected_policy | n_splits |
| --- | --- | --- |
| TabArena | meta_selector_family_prior_budgeted | 38 |
| TabArena | meta_selector_meta_feature_guard_002_k8_rt0 | 10 |
| TabArena | meta_selector_meta_feature_guard_002_k4_rt0 | 7 |
| TabArena | meta_selector_calibrated_budget_10000s | 5 |
| TabArena | meta_selector_meta_feature_guard_002_k12_rt0 | 5 |
| TabArena | meta_selector_meta_feature_guard_002_k12_rt0p03 | 4 |
| TabArena | meta_selector_family_prior_budgeted_1000s | 3 |
| TabArena | meta_selector_meta_feature_guard_002_k8_rt0p1 | 3 |
| TabArena | meta_selector_meta_feature_guard_005_k8_rt0 | 3 |
| TabArena | meta_selector_task_adaptive_budgeted | 3 |
| TabArena | meta_selector_meta_feature_guard_002 | 2 |
| TabArena | meta_selector_meta_feature_guard_002_k4_rt0p03 | 2 |
| TabArena | meta_selector_meta_feature_guard_005 | 2 |
| TabArena | meta_selector_meta_feature_guard_005_k12_rt0p03 | 2 |
| TabArena | meta_selector_meta_feature_guard_005_k8_rt0p1 | 2 |
| TabArena | meta_selector_meta_feature_guard_010_k12_rt0p1 | 2 |
| TabArena | meta_selector_meta_feature_guard_010_k4_rt0p1 | 2 |
| TabArena | meta_selector_meta_feature_guard_002_k4_rt0p1 | 1 |
| TabArena | meta_selector_meta_feature_guard_005_k12_rt0 | 1 |
| TabArena | meta_selector_meta_feature_guard_005_k4_rt0p03 | 1 |
| TabArena | meta_selector_meta_feature_guard_010_k4_rt0 | 1 |
| TabArena | meta_selector_meta_feature_guard_010_k8_rt0p1 | 1 |

## Paired Tests

| artifact | method | reference_method | n_splits | mean_rank_difference | mean_normalized_regret_difference | mean_runtime_difference_seconds | rank_wilcoxon_p_value | rank_holm_p_value |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| TabArena | nested_policy_selected | meta_selector_task_adaptive_budgeted | 100 | -0.153272 | -0.00356447 | 150.066 | 0.271409 | 0.271409 |
| TabArena | nested_policy_selected | meta_selector_meta_feature_guard_005 | 100 | 0.118861 | 0.00276422 | 826.884 | 0.0854608 | 0.170922 |
| TabArena | nested_policy_selected | meta_selector_family_prior_budgeted | 100 | 0.337327 | 0.00784481 | -1145.52 | 0.000438013 | 0.00131404 |
| TabArena | nested_policy_selected | meta_selector_ablate_validation_only | 100 | -3.9922 | -0.092842 | -74895.2 | 1.90345e-17 | 9.51725e-17 |
| TabArena | nested_policy_selected | meta_selector_single | 100 | -3.9922 | -0.092842 | -74895.2 | 1.90345e-17 | 9.51725e-17 |
