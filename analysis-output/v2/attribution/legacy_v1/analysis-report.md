# Fully Nested Score-Level Policy Benchmark

- Command: `scripts/score_level_policy_benchmark.py --config configs/score_level_tabarena_attribution_legacy.yaml --jobs 12`
- Completed artifact/split rows: 100
- Protocol: selector rows are rebuilt from outer-training datasets only; held-out datasets provide validation/runtime artifacts for target-side selection and test metrics for final evaluation only.

## Summary

| artifact | method | mean_rank | tied_best_rate | mean_total_seconds | mean_rank_regret | mean_normalized_rank_regret | n_splits |
| --- | --- | --- | --- | --- | --- | --- | --- |
| TabArena | oracle | 1 | 1 | 46563.3 | 0 | 0 | 100 |
| TabArena | meta_selector_family_prior_budgeted | 8.7874 | 0.134011 | 4269.78 | 7.7874 | 0.181102 | 100 |
| TabArena | meta_selector_meta_feature_guard_002 | 8.93464 | 0.13833 | 2088.95 | 7.93464 | 0.184526 | 100 |
| TabArena | meta_selector_meta_feature_guard_002_k8_rt0 | 8.93884 | 0.136577 | 2634.4 | 7.93884 | 0.184624 | 100 |
| TabArena | meta_selector_meta_feature_guard_002_k12_rt0 | 8.94266 | 0.13687 | 2609.12 | 7.94266 | 0.184713 | 100 |
| TabArena | meta_selector_meta_feature_guard_002_k12_rt0p03 | 8.99382 | 0.138798 | 2047.5 | 7.99382 | 0.185903 | 100 |
| TabArena | meta_selector_meta_feature_guard_010_k12_rt0p1 | 9.00578 | 0.137096 | 2084.86 | 8.00578 | 0.186181 | 100 |
| TabArena | meta_selector_meta_feature_guard_010_k8_rt0p1 | 9.01508 | 0.136511 | 2074.46 | 8.01508 | 0.186397 | 100 |
| TabArena | meta_selector_meta_feature_guard_005 | 9.02631 | 0.13624 | 2296.88 | 8.02631 | 0.186658 | 100 |
| TabArena | meta_selector_meta_feature_guard_005_k12_rt0p03 | 9.04116 | 0.13561 | 2359.92 | 8.04116 | 0.187004 | 100 |
| TabArena | meta_selector_meta_feature_guard_002_k4_rt0 | 9.10018 | 0.131346 | 2598.9 | 8.10018 | 0.188376 | 100 |
| TabArena | meta_selector_meta_feature_guard_005_k8_rt0 | 9.11026 | 0.131203 | 2820.14 | 8.11026 | 0.188611 | 100 |
| TabArena | meta_selector_meta_feature_guard_005_k4_rt0p03 | 9.12489 | 0.13089 | 2410.24 | 8.12489 | 0.188951 | 100 |
| TabArena | meta_selector_meta_feature_guard_005_k12_rt0 | 9.13722 | 0.131513 | 2796.09 | 8.13722 | 0.189238 | 100 |
| TabArena | meta_selector_meta_feature_guard_005_k4_rt0 | 9.14546 | 0.129635 | 2820.29 | 8.14546 | 0.189429 | 100 |
| TabArena | nested_policy_selected | 9.16055 | 0.129718 | 3208.96 | 8.16055 | 0.18978 | 100 |
| TabArena | meta_selector_meta_feature_guard_010_k4_rt0p1 | 9.17085 | 0.132752 | 2184.15 | 8.17085 | 0.19002 | 100 |
| TabArena | meta_selector_meta_feature_guard_010_k4_rt0p03 | 9.20527 | 0.129762 | 2836.02 | 8.20527 | 0.19082 | 100 |
| TabArena | meta_selector_meta_feature_guard_005_k12_rt0p1 | 9.21994 | 0.136564 | 1505.18 | 8.21994 | 0.191161 | 100 |
| TabArena | meta_selector_meta_feature_guard_010 | 9.22202 | 0.130125 | 2864.93 | 8.22202 | 0.19121 | 100 |
| TabArena | meta_selector_meta_feature_guard_010_k4_rt0 | 9.22694 | 0.129588 | 2903.48 | 8.22694 | 0.191324 | 100 |
| TabArena | meta_selector_calibrated_budget_10000s | 9.23293 | 0.125373 | 4266.46 | 8.23293 | 0.191464 | 100 |
| TabArena | meta_selector_meta_feature_guard_020_k4_rt0p1 | 9.26631 | 0.128948 | 2902.03 | 8.26631 | 0.19224 | 100 |
| TabArena | meta_selector_meta_feature_guard_002_k4_rt0p03 | 9.27182 | 0.131704 | 1991.87 | 8.27182 | 0.192368 | 100 |
| TabArena | meta_selector_meta_feature_guard_010_k12_rt0p03 | 9.28886 | 0.129413 | 2906.01 | 8.28886 | 0.192764 | 100 |
| TabArena | meta_selector_meta_feature_guard_020 | 9.29535 | 0.128847 | 2948.55 | 8.29535 | 0.192915 | 100 |
| TabArena | meta_selector_meta_feature_guard_020_k12_rt0 | 9.29535 | 0.128847 | 2948.55 | 8.29535 | 0.192915 | 100 |
| TabArena | meta_selector_meta_feature_guard_020_k12_rt0p03 | 9.29535 | 0.128847 | 2948.55 | 8.29535 | 0.192915 | 100 |
| TabArena | meta_selector_meta_feature_guard_020_k12_rt0p1 | 9.29535 | 0.128847 | 2948.55 | 8.29535 | 0.192915 | 100 |
| TabArena | meta_selector_meta_feature_guard_020_k8_rt0 | 9.29535 | 0.128847 | 2948.55 | 8.29535 | 0.192915 | 100 |
| TabArena | meta_selector_task_adaptive_budgeted | 9.29535 | 0.128847 | 2948.55 | 8.29535 | 0.192915 | 100 |
| TabArena | meta_selector_meta_feature_guard_010_k12_rt0 | 9.29554 | 0.128888 | 2947.9 | 8.29554 | 0.19292 | 100 |
| TabArena | meta_selector_meta_feature_guard_020_k8_rt0p1 | 9.29574 | 0.128847 | 2947.78 | 8.29574 | 0.192924 | 100 |
| TabArena | meta_selector_meta_feature_guard_020_k4_rt0 | 9.29642 | 0.128806 | 2949.2 | 8.29642 | 0.19294 | 100 |
| TabArena | meta_selector_meta_feature_guard_020_k4_rt0p03 | 9.29642 | 0.128806 | 2949.2 | 8.29642 | 0.19294 | 100 |
| TabArena | meta_selector_meta_feature_guard_010_k8_rt0 | 9.29788 | 0.128939 | 2936.03 | 8.29788 | 0.192974 | 100 |
| TabArena | meta_selector_meta_feature_guard_005_k8_rt0p1 | 9.35534 | 0.136003 | 1326.07 | 8.35534 | 0.19431 | 100 |
| TabArena | meta_selector_meta_feature_guard_005_k4_rt0p1 | 9.53578 | 0.135623 | 1163.97 | 8.53578 | 0.198506 | 100 |
| TabArena | meta_selector_meta_feature_guard_002_k12_rt0p1 | 9.6804 | 0.135734 | 947.953 | 8.6804 | 0.20187 | 100 |
| TabArena | meta_selector_meta_feature_guard_002_k8_rt0p1 | 9.76526 | 0.139734 | 706.649 | 8.76526 | 0.203843 | 100 |
| TabArena | meta_selector_meta_feature_guard_002_k4_rt0p1 | 9.80348 | 0.140381 | 632.517 | 8.80348 | 0.204732 | 100 |
| TabArena | meta_selector_family_prior_budgeted_1000s | 9.9602 | 0.1532 | 170.029 | 8.9602 | 0.208377 | 100 |
| TabArena | meta_selector_family_prior_budgeted_500s | 9.9896 | 0.152767 | 76.6919 | 8.9896 | 0.20906 | 100 |
| TabArena | meta_selector_family_prior_budgeted_5000s | 10.282 | 0.134327 | 1652.62 | 9.28204 | 0.215861 | 100 |
| TabArena | meta_selector_family_prior_budgeted_100s | 11.1384 | 0.112485 | 31.062 | 10.1384 | 0.235776 | 100 |
| TabArena | meta_selector_ablate_validation_only | 13.1136 | 0.0980355 | 78032.4 | 12.1136 | 0.281711 | 100 |
| TabArena | meta_selector_single | 13.1136 | 0.0980355 | 78032.4 | 12.1136 | 0.281711 | 100 |

## Selected Policies

| artifact | selected_policy | n_splits |
| --- | --- | --- |
| TabArena | meta_selector_family_prior_budgeted | 35 |
| TabArena | meta_selector_meta_feature_guard_002_k8_rt0 | 10 |
| TabArena | meta_selector_calibrated_budget_10000s | 7 |
| TabArena | meta_selector_meta_feature_guard_002_k4_rt0 | 7 |
| TabArena | meta_selector_meta_feature_guard_002_k12_rt0 | 6 |
| TabArena | meta_selector_meta_feature_guard_002_k4_rt0p03 | 5 |
| TabArena | meta_selector_meta_feature_guard_002_k12_rt0p03 | 4 |
| TabArena | meta_selector_meta_feature_guard_005_k4_rt0 | 3 |
| TabArena | meta_selector_meta_feature_guard_005_k8_rt0 | 3 |
| TabArena | meta_selector_task_adaptive_budgeted | 3 |
| TabArena | meta_selector_family_prior_budgeted_1000s | 2 |
| TabArena | meta_selector_meta_feature_guard_002 | 2 |
| TabArena | meta_selector_meta_feature_guard_005_k12_rt0p03 | 2 |
| TabArena | meta_selector_meta_feature_guard_005_k8_rt0p1 | 2 |
| TabArena | meta_selector_meta_feature_guard_010_k4_rt0p1 | 2 |
| TabArena | meta_selector_meta_feature_guard_002_k4_rt0p1 | 1 |
| TabArena | meta_selector_meta_feature_guard_002_k8_rt0p1 | 1 |
| TabArena | meta_selector_meta_feature_guard_005 | 1 |
| TabArena | meta_selector_meta_feature_guard_005_k12_rt0 | 1 |
| TabArena | meta_selector_meta_feature_guard_005_k4_rt0p03 | 1 |
| TabArena | meta_selector_meta_feature_guard_010_k12_rt0p1 | 1 |
| TabArena | meta_selector_meta_feature_guard_010_k8_rt0p1 | 1 |

## Paired Tests

| artifact | method | reference_method | n_splits | mean_rank_difference | mean_normalized_regret_difference | mean_runtime_difference_seconds | rank_wilcoxon_p_value | rank_holm_p_value |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| TabArena | nested_policy_selected | meta_selector_task_adaptive_budgeted | 100 | -0.134799 | -0.00313486 | 260.412 | 0.536634 | 0.536634 |
| TabArena | nested_policy_selected | meta_selector_meta_feature_guard_005 | 100 | 0.134236 | 0.00312178 | 912.084 | 0.0481033 | 0.0962066 |
| TabArena | nested_policy_selected | meta_selector_family_prior_budgeted | 100 | 0.373149 | 0.00867788 | -1060.82 | 4.11952e-05 | 0.000123586 |
| TabArena | nested_policy_selected | meta_selector_ablate_validation_only | 100 | -3.95302 | -0.0919308 | -74823.4 | 2.01956e-17 | 1.00978e-16 |
| TabArena | nested_policy_selected | meta_selector_single | 100 | -3.95302 | -0.0919308 | -74823.4 | 2.01956e-17 | 1.00978e-16 |
