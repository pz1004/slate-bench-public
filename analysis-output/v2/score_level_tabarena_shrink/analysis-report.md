# Fully Nested Score-Level Policy Benchmark

- Command: `scripts/score_level_policy_benchmark.py --config configs/score_level_tabarena_shrink.yaml --jobs 10`
- Completed artifact/split rows: 100
- Protocol: selector rows are rebuilt from outer-training datasets only; held-out datasets provide validation/runtime artifacts for target-side selection and test metrics for final evaluation only.

## Summary

| artifact | method | mean_rank | tied_best_rate | mean_total_seconds | mean_rank_regret | mean_normalized_rank_regret | n_splits |
| --- | --- | --- | --- | --- | --- | --- | --- |
| TabArena | oracle | 1.01684 | 1 | 46563.3 | 0.0168373 | 0.000391565 | 100 |
| TabArena | meta_selector_family_prior_budgeted | 8.72219 | 0.134403 | 4385.25 | 7.72219 | 0.179586 | 100 |
| TabArena | meta_selector_meta_feature_guard_002 | 8.82612 | 0.130483 | 2647.7 | 7.82612 | 0.182003 | 100 |
| TabArena | meta_selector_meta_feature_guard_005 | 8.90626 | 0.130426 | 3028.83 | 7.90626 | 0.183867 | 100 |
| TabArena | nested_policy_selected | 8.99206 | 0.130192 | 3580.05 | 7.99206 | 0.185862 | 100 |
| TabArena | meta_selector_task_adaptive_budgeted | 9.07425 | 0.128328 | 3466.97 | 8.07425 | 0.187773 | 100 |
| TabArena | meta_selector_calibrated_budget_10000s | 9.09374 | 0.125814 | 4433.2 | 8.09374 | 0.188227 | 100 |
| TabArena | meta_selector_family_prior_budgeted_1000s | 9.99849 | 0.136353 | 237.802 | 8.99849 | 0.209267 | 100 |
| TabArena | meta_selector_ablate_validation_only | 13.1123 | 0.0980355 | 78032.4 | 12.1123 | 0.281681 | 100 |
| TabArena | meta_selector_single | 13.1123 | 0.0980355 | 78032.4 | 12.1123 | 0.281681 | 100 |

## Selected Policies

| artifact | selected_policy | n_splits |
| --- | --- | --- |
| TabArena | meta_selector_family_prior_budgeted | 48 |
| TabArena | meta_selector_meta_feature_guard_002 | 22 |
| TabArena | meta_selector_task_adaptive_budgeted | 15 |
| TabArena | meta_selector_meta_feature_guard_005 | 13 |
| TabArena | meta_selector_calibrated_budget_10000s | 2 |

## Paired Tests

| artifact | method | reference_method | n_splits | mean_rank_difference | mean_normalized_regret_difference | mean_runtime_difference_seconds | rank_wilcoxon_p_value | rank_holm_p_value |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| TabArena | nested_policy_selected | meta_selector_task_adaptive_budgeted | 100 | -0.0821828 | -0.00191123 | 113.08 | 0.239635 | 0.47927 |
| TabArena | nested_policy_selected | meta_selector_meta_feature_guard_005 | 100 | 0.0858015 | 0.00199538 | 551.222 | 0.325358 | 0.47927 |
| TabArena | nested_policy_selected | meta_selector_family_prior_budgeted | 100 | 0.269877 | 0.0062762 | -805.2 | 9.7185e-06 | 2.91555e-05 |
| TabArena | nested_policy_selected | meta_selector_ablate_validation_only | 100 | -4.12021 | -0.0958189 | -74452.3 | 3.89656e-18 | 1.94828e-17 |
| TabArena | nested_policy_selected | meta_selector_single | 100 | -4.12021 | -0.0958189 | -74452.3 | 3.89656e-18 | 1.94828e-17 |
