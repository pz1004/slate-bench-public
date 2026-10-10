# Fully Nested Score-Level Policy Benchmark

- Command: `scripts/score_level_policy_pool_ablation.py --nested-output analysis-output/v2/attribution/inner_shared_tabrepo --output-dir analysis-output/v2/attribution/inner_shared_tabrepo_score_only`
- Completed artifact/split rows: 100
- Protocol: selector rows are rebuilt from outer-training datasets only; held-out datasets provide validation/runtime artifacts for target-side selection and test metrics for final evaluation only.

## Summary

| artifact | method | mean_rank | tied_best_rate | mean_total_seconds | mean_rank_regret | mean_normalized_rank_regret | n_splits |
| --- | --- | --- | --- | --- | --- | --- | --- |
| TabRepo | oracle | 12.217 | 1 | 198.985 | 11.217 | 0.00739002 | 100 |
| TabRepo | meta_selector_meta_feature_guard_005_k8_rt0 | 261.663 | 0.0401333 | 248.985 | 260.663 | 0.171121 | 100 |
| TabRepo | meta_selector_meta_feature_guard_002_k12_rt0 | 261.694 | 0.0408667 | 255.149 | 260.694 | 0.171138 | 100 |
| TabRepo | meta_selector_meta_feature_guard_005_k4_rt0 | 261.988 | 0.0402667 | 251.439 | 260.988 | 0.171333 | 100 |
| TabRepo | meta_selector_meta_feature_guard_010_k4_rt0p03 | 262.026 | 0.0401333 | 248.616 | 261.026 | 0.171358 | 100 |
| TabRepo | meta_selector_meta_feature_guard_005_k4_rt0p03 | 262.054 | 0.0405333 | 243.65 | 261.054 | 0.171372 | 100 |
| TabRepo | meta_selector_meta_feature_guard_005_k12_rt0 | 262.104 | 0.04 | 249.237 | 261.104 | 0.17141 | 100 |
| TabRepo | meta_selector_meta_feature_guard_002_k4_rt0 | 262.131 | 0.0426667 | 252.586 | 261.131 | 0.171426 | 100 |
| TabRepo | meta_selector_meta_feature_guard_005 | 262.188 | 0.0398667 | 245.378 | 261.188 | 0.171464 | 100 |
| TabRepo | meta_selector_meta_feature_guard_010 | 262.239 | 0.04 | 248.984 | 261.239 | 0.171498 | 100 |
| TabRepo | meta_selector_meta_feature_guard_020_k4_rt0p1 | 262.261 | 0.04 | 249 | 261.261 | 0.171511 | 100 |
| TabRepo | meta_selector_meta_feature_guard_010_k4_rt0 | 262.313 | 0.04 | 249.565 | 261.313 | 0.171545 | 100 |
| TabRepo | meta_selector_meta_feature_guard_002_k8_rt0 | 262.31 | 0.0414 | 254.951 | 261.31 | 0.171546 | 100 |
| TabRepo | meta_selector_meta_feature_guard_010_k8_rt0 | 262.336 | 0.04 | 249.479 | 261.336 | 0.171561 | 100 |
| TabRepo | meta_selector_meta_feature_guard_005_k12_rt0p03 | 262.445 | 0.0397333 | 245.807 | 261.445 | 0.171631 | 100 |
| TabRepo | meta_selector_meta_feature_guard_020_k8_rt0p1 | 262.481 | 0.04 | 249.548 | 261.481 | 0.171656 | 100 |
| TabRepo | meta_selector_meta_feature_guard_010_k8_rt0p1 | 262.526 | 0.0385333 | 234.288 | 261.526 | 0.17168 | 100 |
| TabRepo | meta_selector_meta_feature_guard_020_k4_rt0 | 262.581 | 0.04 | 249.545 | 261.581 | 0.171721 | 100 |
| TabRepo | meta_selector_meta_feature_guard_020_k4_rt0p03 | 262.581 | 0.04 | 249.545 | 261.581 | 0.171721 | 100 |
| TabRepo | meta_selector_meta_feature_guard_010_k12_rt0 | 262.595 | 0.04 | 249.538 | 261.595 | 0.17173 | 100 |
| TabRepo | meta_selector_meta_feature_guard_020_k12_rt0p1 | 262.597 | 0.04 | 249.551 | 261.597 | 0.171731 | 100 |
| TabRepo | meta_selector_meta_feature_guard_020 | 262.614 | 0.04 | 249.56 | 261.614 | 0.171743 | 100 |
| TabRepo | meta_selector_meta_feature_guard_020_k12_rt0p03 | 262.628 | 0.04 | 249.564 | 261.628 | 0.171752 | 100 |
| TabRepo | meta_selector_meta_feature_guard_020_k8_rt0 | 262.628 | 0.04 | 249.564 | 261.628 | 0.171752 | 100 |
| TabRepo | meta_selector_meta_feature_guard_010_k12_rt0p03 | 262.632 | 0.04 | 249.408 | 261.632 | 0.171755 | 100 |
| TabRepo | meta_selector_meta_feature_guard_020_k12_rt0 | 262.642 | 0.04 | 249.569 | 261.642 | 0.171761 | 100 |
| TabRepo | meta_selector_task_adaptive_budgeted | 262.642 | 0.04 | 249.569 | 261.642 | 0.171761 | 100 |
| TabRepo | meta_selector_meta_feature_guard_010_k4_rt0p1 | 263.305 | 0.0402 | 221.753 | 262.305 | 0.172194 | 100 |
| TabRepo | meta_selector_meta_feature_guard_010_k12_rt0p1 | 263.546 | 0.0383333 | 234.779 | 262.546 | 0.172349 | 100 |
| TabRepo | meta_selector_family_prior_budgeted_5000s | 264.178 | 0.0418667 | 353.507 | 263.178 | 0.17277 | 100 |
| TabRepo | meta_selector_family_prior_budgeted | 264.219 | 0.0418667 | 353.485 | 263.219 | 0.172797 | 100 |
| TabRepo | nested_policy_selected | 264.679 | 0.042 | 245.726 | 263.679 | 0.1731 | 100 |
| TabRepo | meta_selector_family_prior_budgeted_1000s | 265.265 | 0.0372 | 160.688 | 264.265 | 0.173473 | 100 |
| TabRepo | meta_selector_meta_feature_guard_002 | 265.347 | 0.0366667 | 201.587 | 264.347 | 0.173528 | 100 |
| TabRepo | meta_selector_meta_feature_guard_002_k12_rt0p03 | 266.572 | 0.0354667 | 205.825 | 265.572 | 0.174328 | 100 |
| TabRepo | meta_selector_meta_feature_guard_002_k4_rt0p03 | 267.299 | 0.0396 | 176.361 | 266.299 | 0.174815 | 100 |
| TabRepo | meta_selector_single | 270.581 | 0.0744667 | 268.936 | 269.581 | 0.177041 | 100 |
| TabRepo | meta_selector_ablate_validation_only | 271.59 | 0.076 | 269.645 | 270.59 | 0.177701 | 100 |
| TabRepo | meta_selector_family_prior_budgeted_500s | 272.069 | 0.0332 | 107.331 | 271.069 | 0.177943 | 100 |
| TabRepo | meta_selector_meta_feature_guard_005_k4_rt0p1 | 275.045 | 0.0336667 | 103.957 | 274.045 | 0.179887 | 100 |
| TabRepo | meta_selector_family_prior_budgeted_100s | 275.494 | 0.0274667 | 31.6927 | 274.494 | 0.180178 | 100 |
| TabRepo | meta_selector_meta_feature_guard_005_k8_rt0p1 | 275.733 | 0.031 | 93.0511 | 274.733 | 0.180339 | 100 |
| TabRepo | meta_selector_meta_feature_guard_005_k12_rt0p1 | 275.902 | 0.03 | 87.9183 | 274.902 | 0.180445 | 100 |
| TabRepo | meta_selector_meta_feature_guard_002_k8_rt0p1 | 276.541 | 0.0294 | 42.9101 | 275.541 | 0.180867 | 100 |
| TabRepo | meta_selector_meta_feature_guard_002_k4_rt0p1 | 276.726 | 0.0328 | 56.4351 | 275.726 | 0.180988 | 100 |
| TabRepo | meta_selector_meta_feature_guard_002_k12_rt0p1 | 277.38 | 0.0286 | 39.2407 | 276.38 | 0.181414 | 100 |
| TabRepo | meta_selector_calibrated_budget_10000s | 279.919 | 0.0514667 | 312.124 | 278.919 | 0.183115 | 100 |
| TabRepo | reference_metafeature_knn_validation | 280.744 | 0.0281333 | 279.552 | 279.744 | 0.183672 | 100 |
| TabRepo | reference_optimism_corrected | 287.339 | 0.0396 | 141.944 | 286.339 | 0.188055 | 100 |
| TabRepo | reference_optimism_corrected_budget | 287.339 | 0.0396 | 141.944 | 286.339 | 0.188055 | 100 |
| TabRepo | reference_optimism_corrected_task | 289.599 | 0.0374667 | 159.914 | 288.599 | 0.189586 | 100 |
| TabRepo | reference_single_best_task | 379.512 | 0.0201333 | 453.009 | 378.512 | 0.248722 | 100 |
| TabRepo | reference_single_best_global | 384.095 | 0.0208 | 368.664 | 383.095 | 0.251806 | 100 |
| TabRepo | reference_priors_only | 422.841 | 0.0186 | 163.063 | 421.841 | 0.277335 | 100 |
| TabRepo | reference_metafeature_knn | 461.813 | 0.0197333 | 255.215 | 460.813 | 0.302899 | 100 |

## Selected Policies

| artifact | selected_policy | n_splits |
| --- | --- | --- |
| TabRepo | meta_selector_task_adaptive_budgeted | 19 |
| TabRepo | meta_selector_family_prior_budgeted_1000s | 9 |
| TabRepo | meta_selector_meta_feature_guard_005_k4_rt0 | 9 |
| TabRepo | meta_selector_meta_feature_guard_005_k4_rt0p03 | 9 |
| TabRepo | meta_selector_family_prior_budgeted_5000s | 8 |
| TabRepo | meta_selector_single | 8 |
| TabRepo | meta_selector_meta_feature_guard_002_k4_rt0 | 7 |
| TabRepo | meta_selector_meta_feature_guard_002_k8_rt0 | 4 |
| TabRepo | meta_selector_meta_feature_guard_010_k4_rt0p1 | 4 |
| TabRepo | meta_selector_meta_feature_guard_010_k8_rt0p1 | 4 |
| TabRepo | meta_selector_meta_feature_guard_002 | 3 |
| TabRepo | meta_selector_meta_feature_guard_005 | 3 |
| TabRepo | meta_selector_meta_feature_guard_002_k12_rt0p03 | 2 |
| TabRepo | meta_selector_meta_feature_guard_002_k4_rt0p03 | 2 |
| TabRepo | meta_selector_meta_feature_guard_005_k12_rt0 | 2 |
| TabRepo | meta_selector_meta_feature_guard_005_k8_rt0 | 2 |
| TabRepo | meta_selector_calibrated_budget_10000s | 1 |
| TabRepo | meta_selector_family_prior_budgeted_500s | 1 |
| TabRepo | meta_selector_meta_feature_guard_002_k12_rt0 | 1 |
| TabRepo | meta_selector_meta_feature_guard_005_k8_rt0p1 | 1 |
| TabRepo | meta_selector_meta_feature_guard_010_k4_rt0 | 1 |

## Paired Tests

| artifact | method | reference_method | n_splits | mean_rank_difference | mean_normalized_regret_difference | mean_runtime_difference_seconds | rank_wilcoxon_p_value | rank_holm_p_value |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| TabRepo | nested_policy_selected | meta_selector_task_adaptive_budgeted | 100 | 2.03727 | 0.0013386 | -3.84326 | 0.0615154 | 0.184546 |
| TabRepo | nested_policy_selected | meta_selector_meta_feature_guard_005 | 100 | 2.49147 | 0.0016361 | 0.348278 | 0.0710685 | 0.184546 |
| TabRepo | nested_policy_selected | meta_selector_family_prior_budgeted | 100 | 0.4606 | 0.000302846 | -107.759 | 0.827674 | 0.827674 |
| TabRepo | nested_policy_selected | meta_selector_ablate_validation_only | 100 | -6.91033 | -0.00460109 | -23.9189 | 0.00173765 | 0.00868825 |
| TabRepo | nested_policy_selected | meta_selector_single | 100 | -5.90183 | -0.00394151 | -23.2104 | 0.00604746 | 0.0241898 |
