# Fully Nested Score-Level Policy Benchmark

- Command: `scripts/grouped_inner_splits.py --config configs/score_level_tabrepo_nested.yaml --repeats 1 --split-seed 20261022 --jobs 4 --resume --output-dir analysis-output/v2/score_level_tabrepo_frozen_fit`
- Completed artifact/split rows: 4
- Protocol: selector rows are rebuilt from outer-training datasets only; held-out datasets provide validation/runtime artifacts for target-side selection and test metrics for final evaluation only.

## Summary

| artifact | method | mean_rank | tied_best_rate | mean_total_seconds | mean_rank_regret | mean_normalized_rank_regret | n_splits |
| --- | --- | --- | --- | --- | --- | --- | --- |
| TabRepo | oracle | 12.1225 | 1 | 190.495 | 11.1225 | 0.00732779 | 4 |
| TabRepo | meta_selector_tabrepo_ensemble_k3 | 222.472 | 0.0233333 | 703.067 | 221.472 | 0.145425 | 4 |
| TabRepo | nested_policy_selected | 222.472 | 0.0233333 | 703.067 | 221.472 | 0.145425 | 4 |
| TabRepo | meta_selector_tabrepo_ensemble_k5 | 223.712 | 0.0233333 | 703.067 | 222.712 | 0.146263 | 4 |
| TabRepo | meta_selector_tabrepo_ensemble_k10 | 223.814 | 0.0233333 | 703.067 | 222.814 | 0.146328 | 4 |
| TabRepo | meta_selector_meta_feature_guard_002_k4_rt0p1 | 254.372 | 0.0283333 | 53.7605 | 253.372 | 0.166353 | 4 |
| TabRepo | meta_selector_meta_feature_guard_010_k8_rt0p1 | 255.922 | 0.0333333 | 238.47 | 254.922 | 0.167358 | 4 |
| TabRepo | meta_selector_meta_feature_guard_002_k8_rt0 | 255.941 | 0.0366667 | 251.645 | 254.941 | 0.167367 | 4 |
| TabRepo | meta_selector_family_prior_budgeted_5000s | 256.017 | 0.0366667 | 347.642 | 255.017 | 0.167435 | 4 |
| TabRepo | meta_selector_meta_feature_guard_002_k4_rt0p03 | 256.23 | 0.035 | 194.267 | 255.23 | 0.167575 | 4 |
| TabRepo | meta_selector_meta_feature_guard_010_k4_rt0p1 | 256.311 | 0.0333333 | 220.68 | 255.311 | 0.167623 | 4 |
| TabRepo | meta_selector_family_prior_budgeted | 256.332 | 0.0366667 | 347.61 | 255.332 | 0.167641 | 4 |
| TabRepo | meta_selector_meta_feature_guard_005_k4_rt0p03 | 256.659 | 0.035 | 235.916 | 255.659 | 0.167843 | 4 |
| TabRepo | meta_selector_meta_feature_guard_005_k4_rt0p1 | 257.461 | 0.03 | 103.775 | 256.461 | 0.168375 | 4 |
| TabRepo | meta_selector_meta_feature_guard_002 | 257.643 | 0.0316667 | 227.993 | 256.643 | 0.168478 | 4 |
| TabRepo | meta_selector_meta_feature_guard_010_k12_rt0p1 | 257.727 | 0.0316667 | 244.982 | 256.727 | 0.16854 | 4 |
| TabRepo | meta_selector_meta_feature_guard_002_k12_rt0p03 | 258.16 | 0.0283333 | 219.936 | 257.16 | 0.168827 | 4 |
| TabRepo | meta_selector_meta_feature_guard_002_k4_rt0 | 259.296 | 0.0416667 | 255.559 | 258.296 | 0.169569 | 4 |
| TabRepo | meta_selector_meta_feature_guard_002_k8_rt0p1 | 259.595 | 0.0266667 | 41.2534 | 258.595 | 0.16976 | 4 |
| TabRepo | meta_selector_family_prior_budgeted_1000s | 260.023 | 0.0333333 | 160.685 | 259.023 | 0.170065 | 4 |
| TabRepo | meta_selector_meta_feature_guard_005_k8_rt0p1 | 260.186 | 0.0283333 | 131.42 | 259.186 | 0.170168 | 4 |
| TabRepo | meta_selector_family_prior_budgeted_500s | 260.785 | 0.03 | 108.554 | 259.785 | 0.170585 | 4 |
| TabRepo | meta_selector_meta_feature_guard_005 | 260.839 | 0.0333333 | 286.373 | 259.839 | 0.170593 | 4 |
| TabRepo | meta_selector_meta_feature_guard_005_k4_rt0 | 261.593 | 0.035 | 283.508 | 260.593 | 0.171086 | 4 |
| TabRepo | meta_selector_meta_feature_guard_005_k12_rt0p03 | 262.825 | 0.0316667 | 286.942 | 261.825 | 0.171892 | 4 |
| TabRepo | meta_selector_meta_feature_guard_002_k12_rt0 | 262.851 | 0.035 | 290.5 | 261.851 | 0.171908 | 4 |
| TabRepo | meta_selector_meta_feature_guard_005_k12_rt0 | 263.005 | 0.0316667 | 287.697 | 262.005 | 0.17201 | 4 |
| TabRepo | meta_selector_meta_feature_guard_005_k8_rt0 | 263.005 | 0.0316667 | 287.697 | 262.005 | 0.17201 | 4 |
| TabRepo | meta_selector_meta_feature_guard_010 | 263.005 | 0.0316667 | 287.697 | 262.005 | 0.17201 | 4 |
| TabRepo | meta_selector_meta_feature_guard_010_k12_rt0 | 263.005 | 0.0316667 | 287.697 | 262.005 | 0.17201 | 4 |
| TabRepo | meta_selector_meta_feature_guard_010_k12_rt0p03 | 263.005 | 0.0316667 | 287.697 | 262.005 | 0.17201 | 4 |
| TabRepo | meta_selector_meta_feature_guard_010_k4_rt0 | 263.005 | 0.0316667 | 287.697 | 262.005 | 0.17201 | 4 |
| TabRepo | meta_selector_meta_feature_guard_010_k8_rt0 | 263.005 | 0.0316667 | 287.697 | 262.005 | 0.17201 | 4 |
| TabRepo | meta_selector_meta_feature_guard_020 | 263.005 | 0.0316667 | 287.697 | 262.005 | 0.17201 | 4 |
| TabRepo | meta_selector_meta_feature_guard_020_k12_rt0 | 263.005 | 0.0316667 | 287.697 | 262.005 | 0.17201 | 4 |
| TabRepo | meta_selector_meta_feature_guard_020_k12_rt0p03 | 263.005 | 0.0316667 | 287.697 | 262.005 | 0.17201 | 4 |
| TabRepo | meta_selector_meta_feature_guard_020_k12_rt0p1 | 263.005 | 0.0316667 | 287.697 | 262.005 | 0.17201 | 4 |
| TabRepo | meta_selector_meta_feature_guard_020_k4_rt0 | 263.005 | 0.0316667 | 287.697 | 262.005 | 0.17201 | 4 |
| TabRepo | meta_selector_meta_feature_guard_020_k4_rt0p1 | 263.005 | 0.0316667 | 287.697 | 262.005 | 0.17201 | 4 |
| TabRepo | meta_selector_meta_feature_guard_020_k8_rt0 | 263.005 | 0.0316667 | 287.697 | 262.005 | 0.17201 | 4 |
| TabRepo | meta_selector_meta_feature_guard_020_k8_rt0p1 | 263.005 | 0.0316667 | 287.697 | 262.005 | 0.17201 | 4 |
| TabRepo | meta_selector_task_adaptive_budgeted | 263.005 | 0.0316667 | 287.697 | 262.005 | 0.17201 | 4 |
| TabRepo | meta_selector_meta_feature_guard_010_k4_rt0p03 | 263.328 | 0.0316667 | 288.223 | 262.328 | 0.172222 | 4 |
| TabRepo | meta_selector_meta_feature_guard_020_k4_rt0p03 | 263.328 | 0.0316667 | 288.223 | 262.328 | 0.172222 | 4 |
| TabRepo | meta_selector_meta_feature_guard_002_k12_rt0p1 | 263.81 | 0.025 | 37.4809 | 262.81 | 0.172517 | 4 |
| TabRepo | meta_selector_meta_feature_guard_005_k12_rt0p1 | 264.168 | 0.0283333 | 101.216 | 263.168 | 0.17277 | 4 |
| TabRepo | meta_selector_family_prior_budgeted_100s | 264.221 | 0.025 | 31.2863 | 263.221 | 0.172786 | 4 |
| TabRepo | meta_selector_single | 266.799 | 0.075 | 264.487 | 265.799 | 0.174602 | 4 |
| TabRepo | meta_selector_ablate_validation_only | 267.89 | 0.0766667 | 265.189 | 266.89 | 0.175315 | 4 |
| TabRepo | meta_selector_calibrated_budget_10000s | 273.218 | 0.0566667 | 314.489 | 272.218 | 0.178728 | 4 |
| TabRepo | reference_metafeature_knn_validation | 273.942 | 0.0266667 | 344.334 | 272.942 | 0.17917 | 4 |
| TabRepo | reference_optimism_corrected_task | 292.54 | 0.0366667 | 145.22 | 291.54 | 0.191566 | 4 |
| TabRepo | reference_optimism_corrected | 295.087 | 0.0433333 | 143.106 | 294.087 | 0.193256 | 4 |
| TabRepo | reference_optimism_corrected_budget | 295.087 | 0.0433333 | 143.106 | 294.087 | 0.193256 | 4 |
| TabRepo | reference_single_best_global | 370.444 | 0.0216667 | 331.341 | 369.444 | 0.242838 | 4 |
| TabRepo | reference_single_best_task | 371.7 | 0.02 | 458.753 | 370.7 | 0.243537 | 4 |
| TabRepo | reference_priors_only | 423.837 | 0.0183333 | 172.714 | 422.837 | 0.277918 | 4 |
| TabRepo | reference_metafeature_knn | 491.768 | 0.0166667 | 348.468 | 490.768 | 0.322502 | 4 |

## Selected Policies

| artifact | selected_policy | n_splits |
| --- | --- | --- |
| TabRepo | meta_selector_tabrepo_ensemble_k3 | 4 |

## Paired Tests

| artifact | method | reference_method | n_splits | mean_rank_difference | mean_normalized_regret_difference | mean_runtime_difference_seconds | rank_wilcoxon_p_value | rank_holm_p_value |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| TabRepo | nested_policy_selected | meta_selector_tabrepo_ensemble_k3 | 4 | 0 | 0 | 0 | 1 | 1 |
| TabRepo | nested_policy_selected | meta_selector_task_adaptive_budgeted | 4 | -40.5325 | -0.0265845 | 415.37 | 0.125 | 0.875 |
| TabRepo | nested_policy_selected | meta_selector_meta_feature_guard_005 | 4 | -38.3667 | -0.0251678 | 416.695 | 0.125 | 0.875 |
| TabRepo | nested_policy_selected | meta_selector_family_prior_budgeted | 4 | -33.8592 | -0.0222153 | 355.458 | 0.125 | 0.875 |
| TabRepo | nested_policy_selected | meta_selector_ablate_validation_only | 4 | -45.4175 | -0.0298895 | 437.878 | 0.125 | 0.875 |
| TabRepo | nested_policy_selected | meta_selector_single | 4 | -44.3267 | -0.0291761 | 438.58 | 0.125 | 0.875 |
| TabRepo | nested_policy_selected | meta_selector_tabrepo_ensemble_k5 | 4 | -1.23917 | -0.000837712 | 0 | 0.125 | 0.875 |
