# Third-Artifact Offline Validation

Input artifact: `results/tabarena_ict_retraining_extended/results.csv`.

This is an offline validation artifact, not a real-world deployment. It uses
the compact local retraining table as a third audit surface and keeps
candidate-relative rank regret separate from metric-scale prediction gains.

## Method Summary

| method | label | method_type | n_datasets | n_units | mean_rank | mean_rank_ci_low | mean_rank_ci_high | mean_primary_metric | mean_primary_metric_ci_low | mean_primary_metric_ci_high | mean_val_primary_metric | mean_val_primary_metric_ci_low | mean_val_primary_metric_ci_high | mean_normalized_rank_regret | mean_normalized_rank_regret_ci_low | mean_normalized_rank_regret_ci_high | tied_best_candidate_rate | tied_best_candidate_rate_ci_low | tied_best_candidate_rate_ci_high | mean_total_seconds | mean_total_seconds_ci_low | mean_total_seconds_ci_high |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| meta_selector_local_ensemble | Local soft ensemble | selector | 8 | 24 | 1.29167 | 1.08333 | 1.54167 | 0.25472 | 0.152807 | 0.361341 | 0.251691 | 0.1497 | 0.358012 | 0.0972222 | 0.0277778 | 0.180556 | 0 | 0 | 0 | 0 | 0 | 0 |
| meta_selector_local_validation | Validation-best single model | selector | 8 | 24 | 1.16667 | 1.04167 | 1.33333 | 0.256254 | 0.153903 | 0.363079 | 0.252652 | 0.150443 | 0.35932 | 0.0555556 | 0.0138889 | 0.111111 | 0.875 | 0.75 | 0.958333 | 0 | 0 | 0 |

## Paired Dataset Contrasts

| method | reference_method | method_label | reference_label | endpoint | n_datasets | mean_method_minus_reference | ci95_low | ci95_high | bootstrap_p_value | wilcoxon_p_value | sign_test_p_value | wins_method_lower | ties | losses_method_higher | holm_p_value | bootstrap_holm_p_value |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| meta_selector_local_ensemble | meta_selector_local_validation | Local soft ensemble | Validation-best single model | normalized_rank_regret | 8 | 0.0416667 | 0 | 0.125 | 0.675 | 1 | 1 | 0 | 7 | 1 | 1 | 0.675 |
| meta_selector_local_ensemble | meta_selector_local_validation | Local soft ensemble | Validation-best single model | primary_metric | 8 | -0.00153463 | -0.00243578 | -0.000673013 | 0 | 0.0234375 | 0.0703125 | 7 | 0 | 1 | 0.046875 | 0 |

## Per-Dataset Differences

| dataset | method_mean_rank | method_mean_primary_metric | method_mean_normalized_rank_regret | reference_mean_rank | reference_mean_primary_metric | reference_mean_normalized_rank_regret | rank_difference | normalized_rank_regret_difference | primary_metric_difference |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| APSFailure | 1 | 0.0202063 | 0 | 1 | 0.0204913 | 0 | 0 | 0 | -0.000285023 |
| Amazon_employee_access | 1 | 0.159183 | 0 | 1 | 0.163296 | 0 | 0 | 0 | -0.00411371 |
| Bank_Customer_Churn | 1.33333 | 0.332368 | 0.111111 | 1.33333 | 0.334598 | 0.111111 | 0 | 0 | -0.00222972 |
| E-CommereShippingData | 1 | 0.510144 | 0 | 1 | 0.511581 | 0 | 0 | 0 | -0.0014376 |
| NATICUSdroid | 1 | 0.148671 | 0 | 1 | 0.149775 | 0 | 0 | 0 | -0.00110317 |
| churn | 2 | 0.157649 | 0.333333 | 1 | 0.157123 | 0 | 1 | 0.333333 | 0.000526482 |
| jm1 | 1.33333 | 0.424874 | 0.111111 | 1.33333 | 0.426782 | 0.111111 | 0 | 0 | -0.0019075 |
| website_phishing | 1.66667 | 0.284662 | 0.222222 | 1.66667 | 0.286389 | 0.222222 | 0 | 0 | -0.00172679 |
