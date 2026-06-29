from __future__ import annotations


def evaluate_release(metrics: dict, gate_cfg: dict) -> dict:
    checks = {
        "truth_key_coverage": metrics.get("truth_coverage_error", 1) <= gate_cfg.get("max_truth_coverage_error", 0),
        "negative_forecasts": metrics.get("negative_forecasts", 1) <= gate_cfg.get("max_negative_forecasts", 0),
        "quantile_crossing": metrics.get("quantile_crossing", 1) <= gate_cfg.get("max_quantile_crossing", 0),
        "hierarchy_coherence": metrics.get("hierarchy_max_error", 1.0) <= gate_cfg.get("max_hierarchy_error", 1e-8),
        "scenario_hierarchy_coherence": metrics.get("scenario_hierarchy_error", 1.0) <= gate_cfg.get("max_scenario_hierarchy_error", 1e-8),
        "scenario_quantile_fidelity": metrics.get("scenario_diagnostics", {}).get("normalized_quantile_mae", 1.0) <= gate_cfg.get("max_scenario_quantile_mae", 0.25),
        "candidate_interval_coverage": (
            gate_cfg.get("min_candidate_coverage", 0.70)
            <= metrics.get("test_forecast_metrics", {}).get("production_candidate", {}).get("coverage_80", -1.0)
            <= gate_cfg.get("max_candidate_coverage", 0.92)
        ),
        "recovery_mae_improvement": metrics.get("recovery_diagnostics", {}).get("mae_improvement", -1.0) >= gate_cfg.get("min_recovery_mae_improvement", 0.05),
        "recovery_bias_improvement": metrics.get("recovery_diagnostics", {}).get("absolute_bias_improvement", -1.0) >= gate_cfg.get("min_recovery_bias_improvement", 0.05),
        "recovery_q95_coverage": metrics.get("recovery_diagnostics", {}).get("q95_upper_coverage", 0.0) >= gate_cfg.get("min_recovery_q95_coverage", 0.75),
        "lifecycle_state_recovery": metrics.get("lifecycle_metrics", {}).get("state_macro_f1", 0.0) >= gate_cfg.get("min_lifecycle_state_f1", 0.50),
        "lifecycle_ladt_mae": metrics.get("lifecycle_metrics", {}).get("ladt_mae", 1.0) <= gate_cfg.get("max_ladt_mae", 0.16),
        "lifecycle_ladt_rank": metrics.get("lifecycle_metrics", {}).get("ladt_spearman", 0.0) >= gate_cfg.get("min_ladt_spearman", 0.75),
        "candidate_wape": metrics.get("candidate_wape_improvement", -1.0) >= gate_cfg.get("min_candidate_wape_improvement", -0.02),
        "inventory_cost_win_rate": metrics.get("cost_win_rate", 0.0) >= gate_cfg.get("min_cost_win_rate", 0.5),
        "inventory_cost_regression": metrics.get("candidate_cost_regression", 1.0) <= gate_cfg.get("max_candidate_cost_regression", 0.05),
        "inventory_fill_rate": metrics.get("candidate_fill_rate_degradation", 1.0) <= gate_cfg.get("max_candidate_fill_rate_degradation", 0.01),
        "candidate_interval_sharpness": metrics.get("candidate_interval_width_ratio", 99.0) <= gate_cfg.get("max_candidate_interval_width_ratio", 1.35),
        "forecast_worst_slice": metrics.get("forecast_slice_summary", {}).get("worst_relative_wape_regression", 1.0) <= gate_cfg.get("max_worst_slice_wape_regression", 0.20),
        "forecast_slice_win_rate": metrics.get("forecast_slice_summary", {}).get("slice_win_rate", 0.0) >= gate_cfg.get("min_forecast_slice_win_rate", 0.35),
        "closed_loop_stockout": metrics.get("repeat_stockout_ratio", 1.0) <= gate_cfg.get("max_repeat_stockout_ratio", 0.75),
        "closed_loop_stockout_improvement": metrics.get("repeat_stockout_improvement", -1.0) >= gate_cfg.get("min_repeat_stockout_improvement", 0.0),
        "closed_loop_cost": metrics.get("closed_loop_cost_regression", 1.0) <= gate_cfg.get("max_closed_loop_cost_regression", 0.10),
        "closed_loop_lost_sales": metrics.get("closed_loop_lost_sales_improvement", -1.0) >= gate_cfg.get("min_closed_loop_lost_sales_improvement", -0.02),
        "closed_loop_worst_slice_cost": metrics.get("closed_loop_slice_summary", {}).get("worst_relative_cost_regression", 1.0) <= gate_cfg.get("max_worst_closed_loop_cost_regression", 0.25),
        "closed_loop_worst_slice_fill": metrics.get("closed_loop_slice_summary", {}).get("worst_fill_rate_degradation", 1.0) <= gate_cfg.get("max_worst_closed_loop_fill_degradation", 0.01),
        "recovery_lower_bound": metrics.get("recovery_lower_bound_violations", 1) == 0,
        "recovery_do_no_harm": metrics.get("recovery_do_no_harm_max", 1.0) <= 1e-9,
    }
    critical = [
        "truth_key_coverage",
        "negative_forecasts",
        "quantile_crossing",
        "hierarchy_coherence",
        "scenario_hierarchy_coherence",
        "recovery_lower_bound",
        "recovery_do_no_harm",
    ]
    if not all(checks[name] for name in critical):
        status = "FAIL"
    elif all(checks.values()):
        status = "PASS"
    else:
        status = "ITERATE"
    return {"gate_status": status, "checks": checks, "metrics": metrics}
