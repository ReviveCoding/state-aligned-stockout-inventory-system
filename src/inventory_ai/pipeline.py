from __future__ import annotations

import gc
import os
import shutil
import sys
import tempfile
import time
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from inventory_ai.calibration.conformal import apply_conformal_adjustments, fit_conformal_adjustments, repair_quantiles
from inventory_ai.censoring.recovery import (
    add_controlled_stockouts, recover_latent_demand, recovery_diagnostics, sample_recovery_posterior
)
from inventory_ai.config import PipelineConfig
from inventory_ai.contracts import validate_forecast, validate_panel
from inventory_ai.data.m5 import load_m5_sample
from inventory_ai.data.synthetic import SyntheticConfig, make_lifecycle_benchmark, make_synthetic_retail
from inventory_ai.evaluation.metrics import (
    choose_candidate,
    evaluate_models,
    forecast_slice_report,
    lifecycle_metrics,
    summarize_forecast_slices,
    summarize_validation_model_slices,
    validation_model_slice_report,
)
from inventory_ai.features.basic import add_time_features
from inventory_ai.gates.release import evaluate_release
from inventory_ai.lifecycle.drat import add_drat, estimate_ladt
from inventory_ai.models.baselines import seasonal_naive_forecast, tsb_forecast
from inventory_ai.models.chronos_adapter import check_chronos_capability
from inventory_ai.models.quantile import DirectQuantileForecaster
from inventory_ai.models.router import (
    apply_series_router, learn_decision_aware_router, summarize_routing
)
from inventory_ai.reconciliation.hierarchy import (
    hierarchy_error, reconcile_bottom_up, reconcile_scenarios_bottom_up, scenario_hierarchy_error
)
from inventory_ai.reporting.reports import write_release_report
from inventory_ai.simulation.closed_loop import (
    closed_loop_policy_comparison,
    closed_loop_safety_sweep,
    compare_closed_loop_slices,
    summarize_closed_loop,
    summarize_closed_loop_slice_comparison,
    summarize_closed_loop_slices,
)
from inventory_ai.simulation.inventory import (
    InventoryConfig,
    best_tuned_policy_by_model,
    evaluate_selected_policy_scales,
    run_open_loop_policy,
    summarize_inventory,
    summarize_policy_at_scale,
    tune_policy_scales,
)
from inventory_ai.simulation.scenarios import generate_correlated_scenarios, scenario_diagnostics
from inventory_ai.utils.io import environment_manifest, sha256_file, write_csv, write_json


PIPELINE_REPORT_FILENAMES = (
    "calibration_forecasts.csv",
    "validation_forecasts.csv",
    "validation_inventory_selected_policy.csv",
    "validation_slice_report.csv",
    "routing_map.csv",
    "policy_tuning.csv",
    "forecast_output.csv",
    "reconciled_forecasts.csv",
    "inventory_simulation.csv",
    "closed_loop_replay.csv",
    "closed_loop_sensitivity.csv",
    "closed_loop_slices.csv",
    "closed_loop_slice_comparison.csv",
    "forecast_slice_report.csv",
    "controlled_lifecycle_benchmark.csv",
    "candidate_scenarios.csv",
    "reconciled_scenarios.csv",
    "conformal_adjustments.csv",
    "recovery_posterior_samples.csv",
    "metrics_summary.json",
    "release_gate.json",
    "release_report.md",
)
SQL_REPORT_FILENAMES = ("sql_daily_demand_mart.csv", "sql_inventory_kpi_mart.csv")


def _progress(stage: str, started: float) -> None:
    if os.environ.get("INVENTORY_AI_PROGRESS", "").strip().lower() in {"1", "true", "yes"}:
        elapsed = time.time() - started
        print(f"[inventory-ai] {stage} ({elapsed:.2f}s)", file=sys.stderr, flush=True)


def _reset_owned_outputs(reports: Path, artifacts: Path) -> None:
    # Remove owned final artifacts and orphaned atomic-write temp files from
    # previously interrupted runs.  A killed process can leave ``*.tmp`` files
    # beside valid reports; they are never read by the pipeline, but keeping
    # the output directory clean prevents false release-residue findings and
    # makes repeated local runs deterministic.
    for name in (*PIPELINE_REPORT_FILENAMES, *SQL_REPORT_FILENAMES):
        path = reports / name
        if path.exists():
            path.unlink()
    for directory in (reports, artifacts):
        if directory.exists():
            for temp_path in directory.glob("*.tmp"):
                if temp_path.is_file():
                    temp_path.unlink()
    manifest = artifacts / "run_manifest.json"
    if manifest.exists():
        manifest.unlink()


def _balanced_panel_sample(panel: pd.DataFrame, max_rows: int = 5_000) -> pd.DataFrame:
    """Create a deterministic audit sample with temporal coverage per series.

    Head-only sampling can omit the latest validation/test periods on large
    panels. Evenly spaced indices preserve the beginning, middle, and final
    observation for every represented series while respecting ``max_rows``.
    """
    if len(panel) <= max_rows:
        return panel.sort_values(["series_id", "date_idx"]).reset_index(drop=True).copy()
    series_ids = sorted(panel["series_id"].unique())
    per_series = max(1, max_rows // max(len(series_ids), 1))
    parts: list[pd.DataFrame] = []
    for series_id in series_ids:
        group = panel[panel["series_id"].eq(series_id)].sort_values("date_idx")
        take = min(per_series, len(group))
        if take == 1:
            indices = [len(group) - 1]
        else:
            indices = sorted(set(int(round(value)) for value in pd.Series(range(take)).map(
                lambda idx: idx * (len(group) - 1) / (take - 1)
            )))
        parts.append(group.iloc[indices])
    sampled = pd.concat(parts, ignore_index=False)
    if len(sampled) < max_rows:
        remaining = panel.loc[~panel.index.isin(sampled.index)].sort_values(
            ["date_idx", "series_id"], ascending=[False, True]
        ).head(max_rows - len(sampled))
        sampled = pd.concat([sampled, remaining], ignore_index=False)
    return sampled.sort_values(["series_id", "date_idx"]).head(max_rows).reset_index(drop=True)


def _load_data(cfg: PipelineConfig) -> tuple[pd.DataFrame, str, str | None]:
    source = cfg.data.source
    if source == "synthetic":
        return (
            make_synthetic_retail(SyntheticConfig(cfg.data.n_series, cfg.data.history_days, cfg.seed)),
            "synthetic",
            None,
        )
    if source == "m5":
        return load_m5_sample(
            cfg.data.m5_zip_path,
            cfg.data.n_series,
            cfg.data.history_days,
            series_offset=cfg.data.series_offset,
        ), "m5", None
    try:
        frame = load_m5_sample(
            cfg.data.m5_zip_path,
            cfg.data.n_series,
            cfg.data.history_days,
            series_offset=cfg.data.series_offset,
        )
        return frame, "m5", None
    except FileNotFoundError as exc:
        if not cfg.data.allow_fallback:
            raise
        reason = f"M5 archive not found: {exc}"
        return (
            make_synthetic_retail(SyntheticConfig(cfg.data.n_series, cfg.data.history_days, cfg.seed)),
            "synthetic_fallback",
            reason,
        )


def _prepare_panel(raw: pd.DataFrame, cfg: PipelineConfig) -> pd.DataFrame:
    validate_panel(raw)
    controlled = add_controlled_stockouts(raw, seed=cfg.seed, rate=cfg.data.controlled_stockout_rate)
    recovered = recover_latent_demand(controlled)
    featured = add_time_features(recovered, value_col="recovered_demand_mean")
    featured = add_drat(featured, value_col="recovered_demand_mean")
    validate_panel(featured)
    return featured


def _forecast_at_origin(panel: pd.DataFrame, cfg: PipelineConfig, origin: int) -> pd.DataFrame:
    horizon = cfg.model.horizon
    naive = seasonal_naive_forecast(panel, origin, horizon)
    tsb = tsb_forecast(panel, origin, horizon)
    forecaster = DirectQuantileForecaster(max_iter=cfg.model.max_iter, random_state=cfg.seed)
    forecaster.fit(panel, cutoff=origin, horizon=horizon, max_rows=cfg.model.max_train_rows)
    advanced = forecaster.predict(panel, origin=origin, horizon=horizon)
    combined = repair_quantiles(pd.concat([naive, tsb, advanced], ignore_index=True))
    del forecaster
    gc.collect()
    return combined


def _run_pipeline_impl(cfg: PipelineConfig, root: str | Path) -> dict:
    started = time.time()
    root_path = Path(root)
    reports = root_path / cfg.output_dir
    artifacts = root_path / cfg.artifact_dir
    processed = root_path / "data" / "processed" / cfg.run_name
    for directory in (reports, artifacts, processed):
        directory.mkdir(parents=True, exist_ok=True)
    _reset_owned_outputs(reports, artifacts)

    raw, data_source, fallback_reason = _load_data(cfg)
    panel = _prepare_panel(raw, cfg)
    _progress("data prepared", started)
    write_csv(processed / "training_panel_sample.csv", _balanced_panel_sample(panel))

    horizon = cfg.model.horizon
    max_date = int(panel["date_idx"].max())
    test_origin = max_date - horizon
    # Fit the final test-origin model before rolling validation folds. This keeps
    # peak memory bounded on constrained local environments while preserving
    # strict temporal separation: selection still uses validation folds only.
    test_forecasts_uncalibrated = _forecast_at_origin(panel, cfg, test_origin)
    _progress("test-origin forecasts fitted", started)

    candidate_pretest_origins = [
        test_origin - cfg.model.validation_horizon,
        test_origin - 2 * cfg.model.validation_horizon,
        test_origin - 3 * cfg.model.validation_horizon,
    ]
    pretest_origins = sorted(origin for origin in candidate_pretest_origins if origin > 30)
    if len(pretest_origins) < 2:
        raise ValueError("not enough history for separate calibration, validation, and test origins")

    # The earliest fold is calibration-only. Keeping it disjoint from model and
    # policy selection prevents conformal residuals from improving the same
    # validation observations used to choose the production candidate.
    calibration_origin = pretest_origins[0]
    validation_origins = pretest_origins[1:]
    initial_calibration_forecasts = _forecast_at_origin(panel, cfg, calibration_origin)
    initial_calibration_truth = panel[
        (panel["date_idx"] > calibration_origin)
        & (panel["date_idx"] <= calibration_origin + horizon)
    ]
    validate_forecast(initial_calibration_forecasts, initial_calibration_truth)

    # Sequential adaptive calibration: each validation origin is calibrated
    # only with residuals observed strictly before that origin. After its truth
    # becomes available, the fold is added to the calibration history for the
    # next origin and, ultimately, the held-out test origin.
    calibration_forecast_history = [initial_calibration_forecasts]
    calibration_truth_history = [initial_calibration_truth]
    adjustments = fit_conformal_adjustments(initial_calibration_forecasts, initial_calibration_truth)
    validation_forecast_parts = []
    validation_truth_parts = []
    conformal_update_origins = []
    for validation_origin in validation_origins:
        fold_forecasts_uncalibrated = _forecast_at_origin(panel, cfg, validation_origin)
        fold_truth = panel[
            (panel["date_idx"] > validation_origin)
            & (panel["date_idx"] <= validation_origin + horizon)
        ]
        validate_forecast(fold_forecasts_uncalibrated, fold_truth)
        validation_forecast_parts.append(
            apply_conformal_adjustments(fold_forecasts_uncalibrated, adjustments)
        )
        validation_truth_parts.append(fold_truth)
        calibration_forecast_history.append(fold_forecasts_uncalibrated)
        calibration_truth_history.append(fold_truth)
        conformal_update_origins.append(validation_origin)
        adjustments = fit_conformal_adjustments(
            pd.concat(calibration_forecast_history, ignore_index=True),
            pd.concat(calibration_truth_history, ignore_index=True).drop_duplicates(["series_id", "date_idx"]),
        )
    calibration_forecasts = pd.concat(calibration_forecast_history, ignore_index=True)
    validation_forecasts = pd.concat(validation_forecast_parts, ignore_index=True)
    validation_truth = pd.concat(validation_truth_parts, ignore_index=True).drop_duplicates(["series_id", "date_idx"])
    _progress("calibration and validation forecasts fitted", started)
    source_validation = validation_forecasts
    inventory_cfg = InventoryConfig(**asdict(cfg.inventory))
    source_policy_tuning = tune_policy_scales(source_validation, validation_truth, inventory_cfg)
    source_inventory_summary = best_tuned_policy_by_model(
        source_policy_tuning,
        max_fill_rate_degradation=cfg.model.selection_max_fill_rate_degradation,
    )
    _progress("source policies tuned", started)
    source_policy_scales = {
        model: float(summary["policy_scale"])
        for model, summary in source_inventory_summary.items()
    }
    routing_map = learn_decision_aware_router(
        source_validation,
        validation_truth,
        inventory_cfg,
        source_policy_scales,
        max_relative_wape_regression=cfg.model.router_max_wape_regression,
    )
    validation_router = apply_series_router(source_validation, routing_map)
    calibrated_validation = pd.concat([source_validation, validation_router], ignore_index=True)
    validation_metrics = evaluate_models(calibrated_validation, validation_truth)
    validation_slices = validation_model_slice_report(calibrated_validation, validation_truth)
    validation_slice_summaries = summarize_validation_model_slices(validation_slices)
    _progress("router and validation slices evaluated", started)
    router_tuning = tune_policy_scales(
        pd.concat([
            calibrated_validation[calibrated_validation["model"].eq("seasonal_naive")],
            validation_router,
        ], ignore_index=True),
        validation_truth,
        inventory_cfg,
    )
    router_tuning = router_tuning[router_tuning["model"].eq("reliability_router")]
    policy_tuning = pd.concat([source_policy_tuning, router_tuning], ignore_index=True)
    validation_inventory_raw_summary = summarize_policy_at_scale(policy_tuning, 1.0)
    tuned_policy_choices = best_tuned_policy_by_model(
        policy_tuning,
        max_fill_rate_degradation=cfg.model.selection_max_fill_rate_degradation,
    )
    validation_selected_scales = {
        model: float(summary["policy_scale"])
        for model, summary in tuned_policy_choices.items()
    }
    validation_inventory_selected_sim, validation_inventory_summary = evaluate_selected_policy_scales(
        calibrated_validation,
        validation_truth,
        inventory_cfg,
        validation_selected_scales,
    )
    _progress("validation inventory policies evaluated", started)
    selected_model = choose_candidate(
        validation_metrics,
        validation_inventory_summary,
        slice_summaries=validation_slice_summaries,
        max_worst_slice_regression=cfg.release_gate.max_worst_slice_wape_regression,
        inventory_cost_tolerance=cfg.model.decision_cost_tolerance,
        max_fill_rate_degradation=cfg.model.selection_max_fill_rate_degradation,
        max_interval_width_ratio=cfg.model.selection_max_interval_width_ratio,
    )
    selected_policy_scale = validation_inventory_summary[selected_model]["policy_scale"]

    test_forecasts = apply_conformal_adjustments(test_forecasts_uncalibrated, adjustments)
    test_router = apply_series_router(test_forecasts, routing_map)
    test_forecasts = pd.concat([test_forecasts, test_router], ignore_index=True)
    selected = test_forecasts[test_forecasts["model"] == selected_model].copy()
    selected["source_model"] = selected_model
    selected["model"] = "production_candidate"
    all_forecasts = pd.concat([test_forecasts, selected], ignore_index=True)
    test_truth = panel[(panel["date_idx"] > test_origin) & (panel["date_idx"] <= test_origin + horizon)]
    validate_forecast(all_forecasts, test_truth)
    test_metrics = evaluate_models(all_forecasts, test_truth)
    forecast_slices = forecast_slice_report(all_forecasts, test_truth)
    forecast_slice_summary = summarize_forecast_slices(forecast_slices)
    _progress("held-out forecasts evaluated", started)

    reconciled = reconcile_bottom_up(all_forecasts, panel)
    hierarchy_max_error = hierarchy_error(reconciled)

    candidate_forecast = all_forecasts[all_forecasts["model"] == "production_candidate"]
    scenarios = generate_correlated_scenarios(candidate_forecast, n_scenarios=120, seed=cfg.seed)
    scenario_stats = scenario_diagnostics(scenarios)
    reconciled_scenarios = reconcile_scenarios_bottom_up(scenarios, panel)
    scenario_coherence_error = scenario_hierarchy_error(reconciled_scenarios)
    _progress("scenarios generated and reconciled", started)

    # Apply validation-selected policy scales to every model on the test fold.
    # Otherwise the production candidate would receive a tuned replenishment
    # policy while the seasonal comparator remained at scale=1.0, confounding
    # model quality with policy tuning.
    test_policy_scales = {
        model: float(summary["policy_scale"])
        for model, summary in validation_inventory_summary.items()
    }
    test_policy_scales["production_candidate"] = float(selected_policy_scale)
    inventory_sim = run_open_loop_policy(
        all_forecasts,
        test_truth,
        inventory_cfg,
        policy_scale_by_model=test_policy_scales,
    )
    inventory_summary = summarize_inventory(inventory_sim)
    _progress("held-out inventory simulated", started)
    cost_pivot = inventory_sim.groupby(["model", "series_id"])["cost"].sum().unstack("model")
    if {"production_candidate", "seasonal_naive"}.issubset(cost_pivot.columns):
        cost_win_rate = float((cost_pivot["production_candidate"] <= cost_pivot["seasonal_naive"]).mean())
    else:
        cost_win_rate = 0.0
    candidate_cost = inventory_summary["production_candidate"]["total_cost"]
    baseline_cost = inventory_summary["seasonal_naive"]["total_cost"]
    candidate_cost_regression = float((candidate_cost - baseline_cost) / (baseline_cost + 1e-9))
    candidate_fill_rate_degradation = float(
        inventory_summary["seasonal_naive"]["fill_rate"]
        - inventory_summary["production_candidate"]["fill_rate"]
    )
    candidate_interval_width_ratio = float(
        test_metrics["production_candidate"]["mean_interval_width"]
        / (test_metrics["seasonal_naive"]["mean_interval_width"] + 1e-9)
    )

    closed_loop = closed_loop_policy_comparison(cfg.data.n_series, cfg.closed_loop_cycles, inventory_cfg, cfg.seed, recovery_safety=1.05)
    closed_summary = summarize_closed_loop(closed_loop)
    closed_sensitivity = closed_loop_safety_sweep(cfg.data.n_series, cfg.closed_loop_cycles, inventory_cfg, cfg.seed)
    closed_slices = summarize_closed_loop_slices(closed_loop)
    closed_slice_comparison = compare_closed_loop_slices(closed_slices)
    closed_slice_summary = summarize_closed_loop_slice_comparison(closed_slice_comparison)
    _progress("closed-loop replay completed", started)
    naive_closed = closed_summary["naive_no_recovery"]
    adaptive_closed = closed_summary["recovery_state_aware"]
    repeat_stockout_ratio = adaptive_closed["repeat_stockout_ratio"]
    repeat_stockout_improvement = float(
        (naive_closed["repeat_stockout_ratio"] - adaptive_closed["repeat_stockout_ratio"])
        / (naive_closed["repeat_stockout_ratio"] + 1e-9)
    )
    closed_loop_cost_regression = float(
        (adaptive_closed["total_cost"] - naive_closed["total_cost"])
        / (naive_closed["total_cost"] + 1e-9)
    )
    closed_loop_lost_sales_improvement = float(
        (naive_closed["lost_sales"] - adaptive_closed["lost_sales"])
        / (naive_closed["lost_sales"] + 1e-9)
    )

    lifecycle_raw = make_lifecycle_benchmark(SyntheticConfig(cfg.lifecycle_series, cfg.data.history_days, cfg.seed))
    lifecycle_recovered = recover_latent_demand(lifecycle_raw)
    lifecycle_features = add_time_features(lifecycle_recovered, value_col="recovered_demand_mean")
    lifecycle_estimated = estimate_ladt(lifecycle_features)
    lifecycle_stats = lifecycle_metrics(lifecycle_estimated)

    recovery_stats = recovery_diagnostics(panel)
    recovery_samples = sample_recovery_posterior(panel, n_draws=30, seed=cfg.seed)
    _progress("lifecycle and recovery evaluated", started)
    lower_bound_violations = int((panel["recovered_demand_mean"] + 1e-9 < panel["sales"]).sum())
    negative_forecasts = int((all_forecasts[["q10", "q50", "q90"]] < 0).sum().sum())
    quantile_crossing = int(((all_forecasts["q10"] > all_forecasts["q50"]) | (all_forecasts["q50"] > all_forecasts["q90"])).sum())
    expected_keys = set(map(tuple, test_truth[["series_id", "date_idx"]].to_numpy()))
    coverage_errors = 0
    for _, sub in all_forecasts.groupby("model"):
        coverage_errors += int(set(map(tuple, sub[["series_id", "date_idx"]].to_numpy())) != expected_keys)
    baseline_wape = float(test_metrics["seasonal_naive"]["wape"])
    candidate_wape = float(test_metrics["production_candidate"]["wape"])
    candidate_wape_improvement = float((baseline_wape - candidate_wape) / (baseline_wape + 1e-9))

    metrics = {
        "run_name": cfg.run_name,
        "data_source": data_source,
        "data_fallback_reason": fallback_reason,
        "data_contract": asdict(validate_panel(panel)),
        "selected_model": selected_model,
        "selected_policy_scale": selected_policy_scale,
        "test_policy_scales": test_policy_scales,
        "calibration_origin": calibration_origin,
        "validation_origins": validation_origins,
        "calibration_fold_count": 1,
        "conformal_history_fold_count": len(calibration_forecast_history),
        "validation_fold_count": len(validation_origins),
        "conformal_update_origins": conformal_update_origins,
        "validation_forecast_metrics": validation_metrics,
        "validation_inventory_raw_summary": validation_inventory_raw_summary,
        "validation_inventory_summary": validation_inventory_summary,
        "validation_slice_summaries": validation_slice_summaries,
        "routing_summary": summarize_routing(routing_map),
        "test_forecast_metrics": test_metrics,
        "forecast_slice_summary": forecast_slice_summary,
        "recovery_diagnostics": recovery_stats,
        "lifecycle_metrics": lifecycle_stats,
        "inventory_summary": inventory_summary,
        "closed_loop_summary": closed_summary,
        "closed_loop_slice_summary": closed_slice_summary,
        "scenario_diagnostics": scenario_stats,
        "scenario_hierarchy_error": scenario_coherence_error,
        "hierarchy_max_error": hierarchy_max_error,
        "negative_forecasts": negative_forecasts,
        "quantile_crossing": quantile_crossing,
        "truth_coverage_error": coverage_errors,
        "candidate_wape_improvement": candidate_wape_improvement,
        "cost_win_rate": cost_win_rate,
        "candidate_cost_regression": candidate_cost_regression,
        "candidate_fill_rate_degradation": candidate_fill_rate_degradation,
        "candidate_interval_width_ratio": candidate_interval_width_ratio,
        "repeat_stockout_ratio": repeat_stockout_ratio,
        "repeat_stockout_improvement": repeat_stockout_improvement,
        "closed_loop_cost_regression": closed_loop_cost_regression,
        "closed_loop_lost_sales_improvement": closed_loop_lost_sales_improvement,
        "recovery_lower_bound_violations": lower_bound_violations,
        "recovery_do_no_harm_max": recovery_stats.get("do_no_harm_max") or 0.0,
        "chronos_capability": asdict(check_chronos_capability()),
        "runtime_seconds": float(time.time() - started),
    }
    release = evaluate_release(metrics, asdict(cfg.release_gate))

    write_csv(reports / "calibration_forecasts.csv", calibration_forecasts)
    write_csv(reports / "validation_forecasts.csv", calibrated_validation)
    write_csv(reports / "validation_inventory_selected_policy.csv", validation_inventory_selected_sim)
    write_csv(reports / "validation_slice_report.csv", validation_slices)
    write_csv(reports / "routing_map.csv", routing_map)
    write_csv(reports / "policy_tuning.csv", policy_tuning)
    write_csv(reports / "forecast_output.csv", all_forecasts)
    write_csv(reports / "reconciled_forecasts.csv", reconciled)
    write_csv(reports / "inventory_simulation.csv", inventory_sim)
    write_csv(reports / "closed_loop_replay.csv", closed_loop)
    write_csv(reports / "closed_loop_sensitivity.csv", closed_sensitivity)
    write_csv(reports / "closed_loop_slices.csv", closed_slices)
    write_csv(reports / "closed_loop_slice_comparison.csv", closed_slice_comparison)
    write_csv(reports / "forecast_slice_report.csv", forecast_slices)
    write_csv(reports / "controlled_lifecycle_benchmark.csv", lifecycle_estimated)
    write_csv(reports / "candidate_scenarios.csv", scenarios)
    write_csv(reports / "reconciled_scenarios.csv", reconciled_scenarios)
    write_csv(reports / "conformal_adjustments.csv", adjustments)
    write_csv(reports / "recovery_posterior_samples.csv", recovery_samples)
    write_json(reports / "metrics_summary.json", metrics)
    write_json(reports / "release_gate.json", release)
    write_release_report(reports / "release_report.md", release, metrics)
    _progress("reports written", started)

    input_manifest = {
        "config": cfg.to_dict(),
        "environment": environment_manifest(),
        "data_source": data_source,
        "data_fallback_reason": fallback_reason,
    }
    m5_path = Path(cfg.data.m5_zip_path)
    if data_source == "m5" and m5_path.exists():
        input_manifest["m5_sha256"] = sha256_file(m5_path)
    input_manifest["metrics"] = metrics
    input_manifest["release"] = release
    # Record only artifacts produced by this pipeline stage. Downstream stages
    # such as SQL marts maintain their own provenance section in the same manifest.
    report_files = [reports / name for name in PIPELINE_REPORT_FILENAMES]
    missing_outputs = [path.name for path in report_files if not path.exists()]
    if missing_outputs:
        raise RuntimeError(f"pipeline did not produce required outputs: {missing_outputs}")
    input_manifest["output_files"] = {
        path.name: {"sha256": sha256_file(path), "bytes": path.stat().st_size}
        for path in report_files
    }
    sample_path = processed / "training_panel_sample.csv"
    input_manifest["processed_sample"] = {
        "path": str(sample_path.relative_to(root_path)),
        "sha256": sha256_file(sample_path),
        "bytes": sample_path.stat().st_size,
    }
    write_json(artifacts / "run_manifest.json", input_manifest)
    _progress("manifest written", started)
    return {"release": release, "metrics": metrics}

def _owned_paths_for_run(cfg: PipelineConfig, root_path: Path) -> list[Path]:
    reports = root_path / cfg.output_dir
    artifacts = root_path / cfg.artifact_dir
    processed = root_path / "data" / "processed" / cfg.run_name
    return [
        *(reports / name for name in (*PIPELINE_REPORT_FILENAMES, *SQL_REPORT_FILENAMES)),
        artifacts / "run_manifest.json",
        processed / "training_panel_sample.csv",
    ]


def run_pipeline(cfg: PipelineConfig, root: str | Path) -> dict:
    """Run the pipeline transactionally with rollback of the last valid run.

    All owned artifacts are snapshotted before execution. If any stage fails,
    partial outputs are removed and the previous validated files are restored.
    Unrelated files in the run directories are never touched.
    """
    root_path = Path(root).resolve()
    root_path.mkdir(parents=True, exist_ok=True)
    for orphan in root_path.glob(f".{cfg.run_name}-backup-*"):
        if orphan.is_dir():
            shutil.rmtree(orphan, ignore_errors=True)
    owned_paths = _owned_paths_for_run(cfg, root_path)
    # Keep rollback state outside the repository so an externally killed run
    # cannot contaminate a later source package with orphan backup files.
    with tempfile.TemporaryDirectory(prefix=f"{cfg.run_name}-backup-") as backup_dir:
        backup_root = Path(backup_dir)
        backed_up: dict[Path, Path] = {}
        for path in owned_paths:
            if path.exists() and path.is_file():
                relative = path.relative_to(root_path)
                backup_path = backup_root / relative
                backup_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, backup_path)
                backed_up[path] = backup_path
        try:
            return _run_pipeline_impl(cfg, root_path)
        except Exception:
            for path in owned_paths:
                if path.exists() and path.is_file():
                    path.unlink()
            for destination, source in backed_up.items():
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, destination)
            raise
