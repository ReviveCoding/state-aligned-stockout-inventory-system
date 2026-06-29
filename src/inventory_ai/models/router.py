from __future__ import annotations

import pandas as pd

from inventory_ai.evaluation.metrics import forecast_metrics


ROUTING_COLUMNS = [
    "series_id",
    "selected_model",
    "selected_wape",
    "baseline_wape",
    "relative_improvement",
    "n_validation_rows",
]


def learn_series_router(
    validation_forecasts: pd.DataFrame,
    validation_truth: pd.DataFrame,
    baseline_model: str = "seasonal_naive",
    min_relative_improvement: float = 0.05,
    min_validation_rows: int = 14,
) -> pd.DataFrame:
    """Learn a conservative per-series model router from validation only.

    A challenger is selected only when it improves WAPE by a material margin;
    otherwise the operational baseline remains the fallback. This reduces
    worst-slice regressions without using any held-out test outcomes.
    """
    if baseline_model not in set(validation_forecasts["model"]):
        raise ValueError(f"router baseline is missing: {baseline_model}")
    truth_keys = validation_truth[["series_id", "date_idx"]].drop_duplicates()
    rows: list[dict] = []
    for series_id, truth_sub in validation_truth.groupby("series_id", sort=True):
        forecast_sub = validation_forecasts[validation_forecasts["series_id"].eq(series_id)]
        expected = set(map(tuple, truth_sub[["series_id", "date_idx"]].to_numpy()))
        metrics: dict[str, dict[str, float | int]] = {}
        for model, model_frame in forecast_sub.groupby("model", sort=True):
            actual = set(map(tuple, model_frame[["series_id", "date_idx"]].to_numpy()))
            if actual != expected:
                continue
            metrics[str(model)] = forecast_metrics(model_frame, truth_sub)
        if baseline_model not in metrics:
            raise ValueError(f"router baseline lacks complete validation coverage for {series_id}")
        baseline_wape = float(metrics[baseline_model]["wape"])
        best_model = min(metrics, key=lambda model: float(metrics[model]["wape"]))
        best_wape = float(metrics[best_model]["wape"])
        n_rows = int(metrics[baseline_model]["n"])
        improvement = float((baseline_wape - best_wape) / max(baseline_wape, 1e-9))
        selected = (
            best_model
            if n_rows >= min_validation_rows
            and best_model != baseline_model
            and improvement >= min_relative_improvement
            else baseline_model
        )
        selected_wape = float(metrics[selected]["wape"])
        rows.append(
            {
                "series_id": series_id,
                "selected_model": selected,
                "selected_wape": selected_wape,
                "baseline_wape": baseline_wape,
                "relative_improvement": float(
                    (baseline_wape - selected_wape) / max(baseline_wape, 1e-9)
                ),
                "n_validation_rows": n_rows,
            }
        )
    routing = pd.DataFrame(rows, columns=ROUTING_COLUMNS)
    if set(routing["series_id"]) != set(truth_keys["series_id"]):
        raise ValueError("router did not produce one decision for every series")
    return routing


def apply_series_router(
    forecasts: pd.DataFrame,
    routing: pd.DataFrame,
    output_model: str = "reliability_router",
) -> pd.DataFrame:
    """Apply a frozen routing map to a forecast origin."""
    missing = set(ROUTING_COLUMNS[:2]) - set(routing.columns)
    if missing:
        raise ValueError(f"routing table missing columns: {sorted(missing)}")
    if routing["series_id"].duplicated().any():
        raise ValueError("routing table contains duplicate series decisions")
    available = forecasts.groupby("series_id")["model"].agg(set).to_dict()
    rows = []
    for decision in routing.itertuples(index=False):
        series_id = decision.series_id
        model = decision.selected_model
        if model not in available.get(series_id, set()):
            raise ValueError(f"routed model {model} unavailable for series {series_id}")
        selected = forecasts[
            forecasts["series_id"].eq(series_id) & forecasts["model"].eq(model)
        ].copy()
        selected["routed_from_model"] = model
        selected["model"] = output_model
        rows.append(selected)
    routed = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    if routed.empty:
        raise ValueError("router produced no forecasts")
    expected_keys = set(map(tuple, forecasts[["series_id", "date_idx"]].drop_duplicates().to_numpy()))
    actual_keys = set(map(tuple, routed[["series_id", "date_idx"]].to_numpy()))
    if actual_keys != expected_keys:
        raise ValueError("router key coverage mismatch")
    return routed.sort_values(["series_id", "date_idx"]).reset_index(drop=True)


def summarize_routing(routing: pd.DataFrame) -> dict[str, int | float | dict[str, int]]:
    counts = routing["selected_model"].value_counts().sort_index().to_dict()
    summary = {
        "n_series": int(len(routing)),
        "model_counts": {str(key): int(value) for key, value in counts.items()},
        "mean_validation_improvement": float(routing["relative_improvement"].mean()),
        "fallback_rate": float((routing["selected_model"] == "seasonal_naive").mean()),
    }
    if "relative_cost_improvement" in routing:
        summary["mean_validation_cost_improvement"] = float(
            routing["relative_cost_improvement"].mean()
        )
    return summary


def learn_decision_aware_router(
    validation_forecasts: pd.DataFrame,
    validation_truth: pd.DataFrame,
    inventory_config,
    policy_scale_by_model: dict[str, float],
    baseline_model: str = "seasonal_naive",
    max_relative_wape_regression: float = 0.10,
    min_validation_rows: int = 14,
) -> pd.DataFrame:
    """Learn a validation-only router balancing forecast and inventory quality.

    For every series, models with WAPE no more than the configured regression
    allowance versus the operational baseline are eligible. The lowest-cost
    eligible replenishment decision wins. This turns forecast routing into a
    decision-aware reliability layer without consulting held-out test outcomes.
    """
    from inventory_ai.simulation.inventory import run_open_loop_by_model

    models = sorted(set(validation_forecasts["model"]))
    if baseline_model not in models:
        raise ValueError(f"router baseline is missing: {baseline_model}")
    simulation = run_open_loop_by_model(
        validation_forecasts,
        validation_truth,
        inventory_config,
        policy_scale_by_model=policy_scale_by_model,
    )
    cost_by_series_model = (
        simulation.groupby(["series_id", "model"])["cost"].sum().to_dict()
    )
    rows: list[dict] = []
    for series_id, truth_sub in validation_truth.groupby("series_id", sort=True):
        expected = set(map(tuple, truth_sub[["series_id", "date_idx"]].to_numpy()))
        stats: dict[str, dict[str, float | int]] = {}
        for model, model_frame in validation_forecasts[
            validation_forecasts["series_id"].eq(series_id)
        ].groupby("model", sort=True):
            actual = set(map(tuple, model_frame[["series_id", "date_idx"]].to_numpy()))
            if actual == expected:
                stats[str(model)] = forecast_metrics(model_frame, truth_sub)
        if baseline_model not in stats:
            raise ValueError(f"router baseline lacks complete validation coverage for {series_id}")
        baseline_wape = float(stats[baseline_model]["wape"])
        baseline_cost = float(cost_by_series_model[(series_id, baseline_model)])
        n_rows = int(stats[baseline_model]["n"])
        eligible = []
        for model, values in stats.items():
            model_wape = float(values["wape"])
            if (
                model == baseline_model
                or (n_rows >= min_validation_rows
                    and model_wape <= baseline_wape * (1.0 + max_relative_wape_regression))
            ):
                eligible.append((
                    model,
                    model_wape,
                    float(cost_by_series_model[(series_id, model)]),
                ))
        selected_model, selected_wape, selected_cost = min(
            eligible, key=lambda item: (item[2], item[1], item[0])
        )
        rows.append({
            "series_id": series_id,
            "selected_model": selected_model,
            "selected_wape": selected_wape,
            "baseline_wape": baseline_wape,
            "relative_improvement": float(
                (baseline_wape - selected_wape) / max(baseline_wape, 1e-9)
            ),
            "n_validation_rows": n_rows,
            "selected_cost": selected_cost,
            "baseline_cost": baseline_cost,
            "relative_cost_improvement": float(
                (baseline_cost - selected_cost) / max(baseline_cost, 1e-9)
            ),
            "routing_reason": "lowest_validation_cost_within_wape_guardrail",
        })
    routing = pd.DataFrame(rows)
    expected_series = set(validation_truth["series_id"].unique())
    if set(routing["series_id"]) != expected_series:
        raise ValueError("decision-aware router did not produce one decision for every series")
    return routing.sort_values("series_id").reset_index(drop=True)
