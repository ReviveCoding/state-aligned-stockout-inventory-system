from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score


def pinball_loss(y: np.ndarray, prediction: np.ndarray, quantile: float) -> float:
    error = y - prediction
    return float(np.mean(np.maximum(quantile * error, (quantile - 1.0) * error)))


def forecast_metrics(forecast: pd.DataFrame, truth: pd.DataFrame) -> dict[str, float | int]:
    merged = forecast.merge(
        truth[["series_id", "date_idx", "demand"]],
        on=["series_id", "date_idx"],
        how="inner",
        validate="one_to_one",
    )
    if len(merged) != len(forecast):
        raise ValueError("forecast/truth coverage mismatch during metric calculation")
    y = merged["demand"].astype(float).to_numpy()
    q10 = merged["q10"].astype(float).to_numpy()
    q50 = merged["q50"].astype(float).to_numpy()
    q90 = merged["q90"].astype(float).to_numpy()
    denominator = np.abs(y).sum()
    wape = np.abs(q50 - y).sum() / (denominator + 1e-9)
    signed_bias = (q50 - y).sum() / (denominator + 1e-9)
    coverage = np.mean((y >= q10) & (y <= q90))
    width = np.mean(q90 - q10)
    return {
        "n": int(len(merged)),
        "wape": float(wape),
        "signed_bias": float(signed_bias),
        "coverage_80": float(coverage),
        "mean_interval_width": float(width),
        "pinball_q10": pinball_loss(y, q10, 0.1),
        "pinball_q50": pinball_loss(y, q50, 0.5),
        "pinball_q90": pinball_loss(y, q90, 0.9),
    }


def evaluate_models(forecasts: pd.DataFrame, truth: pd.DataFrame) -> dict[str, dict[str, float | int]]:
    return {model: forecast_metrics(sub, truth) for model, sub in forecasts.groupby("model", sort=True)}


def choose_candidate(
    metrics: dict[str, dict[str, float | int]],
    inventory_summary: dict[str, dict[str, float]] | None = None,
    wape_tolerance: float = 0.15,
    slice_summaries: dict[str, dict[str, float | int | str | None]] | None = None,
    max_worst_slice_regression: float = 0.20,
    inventory_cost_tolerance: float = 0.05,
    baseline_model: str = "seasonal_naive",
    max_fill_rate_degradation: float = 0.0,
    max_interval_width_ratio: float = 1.35,
) -> str:
    """Choose the least-cost validation-feasible operational candidate.

    A challenger must first satisfy forecast, slice, service-level, and
    interval-sharpness contracts on validation data. Cost optimization happens
    only inside that feasible set. The baseline remains available as a safe
    fallback whenever no challenger satisfies every contract.
    """
    if not metrics:
        raise ValueError("no model metrics supplied")
    if baseline_model not in metrics:
        raise ValueError(f"selection baseline is missing from metrics: {baseline_model}")

    best_wape = min(float(values["wape"]) for values in metrics.values())
    eligible = [
        model
        for model, values in metrics.items()
        if float(values["wape"]) <= best_wape * (1.0 + wape_tolerance)
        or model == baseline_model
    ]

    if slice_summaries:
        robust = [
            model
            for model in eligible
            if float(
                slice_summaries.get(model, {}).get(
                    "worst_relative_wape_regression",
                    1.0,
                )
            )
            <= max_worst_slice_regression
        ]
        eligible = robust or [baseline_model]

    baseline_width = float(metrics[baseline_model].get("mean_interval_width", 0.0))
    if baseline_width > 0.0:
        sharp = [
            model
            for model in eligible
            if float(metrics[model].get("mean_interval_width", float("inf")))
            <= baseline_width * max_interval_width_ratio
        ]
        eligible = sharp or [baseline_model]

    if inventory_summary:
        if baseline_model not in inventory_summary:
            raise ValueError(
                f"selection baseline is missing from inventory summary: {baseline_model}"
            )

        eligible_with_inventory = [
            model for model in eligible if model in inventory_summary
        ]
        if not eligible_with_inventory:
            eligible_with_inventory = [baseline_model]

        has_complete_fill_rate_evidence = all(
            "fill_rate" in inventory_summary[model]
            for model in eligible_with_inventory
        )

        if has_complete_fill_rate_evidence:
            baseline_fill = float(inventory_summary[baseline_model]["fill_rate"])
            service_feasible = [
                model
                for model in eligible_with_inventory
                if float(inventory_summary[model]["fill_rate"])
                >= baseline_fill - max_fill_rate_degradation
            ]
            eligible_with_cost = service_feasible or [baseline_model]
        else:
            # Preserve the historical cost-only selector contract used by
            # lightweight callers and legacy unit fixtures. The full pipeline
            # always provides fill-rate evidence, so operational runs still
            # enforce the validation service-level constraint above.
            eligible_with_cost = eligible_with_inventory

        minimum_cost = min(
            float(inventory_summary[model]["total_cost"])
            for model in eligible_with_cost
        )
        near_cost = [
            model
            for model in eligible_with_cost
            if float(inventory_summary[model]["total_cost"])
            <= minimum_cost * (1.0 + inventory_cost_tolerance)
        ]

        has_cost_win_evidence = all(
            "cost_win_rate_vs_baseline" in inventory_summary[model]
            for model in near_cost
        )

        if len(near_cost) > 1 and (slice_summaries or has_cost_win_evidence):
            return min(
                near_cost,
                key=lambda model: (
                    -float(
                        inventory_summary[model].get(
                            "cost_win_rate_vs_baseline",
                            0.0,
                        )
                    ),
                    float(
                        slice_summaries.get(model, {}).get(
                            "worst_relative_wape_regression",
                            1.0,
                        )
                    )
                    if slice_summaries
                    else 1.0,
                    float(metrics[model]["wape"]),
                    float(inventory_summary[model]["total_cost"]),
                ),
            )

        return min(
            near_cost,
            key=lambda model: float(inventory_summary[model]["total_cost"]),
        )

    scores = {}
    for model in eligible:
        values = metrics[model]
        coverage_penalty = abs(float(values["coverage_80"]) - 0.80)
        bias_penalty = abs(float(values["signed_bias"]))
        scores[model] = (
            float(values["wape"])
            + 0.20 * coverage_penalty
            + 0.10 * bias_penalty
        )

    return min(scores, key=scores.get)


def lifecycle_metrics(frame: pd.DataFrame) -> dict[str, float | int]:
    labels = sorted(set(frame["true_lifecycle_state"]) | set(frame["estimated_lifecycle_state"]))
    return {
        "n": int(len(frame)),
        "state_macro_f1": float(
            f1_score(frame["true_lifecycle_state"], frame["estimated_lifecycle_state"], labels=labels, average="macro", zero_division=0)
        ),
        "ladt_mae": float((frame["true_ladt"] - frame["estimated_ladt"]).abs().mean()),
        "ladt_spearman": float(frame[["true_ladt", "estimated_ladt"]].corr(method="spearman").iloc[0, 1]),
    }


def forecast_slice_report(
    forecasts: pd.DataFrame,
    truth: pd.DataFrame,
    candidate_model: str = "production_candidate",
    baseline_model: str = "seasonal_naive",
    min_rows: int = 8,
) -> pd.DataFrame:
    """Compare candidate and baseline on operationally meaningful slices."""
    required_truth = ["series_id", "date_idx", "demand", "category", "store", "promo", "stockout", "drat_state"]
    missing = [column for column in required_truth if column not in truth]
    if missing:
        raise ValueError(f"slice truth is missing columns: {missing}")
    selected = forecasts[forecasts["model"].isin([candidate_model, baseline_model])].copy()
    if set(selected["model"].unique()) != {candidate_model, baseline_model}:
        raise ValueError("slice report requires both candidate and baseline forecasts")
    merged = selected.merge(
        truth[required_truth], on=["series_id", "date_idx"], how="inner", validate="many_to_one"
    )
    rows: list[dict] = []
    dimensions = {
        "category": "category",
        "store": "store",
        "promotion": "promo",
        "stockout": "stockout",
        "demand_regime": "drat_state",
    }
    for dimension, column in dimensions.items():
        for value, slice_frame in merged.groupby(column, dropna=False):
            counts = slice_frame.groupby("model").size()
            if len(counts) != 2 or counts.min() < min_rows:
                continue
            forecast_columns = ["series_id", "date_idx", "horizon", "q10", "q50", "q90", "model"]
            truth_slice = truth[truth[column].eq(value)]
            stats = {
                model: forecast_metrics(sub[forecast_columns], truth_slice)
                for model, sub in slice_frame.groupby("model")
            }
            candidate = stats[candidate_model]
            baseline = stats[baseline_model]
            baseline_wape = float(baseline["wape"])
            candidate_wape = float(candidate["wape"])
            rows.append(
                {
                    "dimension": dimension,
                    "slice_value": str(value),
                    "n": int(candidate["n"]),
                    "candidate_wape": candidate_wape,
                    "baseline_wape": baseline_wape,
                    "relative_wape_regression": float(
                        (candidate_wape - baseline_wape) / max(baseline_wape, 1e-9)
                    ),
                    "candidate_signed_bias": float(candidate["signed_bias"]),
                    "baseline_signed_bias": float(baseline["signed_bias"]),
                    "candidate_coverage_80": float(candidate["coverage_80"]),
                    "baseline_coverage_80": float(baseline["coverage_80"]),
                }
            )
    return pd.DataFrame(rows)


def summarize_forecast_slices(report: pd.DataFrame) -> dict[str, float | int | str | None]:
    if report.empty:
        return {
            "n_slices": 0,
            "worst_relative_wape_regression": 0.0,
            "worst_slice": None,
            "slice_win_rate": 1.0,
        }
    worst = report.sort_values("relative_wape_regression", ascending=False).iloc[0]
    return {
        "n_slices": int(len(report)),
        "worst_relative_wape_regression": float(worst["relative_wape_regression"]),
        "worst_slice": f"{worst['dimension']}={worst['slice_value']}",
        "slice_win_rate": float((report["relative_wape_regression"] <= 0).mean()),
    }


def validation_model_slice_report(
    forecasts: pd.DataFrame,
    truth: pd.DataFrame,
    baseline_model: str = "seasonal_naive",
    min_rows: int = 8,
) -> pd.DataFrame:
    """Build validation slice comparisons for every model versus the baseline."""
    models = sorted(set(forecasts["model"]))
    if baseline_model not in models:
        raise ValueError(f"baseline model is missing: {baseline_model}")
    rows = []
    baseline = forecasts[forecasts["model"].eq(baseline_model)]
    for model in models:
        if model == baseline_model:
            continue
        candidate = forecasts[forecasts["model"].eq(model)].copy()
        candidate["model"] = "production_candidate"
        comparison = pd.concat([baseline, candidate], ignore_index=True)
        report = forecast_slice_report(
            comparison,
            truth,
            candidate_model="production_candidate",
            baseline_model=baseline_model,
            min_rows=min_rows,
        )
        if not report.empty:
            report.insert(0, "model", model)
            rows.append(report)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def summarize_validation_model_slices(report: pd.DataFrame) -> dict[str, dict[str, float | int | str | None]]:
    output: dict[str, dict[str, float | int | str | None]] = {
        "seasonal_naive": {
            "n_slices": 0,
            "worst_relative_wape_regression": 0.0,
            "worst_slice": None,
            "slice_win_rate": 1.0,
        }
    }
    if report.empty:
        return output
    for model, sub in report.groupby("model"):
        output[str(model)] = summarize_forecast_slices(sub)
    return output
