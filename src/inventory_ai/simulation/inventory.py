from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class InventoryConfig:
    holding_cost: float = 1.0
    shortage_cost: float = 5.0
    order_cost: float = 0.1
    lead_time_days: int = 2
    review_period_days: int = 1
    service_level: float = 0.8
    initial_inventory_multiplier: float = 2.0
    order_capacity: float | None = None


def _service_quantile(frame: pd.DataFrame, service_level: float) -> pd.Series:
    """Interpolate a requested service quantile from q10/q50/q90 anchors.

    The previous implementation mapped every service level between 0.20 and
    0.85 to q50, which made a configured 0.80 service target behave like a
    median policy. Piecewise-linear interpolation preserves the configured
    risk preference while remaining deterministic and dependency-free.
    """
    level = float(np.clip(service_level, 0.0, 1.0))
    q10 = frame["q10"].astype(float)
    q50 = frame["q50"].astype(float)
    q90 = frame["q90"].astype(float)
    if level <= 0.10:
        return np.maximum(0.0, q10 * (level / 0.10))
    if level <= 0.50:
        weight = (level - 0.10) / 0.40
        return q10 + weight * (q50 - q10)
    if level <= 0.90:
        weight = (level - 0.50) / 0.40
        return q50 + weight * (q90 - q50)
    # Conservative linear tail extrapolation. Nonnegativity is enforced below.
    tail_slope = (q90 - q50) / 0.40
    return np.maximum(0.0, q90 + (level - 0.90) * tail_slope)


def common_initial_inventory(
    forecast: pd.DataFrame,
    cfg: InventoryConfig,
    reference_model: str = "seasonal_naive",
) -> dict[str, float]:
    """Create one model-independent opening inventory state per series.

    A model comparison is invalid when every model starts with inventory based
    on its own forecast. We use a fixed operational reference model when it is
    available, otherwise the cross-model median, and reuse that state for all
    candidates.
    """
    working = forecast.copy()
    working["service_quantile"] = _service_quantile(working, cfg.service_level)
    if reference_model in set(working["model"]):
        reference = working[working["model"] == reference_model]
    else:
        reference = (
            working.groupby(["series_id", "date_idx", "horizon"], as_index=False)["service_quantile"]
            .median()
        )
    protection = max(1, cfg.lead_time_days + cfg.review_period_days)
    initial: dict[str, float] = {}
    for series_id, sub in reference.groupby("series_id", sort=False):
        target = float(sub.sort_values("date_idx")["service_quantile"].head(protection).sum())
        initial[str(series_id)] = max(0.0, cfg.initial_inventory_multiplier * target)
    return initial


def run_open_loop_policy(
    forecast: pd.DataFrame,
    truth: pd.DataFrame,
    cfg: InventoryConfig,
    policy_scale_by_model: dict[str, float] | None = None,
    initial_inventory_by_series: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Periodic-review lost-sales simulation with delayed replenishment.

    Every model must cover exactly the same truth keys. Orders enter an open-order
    pipeline and arrive only after the configured lead time.
    """
    truth_keys = truth[["series_id", "date_idx"]].drop_duplicates()
    actual_keys = forecast[["series_id", "date_idx"]].drop_duplicates()
    if set(map(tuple, truth_keys.to_numpy())) != set(map(tuple, actual_keys.to_numpy())):
        raise ValueError("forecast and truth must have identical keys")
    rows: list[dict] = []
    policy_scale_by_model = policy_scale_by_model or {}
    working = forecast.copy()
    working["service_quantile"] = _service_quantile(working, cfg.service_level)
    initial_inventory_by_series = initial_inventory_by_series or common_initial_inventory(working, cfg)
    truth_lookup = truth.set_index(["series_id", "date_idx"])["demand"]
    for (model, series_id), sub in working.groupby(["model", "series_id"], sort=False):
        sub = sub.sort_values("date_idx").reset_index(drop=True)
        policy_scale = float(policy_scale_by_model.get(model, 1.0))
        on_hand = float(initial_inventory_by_series[str(series_id)])
        initial_on_hand = on_hand
        open_orders: list[tuple[int, float]] = []
        for idx, row in sub.iterrows():
            date_idx = int(row["date_idx"])
            arrivals = sum(quantity for due, quantity in open_orders if due <= date_idx)
            open_orders = [(due, quantity) for due, quantity in open_orders if due > date_idx]
            on_hand += arrivals
            opening_inventory = on_hand
            demand = float(truth_lookup.loc[(series_id, date_idx)])
            sales = min(on_hand, demand)
            lost_sales = max(0.0, demand - on_hand)
            on_hand = max(0.0, on_hand - sales)
            outstanding = sum(quantity for _, quantity in open_orders)
            inventory_position = on_hand + outstanding

            # Decisions are made after the current period's demand. Therefore
            # the protection target must start at the next period and orders
            # may only be placed on configured review epochs.
            protection_periods = cfg.lead_time_days + cfg.review_period_days
            protection_start = idx + 1
            protection_end = protection_start + protection_periods
            decision_eligible = protection_end <= len(sub)
            is_review_period = (idx % cfg.review_period_days == 0) and decision_eligible
            if decision_eligible:
                target = policy_scale * float(
                    sub.iloc[protection_start:protection_end]["service_quantile"].sum()
                )
            else:
                # Do not place terminal orders whose complete protection window
                # lies outside the evaluation horizon. Such orders incur cost
                # without observable benefit and bias finite-horizon comparisons.
                target = float("nan")
            order_qty = max(0.0, target - inventory_position) if is_review_period else 0.0
            if cfg.order_capacity is not None:
                order_qty = min(order_qty, cfg.order_capacity)
            due = date_idx + cfg.lead_time_days
            if order_qty > 0:
                if cfg.lead_time_days == 0:
                    on_hand += order_qty
                else:
                    open_orders.append((due, order_qty))
            cost = cfg.holding_cost * on_hand + cfg.shortage_cost * lost_sales + cfg.order_cost * order_qty
            rows.append(
                {
                    "model": model,
                    "series_id": series_id,
                    "date_idx": date_idx,
                    "initial_inventory": initial_on_hand,
                    "opening_inventory": opening_inventory,
                    "arrivals": arrivals,
                    "demand": demand,
                    "sales": sales,
                    "lost_sales": lost_sales,
                    "ending_inventory": on_hand,
                    "inventory_position": inventory_position,
                    "target_inventory_position": target,
                    "decision_eligible": int(decision_eligible),
                    "is_review_period": int(is_review_period),
                    "order_qty": order_qty,
                    "open_order_qty": sum(quantity for _, quantity in open_orders),
                    "stockout": int(lost_sales > 1e-9),
                    "cost": cost,
                    "policy_scale": policy_scale,
                }
            )
    result = pd.DataFrame(rows)
    demand_by_model = result.groupby("model")["demand"].sum()
    if demand_by_model.max() - demand_by_model.min() > 1e-8:
        raise AssertionError("models were evaluated on different total demand")
    return result


def run_open_loop_by_model(
    forecast: pd.DataFrame,
    truth: pd.DataFrame,
    cfg: InventoryConfig,
    policy_scale_by_model: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Run model-isolated simulations with one shared initial inventory state.

    This is numerically equivalent to a combined simulation but bounds peak
    intermediate state and avoids long-lived groupby state after repeated model
    fitting in constrained Python/BLAS environments.
    """
    if forecast.empty:
        raise ValueError("forecast is empty")
    policy_scale_by_model = policy_scale_by_model or {}
    initial_state = common_initial_inventory(forecast, cfg)
    parts: list[pd.DataFrame] = []
    for model, sub in forecast.groupby("model", sort=True):
        parts.append(
            run_open_loop_policy(
                sub,
                truth,
                cfg,
                policy_scale_by_model={str(model): float(policy_scale_by_model.get(model, 1.0))},
                initial_inventory_by_series=initial_state,
            )
        )
    result = pd.concat(parts, ignore_index=True)
    return result.sort_values(["model", "series_id", "date_idx"]).reset_index(drop=True)


def summarize_inventory(simulation: pd.DataFrame) -> dict[str, dict[str, float]]:
    summary = {}
    for model, sub in simulation.groupby("model"):
        demand = float(sub["demand"].sum())
        summary[model] = {
            "total_cost": float(sub["cost"].sum()),
            "lost_sales": float(sub["lost_sales"].sum()),
            "fill_rate": float(1.0 - sub["lost_sales"].sum() / (demand + 1e-9)),
            "stockout_rate": float(sub["stockout"].mean()),
            "average_inventory": float(sub["ending_inventory"].mean()),
            "order_volatility": float(sub.groupby("series_id")["order_qty"].std().fillna(0).mean()),
        }
    return summary


def tune_policy_scales(
    forecast: pd.DataFrame,
    truth: pd.DataFrame,
    cfg: InventoryConfig,
    scales: tuple[float, ...] = (0.75, 0.80, 0.85, 0.90, 0.95, 1.0, 1.05, 1.10, 1.15),
    fill_rate_tolerance: float = 0.01,
) -> pd.DataFrame:
    """Tune a replenishment scale on validation data with a fill-rate guardrail."""
    rows = []
    initial_state = common_initial_inventory(forecast, cfg)
    baseline_frame = run_open_loop_policy(
        forecast[forecast["model"] == "seasonal_naive"], truth, cfg,
        initial_inventory_by_series=initial_state,
    )
    baseline_fill = summarize_inventory(baseline_frame)["seasonal_naive"]["fill_rate"]
    for model, sub in forecast.groupby("model"):
        for scale in scales:
            simulation = run_open_loop_policy(
                sub, truth, cfg, {model: scale}, initial_inventory_by_series=initial_state
            )
            summary = summarize_inventory(simulation)[model]
            rows.append(
                {
                    "model": model,
                    "policy_scale": float(scale),
                    "meets_fill_guardrail": bool(summary["fill_rate"] >= baseline_fill - fill_rate_tolerance),
                    **summary,
                }
            )
    return pd.DataFrame(rows)


def best_tuned_policy_by_model(
    tuning: pd.DataFrame,
    baseline_model: str = "seasonal_naive",
    max_fill_rate_degradation: float = 0.0,
) -> dict[str, dict[str, float]]:
    """Choose each model's lowest-cost policy under a baseline-relative service guard.

    The operational baseline is independently tuned first. Challenger scales are
    feasible only when their validation fill rate preserves that tuned baseline
    within the configured degradation allowance.
    """
    if tuning.empty:
        raise ValueError("policy tuning table is empty")
    if baseline_model not in set(tuning["model"]):
        raise ValueError(f"policy baseline is missing: {baseline_model}")

    baseline_rows = tuning[tuning["model"].eq(baseline_model)]
    baseline_guarded = baseline_rows[baseline_rows["meets_fill_guardrail"]]

    if baseline_guarded.empty:
        baseline_guarded = baseline_rows

    baseline_choice = baseline_guarded.sort_values(
        ["total_cost", "policy_scale"]
    ).iloc[0]

    baseline_fill = float(baseline_choice["fill_rate"])
    output: dict[str, dict[str, float]] = {}

    for model, sub in tuning.groupby("model", sort=True):
        service_feasible = sub[
            sub["fill_rate"].astype(float)
            >= baseline_fill - max_fill_rate_degradation
        ]

        is_service_feasible = not service_feasible.empty
        pool = service_feasible if is_service_feasible else sub

        best = pool.sort_values(["total_cost", "policy_scale"]).iloc[0]

        output[model] = {
            "total_cost": float(best["total_cost"]),
            "lost_sales": float(best["lost_sales"]),
            "fill_rate": float(best["fill_rate"]),
            "stockout_rate": float(best["stockout_rate"]),
            "average_inventory": float(best["average_inventory"]),
            "order_volatility": float(best["order_volatility"]),
            "policy_scale": float(best["policy_scale"]),
            "baseline_fill_rate": baseline_fill,
            "fill_rate_degradation_vs_baseline": float(
                baseline_fill - float(best["fill_rate"])
            ),
            "meets_service_guardrail": float(is_service_feasible),
        }

    return output


def evaluate_selected_policy_scales(
    forecast: pd.DataFrame,
    truth: pd.DataFrame,
    cfg: InventoryConfig,
    policy_scale_by_model: dict[str, float],
    baseline_model: str = "seasonal_naive",
) -> tuple[pd.DataFrame, dict[str, dict[str, float]]]:
    """Evaluate validation-selected policy scales with series-level stability.

    Aggregate cost alone can hide that a candidate wins on only a few large
    series. The returned summary therefore includes the fraction of series
    whose cost is no worse than the operational baseline under independently
    tuned, validation-only policy scales.
    """
    simulation = run_open_loop_by_model(
        forecast,
        truth,
        cfg,
        policy_scale_by_model=policy_scale_by_model,
    )
    summary = summarize_inventory(simulation)
    pivot = simulation.groupby(["series_id", "model"])["cost"].sum().unstack("model")
    if baseline_model not in pivot:
        raise ValueError(f"policy evaluation baseline is missing: {baseline_model}")
    baseline_cost = pivot[baseline_model]
    for model, values in summary.items():
        values["policy_scale"] = float(policy_scale_by_model.get(model, 1.0))
        values["cost_win_rate_vs_baseline"] = (
            1.0 if model == baseline_model
            else float((pivot[model] <= baseline_cost).mean())
        )
    return simulation, summary

def summarize_policy_at_scale(
    tuning: pd.DataFrame,
    policy_scale: float = 1.0,
) -> dict[str, dict[str, float]]:
    """Recover per-model diagnostics already computed during policy tuning."""
    output: dict[str, dict[str, float]] = {}
    for model, sub in tuning.groupby("model", sort=True):
        selected = sub[np.isclose(sub["policy_scale"].astype(float), float(policy_scale))]
        if len(selected) != 1:
            raise ValueError(
                f"expected exactly one tuning row for model={model!r}, scale={policy_scale}"
            )
        row = selected.iloc[0]
        output[str(model)] = {
            "total_cost": float(row["total_cost"]),
            "lost_sales": float(row["lost_sales"]),
            "fill_rate": float(row["fill_rate"]),
            "stockout_rate": float(row["stockout_rate"]),
            "average_inventory": float(row["average_inventory"]),
            "order_volatility": float(row["order_volatility"]),
            "policy_scale": float(row["policy_scale"]),
        }
    return output
