from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from inventory_ai.simulation.inventory import InventoryConfig


@dataclass(frozen=True)
class ClosedLoopPath:
    series_id: str
    scenario_type: str
    warmup: tuple[float, ...]
    demand: tuple[float, ...]
    known_event: tuple[int, ...]


def _make_paths(n_series: int, cycles: int, seed: int) -> list[ClosedLoopPath]:
    """Generate deterministic, heterogeneous policy-feedback scenarios."""
    rng = np.random.default_rng(seed + 808)
    scenario_types = ("stable", "transient_promotion", "persistent_shift", "intermittent")
    paths: list[ClosedLoopPath] = []
    for series in range(n_series):
        scenario = scenario_types[series % len(scenario_types)]
        base = rng.uniform(7, 24)
        warmup = tuple(max(0.0, rng.normal(base, max(0.8, 0.12 * base))) for _ in range(8))
        values: list[float] = []
        events: list[int] = []
        for cycle in range(cycles):
            event = int(scenario == "transient_promotion" and cycle in {cycles // 3, 2 * cycles // 3})
            if scenario == "stable":
                mean = base
                value = max(0.0, rng.normal(mean, max(0.8, 0.15 * base)))
            elif scenario == "transient_promotion":
                mean = base * (1.65 if event else 1.0)
                value = max(0.0, rng.normal(mean, max(0.8, 0.16 * base)))
            elif scenario == "persistent_shift":
                mean = base * (1.50 if cycle >= cycles // 2 else 1.0)
                value = max(0.0, rng.normal(mean, max(0.8, 0.14 * base)))
            else:
                occurrence = rng.random() < 0.48
                value = float(rng.gamma(shape=2.2, scale=base / 2.2)) if occurrence else 0.0
            values.append(value)
            events.append(event)
        paths.append(
            ClosedLoopPath(
                series_id=f"closed_{series:03d}",
                scenario_type=scenario,
                warmup=warmup,
                demand=tuple(values),
                known_event=tuple(events),
            )
        )
    return paths


def closed_loop_policy_comparison(
    n_series: int,
    cycles: int,
    cfg: InventoryConfig,
    seed: int = 42,
    recovery_safety: float = 1.05,
) -> pd.DataFrame:
    """Compare policies on identical latent demand under policy-induced censoring.

    The adaptive policy distinguishes a one-period known event from persistent
    upward evidence. A single promotion spike therefore cannot permanently
    inflate its baseline estimate. Persistent shifts require repeated evidence
    or repeated stockouts before the faster update is activated.
    """
    paths = _make_paths(n_series, cycles, seed)
    policies = {
        "naive_no_recovery": {"recovery": False, "safety": 1.05},
        "recovery_state_aware": {"recovery": True, "safety": recovery_safety},
    }
    rows: list[dict] = []
    for policy, settings in policies.items():
        for path in paths:
            estimate = max(0.5, float(np.mean(path.warmup)))
            # Identical opening state for both policies.
            on_hand = max(0.0, estimate * (cfg.lead_time_days + cfg.review_period_days))
            open_orders: list[tuple[int, float]] = []
            previous_estimate = estimate
            stockout_streak = 0
            upward_evidence = 0
            for cycle, (latent, known_event) in enumerate(zip(path.demand, path.known_event)):
                arrivals = sum(quantity for due, quantity in open_orders if due <= cycle)
                open_orders = [(due, quantity) for due, quantity in open_orders if due > cycle]
                on_hand += arrivals
                opening_inventory = on_hand
                observed = min(on_hand, latent)
                lost = max(0.0, latent - on_hand)
                stockout = int(lost > 1e-9)
                stockout_streak = stockout_streak + 1 if stockout else 0
                on_hand = max(0.0, on_hand - observed)

                recovered = observed
                persistent_signal = False
                if settings["recovery"]:
                    # Known promotions are transient: normalize them before the
                    # structural baseline update, while still allowing an upper
                    # demand recovery for the censored event itself.
                    normalized_observed = observed / 1.55 if known_event else observed
                    if stockout:
                        censor_floor = previous_estimate * (1.08 + 0.07 * min(stockout_streak, 3))
                        if known_event:
                            censor_floor = max(censor_floor, previous_estimate * 1.45)
                        recovered = max(observed, censor_floor)
                        normalized_observed = recovered / 1.55 if known_event else recovered
                    high_signal = (not known_event) and (
                        normalized_observed > 1.18 * previous_estimate or stockout_streak >= 2
                    )
                    upward_evidence = min(4, upward_evidence + 1) if high_signal else max(0, upward_evidence - 1)
                    persistent_signal = upward_evidence >= 2
                    alpha = 0.52 if persistent_signal else 0.24
                    estimate = (1.0 - alpha) * previous_estimate + alpha * normalized_observed
                    trend_factor = 1.10 if persistent_signal else 1.0
                else:
                    estimate = 0.72 * previous_estimate + 0.28 * observed
                    trend_factor = 1.0

                forecast = max(0.0, estimate * trend_factor)
                protection_periods = cfg.lead_time_days + cfg.review_period_days
                protection_target = forecast * protection_periods * float(settings["safety"])
                inventory_position = on_hand + sum(quantity for _, quantity in open_orders)
                decision_eligible = cycle + protection_periods < cycles
                is_review_period = (cycle % cfg.review_period_days == 0) and decision_eligible
                order_qty = max(0.0, protection_target - inventory_position) if is_review_period else 0.0
                if cfg.order_capacity is not None:
                    order_qty = min(order_qty, cfg.order_capacity)
                if order_qty > 0:
                    if cfg.lead_time_days == 0:
                        on_hand += order_qty
                    else:
                        open_orders.append((cycle + cfg.lead_time_days, order_qty))
                cost = cfg.shortage_cost * lost + cfg.holding_cost * on_hand + cfg.order_cost * order_qty
                rows.append(
                    {
                        "policy": policy,
                        "series_id": path.series_id,
                        "scenario_type": path.scenario_type,
                        "cycle": cycle,
                        "known_event": known_event,
                        "persistent_signal": int(persistent_signal),
                        "stockout_streak": stockout_streak,
                        "latent_demand": latent,
                        "opening_inventory": opening_inventory,
                        "observed_sales": observed,
                        "recovered_demand": recovered,
                        "forecast": forecast,
                        "decision_eligible": int(decision_eligible),
                        "is_review_period": int(is_review_period),
                        "target_inventory_position": protection_target if decision_eligible else float("nan"),
                        "arrivals": arrivals,
                        "ending_inventory": on_hand,
                        "order_qty": order_qty,
                        "open_order_qty": sum(quantity for _, quantity in open_orders),
                        "stockout": stockout,
                        "lost_sales": lost,
                        "cost": cost,
                    }
                )
                previous_estimate = estimate
    return pd.DataFrame(rows)


def summarize_closed_loop(frame: pd.DataFrame) -> dict[str, dict[str, float]]:
    output: dict[str, dict[str, float]] = {}
    for policy, sub in frame.groupby("policy"):
        ordered = sub.sort_values(["series_id", "cycle"]).copy()
        previous_stockout = ordered.groupby("series_id", sort=False)["stockout"].shift(1).fillna(0).astype(int)
        repeated_events = ordered["stockout"].astype(int).eq(1) & previous_stockout.eq(1)
        total_stockouts = int(ordered["stockout"].sum())
        transition_opportunities = int(
            ordered.groupby("series_id", sort=False).size().sub(1).clip(lower=0).sum()
        )
        output[policy] = {
            "total_cost": float(ordered["cost"].sum()),
            "lost_sales": float(ordered["lost_sales"].sum()),
            "stockout_rate": float(ordered["stockout"].mean()),
            # Frequency of consecutive stockout events over all adjacent-cycle
            # opportunities. Unlike repeats/stockouts, this denominator does
            # not worsen merely because a policy eliminates isolated stockouts.
            "repeat_stockout_ratio": float(repeated_events.sum() / max(transition_opportunities, 1)),
            "repeat_stockout_share": float(repeated_events.sum() / max(total_stockouts, 1)),
            "repeat_stockout_events": int(repeated_events.sum()),
            "cumulative_bias": float((ordered["forecast"] - ordered["latent_demand"]).sum() / (ordered["latent_demand"].sum() + 1e-9)),
            "inventory_oscillation": float(ordered.groupby("series_id")["ending_inventory"].diff().abs().mean()),
        }
    return output


def summarize_closed_loop_slices(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (policy, scenario), sub in frame.groupby(["policy", "scenario_type"]):
        demand = float(sub["latent_demand"].sum())
        rows.append(
            {
                "policy": policy,
                "scenario_type": scenario,
                "total_cost": float(sub["cost"].sum()),
                "lost_sales": float(sub["lost_sales"].sum()),
                "fill_rate": float(1.0 - sub["lost_sales"].sum() / (demand + 1e-9)),
                "stockout_rate": float(sub["stockout"].mean()),
                "average_inventory": float(sub["ending_inventory"].mean()),
            }
        )
    return pd.DataFrame(rows)


def closed_loop_safety_sweep(
    n_series: int,
    cycles: int,
    cfg: InventoryConfig,
    seed: int = 42,
    safety_values: tuple[float, ...] = (0.95, 1.00, 1.05, 1.10, 1.15),
) -> pd.DataFrame:
    rows = []
    for safety in safety_values:
        frame = closed_loop_policy_comparison(n_series, cycles, cfg, seed, recovery_safety=safety)
        summary = summarize_closed_loop(frame)
        for policy, metrics in summary.items():
            rows.append({"recovery_safety": safety, "policy": policy, **metrics})
    return pd.DataFrame(rows)


def compare_closed_loop_slices(
    slice_summary: pd.DataFrame,
    candidate_policy: str = "recovery_state_aware",
    baseline_policy: str = "naive_no_recovery",
) -> pd.DataFrame:
    """Compute per-scenario decision regressions against a common baseline."""
    required = {"policy", "scenario_type", "total_cost", "lost_sales", "fill_rate", "stockout_rate"}
    missing = required - set(slice_summary.columns)
    if missing:
        raise ValueError(f"closed-loop slice summary missing columns: {sorted(missing)}")
    baseline = slice_summary[slice_summary["policy"].eq(baseline_policy)].set_index("scenario_type")
    candidate = slice_summary[slice_summary["policy"].eq(candidate_policy)].set_index("scenario_type")
    if set(baseline.index) != set(candidate.index):
        raise ValueError("closed-loop policies must cover identical scenario slices")
    rows = []
    for scenario in sorted(baseline.index):
        base = baseline.loc[scenario]
        cand = candidate.loc[scenario]
        rows.append(
            {
                "scenario_type": scenario,
                "baseline_cost": float(base["total_cost"]),
                "candidate_cost": float(cand["total_cost"]),
                "relative_cost_regression": float(
                    (cand["total_cost"] - base["total_cost"]) / max(float(base["total_cost"]), 1e-9)
                ),
                "baseline_lost_sales": float(base["lost_sales"]),
                "candidate_lost_sales": float(cand["lost_sales"]),
                "fill_rate_degradation": float(base["fill_rate"] - cand["fill_rate"]),
                "stockout_rate_degradation": float(cand["stockout_rate"] - base["stockout_rate"]),
            }
        )
    return pd.DataFrame(rows)


def summarize_closed_loop_slice_comparison(report: pd.DataFrame) -> dict[str, float | str | int | None]:
    if report.empty:
        return {
            "n_slices": 0,
            "worst_relative_cost_regression": 0.0,
            "worst_cost_slice": None,
            "worst_fill_rate_degradation": 0.0,
            "worst_fill_slice": None,
        }
    worst_cost = report.sort_values("relative_cost_regression", ascending=False).iloc[0]
    worst_fill = report.sort_values("fill_rate_degradation", ascending=False).iloc[0]
    return {
        "n_slices": int(len(report)),
        "worst_relative_cost_regression": float(worst_cost["relative_cost_regression"]),
        "worst_cost_slice": str(worst_cost["scenario_type"]),
        "worst_fill_rate_degradation": float(worst_fill["fill_rate_degradation"]),
        "worst_fill_slice": str(worst_fill["scenario_type"]),
    }
