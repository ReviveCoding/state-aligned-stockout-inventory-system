from __future__ import annotations

import pandas as pd


def reconcile_bottom_up(forecast: pd.DataFrame, metadata: pd.DataFrame) -> pd.DataFrame:
    meta = metadata[["series_id", "category", "store"]].drop_duplicates("series_id")
    bottom = forecast.merge(meta, on="series_id", how="left", validate="many_to_one")
    if bottom[["category", "store"]].isna().any().any():
        raise ValueError("missing hierarchy metadata")
    bottom["level"] = "bottom"
    bottom["node_id"] = bottom["series_id"]
    outputs = [
        bottom[["node_id", "series_id", "date_idx", "horizon", "q10", "q50", "q90", "model", "level", "category", "store"]]
    ]
    levels = {
        "store": ["store"],
        "category": ["category"],
        "store_category": ["store", "category"],
        "total": [],
    }
    for level, dimensions in levels.items():
        group_cols = ["date_idx", "horizon", "model"] + dimensions
        aggregated = bottom.groupby(group_cols, dropna=False)[["q10", "q50", "q90"]].sum().reset_index()
        if dimensions:
            aggregated["node_id"] = level + ":" + aggregated[dimensions].astype(str).agg("|".join, axis=1)
        else:
            aggregated["node_id"] = "total"
        aggregated["series_id"] = aggregated["node_id"]
        aggregated["level"] = level
        for column in ["category", "store"]:
            if column not in aggregated:
                aggregated[column] = "ALL"
        outputs.append(
            aggregated[["node_id", "series_id", "date_idx", "horizon", "q10", "q50", "q90", "model", "level", "category", "store"]]
        )
    return pd.concat(outputs, ignore_index=True)


def hierarchy_error(reconciled: pd.DataFrame) -> float:
    errors = []
    keys = ["date_idx", "horizon", "model"]
    for quantile in ["q10", "q50", "q90"]:
        bottom = reconciled[reconciled["level"] == "bottom"].groupby(keys)[quantile].sum()
        total = reconciled[reconciled["level"] == "total"].set_index(keys)[quantile]
        joined = pd.concat([bottom.rename("bottom"), total.rename("total")], axis=1).dropna()
        if not joined.empty:
            errors.append(float((joined["bottom"] - joined["total"]).abs().max()))
    return max(errors, default=0.0)


def reconcile_scenarios_bottom_up(scenarios: pd.DataFrame, metadata: pd.DataFrame) -> pd.DataFrame:
    """Aggregate bottom-level scenario paths into an exactly coherent hierarchy."""
    meta = metadata[["series_id", "category", "store"]].drop_duplicates("series_id")
    bottom = scenarios.merge(meta, on="series_id", how="left", validate="many_to_one")
    if bottom[["category", "store"]].isna().any().any():
        raise ValueError("missing hierarchy metadata for scenario reconciliation")
    bottom["level"] = "bottom"
    bottom["node_id"] = bottom["series_id"]
    base_cols = ["node_id", "series_id", "scenario", "date_idx", "horizon", "demand", "model", "level", "category", "store"]
    outputs = [bottom[base_cols]]
    levels = {
        "store": ["store"],
        "category": ["category"],
        "store_category": ["store", "category"],
        "total": [],
    }
    for level, dimensions in levels.items():
        group_cols = ["scenario", "date_idx", "horizon", "model"] + dimensions
        aggregated = bottom.groupby(group_cols, dropna=False, as_index=False)["demand"].sum()
        if dimensions:
            aggregated["node_id"] = level + ":" + aggregated[dimensions].astype(str).agg("|".join, axis=1)
        else:
            aggregated["node_id"] = "total"
        aggregated["series_id"] = aggregated["node_id"]
        aggregated["level"] = level
        for column in ["category", "store"]:
            if column not in aggregated:
                aggregated[column] = "ALL"
        outputs.append(aggregated[base_cols])
    return pd.concat(outputs, ignore_index=True)


def scenario_hierarchy_error(reconciled: pd.DataFrame) -> float:
    keys = ["scenario", "date_idx", "horizon", "model"]
    bottom = reconciled[reconciled["level"] == "bottom"].groupby(keys)["demand"].sum()
    total = reconciled[reconciled["level"] == "total"].set_index(keys)["demand"]
    joined = pd.concat([bottom.rename("bottom"), total.rename("total")], axis=1).dropna()
    return float((joined["bottom"] - joined["total"]).abs().max()) if not joined.empty else 0.0
