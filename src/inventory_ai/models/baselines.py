from __future__ import annotations

import numpy as np
import pandas as pd


def seasonal_naive_forecast(panel: pd.DataFrame, origin: int, horizon: int, value_col: str = "sales") -> pd.DataFrame:
    rows: list[dict] = []
    history = panel[panel["date_idx"] <= origin]
    for series_id, sub in history.groupby("series_id", sort=False):
        sub = sub.sort_values("date_idx")
        weekly = sub.set_index("date_idx")[value_col]
        fallback = float(sub[value_col].tail(28).median()) if len(sub) else 0.0
        residual_scale = float(sub[value_col].diff(7).abs().tail(28).median()) if len(sub) >= 8 else max(1.0, fallback * 0.25)
        if not np.isfinite(residual_scale) or residual_scale <= 0:
            residual_scale = max(0.5, fallback * 0.25)
        for step in range(1, horizon + 1):
            date_idx = origin + step
            point = float(weekly.get(date_idx - 7, fallback))
            rows.append(
                {
                    "series_id": series_id,
                    "date_idx": date_idx,
                    "horizon": step,
                    "q10": max(0.0, point - 1.28 * residual_scale),
                    "q50": max(0.0, point),
                    "q90": max(0.0, point + 1.28 * residual_scale),
                    "model": "seasonal_naive",
                }
            )
    return pd.DataFrame(rows)


def tsb_forecast(
    panel: pd.DataFrame,
    origin: int,
    horizon: int,
    alpha: float = 0.2,
    beta: float = 0.2,
    value_col: str = "sales",
) -> pd.DataFrame:
    rows: list[dict] = []
    history = panel[panel["date_idx"] <= origin]
    for series_id, sub in history.groupby("series_id", sort=False):
        values = sub.sort_values("date_idx")[value_col].astype(float).to_numpy()
        positive = values[values > 0]
        size = float(positive[0]) if len(positive) else 0.0
        probability = float((values[: min(14, len(values))] > 0).mean()) if len(values) else 0.0
        for value in values:
            occurrence = float(value > 0)
            probability = probability + beta * (occurrence - probability)
            if value > 0:
                size = size + alpha * (value - size)
        point = max(0.0, probability * size)
        scale = max(0.5, float(np.std(values[-28:])) if len(values) > 1 else point * 0.3)
        for step in range(1, horizon + 1):
            rows.append(
                {
                    "series_id": series_id,
                    "date_idx": origin + step,
                    "horizon": step,
                    "q10": max(0.0, point - 1.28 * scale),
                    "q50": point,
                    "q90": point + 1.28 * scale,
                    "model": "tsb",
                }
            )
    return pd.DataFrame(rows)
