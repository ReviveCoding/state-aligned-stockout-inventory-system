from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

from inventory_ai.features.basic import build_direct_training_frame, build_origin_frame

NUMERIC_FEATURES = [
    "date_idx",
    "series_age",
    "horizon",
    "future_dow",
    "future_price",
    "future_promo",
    "lag_1",
    "lag_7",
    "lag_14",
    "lag_28",
    "roll_mean_7",
    "roll_mean_14",
    "roll_mean_28",
    "roll_std_7",
    "roll_std_14",
    "roll_std_28",
    "zero_ratio_14",
    "price_change",
    "drat_accelerating",
    "drat_stable",
    "drat_decelerating",
    "drat_intermittent",
    "drat_entropy",
    "drat_velocity",
    "drat_progress",
    "recovery_uncertainty",
    "recovery_confidence",
]
CATEGORICAL_FEATURES = ["category", "store"]


@dataclass
class DirectQuantileForecaster:
    quantiles: tuple[float, ...] = (0.1, 0.5, 0.9)
    max_iter: int = 45
    random_state: int = 42
    models: dict[float, GradientBoostingRegressor] = field(default_factory=dict)
    category_maps: dict[str, dict[str, int]] = field(default_factory=dict)

    def _encode(self, frame: pd.DataFrame, *, fit: bool) -> pd.DataFrame:
        encoded = frame[NUMERIC_FEATURES].astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0).copy()
        for column in CATEGORICAL_FEATURES:
            values = frame[column].astype(str)
            if fit:
                unique = sorted(values.unique())
                self.category_maps[column] = {value: idx + 1 for idx, value in enumerate(unique)}
            mapping = self.category_maps.get(column, {})
            encoded[f"{column}_code"] = values.map(mapping).fillna(0).astype(float)
        return encoded

    def fit(
        self,
        featured_panel: pd.DataFrame,
        cutoff: int,
        horizon: int,
        max_rows: int,
        target_col: str = "recovered_demand_mean",
    ) -> "DirectQuantileForecaster":
        training = build_direct_training_frame(
            featured_panel,
            cutoff=cutoff,
            max_horizon=horizon,
            target_col=target_col,
            max_rows=max_rows,
            seed=self.random_state,
        )
        if len(training) < 100:
            raise ValueError("not enough direct-training rows")
        X = self._encode(training, fit=True)
        y = training["target"].astype(float).to_numpy()
        self.models = {}
        for quantile in self.quantiles:
            estimator = GradientBoostingRegressor(
                loss="quantile",
                alpha=quantile,
                learning_rate=0.06,
                n_estimators=self.max_iter,
                max_depth=3,
                min_samples_leaf=12,
                subsample=0.85,
                random_state=self.random_state,
            )
            estimator.fit(X, y)
            self.models[quantile] = estimator
        return self

    def predict(self, featured_panel: pd.DataFrame, origin: int, horizon: int, model_name: str = "quantile_gbm") -> pd.DataFrame:
        if not self.models:
            raise RuntimeError("forecaster must be fitted before predict")
        frame = build_origin_frame(featured_panel, origin=origin, horizon=horizon)
        X = self._encode(frame, fit=False)
        predictions = {q: np.maximum(0.0, model.predict(X)) for q, model in self.models.items()}
        q50 = predictions[0.5]
        return pd.DataFrame(
            {
                "series_id": frame["series_id"].to_numpy(),
                "date_idx": frame["date_idx"].astype(int).to_numpy(),
                "horizon": frame["horizon"].astype(int).to_numpy(),
                "q10": np.minimum(predictions[0.1], q50),
                "q50": q50,
                "q90": np.maximum(predictions[0.9], q50),
                "model": model_name,
            }
        )
