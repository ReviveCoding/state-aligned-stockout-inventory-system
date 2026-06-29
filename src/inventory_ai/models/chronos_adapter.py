from __future__ import annotations

from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from inventory_ai.calibration.conformal import repair_quantiles


@dataclass(frozen=True)
class ChronosCapability:
    available: bool
    version: str | None
    message: str


def check_chronos_capability() -> ChronosCapability:
    """Check optional Chronos-2 support without making it a core dependency."""
    try:
        import chronos  # noqa: F401

        installed = version("chronos-forecasting")
        return ChronosCapability(True, installed, "Chronos optional dependency is available.")
    except (ImportError, PackageNotFoundError) as exc:
        return ChronosCapability(
            False,
            None,
            "Chronos is optional. Install with `pip install -e .[chronos]` before GPU zero-shot or LoRA experiments. "
            f"Detected: {type(exc).__name__}",
        )


def _timestamp(date_idx: pd.Series) -> pd.Series:
    return pd.Timestamp("2000-01-01") + pd.to_timedelta(date_idx.astype(int), unit="D")


def build_chronos_frames(
    panel: pd.DataFrame,
    origin: int,
    horizon: int,
    target_col: str = "recovered_demand_mean",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build Chronos-2 context and known-future covariates.

    A completely absent future panel is a supported production-inference case
    and receives a conservative last-value/no-promotion fallback. A partially
    populated future panel is rejected instead of silently discarding the
    available covariates and fabricating the remainder.
    """
    if horizon < 1:
        raise ValueError("Chronos horizon must be positive")
    required = {"series_id", "date_idx", target_col, "price", "promo", "dow"}
    missing = sorted(required - set(panel.columns))
    if missing:
        raise ValueError(f"missing Chronos panel columns: {missing}")
    if panel[["series_id", "date_idx"]].duplicated().any():
        raise ValueError("Chronos panel contains duplicate series/date keys")
    context = panel[panel["date_idx"] <= origin].copy()
    if context.empty or context["series_id"].nunique() != panel["series_id"].nunique():
        raise ValueError("Chronos context must contain every series")
    past_covariates = [
        column
        for column in [
            "drat_accelerating",
            "drat_stable",
            "drat_decelerating",
            "drat_intermittent",
            "drat_entropy",
            "drat_velocity",
            "recovery_uncertainty",
            "recovery_confidence",
        ]
        if column in context
    ]
    context_columns = ["series_id", "date_idx", target_col, "price", "promo", "dow"] + past_covariates
    if context[context_columns].isna().any().any():
        raise ValueError("Chronos context contains null target or covariates")
    context_df = context[context_columns].copy()
    context_df = context_df.rename(columns={"series_id": "item_id", target_col: "target"})
    context_df["timestamp"] = _timestamp(context_df["date_idx"])
    context_df = context_df.drop(columns="date_idx").sort_values(["item_id", "timestamp"]).reset_index(drop=True)

    future = panel[(panel["date_idx"] > origin) & (panel["date_idx"] <= origin + horizon)][
        ["series_id", "date_idx", "price", "promo", "dow"]
    ].copy()
    series_ids = sorted(panel["series_id"].astype(str).unique())
    expected_keys = {
        (series_id, date_idx)
        for series_id in series_ids
        for date_idx in range(origin + 1, origin + horizon + 1)
    }
    actual_keys = set(
        zip(future["series_id"].astype(str), future["date_idx"].astype(int))
    )
    if future.empty:
        # Production inference may have no future rows in the historical panel.
        last = panel[panel["date_idx"] == origin][["series_id", "price", "dow"]].copy()
        if last["series_id"].nunique() != panel["series_id"].nunique():
            raise ValueError("cannot construct Chronos future covariates at the requested origin")
        future = last.assign(_key=1).merge(
            pd.DataFrame({"step": range(1, horizon + 1), "_key": 1}), on="_key"
        ).drop(columns="_key")
        future["date_idx"] = origin + future["step"]
        future["dow"] = (future["dow"] + future["step"]) % 7
        future["promo"] = 0
        future = future.drop(columns="step")
    elif future[["series_id", "date_idx"]].duplicated().any() or actual_keys != expected_keys:
        missing_keys = sorted(expected_keys - actual_keys)[:5]
        unexpected_keys = sorted(actual_keys - expected_keys)[:5]
        raise ValueError(
            "Chronos future covariates are partially populated; "
            f"missing={missing_keys}, unexpected={unexpected_keys}"
        )
    if future[["price", "promo", "dow"]].isna().any().any():
        raise ValueError("Chronos future covariates contain null values")
    future_df = future.rename(columns={"series_id": "item_id"})
    future_df["timestamp"] = _timestamp(future_df["date_idx"])
    future_df = future_df.drop(columns="date_idx").sort_values(["item_id", "timestamp"]).reset_index(drop=True)
    return context_df, future_df


def prepare_chronos_finetune_inputs(
    panel: pd.DataFrame,
    cutoff: int,
    target_col: str = "recovered_demand_mean",
) -> list[dict[str, Any]]:
    """Convert a causal panel prefix into Chronos-2 fine-tuning inputs."""
    history = panel[panel["date_idx"] <= cutoff].sort_values(["series_id", "date_idx"])
    known_covariates = ["price", "promo", "dow"]
    past_only = [
        column
        for column in [
            "drat_accelerating",
            "drat_stable",
            "drat_decelerating",
            "drat_intermittent",
            "drat_entropy",
            "drat_velocity",
            "recovery_uncertainty",
            "recovery_confidence",
        ]
        if column in history
    ]
    inputs: list[dict[str, Any]] = []
    for _, group in history.groupby("series_id", sort=False):
        inputs.append(
            {
                "target": group[target_col].astype(np.float32).to_numpy(),
                "past_covariates": {
                    column: group[column].astype(np.float32).to_numpy()
                    for column in past_only + known_covariates
                },
                "future_covariates": {column: None for column in known_covariates},
            }
        )
    return inputs


class Chronos2Forecaster:
    """Optional Chronos-2 zero-shot and LoRA adapter.

    A pipeline object may be injected for deterministic unit tests. In normal
    use it is lazily loaded from the official Hugging Face model identifier.
    """

    def __init__(
        self,
        model_id: str = "amazon/chronos-2",
        device_map: str = "cpu",
        pipeline: Any | None = None,
    ) -> None:
        self.model_id = model_id
        self.device_map = device_map
        self._pipeline = pipeline

    @property
    def pipeline(self):
        if self._pipeline is None:
            capability = check_chronos_capability()
            if not capability.available:
                raise RuntimeError(capability.message)
            from chronos import BaseChronosPipeline

            self._pipeline = BaseChronosPipeline.from_pretrained(self.model_id, device_map=self.device_map)
        return self._pipeline

    def predict(
        self,
        panel: pd.DataFrame,
        origin: int,
        horizon: int,
        target_col: str = "recovered_demand_mean",
        model_name: str = "chronos2_zero_shot",
    ) -> pd.DataFrame:
        context_df, future_df = build_chronos_frames(panel, origin, horizon, target_col)
        prediction = self.pipeline.predict_df(
            context_df,
            future_df=future_df,
            id_column="item_id",
            timestamp_column="timestamp",
            target="target",
            prediction_length=horizon,
            quantile_levels=[0.1, 0.5, 0.9],
        )
        quantile_columns: dict[float, object] = {}
        for quantile in (0.1, 0.5, 0.9):
            candidates = [quantile, str(quantile)]
            match = next((candidate for candidate in candidates if candidate in prediction.columns), None)
            if match is None:
                raise ValueError(f"Chronos prediction is missing quantile column {quantile}")
            quantile_columns[quantile] = match
        required = {"item_id", "timestamp"}
        missing = required - set(prediction.columns)
        if missing:
            raise ValueError(f"Chronos prediction is missing columns: {sorted(missing)}")
        selected_columns = ["item_id", "timestamp"] + [quantile_columns[q] for q in (0.1, 0.5, 0.9)]
        output = prediction[selected_columns].copy()
        output = output.rename(
            columns={
                "item_id": "series_id",
                quantile_columns[0.1]: "q10",
                quantile_columns[0.5]: "q50",
                quantile_columns[0.9]: "q90",
            }
        )
        output["timestamp"] = pd.to_datetime(output["timestamp"])
        output["date_idx"] = (output["timestamp"] - pd.Timestamp("2000-01-01")).dt.days.astype(int)
        output["horizon"] = output["date_idx"] - int(origin)
        output["model"] = model_name
        output = output.drop(columns="timestamp")
        output = output[output["horizon"].between(1, horizon)]
        output = output.sort_values(["series_id", "horizon"]).reset_index(drop=True)
        if output[["series_id", "date_idx"]].duplicated().any():
            raise ValueError("Chronos prediction contains duplicate series/date keys")
        expected_keys = {
            (str(series_id), origin + step)
            for series_id in panel["series_id"].astype(str).unique()
            for step in range(1, horizon + 1)
        }
        actual_keys = set(zip(output["series_id"].astype(str), output["date_idx"].astype(int)))
        if actual_keys != expected_keys:
            raise ValueError(
                "Chronos prediction key coverage mismatch: "
                f"missing={sorted(expected_keys-actual_keys)[:5]}, "
                f"unexpected={sorted(actual_keys-expected_keys)[:5]}"
            )
        if output[["q10", "q50", "q90"]].isna().any().any():
            raise ValueError("Chronos prediction contains null quantiles")
        return repair_quantiles(output)

    def fit_lora(
        self,
        train_inputs: list[dict[str, Any]],
        prediction_length: int,
        output_dir: str | Path,
        validation_inputs: list[dict[str, Any]] | None = None,
        learning_rate: float = 1e-5,
        num_steps: int = 200,
        batch_size: int = 32,
    ) -> "Chronos2Forecaster":
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        fitted = self.pipeline.fit(
            inputs=train_inputs,
            prediction_length=prediction_length,
            validation_inputs=validation_inputs,
            finetune_mode="lora",
            learning_rate=learning_rate,
            num_steps=num_steps,
            batch_size=batch_size,
            output_dir=output_path,
        )
        return Chronos2Forecaster(self.model_id, self.device_map, pipeline=fitted)
