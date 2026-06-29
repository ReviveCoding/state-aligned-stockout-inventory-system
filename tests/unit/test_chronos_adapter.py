from __future__ import annotations

import numpy as np
import pandas as pd

from inventory_ai.censoring.recovery import recover_latent_demand
from inventory_ai.data.synthetic import SyntheticConfig, make_synthetic_retail
from inventory_ai.features.basic import add_time_features
from inventory_ai.lifecycle.drat import add_drat
from inventory_ai.models.chronos_adapter import (
    Chronos2Forecaster,
    build_chronos_frames,
    prepare_chronos_finetune_inputs,
)


def _panel():
    frame = recover_latent_demand(make_synthetic_retail(SyntheticConfig(4, 60, 91)))
    return add_drat(add_time_features(frame, value_col="recovered_demand_mean"))


def test_chronos_frames_respect_origin_and_future_contract():
    panel = _panel()
    context, future = build_chronos_frames(panel, origin=50, horizon=7)
    assert context["item_id"].nunique() == 4
    assert future["item_id"].nunique() == 4
    assert len(future) == 28
    assert context["timestamp"].max() == pd.Timestamp("2000-02-20")
    assert future["timestamp"].min() == pd.Timestamp("2000-02-21")
    assert set(["price", "promo", "dow"]).issubset(future.columns)


def test_chronos_finetune_inputs_mark_known_future_covariates():
    inputs = prepare_chronos_finetune_inputs(_panel(), cutoff=50)
    assert len(inputs) == 4
    assert set(inputs[0]["future_covariates"]) == {"price", "promo", "dow"}
    assert all(value is None for value in inputs[0]["future_covariates"].values())
    assert len(inputs[0]["target"]) == 51


class _MockChronosPipeline:
    def predict_df(self, context_df, future_df, **kwargs):
        rows = []
        for item_id, sub in future_df.groupby("item_id"):
            for _, row in sub.iterrows():
                rows.append(
                    {
                        "item_id": item_id,
                        "timestamp": row["timestamp"],
                        "target_name": "target",
                        "predictions": 5.0,
                        0.1: 3.0,
                        0.5: 5.0,
                        0.9: 8.0,
                    }
                )
        return pd.DataFrame(rows)

    def fit(self, **kwargs):
        assert kwargs["finetune_mode"] == "lora"
        return self


def test_chronos_mock_prediction_converts_to_core_contract(tmp_path):
    panel = _panel()
    forecaster = Chronos2Forecaster(pipeline=_MockChronosPipeline())
    forecast = forecaster.predict(panel, origin=50, horizon=7)
    assert len(forecast) == 28
    assert forecast["horizon"].between(1, 7).all()
    assert (forecast["q10"] <= forecast["q50"]).all()
    fitted = forecaster.fit_lora(
        prepare_chronos_finetune_inputs(panel, 45),
        prediction_length=7,
        output_dir=tmp_path,
        num_steps=1,
        batch_size=2,
    )
    assert isinstance(fitted, Chronos2Forecaster)


def test_chronos_rejects_partially_populated_future_covariates():
    panel = _panel()
    partial = panel.drop(
        panel[(panel["series_id"] == panel["series_id"].iloc[0]) & (panel["date_idx"] == 54)].index
    )
    import pytest

    with pytest.raises(ValueError, match="partially populated"):
        build_chronos_frames(partial, origin=50, horizon=7)


def test_chronos_allows_completely_absent_future_with_conservative_fallback():
    panel = _panel()
    history_only = panel[panel["date_idx"] <= 50].copy()
    _, future = build_chronos_frames(history_only, origin=50, horizon=7)
    assert len(future) == 28
    assert future["promo"].eq(0).all()


class _PartialPredictionPipeline(_MockChronosPipeline):
    def predict_df(self, context_df, future_df, **kwargs):
        return super().predict_df(context_df, future_df, **kwargs).iloc[:-1].copy()


class _StringQuantilePipeline(_MockChronosPipeline):
    def predict_df(self, context_df, future_df, **kwargs):
        frame = super().predict_df(context_df, future_df, **kwargs)
        return frame.rename(columns={0.1: "0.1", 0.5: "0.5", 0.9: "0.9"})


def test_chronos_prediction_requires_complete_key_coverage():
    import pytest

    panel = _panel()
    forecaster = Chronos2Forecaster(pipeline=_PartialPredictionPipeline())
    with pytest.raises(ValueError, match="key coverage mismatch"):
        forecaster.predict(panel, origin=50, horizon=7)


def test_chronos_accepts_string_quantile_column_names():
    panel = _panel()
    forecaster = Chronos2Forecaster(pipeline=_StringQuantilePipeline())
    forecast = forecaster.predict(panel, origin=50, horizon=7)
    assert len(forecast) == 28
    assert forecast[["q10", "q50", "q90"]].notna().all().all()
