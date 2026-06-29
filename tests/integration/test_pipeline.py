from __future__ import annotations

from inventory_ai.config import (
    DataConfig,
    InventoryConfigData,
    ModelConfig,
    PipelineConfig,
    ReleaseGateConfig,
)
from inventory_ai.pipeline import run_pipeline
from inventory_ai.reporting.sql_marts import build_sql_marts
from inventory_ai.utils.io import sha256_file


def test_end_to_end_pipeline_writes_connected_artifacts(tmp_path):
    cfg = PipelineConfig(
        seed=31,
        run_name="pytest_smoke",
        data=DataConfig(source="synthetic", n_series=6, history_days=84, controlled_stockout_rate=0.15),
        model=ModelConfig(horizon=7, max_train_rows=2500, max_iter=8, validation_horizon=7),
        inventory=InventoryConfigData(lead_time_days=2),
        release_gate=ReleaseGateConfig(min_cost_win_rate=0.0, max_candidate_cost_regression=1.0),
        closed_loop_cycles=10,
        lifecycle_series=8,
    )
    result = run_pipeline(cfg, tmp_path)
    assert result["release"]["gate_status"] in {"PASS", "ITERATE"}
    metrics = result["metrics"]
    assert metrics["calibration_fold_count"] == 1
    assert metrics["validation_fold_count"] >= 1
    assert metrics["calibration_origin"] < min(metrics["validation_origins"])
    required = [
        "reports/calibration_forecasts.csv",
        "reports/forecast_output.csv",
        "reports/reconciled_forecasts.csv",
        "reports/inventory_simulation.csv",
        "reports/closed_loop_replay.csv",
        "reports/closed_loop_slices.csv",
        "reports/reconciled_scenarios.csv",
        "reports/recovery_posterior_samples.csv",
        "reports/metrics_summary.json",
        "reports/release_gate.json",
        "artifacts/run_manifest.json",
    ]
    for relative in required:
        assert (tmp_path / relative).exists(), relative

    forecast_path = tmp_path / "reports/forecast_output.csv"
    first_hash = sha256_file(forecast_path)
    first_rows = sum(1 for _ in forecast_path.open(encoding="utf-8"))
    rerun = run_pipeline(cfg, tmp_path)
    assert rerun["release"]["gate_status"] in {"PASS", "ITERATE"}
    assert sha256_file(forecast_path) == first_hash
    assert sum(1 for _ in forecast_path.open(encoding="utf-8")) == first_rows


def test_sql_marts_library_requires_completed_pipeline(tmp_path):
    import pytest

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "run_name: missing_run\ndata:\n  source: synthetic\n  n_series: 4\n  history_days: 84\nmodel:\n  horizon: 7\n  validation_horizon: 7\n",
        encoding="utf-8",
    )
    with pytest.raises(FileNotFoundError, match="run the forecasting pipeline"):
        build_sql_marts(tmp_path, config_path)
