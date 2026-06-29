from __future__ import annotations

import pandas as pd
import pytest

from inventory_ai.config import DataConfig, ModelConfig, PipelineConfig, validate_config
from inventory_ai.contracts import validate_forecast, validate_panel
from inventory_ai.data.synthetic import SyntheticConfig, make_synthetic_retail


def test_config_rejects_short_history():
    cfg = PipelineConfig(data=DataConfig(history_days=30), model=ModelConfig(horizon=14))
    with pytest.raises(ValueError, match="disjoint calibration"):
        validate_config(cfg)


def test_panel_contract_rejects_duplicate_keys():
    frame = make_synthetic_retail(SyntheticConfig(4, 50, 1))
    duplicated = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate"):
        validate_panel(duplicated)


def test_forecast_contract_rejects_key_coverage_mismatch():
    frame = make_synthetic_retail(SyntheticConfig(4, 50, 2))
    truth = frame[frame["date_idx"].between(45, 49)][["series_id", "date_idx"]]
    forecast = truth.copy()
    forecast["horizon"] = forecast["date_idx"] - 44
    forecast["q10"] = 1.0
    forecast["q50"] = 2.0
    forecast["q90"] = 3.0
    forecast["model"] = "x"
    validate_forecast(forecast, truth)
    with pytest.raises(ValueError, match="coverage mismatch"):
        validate_forecast(forecast.iloc[:-1], truth)


def test_config_rejects_unsafe_paths_and_overlapping_validation():
    with pytest.raises(ValueError, match="validation_horizon"):
        validate_config(PipelineConfig(model=ModelConfig(horizon=14, validation_horizon=7)))
    with pytest.raises(ValueError, match="output_dir"):
        validate_config(PipelineConfig(output_dir="../outside"))


def test_auto_source_falls_back_only_for_missing_archive(tmp_path):
    from dataclasses import replace
    from inventory_ai.pipeline import _load_data

    cfg = PipelineConfig(
        data=DataConfig(source="auto", m5_zip_path=str(tmp_path / "missing.zip"), n_series=4, history_days=50, allow_fallback=True),
        model=ModelConfig(horizon=10, validation_horizon=10),
    )
    frame, source, reason = _load_data(cfg)
    assert source == "synthetic_fallback"
    assert reason and "not found" in reason
    assert not frame.empty


def test_auto_source_does_not_hide_corrupt_archive(tmp_path):
    from inventory_ai.pipeline import _load_data

    archive = tmp_path / "corrupt.zip"
    archive.write_text("not a zip", encoding="utf-8")
    cfg = PipelineConfig(
        data=DataConfig(source="auto", m5_zip_path=str(archive), n_series=4, history_days=50, allow_fallback=True),
        model=ModelConfig(horizon=10, validation_horizon=10),
    )
    with pytest.raises(Exception):
        _load_data(cfg)


def test_optional_spark_path_fails_with_actionable_message_when_unavailable():
    from inventory_ai.data.spark_features import require_spark, spark_available

    if spark_available():
        return
    with pytest.raises(RuntimeError, match=r"pip install -e \.\[spark\]"):
        require_spark()


def test_panel_contract_rejects_gaps_and_unaligned_series():
    frame = make_synthetic_retail(SyntheticConfig(4, 50, 71))
    with_gap = frame.drop(frame[(frame["series_id"] == frame["series_id"].iloc[0]) & (frame["date_idx"] == 12)].index)
    with pytest.raises(ValueError, match="contiguous"):
        validate_panel(with_gap)
    shortened = frame.drop(frame[(frame["series_id"] == frame["series_id"].iloc[0]) & (frame["date_idx"] == 49)].index)
    with pytest.raises(ValueError, match="common min/max"):
        validate_panel(shortened)


def test_forecast_contract_rejects_inconsistent_horizon_origin():
    frame = make_synthetic_retail(SyntheticConfig(4, 50, 72))
    truth = frame[frame["date_idx"].between(45, 49)][["series_id", "date_idx"]]
    forecast = truth.copy()
    forecast["horizon"] = forecast["date_idx"] - 44
    forecast["q10"] = 1.0
    forecast["q50"] = 2.0
    forecast["q90"] = 3.0
    forecast["model"] = "x"
    bad = forecast.copy()
    bad.loc[bad.index[0], "horizon"] = 2
    with pytest.raises(ValueError, match="noncontiguous|inconsistent"):
        validate_forecast(bad, truth)


def test_m5_price_imputation_is_future_invariant():
    from inventory_ai.data.m5 import impute_prices_causally

    prefix = pd.DataFrame({
        "id": ["a", "b", "a", "b"],
        "store_id": ["s1", "s1", "s1", "s1"],
        "item_id": ["i1", "i2", "i1", "i2"],
        "wm_yr_wk": [1, 1, 2, 2],
        "date_idx": [0, 0, 1, 1],
        "sell_price": [float("nan"), 2.0, float("nan"), 3.0],
    })
    extended = pd.concat([
        prefix,
        pd.DataFrame({
            "id": ["a", "b"],
            "store_id": ["s1", "s1"],
            "item_id": ["i1", "i2"],
            "wm_yr_wk": [3, 3],
            "date_idx": [2, 2],
            "sell_price": [1000.0, 1200.0],
        }),
    ], ignore_index=True)
    prefix_price = impute_prices_causally(prefix)
    extended_price = impute_prices_causally(extended).iloc[: len(prefix)]
    assert prefix_price.tolist() == extended_price.tolist()
    assert prefix_price.tolist() == [2.0, 2.0, 3.0, 3.0]


def test_release_gate_rejects_excessive_open_loop_fill_degradation():
    from dataclasses import asdict
    from inventory_ai.config import ReleaseGateConfig
    from inventory_ai.gates.release import evaluate_release

    metrics = {
        "truth_coverage_error": 0,
        "negative_forecasts": 0,
        "quantile_crossing": 0,
        "hierarchy_max_error": 0.0,
        "scenario_hierarchy_error": 0.0,
        "scenario_diagnostics": {"normalized_quantile_mae": 0.0},
        "test_forecast_metrics": {"production_candidate": {"coverage_80": 0.8}},
        "recovery_diagnostics": {"mae_improvement": 1.0, "absolute_bias_improvement": 1.0, "q95_upper_coverage": 1.0},
        "lifecycle_metrics": {"state_macro_f1": 1.0, "ladt_mae": 0.0, "ladt_spearman": 1.0},
        "candidate_wape_improvement": 0.1,
        "cost_win_rate": 1.0,
        "candidate_cost_regression": -0.1,
        "candidate_fill_rate_degradation": 0.02,
        "forecast_slice_summary": {"worst_relative_wape_regression": 0.0, "slice_win_rate": 1.0},
        "repeat_stockout_ratio": 0.0,
        "repeat_stockout_improvement": 1.0,
        "closed_loop_cost_regression": 0.0,
        "closed_loop_lost_sales_improvement": 1.0,
        "closed_loop_slice_summary": {"worst_relative_cost_regression": 0.0, "worst_fill_rate_degradation": 0.0},
        "recovery_lower_bound_violations": 0,
        "recovery_do_no_harm_max": 0.0,
    }
    result = evaluate_release(metrics, asdict(ReleaseGateConfig()))
    assert result["checks"]["inventory_fill_rate"] is False
    assert result["gate_status"] == "ITERATE"


def test_package_version_is_single_sourced():
    import tomllib
    from pathlib import Path
    from inventory_ai import __version__

    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    assert "version" in pyproject["project"]["dynamic"]
    assert pyproject["tool"]["setuptools"]["dynamic"]["version"]["attr"] == "inventory_ai.__version__"
    assert __version__ == "0.6.2"


def test_cli_help_and_version_do_not_import_numerical_pipeline():
    import os
    import subprocess
    import sys
    from pathlib import Path

    # Pytest's ``pythonpath`` option only modifies the current interpreter.
    # A fresh subprocess must receive the source path explicitly so this
    # contract is valid before an editable install as well as after one.
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[2] / "src")

    version = subprocess.run(
        [sys.executable, "-m", "inventory_ai.cli", "--version"],
        check=True, capture_output=True, text=True, env=env,
    )
    assert version.stdout.strip() == "0.6.2"
    help_result = subprocess.run(
        [sys.executable, "-m", "inventory_ai.cli", "--help"],
        check=True, capture_output=True, text=True, env=env,
    )
    assert "--config" in help_result.stdout


def test_config_rejects_history_too_short_for_disjoint_folds(tmp_path):
    from inventory_ai.config import load_config
    import pytest

    path = tmp_path / "short.yaml"
    path.write_text(
        """
run_name: too_short
data:
  source: synthetic
  n_series: 4
  history_days: 70
model:
  horizon: 14
  validation_horizon: 14
""",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="disjoint calibration"):
        load_config(path)


def test_release_gate_rejects_excessive_interval_width():
    from dataclasses import asdict
    from inventory_ai.config import ReleaseGateConfig
    from inventory_ai.gates.release import evaluate_release

    metrics = {
        "truth_coverage_error": 0,
        "negative_forecasts": 0,
        "quantile_crossing": 0,
        "hierarchy_max_error": 0.0,
        "scenario_hierarchy_error": 0.0,
        "scenario_diagnostics": {"normalized_quantile_mae": 0.0},
        "test_forecast_metrics": {"production_candidate": {"coverage_80": 0.8}},
        "recovery_diagnostics": {"mae_improvement": 1.0, "absolute_bias_improvement": 1.0, "q95_upper_coverage": 1.0},
        "lifecycle_metrics": {"state_macro_f1": 1.0, "ladt_mae": 0.0, "ladt_spearman": 1.0},
        "candidate_wape_improvement": 0.1,
        "cost_win_rate": 1.0,
        "candidate_cost_regression": -0.1,
        "candidate_fill_rate_degradation": 0.0,
        "candidate_interval_width_ratio": 2.0,
        "forecast_slice_summary": {"worst_relative_wape_regression": 0.0, "slice_win_rate": 1.0},
        "repeat_stockout_ratio": 0.0,
        "repeat_stockout_improvement": 1.0,
        "closed_loop_cost_regression": 0.0,
        "closed_loop_lost_sales_improvement": 1.0,
        "closed_loop_slice_summary": {"worst_relative_cost_regression": 0.0, "worst_fill_rate_degradation": 0.0},
        "recovery_lower_bound_violations": 0,
        "recovery_do_no_harm_max": 0.0,
    }
    result = evaluate_release(metrics, asdict(ReleaseGateConfig()))
    assert result["checks"]["candidate_interval_sharpness"] is False
    assert result["gate_status"] == "ITERATE"


def test_archive_extraction_rejects_path_traversal_and_multiple_roots(tmp_path):
    import zipfile
    from scripts._archive_safety import safe_extract_zip

    traversal = tmp_path / "traversal.zip"
    with zipfile.ZipFile(traversal, "w") as archive:
        archive.writestr("repo/ok.txt", "ok")
        archive.writestr("../escape.txt", "bad")
    with pytest.raises(ValueError, match="unsafe archive member"):
        safe_extract_zip(traversal, tmp_path / "out1")
    assert not (tmp_path / "escape.txt").exists()

    multi_root = tmp_path / "multi.zip"
    with zipfile.ZipFile(multi_root, "w") as archive:
        archive.writestr("repo_a/a.txt", "a")
        archive.writestr("repo_b/b.txt", "b")
    with pytest.raises(ValueError, match="exactly one top-level root"):
        safe_extract_zip(multi_root, tmp_path / "out2")


def test_archive_extraction_accepts_single_clean_root(tmp_path):
    import zipfile
    from scripts._archive_safety import safe_extract_zip

    archive_path = tmp_path / "clean.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("repo/README.md", "hello")
        archive.writestr("repo/src/module.py", "VALUE = 1\n")
    root = safe_extract_zip(archive_path, tmp_path / "out")
    assert root.name == "repo"
    assert (root / "README.md").read_text(encoding="utf-8") == "hello"


def test_docker_build_context_includes_license_before_install():
    from pathlib import Path

    dockerfile = Path("Dockerfile").read_text(encoding="utf-8")
    copy_line = "COPY pyproject.toml README.md LICENSE ./"
    assert copy_line in dockerfile
    assert dockerfile.index(copy_line) < dockerfile.index("python -m pip install .")


def test_build_backend_supports_pep639_license_metadata():
    import re
    import tomllib
    from pathlib import Path

    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    setuptools_req = next(
        item for item in pyproject["build-system"]["requires"] if item.startswith("setuptools")
    )
    match = re.search(r">=([0-9]+)\.([0-9]+)\.([0-9]+)", setuptools_req)
    assert match is not None
    assert tuple(map(int, match.groups())) >= (77, 0, 3)
    assert pyproject["project"]["license"] == "MIT"
    assert "LICENSE" in pyproject["project"]["license-files"]


def test_balanced_panel_sample_preserves_latest_periods():
    from inventory_ai.pipeline import _balanced_panel_sample

    panel = make_synthetic_retail(SyntheticConfig(8, 200, 101))
    sample = _balanced_panel_sample(panel, max_rows=80)
    assert len(sample) <= 80
    original_bounds = panel.groupby("series_id")["date_idx"].agg(["min", "max"])
    sample_bounds = sample.groupby("series_id")["date_idx"].agg(["min", "max"])
    pd.testing.assert_series_equal(sample_bounds["min"], original_bounds["min"], check_names=False)
    pd.testing.assert_series_equal(sample_bounds["max"], original_bounds["max"], check_names=False)


def test_pipeline_failure_restores_previous_owned_outputs(tmp_path, monkeypatch):
    import inventory_ai.pipeline as pipeline_module
    from inventory_ai.config import DataConfig, ModelConfig, PipelineConfig

    cfg = PipelineConfig(
        run_name="rollback_case",
        output_dir="reports/rollback_case",
        artifact_dir="artifacts/rollback_case",
        data=DataConfig(source="synthetic", n_series=4, history_days=70),
        model=ModelConfig(horizon=7, validation_horizon=7, max_iter=2),
        closed_loop_cycles=4,
        lifecycle_series=4,
    )
    report = tmp_path / cfg.output_dir / "metrics_summary.json"
    manifest = tmp_path / cfg.artifact_dir / "run_manifest.json"
    sample = tmp_path / "data" / "processed" / cfg.run_name / "training_panel_sample.csv"
    for path, text in [(report, "old-report"), (manifest, "old-manifest"), (sample, "old-sample")]:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")

    def fail_after_reset(*args, **kwargs):
        raise RuntimeError("simulated pipeline failure")

    monkeypatch.setattr(pipeline_module, "_load_data", fail_after_reset)
    with pytest.raises(RuntimeError, match="simulated pipeline failure"):
        pipeline_module.run_pipeline(cfg, tmp_path)
    assert report.read_text(encoding="utf-8") == "old-report"
    assert manifest.read_text(encoding="utf-8") == "old-manifest"
    assert sample.read_text(encoding="utf-8") == "old-sample"


def test_package_filter_excludes_orphan_pipeline_backups():
    from pathlib import Path
    from scripts.package_release import include

    assert include(Path(".m5_smoke-backup-orphan/reports/metrics_summary.json")) is False


def test_ci_uses_least_privilege_checkout_credentials():
    from pathlib import Path

    workflow = Path(".github/workflows/ci.yml").read_text(encoding="utf-8")
    assert "permissions:\n  contents: read" in workflow
    assert "persist-credentials: false" in workflow


def test_package_filter_excludes_local_and_secret_residue():
    from pathlib import Path
    from scripts.package_release import include

    for path in [
        Path(".coverage"),
        Path(".coverage.worker-1"),
        Path(".env"),
        Path(".DS_Store"),
        Path(".vscode/settings.json"),
        Path(".idea/workspace.xml"),
        Path("notebooks/.ipynb_checkpoints/example.ipynb"),
        Path("qualification_manifest.json"),
        Path("release_bundle_manifest.json"),
    ]:
        assert include(path) is False
    assert include(Path(".gitignore")) is True
    assert include(Path(".github/workflows/ci.yml")) is True


def test_archive_extraction_rejects_duplicate_members(tmp_path):
    import zipfile
    import pytest
    from scripts._archive_safety import safe_extract_zip

    archive = tmp_path / "duplicate.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("repo/file.txt", "first")
        with pytest.warns(UserWarning, match="Duplicate name"):
            handle.writestr("repo/file.txt", "second")
    with pytest.raises(ValueError, match="duplicate archive member"):
        safe_extract_zip(archive, tmp_path / "out")


def test_package_filter_excludes_interrupted_write_tmp_residue():
    from pathlib import Path
    from scripts.package_release import include

    assert include(Path("reports/synthetic_smoke/inventory_simulation.csv.partial.tmp")) is False
    assert include(Path("artifacts/run_manifest.json.tmp")) is False


def test_cli_missing_config_returns_actionable_error_without_traceback(tmp_path):
    import os
    import subprocess
    import sys
    from pathlib import Path

    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[2] / "src")
    result = subprocess.run(
        [sys.executable, "-m", "inventory_ai.cli", "--config", "configs/does_not_exist.yaml"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        env=env,
    )
    assert result.returncode != 0
    assert "configuration file not found" in result.stderr
    assert "Traceback" not in result.stderr


def test_cli_relative_config_defaults_to_current_working_directory(tmp_path):
    import argparse
    from inventory_ai.cli import _resolve_root_and_config

    config = tmp_path / "configs" / "smoke.yaml"
    config.parent.mkdir(parents=True)
    config.write_text("run_name: smoke\n", encoding="utf-8")

    parser = argparse.ArgumentParser()
    root, resolved = _resolve_root_and_config(str(tmp_path), "configs/smoke.yaml", parser)
    assert root == tmp_path.resolve()
    assert resolved == config.resolve()


def test_runtime_smoke_exposes_library_function():
    from scripts.run_runtime_smoke import run_connected_smoke

    assert callable(run_connected_smoke)


def test_reset_owned_outputs_removes_interrupted_atomic_temps(tmp_path):
    from inventory_ai.pipeline import _reset_owned_outputs

    reports = tmp_path / "reports"
    artifacts = tmp_path / "artifacts"
    reports.mkdir()
    artifacts.mkdir()
    (reports / "forecast_output.csv.tmp").write_text("partial", encoding="utf-8")
    (artifacts / "run_manifest.json.tmp").write_text("partial", encoding="utf-8")
    (reports / "forecast_output.csv").write_text("old", encoding="utf-8")
    (artifacts / "run_manifest.json").write_text("old", encoding="utf-8")
    _reset_owned_outputs(reports, artifacts)
    assert not (reports / "forecast_output.csv").exists()
    assert not (reports / "forecast_output.csv.tmp").exists()
    assert not (artifacts / "run_manifest.json").exists()
    assert not (artifacts / "run_manifest.json.tmp").exists()
