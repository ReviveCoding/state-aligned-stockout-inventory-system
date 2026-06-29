from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class DataConfig:
    source: str = "synthetic"
    m5_zip_path: str = "/mnt/data/m5-forecasting-accuracy.zip"
    n_series: int = 24
    series_offset: int = 0
    history_days: int = 140
    controlled_stockout_rate: float = 0.12
    allow_fallback: bool = False


@dataclass(frozen=True)
class ModelConfig:
    horizon: int = 14
    quantiles: tuple[float, ...] = (0.1, 0.5, 0.9)
    max_train_rows: int = 60_000
    max_iter: int = 80
    validation_horizon: int = 14
    decision_cost_tolerance: float = 0.05
    router_max_wape_regression: float = 0.10
    selection_max_fill_rate_degradation: float = 0.0
    selection_max_interval_width_ratio: float = 1.35


@dataclass(frozen=True)
class InventoryConfigData:
    holding_cost: float = 1.0
    shortage_cost: float = 5.0
    order_cost: float = 0.1
    lead_time_days: int = 2
    review_period_days: int = 1
    service_level: float = 0.80
    initial_inventory_multiplier: float = 1.0
    order_capacity: float | None = None


@dataclass(frozen=True)
class ReleaseGateConfig:
    max_quantile_crossing: int = 0
    max_negative_forecasts: int = 0
    max_hierarchy_error: float = 1e-8
    max_scenario_hierarchy_error: float = 1e-8
    max_scenario_quantile_mae: float = 0.25
    min_candidate_coverage: float = 0.70
    max_candidate_coverage: float = 0.93
    min_recovery_mae_improvement: float = 0.05
    min_recovery_bias_improvement: float = 0.05
    min_recovery_q95_coverage: float = 0.75
    min_lifecycle_state_f1: float = 0.50
    max_ladt_mae: float = 0.16
    min_ladt_spearman: float = 0.75
    min_candidate_wape_improvement: float = -0.02
    min_cost_win_rate: float = 0.50
    max_candidate_cost_regression: float = 0.05
    max_candidate_fill_rate_degradation: float = 0.01
    max_candidate_interval_width_ratio: float = 1.35
    max_worst_slice_wape_regression: float = 0.20
    min_forecast_slice_win_rate: float = 0.35
    max_repeat_stockout_ratio: float = 0.75
    min_repeat_stockout_improvement: float = 0.0
    max_closed_loop_cost_regression: float = 0.10
    min_closed_loop_lost_sales_improvement: float = -0.02
    max_worst_closed_loop_cost_regression: float = 0.25
    max_worst_closed_loop_fill_degradation: float = 0.01
    max_truth_coverage_error: int = 0


@dataclass(frozen=True)
class PipelineConfig:
    seed: int = 42
    run_name: str = "smoke"
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    inventory: InventoryConfigData = field(default_factory=InventoryConfigData)
    release_gate: ReleaseGateConfig = field(default_factory=ReleaseGateConfig)
    closed_loop_cycles: int = 12
    lifecycle_series: int = 18
    output_dir: str = "reports"
    artifact_dir: str = "artifacts"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _section(cls, raw: dict[str, Any] | None):
    return cls(**(raw or {}))


def load_config(path: str | Path) -> PipelineConfig:
    config_path = Path(path)
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    cfg = PipelineConfig(
        seed=int(raw.get("seed", 42)),
        run_name=str(raw.get("run_name", config_path.stem)),
        data=_section(DataConfig, raw.get("data")),
        model=ModelConfig(**{**(raw.get("model") or {}), "quantiles": tuple((raw.get("model") or {}).get("quantiles", (0.1, 0.5, 0.9)))}),
        inventory=_section(InventoryConfigData, raw.get("inventory")),
        release_gate=_section(ReleaseGateConfig, raw.get("release_gate")),
        closed_loop_cycles=int(raw.get("closed_loop_cycles", 12)),
        lifecycle_series=int(raw.get("lifecycle_series", 18)),
        output_dir=str(raw.get("output_dir", "reports")),
        artifact_dir=str(raw.get("artifact_dir", "artifacts")),
    )
    validate_config(cfg)
    return cfg


def validate_config(cfg: PipelineConfig) -> None:
    if cfg.data.source not in {"synthetic", "m5", "auto"}:
        raise ValueError("data.source must be synthetic, m5, or auto")
    if cfg.data.n_series < 4:
        raise ValueError("data.n_series must be at least 4")
    if cfg.data.series_offset < 0:
        raise ValueError("data.series_offset must be nonnegative")
    minimum_history = 32 + cfg.model.horizon + 2 * cfg.model.validation_horizon
    if cfg.data.history_days < minimum_history:
        raise ValueError(
            "history_days is too short for disjoint calibration, validation, and test folds; "
            f"need at least {minimum_history} days"
        )
    if not 0 <= cfg.data.controlled_stockout_rate < 0.8:
        raise ValueError("controlled_stockout_rate must be in [0, 0.8)")
    if cfg.model.horizon < 2:
        raise ValueError("model.horizon must be at least 2")
    if tuple(sorted(cfg.model.quantiles)) != cfg.model.quantiles:
        raise ValueError("quantiles must be sorted")
    if cfg.model.quantiles != (0.1, 0.5, 0.9):
        raise ValueError("core pipeline currently requires quantiles (0.1, 0.5, 0.9)")
    if cfg.inventory.lead_time_days < 0 or cfg.inventory.review_period_days < 1:
        raise ValueError("lead time must be nonnegative and review period positive")
    if not 0.0 <= cfg.inventory.service_level <= 1.0:
        raise ValueError("inventory.service_level must be in [0, 1]")
    if min(cfg.inventory.holding_cost, cfg.inventory.shortage_cost, cfg.inventory.order_cost) < 0:
        raise ValueError("inventory costs must be nonnegative")
    if cfg.inventory.initial_inventory_multiplier < 0:
        raise ValueError("initial_inventory_multiplier must be nonnegative")
    if cfg.inventory.order_capacity is not None and cfg.inventory.order_capacity <= 0:
        raise ValueError("order_capacity must be positive when configured")
    if cfg.model.max_train_rows < 100 or cfg.model.max_iter < 1:
        raise ValueError("model training budget is invalid")
    if cfg.model.validation_horizon < cfg.model.horizon:
        raise ValueError("validation_horizon must be at least model.horizon")
    if not 0.0 <= cfg.model.decision_cost_tolerance <= 0.25:
        raise ValueError("model.decision_cost_tolerance must be in [0, 0.25]")
    if not 0.0 <= cfg.model.router_max_wape_regression <= 0.50:
        raise ValueError("model.router_max_wape_regression must be in [0, 0.50]")
    if not 0.0 <= cfg.model.selection_max_fill_rate_degradation <= 0.05:
        raise ValueError("model.selection_max_fill_rate_degradation must be in [0, 0.05]")
    if not 1.0 <= cfg.model.selection_max_interval_width_ratio <= 2.0:
        raise ValueError("model.selection_max_interval_width_ratio must be in [1.0, 2.0]")
    if not cfg.run_name or any(char in cfg.run_name for char in ("/", "\\")):
        raise ValueError("run_name must be a simple non-empty name")
    for label, value in (("output_dir", cfg.output_dir), ("artifact_dir", cfg.artifact_dir)):
        path = Path(value)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError(f"{label} must be a relative path inside the repository")
