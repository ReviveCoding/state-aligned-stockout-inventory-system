from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from inventory_ai.config import load_config
from inventory_ai.contracts import validate_forecast
from inventory_ai.evaluation.metrics import forecast_metrics
from inventory_ai.models.chronos_adapter import (
    Chronos2Forecaster,
    check_chronos_capability,
    prepare_chronos_finetune_inputs,
)
from inventory_ai.pipeline import _load_data, _prepare_panel
from inventory_ai.utils.io import write_csv, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Run optional Chronos-2 zero-shot or LoRA forecasting.")
    parser.add_argument("--config", default="configs/smoke.yaml")
    parser.add_argument("--mode", choices=["zero-shot", "lora"], default="zero-shot")
    parser.add_argument("--device-map", default="cuda")
    parser.add_argument("--num-steps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--output-dir", default="reports/chronos")
    args = parser.parse_args()

    capability = check_chronos_capability()
    if not capability.available:
        raise SystemExit(capability.message)
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    cfg = load_config(config_path)
    raw, source, fallback_reason = _load_data(cfg)
    panel = _prepare_panel(raw, cfg)
    origin = int(panel["date_idx"].max()) - cfg.model.horizon
    forecaster = Chronos2Forecaster(device_map=args.device_map)
    model_name = "chronos2_zero_shot"
    if args.mode == "lora":
        validation_cutoff = origin - cfg.model.horizon
        train_inputs = prepare_chronos_finetune_inputs(panel, validation_cutoff)
        validation_inputs = prepare_chronos_finetune_inputs(panel, origin)
        forecaster = forecaster.fit_lora(
            train_inputs,
            prediction_length=cfg.model.horizon,
            validation_inputs=validation_inputs,
            output_dir=ROOT / "artifacts" / "chronos_lora",
            num_steps=args.num_steps,
            batch_size=args.batch_size,
        )
        model_name = "chronos2_lora"
    forecast = forecaster.predict(panel, origin, cfg.model.horizon, model_name=model_name)
    truth = panel[(panel["date_idx"] > origin) & (panel["date_idx"] <= origin + cfg.model.horizon)]
    validate_forecast(forecast, truth)
    metrics = forecast_metrics(forecast, truth)
    output_dir = ROOT / args.output_dir
    write_csv(output_dir / "forecast.csv", forecast)
    write_json(
        output_dir / "metrics.json",
        {
            "mode": args.mode,
            "device_map": args.device_map,
            "data_source": source,
            "fallback_reason": fallback_reason,
            "capability": capability.__dict__,
            "metrics": metrics,
        },
    )
    print(json.dumps({"model": model_name, "metrics": metrics}, indent=2))


if __name__ == "__main__":
    main()
