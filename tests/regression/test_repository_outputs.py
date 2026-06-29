from __future__ import annotations

import json
from pathlib import Path


def test_checked_in_release_metrics_are_structurally_valid():
    paths = sorted(Path("reports").glob("*/metrics_summary.json"))
    legacy = Path("reports/metrics_summary.json")
    if legacy.exists():
        paths.append(legacy)
    for path in paths:
        metrics = json.loads(path.read_text(encoding="utf-8"))
        assert metrics["truth_coverage_error"] == 0, path
        assert metrics["negative_forecasts"] == 0, path
        assert metrics["quantile_crossing"] == 0, path
        assert metrics["hierarchy_max_error"] <= 1e-8, path
