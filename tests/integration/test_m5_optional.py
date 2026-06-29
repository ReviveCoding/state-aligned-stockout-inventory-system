from __future__ import annotations

import os
from pathlib import Path

import pytest

from inventory_ai.contracts import validate_panel
from inventory_ai.data.m5 import load_m5_sample


@pytest.mark.optional
def test_m5_adapter_when_explicitly_enabled():
    if os.environ.get("RUN_M5_TESTS") != "1":
        pytest.skip("set RUN_M5_TESTS=1 to run the local M5 adapter test")
    path = Path("/mnt/data/m5-forecasting-accuracy.zip")
    if not path.exists():
        pytest.skip("M5 archive is not available")
    frame = load_m5_sample(path, n_series=8, history_days=70)
    result = validate_panel(frame)
    assert result.series == 8
    assert frame.groupby("series_id")["demand"].sum().gt(0).all()
