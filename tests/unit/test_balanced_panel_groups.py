import numpy as np
import pandas as pd

from inventory_ai.pipeline import _balanced_panel_sample


def test_balanced_sample_keeps_first_and_last_per_series_and_cap():
    frame = pd.DataFrame(
        {
            "series_id": np.repeat(["z", "a", "b"], 20),
            "date_idx": np.tile(np.arange(20), 3),
            "value": np.arange(60),
        }
    )
    result = _balanced_panel_sample(frame.sample(frac=1, random_state=5), max_rows=12)
    assert len(result) == 12
    for _, group in result.groupby("series_id"):
        assert group["date_idx"].tolist() == [0, 6, 13, 19]
    pd.testing.assert_frame_equal(
        result, result.sort_values(["series_id", "date_idx"]).reset_index(drop=True)
    )
