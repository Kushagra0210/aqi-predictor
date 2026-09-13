from pathlib import Path

import pandas as pd
import pytest

from aqi_core import DatasetError, aqi_category, load_dataset, validate_dataset


def sample_frame(rows: int = 120) -> pd.DataFrame:
    return pd.DataFrame({"Date": pd.date_range("2020-01-01", periods=rows), "City": ["Delhi", "Mumbai"] * (rows // 2), "AQI": range(50, 50 + rows), "PM2.5": [40.0] * rows, "PM10": [80.0] * rows, "NO2": [25.0] * rows, "CO": [1.0] * rows, "O3": [30.0] * rows})


def test_validate_dataset_rejects_missing_columns():
    with pytest.raises(DatasetError, match="missing required columns"):
        validate_dataset(pd.DataFrame({"AQI": [10]}))


def test_local_dataset_loading(tmp_path: Path):
    path = tmp_path / "sample.csv"
    sample_frame().to_csv(path, index=False)
    data, source = load_dataset(path, allow_download=False)
    assert source == "bundled snapshot"
    assert len(data) == 120


@pytest.mark.parametrize("value,expected", [(25, "Good"), (75, "Satisfactory"), (150, "Moderate"), (250, "Poor"), (350, "Very poor"), (450, "Severe")])
def test_categories(value, expected):
    assert aqi_category(value)[0] == expected
