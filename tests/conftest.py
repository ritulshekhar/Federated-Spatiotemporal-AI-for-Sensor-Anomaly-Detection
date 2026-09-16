"""Shared fixtures.

Everything here is deliberately tiny and hand-checkable. These tests guard the
LOCKED decisions in CLAUDE.md §2, so a fixture whose correct answer is not
obvious by inspection would defeat the purpose.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.preprocessing.indexing import (
    IMPUTED_SUFFIX, LABEL_SUFFIX, VERIFICATION_SUFFIX,
)

TARGET = "NO2"


@pytest.fixture
def hourly_index() -> pd.DatetimeIndex:
    """1000 contiguous UTC hours."""
    return pd.date_range("2020-01-01", periods=1000, freq="h", tz="UTC",
                         name="timestamp_utc")


@pytest.fixture
def station_frame(hourly_index) -> pd.DataFrame:
    """A minimal post-§6.4 station frame: target, one feature, and metadata."""
    n = len(hourly_index)
    rng = np.random.default_rng(0)
    frame = pd.DataFrame(
        {
            TARGET: 40 + 10 * np.sin(np.arange(n) * 2 * np.pi / 24) + rng.normal(0, 2, n),
            "cyc_hour_sin": np.sin(hourly_index.hour.to_numpy() * 2 * np.pi / 24),
            TARGET + IMPUTED_SUFFIX: np.zeros(n, dtype=bool),
            TARGET + LABEL_SUFFIX: np.where(np.arange(n) % 97 == 0,
                                            "fault", "trustworthy"),
            TARGET + VERIFICATION_SUFFIX: np.ones(n, dtype=np.int8),
        },
        index=hourly_index,
    )
    frame.attrs.update(
        station_id="TEST01", region="testland", station_class="urban_background",
        label_source="real", pollutants=[TARGET],
    )
    return frame


@pytest.fixture
def feature_columns() -> list[str]:
    return [TARGET, "cyc_hour_sin"]
