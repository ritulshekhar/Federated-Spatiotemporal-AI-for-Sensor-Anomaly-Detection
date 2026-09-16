"""The §6 invariants that are expensive to violate silently.

Each test here corresponds to a specific failure mode CLAUDE.md calls out, not
to a function's happy path.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.preprocessing.imputation import impute_frame, imputed_mask
from src.preprocessing.indexing import IMPUTED_SUFFIX, build_station_frame
from src.preprocessing.normalize import (
    NEVER_TRANSMIT, contains_normalization_statistics, fit_local_scaler,
)
from src.preprocessing.windowing import compute_split_boundaries, make_windows

TARGET = "NO2"


# ---------------------------------------------------------------------------
# §6.2 — the mask is built BEFORE filling
# ---------------------------------------------------------------------------
def test_is_imputed_marks_exactly_the_missing_points(hourly_index):
    """The detector must never be able to flag its own interpolation."""
    n = len(hourly_index)
    values = np.arange(n, dtype=float)
    gap = slice(100, 103)              # a 3h gap -> linear interpolation
    values[gap] = np.nan
    frame = pd.DataFrame({TARGET: values}, index=hourly_index)
    frame.attrs["station_id"] = "TEST01"

    out, _ = impute_frame(frame, [TARGET], short_gap_hours=3)
    mask = imputed_mask(out, TARGET).to_numpy()

    assert mask[gap].all(), "the gap must be marked imputed"
    assert mask.sum() == 3, "nothing outside the gap may be marked imputed"
    assert out[TARGET].notna().all(), "the gap should have been filled"
    # And the fill is the linear one, not a copy of a neighbour.
    np.testing.assert_allclose(out[TARGET].to_numpy()[gap], [100.0, 101.0, 102.0])


def test_long_gaps_are_left_nan_not_invented(hourly_index):
    """Beyond max_fill_hours the honest answer is 'no data', not a guess."""
    values = np.arange(len(hourly_index), dtype=float)
    values[200:500] = np.nan           # 300h, beyond a 168h limit
    frame = pd.DataFrame({TARGET: values}, index=hourly_index)
    frame.attrs["station_id"] = "TEST01"

    out, reports = impute_frame(frame, [TARGET], short_gap_hours=3,
                                max_fill_hours=168)

    assert out[TARGET].iloc[200:500].isna().all()
    assert reports[0].n_left_nan == 300
    # A point that was never filled is not "imputed".
    assert not out[TARGET + IMPUTED_SUFFIX].iloc[200:500].any()


def test_imputed_mask_raises_if_imputation_never_ran(station_frame):
    frame = station_frame.drop(columns=[TARGET + IMPUTED_SUFFIX])
    with pytest.raises(KeyError, match="§6.2"):
        imputed_mask(frame, TARGET)


# ---------------------------------------------------------------------------
# §6.6 — chronological split, and no window straddles a boundary
# ---------------------------------------------------------------------------
def test_splits_are_chronological_and_disjoint(station_frame, feature_columns):
    b = compute_split_boundaries(station_frame.index)
    windows = make_windows(station_frame, feature_columns, TARGET, b,
                           raw_target=station_frame[TARGET],
                           input_hours=24, horizon_hours=6)

    train, val, test = windows["train"], windows["val"], windows["test"]
    assert len(train) and len(val) and len(test)

    # Every training target precedes every validation target, and so on. A
    # shuffled split would break this immediately.
    assert train.t_target.max() < val.t_target.min()
    assert val.t_target.max() < test.t_target.min()


def test_no_window_spans_a_split_boundary(station_frame, feature_columns):
    """Assigning by target timestamp alone is not enough (see windowing.py).

    A window targeting the first validation hour still *reads* 24 hours of
    training data. Those windows must be dropped, not reassigned.
    """
    b = compute_split_boundaries(station_frame.index)
    windows = make_windows(station_frame, feature_columns, TARGET, b,
                           raw_target=station_frame[TARGET],
                           input_hours=24, horizon_hours=6)

    train_end = b.train_end.tz_convert("UTC").tz_localize(None).to_datetime64()
    val_end = b.val_end.tz_convert("UTC").tz_localize(None).to_datetime64()

    # The full span of a window is [t0 - 23h, last target hour].
    span_start = windows["train"].t0 - np.timedelta64(23, "h")
    assert (windows["train"].t_target.max(axis=1) <= train_end).all()

    val_span_start = windows["val"].t0 - np.timedelta64(23, "h")
    assert (val_span_start > train_end).all(), \
        "a validation window reads training hours"
    assert (windows["val"].t_target.max(axis=1) <= val_end).all()

    test_span_start = windows["test"].t0 - np.timedelta64(23, "h")
    assert (test_span_start > val_end).all(), \
        "a test window reads validation hours"
    assert span_start.size == len(windows["train"])


def test_split_fractions_must_sum_to_one(hourly_index):
    with pytest.raises(ValueError, match="sum to 1.0"):
        compute_split_boundaries(hourly_index, train=0.8, val=0.15, test=0.15)


# ---------------------------------------------------------------------------
# §6.5 / §2.2 — the scaler sees the training window only, and never leaves
# ---------------------------------------------------------------------------
def test_scaler_is_fitted_on_the_train_window_only(station_frame, feature_columns):
    b = compute_split_boundaries(station_frame.index)
    scaler = fit_local_scaler(station_frame, feature_columns, b.train_end, "TEST01")

    train_slice = station_frame.loc[station_frame.index <= b.train_end]
    assert scaler.n_fit_rows == len(train_slice)
    # The fitted mean must equal the TRAIN mean, not the whole-series mean.
    assert scaler.mean_[TARGET] == pytest.approx(float(train_slice[TARGET].mean()))
    whole = float(station_frame[TARGET].mean())
    assert scaler.mean_[TARGET] != pytest.approx(whole, abs=1e-12) or \
        len(train_slice) == len(station_frame)


def test_constant_column_does_not_explode(hourly_index):
    """std ~ 0 would amplify float noise into huge spurious z-scores."""
    frame = pd.DataFrame({TARGET: np.ones(len(hourly_index))}, index=hourly_index)
    scaler = fit_local_scaler(frame, [TARGET], hourly_index[-1], "TEST01")
    assert scaler.std_[TARGET] == 1.0
    assert np.isfinite(scaler.transform(frame)[TARGET]).all()


def test_scaler_is_recognised_as_untransmittable(station_frame, feature_columns):
    """The federated privacy guard must catch a scaler in any payload shape."""
    b = compute_split_boundaries(station_frame.index)
    scaler = fit_local_scaler(station_frame, feature_columns, b.train_end, "TEST01")

    assert contains_normalization_statistics(scaler)
    assert contains_normalization_statistics({"weights": [1, 2], "scaler": scaler})
    assert contains_normalization_statistics({"mean": {"NO2": 1.0}, "std": {"NO2": 2.0}})
    assert contains_normalization_statistics([NEVER_TRANSMIT])
    # A legitimate payload must not trip it.
    assert not contains_normalization_statistics(
        {"gru_weight_update": [0.1, 0.2], "encoder_hidden_state": [0.3], "n_samples": 10}
    )


# ---------------------------------------------------------------------------
# §6.1 — reindexing makes gaps visible
# ---------------------------------------------------------------------------
def test_missing_rows_become_explicit_gaps():
    """A station whose file simply lacks 03:00 looks continuous until reindexed."""
    times = pd.to_datetime(
        ["2020-01-01T00:00Z", "2020-01-01T01:00Z", "2020-01-01T04:00Z"], utc=True
    )
    long_df = pd.DataFrame({
        "station_id": "TEST01", "region": "testland", "station_class": "urban_background",
        "label_source": "real", "timestamp_utc": times, "pollutant": TARGET,
        "concentration": [10.0, 11.0, 14.0], "unit": "ug/m3",
        "label": "trustworthy", "validity": 1, "verification": 1,
    })
    frame, reports = build_station_frame(long_df, "TEST01", [TARGET])

    assert len(frame) == 5                      # 00,01,02,03,04
    assert frame[TARGET].isna().sum() == 2      # 02 and 03 are now visible
    assert reports[0].longest_gap_hours == 2
    assert reports[0].n_observed == 3
