"""Ground-truth labelling, the unit trap, and the no-pooling rule.

These cover CLAUDE.md §2.1, §2.7, §3.1, §5.5, §8.2 and §8.3.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.harmonize import (
    FAULT_VALIDITY_CODES, VALID_VALIDITY_CODES, _canonicalise_units,
    label_from_validity, real_fault_rate,
)
from src.evaluation.metrics import anomaly_metrics, extreme_mask, forecast_metrics
from src.evaluation.report import PoolingError, anomaly_table, forecast_table
from src.preprocessing.fault_labels import (
    SYNTHETIC_LABEL_COL_SUFFIX, inject_faults, measured_rate_from,
)
from src.preprocessing.indexing import IMPUTED_SUFFIX, LABEL_SUFFIX

TARGET = "NO2"


# ---------------------------------------------------------------------------
# §3.1 — the Validity vocabulary
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("code", FAULT_VALIDITY_CODES)
def test_negative_validity_is_a_fault(code):
    assert label_from_validity(code) == "fault"


@pytest.mark.parametrize("code", VALID_VALIDITY_CODES)
def test_documented_valid_codes_are_trustworthy(code):
    assert label_from_validity(code) == "trustworthy"


def test_unit_mismatch_raises_rather_than_coercing():
    """§3.1's CO trap: CO is mg/m3, everything else ug/m3."""
    df = pd.DataFrame({"pollutant": ["CO"], "unit": ["ug/m3"]})
    with pytest.raises(ValueError, match="unit mismatch"):
        _canonicalise_units(df)


def test_known_unit_spellings_are_normalised():
    df = pd.DataFrame({"pollutant": ["NO2", "PM10", "CO"],
                       "unit": ["µg/m3", "ug.m-3", "mg.m-3"]})
    assert _canonicalise_units(df)["unit"].tolist() == ["ug/m3", "ug/m3", "mg/m3"]


def test_real_fault_rate_refuses_to_guess_without_real_rows():
    df = pd.DataFrame({"label_source": ["synthetic"], "label": ["fault"]})
    with pytest.raises(ValueError, match="label_source == 'real'"):
        real_fault_rate(df)


# ---------------------------------------------------------------------------
# §2.7 / §5.5 — injection is rate-matched, and never overwrites real labels
# ---------------------------------------------------------------------------
def _injectable_frame(n: int = 4000) -> pd.DataFrame:
    idx = pd.date_range("2020-01-01", periods=n, freq="h", tz="UTC")
    rng = np.random.default_rng(1)
    frame = pd.DataFrame({
        TARGET: 40 + rng.normal(0, 8, n),
        TARGET + LABEL_SUFFIX: "trustworthy",
        TARGET + IMPUTED_SUFFIX: np.zeros(n, dtype=bool),
    }, index=idx)
    frame.loc[frame.index[:50], TARGET + LABEL_SUFFIX] = "fault"   # real faults
    frame.attrs["station_id"] = "TEST01"
    return frame


def test_injection_has_no_default_rate():
    """An arbitrary default lets the model learn 'region -> fault likelihood'."""
    with pytest.raises(TypeError):
        inject_faults(_injectable_frame(), TARGET)      # type: ignore[call-arg]


@pytest.mark.parametrize("bad", [0.0, 1.0, -0.1, 1.5])
def test_injection_rejects_impossible_rates(bad):
    with pytest.raises(ValueError, match="measured real fault rate"):
        inject_faults(_injectable_frame(), TARGET, target_rate=bad)


def test_injection_matches_the_measured_rate():
    frame = _injectable_frame()
    target_rate = 0.028                       # the measured EEA rate
    out, rep = inject_faults(frame, TARGET, target_rate=target_rate, seed=7)

    assert rep.achieved_rate == pytest.approx(target_rate, abs=0.005)
    assert rep.target_rate == pytest.approx(target_rate)


def test_injection_never_touches_real_faults_or_imputed_points():
    frame = _injectable_frame()
    frame.loc[frame.index[100:200], TARGET + IMPUTED_SUFFIX] = True
    before = frame[TARGET].to_numpy().copy()

    out, _ = inject_faults(frame, TARGET, target_rate=0.05, seed=3)
    injected = (out[TARGET + SYNTHETIC_LABEL_COL_SUFFIX] == "fault").to_numpy()

    assert not injected[:50].any(), "injected on top of a real fault"
    assert not injected[100:200].any(), "injected into an imputed stretch"
    # Real labels survive untouched, in their own column.
    assert (out[TARGET + LABEL_SUFFIX] == frame[TARGET + LABEL_SUFFIX]).all()
    np.testing.assert_array_equal(before[~injected], out[TARGET].to_numpy()[~injected])


def test_measured_rate_from_refuses_an_empty_series():
    with pytest.raises(ValueError, match="arbitrary default"):
        measured_rate_from(pd.Series([], dtype=object))


# ---------------------------------------------------------------------------
# §8.2 — the false-positive control is mandatory
# ---------------------------------------------------------------------------
def test_anomaly_metrics_requires_the_extreme_mask():
    with pytest.raises(TypeError):
        anomaly_metrics(np.array([1, 0]), np.array([1.0, 0.0]), 0.5)  # type: ignore[call-arg]


def test_detector_that_flags_every_high_reading_is_exposed():
    """Perfect recall, and fpr_on_extremes shows why that is worthless.

    This is the Gupchup (2019) failure mode in miniature.
    """
    n = 200
    label = np.zeros(n, dtype=int)
    label[:10] = 1                            # 10 genuine faults
    score = np.zeros(n)
    score[:10] = 10.0                         # faults score high
    is_extreme = np.zeros(n, dtype=bool)
    is_extreme[10:40] = True                  # 30 genuine high readings
    score[10:40] = 10.0                       # ...which the detector also flags

    m = anomaly_metrics(label, score, threshold=1.0, is_extreme=is_extreme)

    assert m.recall == pytest.approx(1.0)
    assert m.fpr_on_extremes == pytest.approx(1.0), \
        "every genuine extreme was called a fault and the control must say so"
    assert m.precision == pytest.approx(10 / 40)


def test_unknown_labels_are_dropped_not_counted_as_trustworthy():
    label = np.array([1, 0, -1, -1])
    m = anomaly_metrics(label, np.array([9.0, 0.0, 9.0, 0.0]), threshold=1.0,
                        is_extreme=np.zeros(4, dtype=bool))
    assert m.n == 2


def test_mape_excludes_near_zero_denominators():
    m = forecast_metrics(np.array([0.1, 50.0]), np.array([0.2, 55.0]))
    assert m.mape_n_excluded == 1
    assert m.mape == pytest.approx(10.0)


def test_extreme_mask_uses_the_reference_not_the_test_set():
    train = np.arange(100, dtype=float)
    test = np.array([50.0, 200.0])
    mask = extreme_mask(test, quantile=0.95, reference=train)
    assert mask.tolist() == [False, True]


# ---------------------------------------------------------------------------
# §8.3 — pooling is structurally unreachable
# ---------------------------------------------------------------------------
def _records(n: int = 40) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "model": "1_persistence",
        "region": "catalonia",
        "station_id": np.where(np.arange(n) % 2 == 0, "ES0001A", "ES0002A"),
        "station_class": "urban_background",
        "label_source": "real",
        # repeat, not tile: station_id alternates on index parity, so tiling the
        # horizon the same way would make the two perfectly correlated and
        # collapse the grouping to 2 cells instead of 4.
        "horizon": np.repeat([1, 3], n // 2),
        "t_target": pd.date_range("2024-01-01", periods=n, freq="h"),
        "y_true": rng.normal(40, 5, n),
        "y_pred": rng.normal(40, 5, n),
        "is_imputed": False,
        "label": np.where(np.arange(n) % 10 == 0, "fault", "trustworthy"),
        "is_extreme": False,
        "score": rng.random(n),
    })


@pytest.mark.parametrize("dropped", ["region", "station_class", "label_source"])
def test_missing_group_key_is_refused(dropped):
    with pytest.raises(PoolingError, match="required group key"):
        forecast_table(_records().drop(columns=[dropped]))


def test_forecast_table_is_grouped_per_station_and_horizon():
    out = forecast_table(_records())
    assert set(out["station_id"]) == {"ES0001A", "ES0002A"}
    assert {"region", "station_class", "label_source"} <= set(out.columns)
    # One row per (station, horizon) — never one row for the whole region.
    assert len(out) == 4


def test_anomaly_table_requires_a_per_model_station_threshold():
    """A shared threshold is the §5.8 miscalibration failure mode."""
    with pytest.raises(PoolingError, match="no calibrated threshold"):
        anomaly_table(_records(), thresholds={("1_persistence", "ES0001A"): 0.5})
