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
    SYNTHETIC_LABEL_COL_SUFFIX, _apply_fault, inject_faults, measured_rate_from,
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


@pytest.mark.parametrize("bad", [(0.0, 1.0), (-1.0, 1.0), (2.0, 0.5)])
def test_injection_rejects_impossible_drift_rates(bad):
    with pytest.raises(ValueError, match="sigma per 72 hours"):
        inject_faults(_injectable_frame(), TARGET, target_rate=0.05,
                      drift_sigma_per_72h=bad)


def test_drift_is_injected_at_a_rate_not_a_fixed_total():
    """Drift magnitude must scale with episode length, and stay in its range.

    The original injector drew a fixed 0.8-2.5 sigma total regardless of
    duration, which made a long episode drift *more slowly* than a short one
    and put every episode far above the ~0.01 sigma/72h that CONTEXT.md entry
    17 measured on working sensors.
    """
    rate_hi = 0.4
    frame = _injectable_frame(n=20_000)
    out, rep = inject_faults(frame, TARGET, target_rate=0.08, seed=11,
                             drift_sigma_per_72h=(0.2, rate_hi))

    assert (rep.drift_sigma_lo, rep.drift_sigma_hi) == (0.2, rate_hi)

    kind = out[TARGET + "__synthetic_fault_type"].to_numpy()
    scale = float(np.nanstd(frame[TARGET].to_numpy()))
    delta = out[TARGET].to_numpy() - frame[TARGET].to_numpy()

    # Episodes are contiguous runs of the same subtype.
    is_drift = kind == "drift"
    assert is_drift.any(), "no drift episode was injected"
    edges = np.flatnonzero(np.diff(np.r_[False, is_drift, False].astype(int)))
    for start, stop in zip(edges[0::2], edges[1::2]):
        hours = stop - start
        observed = abs(delta[stop - 1]) / scale            # sigma, end of ramp
        assert observed <= rate_hi * (hours / 72.0) + 1e-6, (
            f"a {hours}h drift episode moved {observed:.3f} sigma, faster than "
            f"the configured ceiling of {rate_hi} sigma/72h"
        )


def test_drift_rates_are_spread_log_uniformly_over_the_range():
    """The slow decade must actually be sampled.

    Uniform sampling over [0.01, 2.0] would put half the episodes above
    1 sigma/72h and essentially none near the measured real rate, leaving the
    per-magnitude recall table with nothing to say about the band the detector
    cannot currently reach.
    """
    rng = np.random.default_rng(0)
    scale, hours = 10.0, 72
    seg = np.full(hours, 50.0)
    full, eligible = np.full(4000, 50.0), np.ones(4000, dtype=bool)
    full[:2] = [50.0 - scale, 50.0 + scale]      # give nanstd something to see

    rates = []
    for _ in range(400):
        out = _apply_fault("drift", seg, full, eligible, rng,
                           drift_sigma_per_72h=(0.01, 2.0))
        rates.append(abs(out[-1] - out[0]) / float(np.nanstd(full[eligible])))

    rates = np.array(rates)
    assert rates.min() >= 0.01 * 0.9 and rates.max() <= 2.0 * 1.1
    # Log-uniform over two decades puts ~half the mass below the geometric
    # mean (0.141); uniform would put ~7% there.
    below_geometric_mean = float((rates < np.sqrt(0.01 * 2.0)).mean())
    assert 0.35 < below_geometric_mean < 0.65, (
        f"{below_geometric_mean:.2f} of drift rates fell in the slow decade — "
        "the rate does not look log-uniform"
    )


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
