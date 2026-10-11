"""The §7.4 decision layer: four detectors, each on its own budget.

Entry 10 measured what happens without per-detector budgets — recall doubled
but precision fell from 0.722 to 0.173, because combining four detectors with
a maximum and one threshold gives every reading four chances to clear one bar.
These tests pin the fix.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models.decision_layer import (
    SENSOR_FAULT, TRUSTWORTHY_BASELINE, TRUSTWORTHY_EVENT, DecisionLayer,
    low_variance_score, normalise,
)


def _frame(n: int = 600, seed: int = 0, with_spatial: bool = True) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq="h")
    data = {
        "residual": np.abs(rng.normal(0, 1, n)),
        "y_true": 30 + rng.normal(0, 8, n),
    }
    if with_spatial:
        data["spatial"] = np.abs(rng.normal(0, 1, n))
    return pd.DataFrame(data, index=idx)


def _labels(n: int = 600, rate: float = 0.05, seed: int = 1):
    rng = np.random.default_rng(seed)
    labels = (rng.random(n) < rate).astype(int)
    is_extreme = np.zeros(n, dtype=bool)
    is_extreme[rng.choice(n, size=n // 10, replace=False)] = True
    return labels, is_extreme


# ---------------------------------------------------------------------------
# Per-detector budgeting
# ---------------------------------------------------------------------------
def test_budget_is_split_across_detectors():
    """Four detectors sharing a 0.10 budget get 0.025 each, not 0.10 each."""
    frame = _frame()
    labels, extreme = _labels()
    layer = DecisionLayer().fit(frame, labels, extreme, target_fpr_extremes=0.10)
    assert len(layer._thresholds) == 4
    # Each threshold must be strict enough that its own FP rate is within budget.
    parts = layer._parts(frame)
    genuine_extreme = extreme & (labels == 0)
    for key, thr in layer._thresholds.items():
        std = normalise(parts[key], layer._scales[key])
        fired = np.nan_to_num(std, nan=-np.inf) > thr
        rate = float((fired & genuine_extreme).sum() / genuine_extreme.sum())
        assert rate <= 0.025 + 1e-9, f"{key} exceeded its share of the budget"


def test_combined_score_crosses_zero_at_the_operating_point():
    """The combined score is a margin, so 0 is the natural cut."""
    frame = _frame()
    labels, extreme = _labels()
    layer = DecisionLayer().fit(frame, labels, extreme)
    out = layer.score(frame)
    fired_any = np.vstack([
        normalise(v, layer._scales[k]) > layer._thresholds[k]
        for k, v in layer._parts(frame).items() if k in layer._thresholds
    ]).any(axis=0)
    assert np.array_equal(np.nan_to_num(out.combined, nan=-np.inf) > 0, fired_any)


def test_scoring_before_fitting_is_refused():
    """Thresholds come from validation; scoring test first would invert that."""
    with pytest.raises(RuntimeError, match="fit"):
        DecisionLayer().score(_frame())


def test_detector_attribution_is_kept():
    frame = _frame()
    labels, extreme = _labels()
    out = DecisionLayer().fit(frame, labels, extreme).score(frame)
    assert set(np.unique(out.which)) <= {
        "residual", "accumulated", "low_variance", "spatial"}


def test_missing_spatial_is_handled():
    """Rungs without a server GNN still run the other three detectors."""
    frame = _frame(with_spatial=False)
    labels, extreme = _labels()
    layer = DecisionLayer().fit(frame, labels, extreme)
    assert "spatial" not in layer._thresholds
    assert np.isfinite(layer.score(frame).combined).any()


# ---------------------------------------------------------------------------
# The stuck-at detector
# ---------------------------------------------------------------------------
def test_low_variance_fires_on_a_frozen_sensor():
    """Entry 7: stuck-at scores *below* normal on every residual-based score."""
    n = 400
    y = pd.Series(30 + np.random.default_rng(0).normal(0, 6, n))
    y.iloc[200:260] = 41.7                       # frozen
    score = low_variance_score(y, window=12)
    assert np.nanmedian(score[210:260]) > 5 * np.nanmedian(score[:190])


def test_a_quiet_but_working_sensor_scores_far_below_a_frozen_one():
    """A calm stretch must not look like a dead sensor.

    "Quiet" means small but non-zero variation. A literally constant stretch is
    a stuck sensor, which is precisely what this detector should flag.
    """
    rng = np.random.default_rng(1)
    quiet = 30.0 + rng.normal(0, 0.4, 200)      # low noise, still alive
    frozen = np.full(60, 25.0)                  # genuinely stuck
    score = low_variance_score(pd.Series(np.r_[quiet, frozen]), window=12)

    warm = score[12:]                           # rolling windows start as NaN
    assert np.isfinite(warm).all()
    assert np.nanmedian(score[220:]) > 10 * np.nanmedian(score[12:190])


# ---------------------------------------------------------------------------
# §7.4's three-way call
# ---------------------------------------------------------------------------
def test_high_agreement_means_real_event_not_fault():
    """The Gupchup (2019) control, made mechanical."""
    combined = np.array([-1.0, 2.0, 2.0])
    agreement = np.array([0.0, 0.9, 0.1])
    out = DecisionLayer.classify(combined, threshold=0.0, agreement=agreement,
                                 high_agreement=0.6)
    assert out[0] == TRUSTWORTHY_BASELINE
    assert out[1] == TRUSTWORTHY_EVENT, "neighbours agree -> a real episode"
    assert out[2] == SENSOR_FAULT, "station is alone -> a fault"


def test_without_neighbours_everything_flagged_is_called_a_fault():
    out = DecisionLayer.classify(np.array([-1.0, 5.0]), threshold=0.0)
    assert out.tolist() == [TRUSTWORTHY_BASELINE, SENSOR_FAULT]


def test_normalise_is_robust_to_outliers():
    """MAD scaling, so one 1000x spike cannot flatten everything else."""
    x = np.r_[np.random.default_rng(2).normal(0, 1, 500), [1000.0]]
    z = normalise(x)
    assert abs(float(np.median(z))) < 0.1
    assert z[-1] > 50
