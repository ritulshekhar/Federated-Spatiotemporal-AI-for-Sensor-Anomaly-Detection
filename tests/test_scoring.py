"""The detector's scoring and threshold logic.

These guard the measures added in response to the §8.2 failure measured in
CONTEXT.md entry 5: a plain residual threshold called up to 66.7% of genuine
high-pollution readings sensor faults.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.evaluation.metrics import (
    calibrate_threshold, residual_score, volatility_scaled_score,
)


# ---------------------------------------------------------------------------
# Volatility-scaled score
# ---------------------------------------------------------------------------
def test_volatility_scaling_separates_an_event_from_a_fault():
    """The point of the whole change, in two readings.

    Both miss by the same 40 ug/m3. One lands during a turbulent stretch (a
    building pollution episode), the other on a calm one (a sensor glitch). A
    plain residual cannot tell them apart; dividing by recent volatility can.
    """
    y_true = np.array([100.0, 100.0])
    y_pred = np.array([60.0, 60.0])
    volatility = np.array([40.0, 2.0])      # turbulent vs calm recent past

    plain = residual_score(y_true, y_pred, scale=20.0)
    scaled = volatility_scaled_score(y_true, y_pred, volatility, floor=1.0)

    assert plain[0] == pytest.approx(plain[1]), "plain residual cannot separate them"
    assert scaled[1] > 10 * scaled[0], "volatility scaling must separate them"


def test_volatility_floor_prevents_division_blowup():
    scaled = volatility_scaled_score(
        np.array([10.0]), np.array([10.5]), np.array([0.0]), floor=2.0
    )
    assert np.isfinite(scaled).all()
    assert scaled[0] == pytest.approx(0.25)


# ---------------------------------------------------------------------------
# Threshold calibration
# ---------------------------------------------------------------------------
def _val_scores(n: int = 1000, fault_rate: float = 0.05, seed: int = 0):
    rng = np.random.default_rng(seed)
    labels = (rng.random(n) < fault_rate).astype(int)
    score = rng.normal(1.0, 0.3, n)
    score[labels == 1] = rng.normal(4.0, 0.5, int(labels.sum()))
    return score, labels


def test_f1_sweep_beats_a_fixed_quantile_when_the_rate_is_mismatched():
    """A 0.98 quantile assumes 2% faults. Here there are 5%, so it under-flags."""
    score, labels = _val_scores(fault_rate=0.05)

    t_quantile = calibrate_threshold(score, method="quantile", quantile=0.98)
    t_sweep = calibrate_threshold(score, method="val_f1_sweep", labels=labels)

    def f1_at(t):
        pred = score > t
        tp = int((pred & (labels == 1)).sum())
        if tp == 0:
            return 0.0
        p, r = tp / int(pred.sum()), tp / int((labels == 1).sum())
        return 2 * p * r / (p + r)

    assert f1_at(t_sweep) >= f1_at(t_quantile)
    assert t_sweep < t_quantile, "the sweep should flag more at a 5% fault rate"


def test_fpr_extremes_method_respects_its_budget():
    """The threshold must hold false positives on genuine extremes under target."""
    rng = np.random.default_rng(1)
    n = 2000
    labels = np.zeros(n, dtype=int)
    labels[:100] = 1
    score = rng.normal(1.0, 0.3, n)
    score[:100] = rng.normal(3.0, 0.5, 100)
    is_extreme = np.zeros(n, dtype=bool)
    is_extreme[100:400] = True
    score[100:400] = rng.normal(2.0, 0.5, 300)   # real events score high too

    t = calibrate_threshold(score, method="val_fpr_extremes", labels=labels,
                            is_extreme=is_extreme, target_fpr_extremes=0.10)

    genuine_extreme = is_extreme & (labels == 0)
    achieved = float(((score > t) & genuine_extreme).sum() / genuine_extreme.sum())
    assert achieved <= 0.10 + 1e-9


def test_label_requiring_methods_refuse_to_run_blind():
    score, _ = _val_scores()
    with pytest.raises(ValueError, match="needs validation labels"):
        calibrate_threshold(score, method="val_f1_sweep")


def test_sweep_falls_back_when_validation_has_no_faults():
    score = np.random.default_rng(2).normal(1.0, 0.3, 500)
    labels = np.zeros(500, dtype=int)
    t = calibrate_threshold(score, method="val_f1_sweep", labels=labels,
                            quantile=0.98)
    assert np.isfinite(t)
    assert t == pytest.approx(float(np.quantile(score, 0.98)))


def test_unknown_labels_are_excluded_from_calibration():
    score = np.concatenate([np.ones(100), np.full(100, 99.0)])
    labels = np.concatenate([np.zeros(100, dtype=int), np.full(100, -1)])
    t = calibrate_threshold(score, method="val_f1_sweep", labels=labels,
                            quantile=0.98)
    assert t < 99.0, "unlabelled rows must not drag the threshold"
