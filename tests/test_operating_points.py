"""Magnitude stratification and the two-tier operating points."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.evaluation.magnitude import BIN_LABELS, stratified_recall
from src.evaluation.operating_points import detector_columns, score_at_budget


def _records(n: int = 600, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    half = n // 2
    return pd.DataFrame({
        "model": "6_proposed",
        "region": "catalonia",
        "station_id": np.where(np.arange(n) % 2 == 0, "ES0001A", "ES0002A"),
        "station_class": "urban_background",
        "label_source": "synthetic",
        "split": np.r_[np.repeat("val", half), np.repeat("test", n - half)],
        "horizon": 1,
        "t_target": pd.date_range("2024-01-01", periods=n, freq="h"),
        "y_true": rng.normal(30, 8, n),
        "y_pred": rng.normal(30, 8, n),
        "is_imputed": False,
        "label": np.where(rng.random(n) < 0.2, "fault", "trustworthy"),
        "fault_type": "spike",
        "is_extreme": rng.random(n) < 0.1,
        "local_volatility": np.abs(rng.normal(5, 1, n)),
        "score": rng.normal(0, 1, n),
        "det_residual": rng.normal(0, 1, n),
        "det_spatial": rng.normal(0, 1, n),
    })


# ---------------------------------------------------------------------------
# Magnitude stratification
# ---------------------------------------------------------------------------
def test_recall_is_reported_per_magnitude_band():
    """A detector that only sees big faults must show exactly that."""
    rec = _records()
    rec = rec[rec.label == "fault"].copy()
    rec["split"] = "test"
    # Small faults score low, large ones score high.
    rec["magnitude_sigma"] = np.linspace(0.1, 10.0, len(rec))
    rec["score"] = np.where(rec["magnitude_sigma"] > 5, 5.0, -5.0)

    out = stratified_recall(rec, {("6_proposed", "ES0001A"): 0.0,
                                  ("6_proposed", "ES0002A"): 0.0})
    by_band = out.set_index("band")["recall"]
    assert by_band.get(">5σ", 0) == pytest.approx(1.0)
    assert by_band.get("<0.5σ", 1) == pytest.approx(0.0)


def test_only_genuine_faults_are_counted():
    """Trustworthy readings have no magnitude band to belong to."""
    rec = _records()
    rec["split"] = "test"
    rec["magnitude_sigma"] = 3.0
    out = stratified_recall(rec, {("6_proposed", "ES0001A"): 0.0,
                                  ("6_proposed", "ES0002A"): 0.0})
    assert out["n_faults"].sum() == int((rec.label == "fault").sum())


def test_imputed_points_are_excluded():
    rec = _records()
    rec["split"] = "test"
    rec["magnitude_sigma"] = 3.0
    rec.loc[rec.index[: len(rec) // 2], "is_imputed"] = True
    out = stratified_recall(rec, {("6_proposed", "ES0001A"): 0.0,
                                  ("6_proposed", "ES0002A"): 0.0})
    kept = int(((rec.label == "fault") & (~rec.is_imputed)).sum())
    assert out["n_faults"].sum() == kept


def test_bands_cover_every_magnitude():
    rec = _records()
    rec = rec[rec.label == "fault"].copy()
    rec["split"] = "test"
    rec["magnitude_sigma"] = np.linspace(0.0, 50.0, len(rec))
    out = stratified_recall(rec, {("6_proposed", "ES0001A"): 0.0,
                                  ("6_proposed", "ES0002A"): 0.0})
    assert out["n_faults"].sum() == len(rec), "a magnitude fell outside every band"
    assert set(out["band"]) <= set(BIN_LABELS)


# ---------------------------------------------------------------------------
# Operating points
# ---------------------------------------------------------------------------
def test_detector_columns_ignores_empty_detectors():
    rec = _records()
    rec["det_low_variance"] = np.nan        # detector that produced nothing
    assert set(detector_columns(rec)) == {"det_residual", "det_spatial"}


def test_a_looser_budget_never_flags_less():
    """Relaxing the budget can only lower thresholds, so recall cannot fall."""
    rec = _records()
    tight = score_at_budget(rec, 0.01)
    loose = score_at_budget(rec, 0.40)
    assert (loose["score"] >= tight["score"] - 1e-9).all()


def test_scoring_without_detector_columns_is_refused():
    rec = _records().drop(columns=["det_residual", "det_spatial"])
    with pytest.raises(ValueError, match="det_"):
        score_at_budget(rec, 0.01)


def test_thresholds_come_from_validation_only():
    """Changing test rows must not move the thresholds."""
    rec = _records()
    a = score_at_budget(rec, 0.05)
    altered = rec.copy()
    mask = altered["split"] == "test"
    altered.loc[mask, "det_residual"] = altered.loc[mask, "det_residual"] + 100
    b = score_at_budget(altered, 0.05)
    val = rec["split"] == "val"
    np.testing.assert_allclose(a.loc[val, "score"], b.loc[val, "score"])
