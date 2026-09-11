"""§8.2 — Metrics.

Forecasting: RMSE, MAE, MAPE, in physical units (ug/m3), per station, per
horizon, per station class.

Anomaly: precision, recall, F1, AUC-PR against the real `Validity` labels
primarily, and separately against synthetic-injected faults where used.

**And always a false-positive rate on genuine extreme readings.** §8.2 makes
this non-optional and gives the reason: Gupchup et al. (2019) found naive
detectors misclassify up to 45% of real pollution events as sensor faults.
Recall without that control is not a result — a detector that flags every high
reading scores perfect recall and is useless. ``anomaly_metrics`` therefore
*requires* the extreme-reading mask; there is no way to call it without one.

Points imputed by §6.2 are excluded from every anomaly metric. Scoring a
detector on its own interpolation measures the interpolator.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np

# MAPE denominator floor, in ug/m3. Hourly NO2 legitimately approaches zero at
# night; dividing by 0.3 ug/m3 produces a five-figure percentage that dominates
# the mean and says nothing. Points below the floor are excluded and counted.
MAPE_FLOOR = 1.0


@dataclass
class ForecastMetrics:
    n: int
    rmse: float
    mae: float
    mape: float
    mape_n_excluded: int
    bias: float

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class AnomalyMetrics:
    n: int
    n_positive: int
    precision: float
    recall: float
    f1: float
    auc_pr: float
    # The §8.2 control: of genuine (real, non-faulty) EXTREME readings, what
    # fraction did the detector wrongly call a fault?
    fpr_on_extremes: float
    n_extremes: int
    # Overall false-positive rate on all genuine readings, for context.
    fpr_overall: float

    def as_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
def forecast_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                     mask: np.ndarray | None = None) -> ForecastMetrics:
    """RMSE / MAE / MAPE / bias in physical units.

    ``mask`` selects the points to score (True = include). Callers pass
    ``~is_imputed`` to keep invented values out of the forecast error too — a
    model that predicts an interpolation well has learned the interpolator.
    """
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    keep = np.isfinite(y_true) & np.isfinite(y_pred)
    if mask is not None:
        keep &= np.asarray(mask, dtype=bool).ravel()

    yt, yp = y_true[keep], y_pred[keep]
    if yt.size == 0:
        return ForecastMetrics(0, float("nan"), float("nan"), float("nan"), 0,
                               float("nan"))

    err = yp - yt
    big_enough = np.abs(yt) >= MAPE_FLOOR
    mape = (float(np.mean(np.abs(err[big_enough] / yt[big_enough])) * 100.0)
            if big_enough.any() else float("nan"))

    return ForecastMetrics(
        n=int(yt.size),
        rmse=float(np.sqrt(np.mean(err ** 2))),
        mae=float(np.mean(np.abs(err))),
        mape=mape,
        mape_n_excluded=int((~big_enough).sum()),
        bias=float(np.mean(err)),
    )


def anomaly_metrics(
    y_label: np.ndarray,
    score: np.ndarray,
    threshold: float,
    is_extreme: np.ndarray,
    mask: np.ndarray | None = None,
) -> AnomalyMetrics:
    """Detection metrics plus the mandatory false-positive control.

    Parameters
    ----------
    y_label
        1 = fault, 0 = trustworthy, -1 = unknown. Unknowns are dropped.
    score
        Continuous anomaly score; higher = more anomalous.
    threshold
        Decision threshold, calibrated on VALIDATION only.
    is_extreme
        True where the reading is a genuine high-pollution reading. Required —
        see the module docstring. Build it with ``extreme_mask``.
    mask
        Points to score at all; callers pass ``~is_imputed``.
    """
    y_label = np.asarray(y_label).ravel()
    score = np.asarray(score, dtype=float).ravel()
    is_extreme = np.asarray(is_extreme, dtype=bool).ravel()

    keep = (y_label != -1) & np.isfinite(score)
    if mask is not None:
        keep &= np.asarray(mask, dtype=bool).ravel()

    y = (y_label[keep] == 1)
    s = score[keep]
    ext = is_extreme[keep]

    if y.size == 0:
        nan = float("nan")
        return AnomalyMetrics(0, 0, nan, nan, nan, nan, nan, 0, nan)

    pred = s > threshold
    tp = int((pred & y).sum())
    fp = int((pred & ~y).sum())
    fn = int((~pred & y).sum())
    tn = int((~pred & ~y).sum())

    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    recall = tp / (tp + fn) if (tp + fn) else float("nan")
    f1 = (2 * precision * recall / (precision + recall)
          if (tp + fp) and (tp + fn) and (precision + recall) > 0 else float("nan"))

    # The control. Genuine, non-faulty, extreme readings wrongly called faults.
    genuine_extreme = ext & ~y
    n_ext = int(genuine_extreme.sum())
    fpr_ext = (float((pred & genuine_extreme).sum() / n_ext) if n_ext
               else float("nan"))

    return AnomalyMetrics(
        n=int(y.size),
        n_positive=int(y.sum()),
        precision=precision,
        recall=recall,
        f1=f1,
        auc_pr=average_precision(y, s),
        fpr_on_extremes=fpr_ext,
        n_extremes=n_ext,
        fpr_overall=fp / (fp + tn) if (fp + tn) else float("nan"),
    )


def average_precision(y_true: np.ndarray, score: np.ndarray) -> float:
    """Area under the precision-recall curve (AUC-PR).

    §8.2 asks for AUC-PR rather than ROC-AUC because faults are ~3% of readings
    and ROC-AUC flatters a detector on that kind of imbalance.
    """
    y_true = np.asarray(y_true, dtype=bool).ravel()
    score = np.asarray(score, dtype=float).ravel()
    if y_true.size == 0 or y_true.sum() == 0 or y_true.all():
        return float("nan")
    try:
        from sklearn.metrics import average_precision_score
        return float(average_precision_score(y_true, score))
    except ImportError:
        order = np.argsort(-score)
        y = y_true[order]
        tp = np.cumsum(y)
        precision = tp / np.arange(1, y.size + 1)
        return float((precision * y).sum() / y.sum())


# ---------------------------------------------------------------------------
def extreme_mask(y_true: np.ndarray, quantile: float = 0.95,
                 reference: np.ndarray | None = None) -> np.ndarray:
    """Genuine high-pollution readings, for the §8.2 false-positive control.

    The quantile is taken over ``reference`` (pass the *training* target so the
    definition of "extreme" is fixed before any test data is seen) or over
    ``y_true`` itself when no reference is given.
    """
    y_true = np.asarray(y_true, dtype=float)
    ref = np.asarray(reference if reference is not None else y_true, dtype=float)
    ref = ref[np.isfinite(ref)]
    if ref.size == 0:
        return np.zeros_like(y_true, dtype=bool)
    return y_true >= float(np.quantile(ref, quantile))


def residual_score(y_true: np.ndarray, y_pred: np.ndarray,
                   scale: float | None = None) -> np.ndarray:
    """|y - y_hat|, optionally divided by a station-local scale.

    This is the anomaly score of the baseline detector and the first term of
    the §7.4 decision layer. ``scale`` is the station's own local std — it stays
    on the node and is never uploaded (§6.5).
    """
    resid = np.abs(np.asarray(y_pred, dtype=float) - np.asarray(y_true, dtype=float))
    if scale is None:
        return resid
    return resid / max(float(scale), 1e-8)


def calibrate_threshold(score: np.ndarray, method: str = "quantile",
                        quantile: float = 0.98,
                        mask: np.ndarray | None = None) -> float:
    """Pick a residual threshold from VALIDATION scores only.

    Never from test, and — once a second region exists — never one global
    threshold inherited from whichever region trained first (§5.8, §7.4).
    """
    s = np.asarray(score, dtype=float).ravel()
    keep = np.isfinite(s)
    if mask is not None:
        keep &= np.asarray(mask, dtype=bool).ravel()
    s = s[keep]
    if s.size == 0:
        raise ValueError("no finite validation scores to calibrate a threshold on")

    if method == "quantile":
        return float(np.quantile(s, quantile))
    if method == "mad":
        median = float(np.median(s))
        mad = float(np.median(np.abs(s - median)))
        return median + 3.0 * 1.4826 * mad
    raise ValueError(f"unknown threshold method {method!r}")
