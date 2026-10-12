"""§7.4 — residual + spatial agreement → trustworthy / sensor-faulty.

§7.4 sketches a single per-reading rule: threshold the residual, then let
spatial agreement arbitrate. Measurement says that rule cannot work as written,
and this module is the design the measurements force instead.

**Why one per-reading score is not enough (CONTEXT.md entries 6–8).** Against
rate-matched synthetic faults, a residual threshold catches spikes (0.59) and
noise (0.40) and is blind to the rest: drift 0.000, bias 0.024, stuck-at 0.000,
dropout 0.081. Those four produce scores *inside the distribution of normal
readings* — stuck-at actually scores **below** the normal median, because a
frozen sensor is more predictable than a working one. No threshold on any
per-reading score separates them, and swapping in an uncontaminated temporal
reference changed nothing. What did move them was changing the **timescale**
and the **statistic**:

| subtype | what finds it | measured |
|---|---|---|
| spike, noise | per-reading residual | 0.70 / 0.67 |
| bias | residual accumulated over ~6h | 0.024 → 0.261 |
| dropout | accumulated over ~24h | 0.081 → 0.228 |
| drift | neighbour comparison over ~24h | 0.004 → 0.149 |
| stuck-at | *low* variance, not a high score | nothing else works |

So the decision layer runs four detectors at the timescale each subtype needs
and takes the strongest signal. A single global per-reading threshold would
reproduce the 0.000s and look like the graph layer failing, when it would
really be the timescale failing.

**The §7.4 arbitration survives, and matters.** Once something is flagged, high
spatial agreement means the neighbourhood is seeing it too, so it is a real
event; low agreement means the station is alone, which is what a fault looks
like. That is the Gupchup (2019) control made mechanical, and entry 8 measured
why it should work: differencing against neighbours cancels regional episodes,
keeping `fpr_on_extremes` at 0.060–0.150 across windows.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.evaluation.metrics import calibrate_threshold
from src.utils.logging_setup import get_logger

log = get_logger(__name__)

TRUSTWORTHY_BASELINE = "trustworthy — baseline"
TRUSTWORTHY_EVENT = "trustworthy — real event"
SENSOR_FAULT = "sensor fault"

#: Windows measured in entries 7–8, per subtype. Not tuned on test.
DEFAULT_RESIDUAL_WINDOWS = (1, 6, 24)
DEFAULT_SPATIAL_WINDOWS = (1, 6, 24)
DEFAULT_VARIANCE_WINDOW = 12


@dataclass
class DetectorOutput:
    """Each detector's contribution, kept separate so attribution stays possible."""
    residual: np.ndarray
    accumulated: np.ndarray
    low_variance: np.ndarray
    spatial: np.ndarray
    combined: np.ndarray
    which: np.ndarray            # which detector fired, per reading

    def as_frame(self, index: pd.Index) -> pd.DataFrame:
        return pd.DataFrame({
            "score_residual": self.residual,
            "score_accumulated": self.accumulated,
            "score_low_variance": self.low_variance,
            "score_spatial": self.spatial,
            "score": self.combined,
            "detector": self.which,
        }, index=index)


def _rolling(values: pd.Series, window: int) -> np.ndarray:
    if window <= 1:
        return values.to_numpy(dtype=float)
    return (values.rolling(window, min_periods=max(2, window // 4))
            .mean().to_numpy(dtype=float))


def low_variance_score(y: pd.Series, window: int = DEFAULT_VARIANCE_WINDOW,
                       floor: float = 1e-3) -> np.ndarray:
    """High when the sensor is *too* steady — the stuck-at detector.

    Every other detector asks "is this reading far from expectation?", which a
    frozen sensor answers with a confident no. This asks the opposite question.
    A genuinely stuck sensor has exactly zero rolling variance, so the signal is
    unambiguous; the floor keeps a quiet-but-working night from dividing by ~0.
    """
    rolling_std = y.rolling(window, min_periods=max(2, window // 4)).std()
    typical = float(np.nanmedian(rolling_std.to_numpy())) or 1.0
    return (typical / np.maximum(rolling_std.to_numpy(), floor * typical)).clip(0, 50)


def normalise(score: np.ndarray, reference: np.ndarray | None = None) -> np.ndarray:
    """Put a detector on a common scale: median-centred, MAD-scaled.

    Combining raw scores would let whichever detector happens to have the
    largest units dominate the maximum. ``reference`` is the validation slice,
    so the scale never comes from test.
    """
    ref = reference if reference is not None else score
    ref = ref[np.isfinite(ref)]
    if ref.size == 0:
        # An all-NaN detector (e.g. spatial, with no server GNN) must stay NaN
        # so nanmax ignores it. Returning zeros would make an absent detector
        # look like a live one sitting exactly at its threshold.
        return np.full_like(score, np.nan, dtype=float)
    median = float(np.median(ref))
    mad = float(np.median(np.abs(ref - median))) or 1e-9
    return (score - median) / (1.4826 * mad)


class DecisionLayer:
    """The four-detector ensemble, plus §7.4's spatial arbitration."""

    def __init__(self, residual_windows=DEFAULT_RESIDUAL_WINDOWS,
                 spatial_windows=DEFAULT_SPATIAL_WINDOWS,
                 variance_window: int = DEFAULT_VARIANCE_WINDOW,
                 use_low_variance: bool = True):
        self.residual_windows = tuple(residual_windows)
        self.spatial_windows = tuple(spatial_windows)
        self.variance_window = variance_window
        self.use_low_variance = use_low_variance
        self._scales: dict[str, np.ndarray] = {}
        self._thresholds: dict[str, float] = {}

    def fit(self, frame: pd.DataFrame, labels: np.ndarray,
            is_extreme: np.ndarray, target_fpr_extremes: float = 0.10,
            budget_on: str = "overall",
            target_fpr_overall: float = 0.02) -> "DecisionLayer":
        """Calibrate each detector separately, on validation, to its own budget.

        This is the fix for the precision collapse measured in entry 10.
        Combining four standardised detectors with a maximum and then applying
        *one* threshold gives every reading four chances to clear one bar, so
        false positives compound — recall doubled but precision fell from 0.722
        to 0.173.

        Each detector gets ``budget / k``, so by the union bound the combined
        false-positive rate stays under the target no matter how many detectors
        run. Each detector's score is then expressed as its *margin* over its
        own threshold, which puts them on a comparable footing: the combined
        score crosses zero exactly when some detector has exceeded its own
        budgeted operating point.

        **``budget_on`` must be "overall", and entry 11 is why.** Budgeting on
        *extremes* looks right — §8.2 cares most about crying wolf on real
        episodes — but extremes are only the top few percent of readings, so
        capping false positives there leaves the other 95% entirely
        unconstrained. Four detectors each taking the lowest threshold that
        satisfied an extremes-only budget drove precision to **0.027**, worse
        than the single-threshold version it was meant to fix. Budgeting on all
        trustworthy readings is what actually controls precision.
        """
        parts = self._parts(frame)
        self._scales = {k: v.copy() for k, v in parts.items()}
        standardised = {k: normalise(v, self._scales[k]) for k, v in parts.items()}

        # Usability is decided on the RAW outputs: a detector that produced
        # nothing must not claim a share of the budget.
        usable = [k for k, v in parts.items() if np.isfinite(v).any()]
        if budget_on == "overall":
            method, total = "val_fpr_overall", target_fpr_overall
        elif budget_on == "extremes":
            method, total = "val_fpr_extremes", target_fpr_extremes
        else:
            raise ValueError(f"budget_on must be 'overall' or 'extremes', got {budget_on!r}")
        budget = total / max(len(usable), 1)
        log.info("decision layer: %d detector(s), %s budget %.4f each",
                 len(usable), budget_on, budget)

        self._thresholds = {}
        for key in usable:
            try:
                self._thresholds[key] = calibrate_threshold(
                    standardised[key], method=method,
                    labels=labels, is_extreme=is_extreme,
                    target_fpr_extremes=budget,
                )
            except ValueError:
                # No labelled extremes for this detector; fall back to a plain
                # quantile at the same budget rather than leaving it uncapped.
                finite = standardised[key][np.isfinite(standardised[key])]
                self._thresholds[key] = (float(np.quantile(finite, 1 - budget))
                                         if finite.size else np.inf)
        return self

    def _parts(self, frame: pd.DataFrame) -> dict[str, np.ndarray]:
        """Raw (unstandardised) output of each detector."""
        residual = frame["residual"].astype(float)

        # Rolling windows leave NaN at the head of a series, so an all-NaN
        # column there is expected rather than a problem worth warning about.
        warnings.filterwarnings("ignore", message="All-NaN slice encountered")

        accumulated = np.nanmax(
            np.vstack([_rolling(residual, w) for w in self.residual_windows
                       if w > 1]) if len(self.residual_windows) > 1
            else residual.to_numpy(dtype=float)[None, :], axis=0)

        if "spatial" in frame.columns and frame["spatial"].notna().any():
            sp = frame["spatial"].astype(float)
            spatial = np.nanmax(
                np.vstack([_rolling(sp, w) for w in self.spatial_windows]), axis=0)
        else:
            spatial = np.full(len(frame), np.nan)

        low_var = (low_variance_score(frame["y_true"].astype(float),
                                      self.variance_window)
                   if self.use_low_variance else np.full(len(frame), np.nan))

        return {
            "residual": residual.to_numpy(dtype=float),
            "accumulated": accumulated,
            "low_variance": low_var,
            "spatial": spatial,
        }

    def score(self, frame: pd.DataFrame) -> DetectorOutput:
        """Score one station's time-ordered frame. ``fit`` must have run first.

        Expects ``residual`` (|y - ŷ| / scale), ``y_true`` and optionally
        ``spatial`` (per-window disagreement from the server GNN). Rows must be
        in time order; the rolling statistics depend on it.
        """
        if not self._thresholds:
            raise RuntimeError(
                "DecisionLayer.fit() must run on the validation split before "
                "scoring — per-detector thresholds are calibrated there, never "
                "on test (§7.4)."
            )
        parts = self._parts(frame)
        standardised = {k: normalise(v, self._scales.get(k))
                        for k, v in parts.items()}

        # Margin over each detector's own budgeted threshold: positive means
        # that detector has fired. Zero is therefore the natural cut point.
        margins = {k: standardised[k] - t for k, t in self._thresholds.items()}
        stacked = np.vstack(list(margins.values()))
        combined = np.nanmax(stacked, axis=0)
        which = np.array(list(margins.keys()))[np.nanargmax(
            np.nan_to_num(stacked, nan=-np.inf), axis=0)]

        return DetectorOutput(
            residual=standardised["residual"],
            accumulated=standardised["accumulated"],
            low_variance=standardised["low_variance"],
            spatial=standardised["spatial"],
            combined=combined,
            which=which,
        )

    @staticmethod
    def classify(combined: np.ndarray, threshold: float,
                 agreement: np.ndarray | None = None,
                 high_agreement: float = 0.6) -> np.ndarray:
        """§7.4's three-way call, verbatim in structure.

        Below threshold: trustworthy, nothing unusual. Above it, the
        neighbourhood decides — if the neighbours agree, the anomaly is a real
        event; if the station is alone, it is a fault.
        """
        labels = np.full(len(combined), TRUSTWORTHY_BASELINE, dtype=object)
        flagged = combined > threshold
        if agreement is None:
            labels[flagged] = SENSOR_FAULT
            return labels
        agrees = np.nan_to_num(agreement, nan=-np.inf) >= high_agreement
        labels[flagged & agrees] = TRUSTWORTHY_EVENT
        labels[flagged & ~agrees] = SENSOR_FAULT
        return labels
