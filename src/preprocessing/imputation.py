"""§6.2 — Missing values, with mask retention.

The whole point of this module is the *order of operations*:

    build is_imputed  ->  then fill

not the other way round. Once a series is filled you cannot recover which
points were real, and an anomaly detector that scores its own interpolation is
measuring itself. The mask is carried all the way through windowing so §7.4 can
exclude imputed points from anomaly scoring.

Fill policy (§6.2):

* gap <= ``short_gap_hours`` (default 3h)  -> linear interpolation
* longer gaps                              -> seasonal-naive: same hour,
                                              previous day; walking back day by
                                              day until an observed value is
                                              found
* gap > ``max_fill_hours`` (default 168h)  -> left NaN on purpose. A week-long
                                              outage is not something to invent
                                              a week of data for; windowing
                                              drops any window touching it.

Seasonal-naive is chosen over a smarter imputer deliberately. It preserves the
diurnal shape, it is auditable, and being obviously simple it does not tempt
anyone into reading imputed stretches as signal.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.preprocessing.indexing import IMPUTED_SUFFIX, gap_run_lengths
from src.utils.logging_setup import get_logger

log = get_logger(__name__)


@dataclass
class ImputationReport:
    station_id: str
    pollutant: str
    n_total: int
    n_missing: int
    n_linear: int
    n_seasonal: int
    n_left_nan: int

    def as_dict(self) -> dict:
        d = self.__dict__.copy()
        d["imputed_fraction"] = round(
            (self.n_linear + self.n_seasonal) / self.n_total, 4
        ) if self.n_total else 0.0
        return d


def impute_frame(
    frame: pd.DataFrame,
    pollutants: list[str],
    short_gap_hours: int = 3,
    seasonal_period_hours: int = 24,
    max_fill_hours: int = 168,
) -> tuple[pd.DataFrame, list[ImputationReport]]:
    """Fill each pollutant column in place, adding ``<POL>__is_imputed`` first."""
    frame = frame.copy()
    station_id = frame.attrs.get("station_id", "?")
    reports: list[ImputationReport] = []

    for pol in pollutants:
        if pol not in frame.columns:
            continue
        series = frame[pol]

        # ---- the mask, BEFORE any filling -----------------------------------
        was_missing = series.isna().to_numpy()
        gap_len = gap_run_lengths(series)

        short = was_missing & (gap_len <= short_gap_hours)
        long_ = was_missing & (gap_len > short_gap_hours) & (gap_len <= max_fill_hours)
        too_long = was_missing & (gap_len > max_fill_hours)

        filled = series.copy()

        # ---- short gaps: linear interpolation --------------------------------
        if short.any():
            interpolated = series.interpolate(method="time", limit_area="inside")
            filled = filled.where(~short, interpolated)

        # ---- long gaps: seasonal-naive ---------------------------------------
        if long_.any():
            filled = _seasonal_naive_fill(
                filled, mask=long_, period=seasonal_period_hours
            )

        # A short gap at the very start/end has no interior to interpolate
        # across, so limit_area="inside" leaves it NaN. Fall back to seasonal
        # for those rather than silently reclassifying them as unfillable.
        still_short = short & filled.isna().to_numpy()
        if still_short.any():
            filled = _seasonal_naive_fill(
                filled, mask=still_short, period=seasonal_period_hours
            )

        # ---- the mask is true only where a value was ACTUALLY produced -------
        actually_filled = was_missing & filled.notna().to_numpy()

        frame[pol] = filled
        frame[pol + IMPUTED_SUFFIX] = actually_filled

        n_linear = int((short & actually_filled).sum())
        n_seasonal = int((long_ & actually_filled).sum()) + int(
            (still_short & actually_filled).sum()
        )
        rep = ImputationReport(
            station_id=station_id,
            pollutant=pol,
            n_total=len(series),
            n_missing=int(was_missing.sum()),
            n_linear=n_linear,
            n_seasonal=n_seasonal,
            n_left_nan=int(filled.isna().sum()),
        )
        reports.append(rep)
        log.info(
            "%s %-5s | missing %d (%.1f%%) -> linear %d, seasonal %d, left NaN %d "
            "(%d beyond the %dh fill limit)",
            station_id, pol, rep.n_missing, 100 * rep.n_missing / max(rep.n_total, 1),
            rep.n_linear, rep.n_seasonal, rep.n_left_nan,
            int(too_long.sum()), max_fill_hours,
        )

    return frame, reports


def _seasonal_naive_fill(series: pd.Series, mask: np.ndarray,
                         period: int = 24, max_lookback_days: int = 14) -> pd.Series:
    """Fill ``mask`` positions with the same hour on the nearest earlier day.

    Walks back one period at a time so a multi-day outage is filled from the
    last genuinely observed matching hour, not from another imputed value.
    """
    values = series.to_numpy(dtype=float, copy=True)
    observed = ~np.isnan(values)
    remaining = mask.copy()

    for step in range(1, max_lookback_days + 1):
        lag = step * period
        idx = np.flatnonzero(remaining)
        idx = idx[idx >= lag]
        if idx.size == 0:
            break
        src = idx - lag
        usable = observed[src]
        values[idx[usable]] = values[src[usable]]
        remaining[idx[usable]] = False
        if not remaining.any():
            break

    # Anything still unfilled had no matching hour in the preceding fortnight —
    # try walking *forward* before giving up.
    if remaining.any():
        for step in range(1, max_lookback_days + 1):
            lag = step * period
            idx = np.flatnonzero(remaining)
            idx = idx[idx + lag < len(values)]
            if idx.size == 0:
                break
            src = idx + lag
            usable = observed[src]
            values[idx[usable]] = values[src[usable]]
            remaining[idx[usable]] = False
            if not remaining.any():
                break

    return pd.Series(values, index=series.index, name=series.name)


def imputed_mask(frame: pd.DataFrame, pollutant: str) -> pd.Series:
    """The retained mask for one pollutant. Raises if imputation never ran."""
    col = pollutant + IMPUTED_SUFFIX
    if col not in frame.columns:
        raise KeyError(
            f"{col} missing — impute_frame() must run before anything that scores "
            "anomalies, so the detector never flags its own interpolation (§6.2)."
        )
    return frame[col].astype(bool)
