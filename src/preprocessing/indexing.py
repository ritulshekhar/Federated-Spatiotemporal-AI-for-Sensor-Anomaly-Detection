"""§6.1 — Ingestion and indexing.

Turns the long-format common schema into one *wide, contiguous, hourly* frame
per station and reports the gap structure it had to create to get there.

The wide frame is the object every later stage operates on. Its column
convention, fixed here and relied on everywhere downstream:

    <POLLUTANT>                 float   concentration
    <POLLUTANT>__validity       Int64   raw EEA Validity code
    <POLLUTANT>__verification   Int64   raw EEA Verification code
    <POLLUTANT>__label          object  "fault" / "trustworthy" / None

Stages then add, in order:

    <POLLUTANT>__is_imputed     bool    §6.2, built BEFORE any filling
    <POLLUTANT>__physical_flag  bool    §6.3

Reindexing onto a complete hourly range is what makes gaps *visible*. A station
whose file simply has no row for 03:00 looks continuous until you do this; here
it becomes an explicit NaN that §6.2 has to make a decision about.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.utils.logging_setup import get_logger

log = get_logger(__name__)

VALIDITY_SUFFIX = "__validity"
VERIFICATION_SUFFIX = "__verification"
LABEL_SUFFIX = "__label"
IMPUTED_SUFFIX = "__is_imputed"
PHYSICAL_SUFFIX = "__physical_flag"


@dataclass
class CoverageReport:
    """Per-station gap structure, logged and carried into the run manifest."""
    station_id: str
    pollutant: str
    first: pd.Timestamp
    last: pd.Timestamp
    n_expected: int
    n_observed: int
    n_missing: int
    longest_gap_hours: int
    n_gaps: int

    @property
    def observed_fraction(self) -> float:
        return self.n_observed / self.n_expected if self.n_expected else 0.0

    def as_dict(self) -> dict:
        return {
            "station_id": self.station_id,
            "pollutant": self.pollutant,
            "first": self.first,
            "last": self.last,
            "n_expected": self.n_expected,
            "n_observed": self.n_observed,
            "n_missing": self.n_missing,
            "observed_fraction": round(self.observed_fraction, 4),
            "longest_gap_hours": self.longest_gap_hours,
            "n_gaps": self.n_gaps,
        }


def build_station_frame(
    long_df: pd.DataFrame,
    station_id: str,
    pollutants: list[str],
) -> tuple[pd.DataFrame, list[CoverageReport]]:
    """One station's long rows -> a contiguous hourly wide frame.

    ``long_df`` is the common schema from ``src.data.harmonize``. It may contain
    every station; only ``station_id`` is used. Pollutants the station does not
    report are simply absent from the output — they are not filled with NaN
    columns, because a station that never measured O3 is a different thing from
    one whose O3 sensor was offline.
    """
    sub = long_df[long_df["station_id"] == station_id]
    if sub.empty:
        raise ValueError(f"no rows for station {station_id!r}")

    sub = sub[sub["pollutant"].isin(pollutants)]
    if sub.empty:
        raise ValueError(
            f"station {station_id!r} reports none of the requested pollutants "
            f"{pollutants}; it reports {sorted(long_df.loc[long_df['station_id'] == station_id, 'pollutant'].unique())}"
        )

    ts = pd.to_datetime(sub["timestamp_utc"], utc=True)
    # Snap to the hour. EEA is hourly by construction, but a stray sub-hourly
    # timestamp would otherwise silently create a one-row-wide reindex gap.
    ts_floor = ts.dt.floor("h")
    if not ts.equals(ts_floor):
        n_off = int((ts != ts_floor).sum())
        log.warning("%s: %d timestamp(s) were not on the hour; floored", station_id, n_off)
    sub = sub.assign(timestamp_utc=ts_floor)

    full_index = pd.date_range(
        sub["timestamp_utc"].min(), sub["timestamp_utc"].max(), freq="h", tz="UTC",
        name="timestamp_utc",
    )

    frame = pd.DataFrame(index=full_index)
    reports: list[CoverageReport] = []

    for pol in pollutants:
        p = sub[sub["pollutant"] == pol]
        if p.empty:
            continue
        p = p.drop_duplicates(subset="timestamp_utc", keep="last").set_index("timestamp_utc")

        conc = p["concentration"].reindex(full_index).astype(float)
        frame[pol] = conc
        frame[pol + VALIDITY_SUFFIX] = (
            p["validity"].reindex(full_index).astype("Int64")
        )
        frame[pol + VERIFICATION_SUFFIX] = (
            p["verification"].reindex(full_index).astype("Int64")
        )
        frame[pol + LABEL_SUFFIX] = p["label"].reindex(full_index)

        reports.append(_coverage(station_id, pol, conc, full_index))

    frame.attrs["station_id"] = station_id
    frame.attrs["region"] = str(sub["region"].iloc[0])
    frame.attrs["station_class"] = str(sub["station_class"].iloc[0])
    frame.attrs["label_source"] = str(sub["label_source"].iloc[0])
    frame.attrs["pollutants"] = [p for p in pollutants if p in frame.columns]

    for r in reports:
        log.info(
            "%s %-5s | %s..%s | observed %.1f%% | %d gap(s), longest %dh",
            station_id, r.pollutant, r.first.date(), r.last.date(),
            100 * r.observed_fraction, r.n_gaps, r.longest_gap_hours,
        )
    return frame, reports


def _coverage(station_id: str, pollutant: str, series: pd.Series,
              index: pd.DatetimeIndex) -> CoverageReport:
    missing = series.isna().to_numpy()
    gap_lengths = _run_lengths(missing)
    return CoverageReport(
        station_id=station_id,
        pollutant=pollutant,
        first=index[0],
        last=index[-1],
        n_expected=len(index),
        n_observed=int((~missing).sum()),
        n_missing=int(missing.sum()),
        longest_gap_hours=int(max(gap_lengths, default=0)),
        n_gaps=len(gap_lengths),
    )


def _run_lengths(mask: np.ndarray) -> list[int]:
    """Lengths of each maximal run of True in ``mask``."""
    if mask.size == 0 or not mask.any():
        return []
    padded = np.concatenate(([False], mask, [False]))
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    return (edges[1::2] - edges[0::2]).tolist()


def gap_run_lengths(series: pd.Series) -> np.ndarray:
    """Per-element gap length: for each NaN, the length of the run it belongs to.

    §6.2 needs this to choose between linear interpolation (short gaps) and
    seasonal-naive filling (long ones) *per gap*, not per series.
    """
    missing = series.isna().to_numpy()
    out = np.zeros(len(series), dtype=int)
    if not missing.any():
        return out
    padded = np.concatenate(([False], missing, [False]))
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    for start, end in zip(edges[0::2], edges[1::2]):
        out[start:end] = end - start
    return out
