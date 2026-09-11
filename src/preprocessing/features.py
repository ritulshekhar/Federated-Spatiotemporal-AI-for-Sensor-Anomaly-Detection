"""§6.4 — Feature engineering.

Adds, per station:

* cyclical encodings of hour / day-of-week / month (sin+cos pairs, so 23:00 and
  00:00 are adjacent rather than maximally distant);
* wind direction as sin/cos of degrees, accepting either degrees or a 16-point
  compass string (§6.4);
* lags at t-1, t-24, t-168 — the previous hour, the same hour yesterday, the
  same hour last week;
* rolling mean and std over 6h and 24h.

Every feature here is **causal**: it depends only on values at or before t. The
model sees a 24h window ending at t and forecasts t+1..t+6, so including the
value at t itself is legitimate. Nothing in this module may look forward — that
would leak the future into the past just as surely as shuffling the split would
(§6.6).

Derived features leave NaN at the head of the series (a t-168 lag needs a week
of history). Those rows are dropped by windowing, not filled — filling them
would manufacture a week of fictitious history at every station.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils.logging_setup import get_logger

log = get_logger(__name__)

# §6.4: "wind direction from 16-point compass to sine/cosine of degrees".
COMPASS_16 = {
    "N": 0.0, "NNE": 22.5, "NE": 45.0, "ENE": 67.5,
    "E": 90.0, "ESE": 112.5, "SE": 135.0, "SSE": 157.5,
    "S": 180.0, "SSW": 202.5, "SW": 225.0, "WSW": 247.5,
    "W": 270.0, "WNW": 292.5, "NW": 315.0, "NNW": 337.5,
}

CYCLICAL_PERIODS = {"hour": 24, "dayofweek": 7, "month": 12}


def add_calendar_features(frame: pd.DataFrame,
                          which: list[str] | None = None) -> pd.DataFrame:
    """sin/cos encodings for each requested calendar field."""
    which = which or ["hour", "dayofweek", "month"]
    frame = frame.copy()
    idx = frame.index

    for field in which:
        period = CYCLICAL_PERIODS[field]
        if field == "hour":
            raw = idx.hour.to_numpy()
        elif field == "dayofweek":
            raw = idx.dayofweek.to_numpy()
        elif field == "month":
            raw = idx.month.to_numpy() - 1   # 1..12 -> 0..11
        else:
            raise KeyError(f"unsupported cyclical field {field!r}")
        angle = 2 * np.pi * raw / period
        frame[f"cyc_{field}_sin"] = np.sin(angle)
        frame[f"cyc_{field}_cos"] = np.cos(angle)

    # A plain weekend indicator on top of the cyclical day encoding: traffic
    # NO2 has a sharp weekday/weekend step that a smooth sin/cos pair needs
    # several harmonics to express.
    frame["is_weekend"] = (idx.dayofweek >= 5).astype(float)
    return frame


def encode_wind_direction(values: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Wind direction -> (sin, cos). Accepts degrees or 16-point compass labels."""
    if values.dtype == object:
        degrees = values.astype(str).str.strip().str.upper().map(COMPASS_16)
        unknown = values.notna() & degrees.isna()
        if unknown.any():
            log.warning("%d unrecognised wind-direction label(s), left as NaN",
                        int(unknown.sum()))
    else:
        degrees = pd.to_numeric(values, errors="coerce")

    radians = np.deg2rad(degrees.astype(float))
    return np.sin(radians), np.cos(radians)


def add_meteo_features(frame: pd.DataFrame,
                       meteo: pd.DataFrame | None) -> pd.DataFrame:
    """Left-join met covariates onto the station frame.

    ``meteo`` is what ``src.data.fetch_meteo.load_station_meteo`` returns, or
    None when meteo is disabled (the default — see that module's flagged
    deviation note). Absent met data is not an error; it is the default state.
    """
    if meteo is None or meteo.empty:
        return frame

    frame = frame.copy()
    met = meteo.copy()
    met.index = pd.to_datetime(met.index, utc=True)
    met = met[~met.index.duplicated(keep="last")].reindex(frame.index)

    if "wind_direction_deg" in met.columns:
        sin, cos = encode_wind_direction(met["wind_direction_deg"])
        met["wind_dir_sin"], met["wind_dir_cos"] = sin, cos
        met = met.drop(columns=["wind_direction_deg"])

    for col in met.columns:
        frame[f"met_{col}"] = met[col].astype(float)
    log.info("%s | joined %d met feature(s)",
             frame.attrs.get("station_id", "?"), len(met.columns))
    return frame


def add_lag_features(frame: pd.DataFrame, columns: list[str],
                     lags: list[int]) -> pd.DataFrame:
    """t-k lags. Causal by construction."""
    frame = frame.copy()
    for col in columns:
        if col not in frame.columns:
            continue
        for lag in lags:
            frame[f"{col}_lag{lag}"] = frame[col].shift(lag)
    return frame


def add_rolling_features(frame: pd.DataFrame, columns: list[str],
                         windows: list[int]) -> pd.DataFrame:
    """Rolling mean and std over trailing windows ending at t (inclusive)."""
    frame = frame.copy()
    for col in columns:
        if col not in frame.columns:
            continue
        for w in windows:
            roll = frame[col].rolling(window=w, min_periods=max(2, w // 2))
            frame[f"{col}_rollmean{w}"] = roll.mean()
            frame[f"{col}_rollstd{w}"] = roll.std()
    return frame


def build_features(
    frame: pd.DataFrame,
    pollutants: list[str],
    cyclical: list[str],
    lags: list[int],
    rolling_windows: list[int],
    meteo: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Run the full §6.4 stage. Returns ``(frame, feature_column_names)``.

    The returned name list is what ``normalize`` and ``windowing`` consume, so
    flag/label/mask columns can never accidentally end up as model inputs.
    """
    present = [p for p in pollutants if p in frame.columns]

    frame = add_calendar_features(frame, cyclical)
    frame = add_meteo_features(frame, meteo)
    frame = add_lag_features(frame, present, lags)
    frame = add_rolling_features(frame, present, rolling_windows)

    feature_cols = (
        present
        + [c for c in frame.columns if c.startswith("cyc_")]
        + ["is_weekend"]
        + [c for c in frame.columns if c.startswith("met_")]
        + [c for c in frame.columns
           if any(c.startswith(f"{p}_lag") or c.startswith(f"{p}_roll")
                  for p in present)]
    )
    # Preserve order, drop duplicates.
    seen: set[str] = set()
    feature_cols = [c for c in feature_cols if not (c in seen or seen.add(c))]

    frame.attrs["feature_columns"] = feature_cols
    log.info("%s | %d feature column(s) built from %d pollutant(s)",
             frame.attrs.get("station_id", "?"), len(feature_cols), len(present))
    return frame, feature_cols
