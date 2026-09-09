"""SIMULATED station data in the EEA schema — for pipeline exercise only.

    ############################################################
    #  NOTHING PRODUCED BY THIS MODULE IS A REPORTABLE RESULT.  #
    ############################################################

Why it exists: the preprocessing pipeline (§6), the baseline ladder (§8.1) and
the reporting rules (§8.3) all need to be runnable and testable before an EEA
download is available on a given machine. This generator makes that possible
without pretending to be data.

How it stays honest — three independent guards, all of which the reporting layer
groups on, so a simulated number can never be silently averaged into a real one:

  * ``region``       = ``"demo"``      (never ``"catalonia"``)
  * ``label_source`` = ``"simulated"`` (never ``"real"``, never ``"synthetic"``)
  * every station id is prefixed ``DEMO-``

The generator is deliberately *not* a good model of air quality. It reproduces
only the structure the pipeline must cope with: hourly cadence with gaps, a
diurnal + weekly + seasonal cycle, a shared regional component that creates
genuine cross-station spatial correlation (so the Phase 2 graph layer has
something to find), real high-pollution episodes that are NOT faults, and
Validity-flagged faults whose Concentration is still present — the property
that makes real EEA fault labels usable at all (§3.1).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils.logging_setup import get_logger

log = get_logger(__name__)

DEMO_REGION = "demo"
DEMO_LABEL_SOURCE = "simulated"

# Per-station character: (class, baseline ug/m3, traffic amplitude, noise sd).
_DEMO_STATIONS: list[tuple[str, str, float, float, float]] = [
    ("DEMO-BCN01", "urban_traffic", 52.0, 22.0, 7.0),
    ("DEMO-BCN02", "urban_traffic", 48.0, 20.0, 7.5),
    ("DEMO-BCN03", "urban_traffic", 45.0, 19.0, 6.5),
    ("DEMO-BCN04", "urban_background", 30.0, 9.0, 5.5),
    ("DEMO-BCN05", "urban_background", 28.0, 8.0, 5.0),
    ("DEMO-BCN06", "urban_background", 26.0, 8.5, 5.5),
    ("DEMO-BCN07", "urban_background", 31.0, 9.5, 5.0),
    ("DEMO-BCN08", "suburban_background", 18.0, 5.0, 4.5),
    ("DEMO-BCN09", "urban_industrial", 38.0, 12.0, 9.0),
    ("DEMO-BCN10", "urban_industrial", 41.0, 13.0, 9.5),
]

# Roughly the fault prevalence seen in EEA Catalonia series. The point is not
# the exact value — it is that faults are rare, so the anomaly metrics in §8.2
# operate on the imbalanced regime the real problem has.
_FAULT_RATE = 0.030          # Validity = -1
_MAINTENANCE_RATE = 0.008    # Validity = -99
_GAP_RATE = 0.012            # row absent entirely
_EPISODE_RATE = 0.004        # genuine high-pollution event, Validity stays 1


def demo_station_table() -> pd.DataFrame:
    """Station id / class table matching what the generator emits."""
    return pd.DataFrame(
        [{"eoi_code": sid, "name": sid, "station_class": cls}
         for sid, cls, *_ in _DEMO_STATIONS]
    )


def generate_demo_raw(cfg: dict, years: int = 3, seed: int = 1337) -> pd.DataFrame:
    """Emit EEA-schema rows for the demo station set.

    Returns the same columns ``src.data.fetch_eea.REQUIRED_COLUMNS`` guarantees,
    so everything downstream is exercised on the real code path.
    """
    rng = np.random.default_rng(seed)
    pollutants = ["NO2", "PM10", "PM2.5", "O3"]

    end = pd.Timestamp("2024-12-31 23:00", tz="UTC")
    start = end - pd.Timedelta(hours=years * 8760 - 1)
    index = pd.date_range(start, end, freq="h", tz="UTC")
    n = len(index)

    hour = index.hour.to_numpy()
    dow = index.dayofweek.to_numpy()
    doy = index.dayofyear.to_numpy()

    # Shared regional driver: synoptic weather on a multi-day scale. This is the
    # component that makes stations genuinely correlated, which is the signal
    # the Phase 2 server GNN is supposed to recover from embeddings alone.
    regional = _ou_process(n, theta=0.02, sigma=1.0, rng=rng)
    regional = (regional - regional.mean()) / (regional.std() + 1e-9)

    # Winter inversion -> higher NO2; summer sun -> higher O3.
    winter = np.cos(2 * np.pi * doy / 365.25)
    # Morning and evening traffic peaks, damped at the weekend.
    diurnal = (np.exp(-0.5 * ((hour - 8) / 1.8) ** 2)
               + 0.85 * np.exp(-0.5 * ((hour - 20) / 2.2) ** 2))
    weekly = np.where(dow >= 5, 0.6, 1.0)

    frames: list[pd.DataFrame] = []
    for sid, _cls, base, amp, noise_sd in _DEMO_STATIONS:
        local = _ou_process(n, theta=0.15, sigma=1.0, rng=rng)
        local = (local - local.mean()) / (local.std() + 1e-9)

        for pol in pollutants:
            series = _pollutant_series(
                pol, base, amp, noise_sd, diurnal, weekly, winter,
                regional, local, rng, n,
            )
            validity = np.ones(n, dtype=int)

            # Genuine episodes: a real, large, spatially-coherent excursion that
            # is NOT a fault. §8.2 requires a false-positive rate on exactly
            # these — a detector that flags them is the Gupchup (2019) failure.
            for s, dur in _runs(n, _EPISODE_RATE, 6, 36, rng):
                series[s:s + dur] *= rng.uniform(1.8, 3.2)

            # Faults. Concentration REMAINS PRESENT (§3.1) — that is the whole
            # reason Validity gives usable per-reading labels.
            for s, dur in _runs(n, _FAULT_RATE, 2, 24, rng):
                kind = rng.choice(["stuck", "spike", "bias", "noise"])
                seg = series[s:s + dur]
                if kind == "stuck":
                    seg[:] = seg[0]
                elif kind == "spike":
                    seg[:] = seg * rng.uniform(4.0, 9.0)
                elif kind == "bias":
                    seg[:] = seg + rng.uniform(30.0, 90.0)
                else:
                    seg[:] = np.abs(seg + rng.normal(0, seg.std() * 3 + 5, dur))
                validity[s:s + dur] = -1

            # Maintenance: the most valuable label in the dataset (§3.1).
            for s, dur in _runs(n, _MAINTENANCE_RATE, 3, 48, rng):
                series[s:s + dur] = rng.uniform(0.0, 2.0, dur)
                validity[s:s + dur] = -99

            keep = rng.random(n) > _GAP_RATE
            frames.append(pd.DataFrame({
                "AirQualityStationEoICode": sid,
                "Countrycode": "ES",
                "AirPollutant": pol,
                "AveragingTime": "hour",
                "Concentration": np.round(np.maximum(series[keep], 0.0), 3),
                "UnitOfMeasurement": "ug/m3",
                "DatetimeBegin": index[keep],
                "DatetimeEnd": index[keep] + pd.Timedelta(hours=1),
                "Validity": validity[keep],
                "Verification": 1,
            }))

    df = pd.concat(frames, ignore_index=True)
    fault_rate = df["Validity"].isin([-1, -99]).mean()
    log.warning(
        "SIMULATED DATA: %s rows, %d stations, %d pollutants, fault rate %.3f. "
        "region='%s', label_source='%s'. Not a reportable result.",
        f"{len(df):,}", df["AirQualityStationEoICode"].nunique(),
        df["AirPollutant"].nunique(), fault_rate, DEMO_REGION, DEMO_LABEL_SOURCE,
    )
    return df


# ---------------------------------------------------------------------------
def _pollutant_series(pol, base, amp, noise_sd, diurnal, weekly, winter,
                      regional, local, rng, n) -> np.ndarray:
    """Per-pollutant composition of the shared components."""
    if pol == "NO2":
        s = (base + amp * diurnal * weekly + 8.0 * winter
             + 9.0 * regional + 4.0 * local)
    elif pol == "PM10":
        s = (0.7 * base + 0.5 * amp * diurnal * weekly + 4.0 * winter
             + 11.0 * regional + 5.0 * local)
    elif pol == "PM2.5":
        # Kept strictly below PM10 in the mean so §6.3's PM2.5 > PM10 screen
        # fires on genuine inconsistencies rather than on the generator itself.
        s = (0.42 * base + 0.3 * amp * diurnal * weekly + 3.0 * winter
             + 7.0 * regional + 3.0 * local)
    else:  # O3 — anticorrelated with NO2, peaks midday in summer
        s = (55.0 - 0.35 * amp * diurnal * weekly - 14.0 * winter
             + 6.0 * regional + 3.0 * local)
    return np.maximum(s + rng.normal(0.0, noise_sd, n), 0.2)


def _ou_process(n: int, theta: float, sigma: float, rng) -> np.ndarray:
    """Ornstein-Uhlenbeck: autocorrelated noise, a cheap stand-in for weather."""
    x = np.zeros(n)
    eps = rng.normal(0.0, sigma, n)
    for i in range(1, n):
        x[i] = x[i - 1] + theta * (0.0 - x[i - 1]) + eps[i]
    return x


def _runs(n: int, rate: float, min_len: int, max_len: int, rng):
    """Yield ``(start, length)`` for episodes covering ~``rate`` of the series."""
    mean_len = (min_len + max_len) / 2
    n_runs = max(int(n * rate / mean_len), 1)
    for _ in range(n_runs):
        length = int(rng.integers(min_len, max_len + 1))
        start = int(rng.integers(0, max(n - length, 1)))
        yield start, length
