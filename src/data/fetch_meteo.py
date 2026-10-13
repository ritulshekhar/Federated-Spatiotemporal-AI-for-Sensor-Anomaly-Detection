"""Meteorological covariates for each station.

    ####################################################################
    #  FLAGGED DEVIATION — read this before enabling meteo in configs. #
    ####################################################################

CLAUDE.md objective 3 requires forecasting from "pollutant + meteorological
variables". **EEA Air Quality e-Reporting contains no meteorological
variables.** The E1a delivery is pollutant concentrations plus QA flags; there
is no temperature, wind, pressure or humidity column to use.

So meteorology has to come from somewhere else, and that is a design decision
CLAUDE.md does not make. It is therefore:

  * **opt-in** — ``meteo.enabled: false`` in configs/data.yaml by default;
  * **external** — Open-Meteo's ERA5 reanalysis archive (free, no API key,
    hourly, 1940-present), sampled at each station's registry coordinates;
  * **recorded** — listed under "Open items" in CONTEXT.md rather than quietly
    absorbed.

Three consequences worth being explicit about:

1. ERA5 is a ~25 km reanalysis grid. Every Barcelona station falls in one or
   two grid cells, so met features are near-identical across the node set and
   carry almost no *spatial* discriminative power. They help the temporal
   forecast; they will not help the graph layer distinguish stations.
2. It is a third-party dependency on a network service, unlike the rest of the
   Phase 1 pipeline.
3. Under the privacy contract (§2.4, §7.1) this is fine: the server fetches met
   data for public coordinates it already holds from the EEA registry. No node
   uploads anything to obtain it.

Wind direction is returned in degrees and converted to sin/cos in
``src/preprocessing/features.py`` (§6.4), not here.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd

from src.data.station_metadata import configured_stations, coordinates
from src.utils.logging_setup import get_logger
from src.utils.paths import data_paths, load_config

log = get_logger(__name__)

OPEN_METEO_ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"

# Open-Meteo variable name -> our feature name. Kept explicit so a renamed
# upstream variable is a KeyError, not a silently-missing feature column.
VARIABLE_RENAME = {
    "temperature_2m": "temperature_c",
    "relative_humidity_2m": "relative_humidity",
    "surface_pressure": "pressure_hpa",
    "wind_speed_10m": "wind_speed_ms",
    "wind_direction_10m": "wind_direction_deg",
    "precipitation": "precipitation_mm",
    "boundary_layer_height": "blh_m",
}


def _fetch_chunk(lat: float, lon: float, start: str, end: str,
                 variables: list[str], timeout: int, max_retries: int = 5) -> pd.DataFrame:
    """One Open-Meteo request, with backoff on rate limiting."""
    import requests

    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start,
        "end_date": end,
        "hourly": ",".join(variables),
        "timezone": "UTC",          # §5.2 — UTC at the source, not converted later
        "wind_speed_unit": "ms",
    }
    delay = 5.0
    for attempt in range(max_retries):
        try:
            resp = requests.get(OPEN_METEO_ARCHIVE, params=params, timeout=timeout)
        except requests.exceptions.RequestException as exc:
            # A free public API drops connections under load as readily as it
            # returns 429; both are "try again later".
            log.warning("connection problem (%s..%s): %s; waiting %.0fs",
                        start, end, type(exc).__name__, delay)
            time.sleep(delay)
            delay *= 2
            continue
        if resp.status_code == 429:
            log.warning("rate limited (%s..%s), waiting %.0fs", start, end, delay)
            time.sleep(delay)
            delay *= 2
            continue
        if resp.status_code >= 500:
            # A 5xx from a free public API means "try again later" just as much
            # as a 429 does, and it was NOT retried before: a single 502 killed
            # a 34-of-78-station fetch outright. Open-Meteo returns 502/503
            # intermittently under load, so this is routine rather than
            # exceptional.
            log.warning("server error %d (%s..%s), waiting %.0fs",
                        resp.status_code, start, end, delay)
            time.sleep(delay)
            delay *= 2
            continue
        resp.raise_for_status()
        payload = resp.json()
        if "hourly" not in payload:
            raise ValueError(f"unexpected Open-Meteo response: {payload}")
        return pd.DataFrame(payload["hourly"])
    raise RuntimeError(
        f"Open-Meteo would not serve {start}..{end} after {max_retries} tries. "
        "The free tier weights long ranges heavily; reduce the year span or the "
        "variable list in configs/data.yaml. Stations already downloaded are "
        "cached, so re-running resumes rather than restarting."
    )


def fetch_station_meteo(lat: float, lon: float, start: str, end: str,
                        variables: list[str], timeout: int = 120,
                        chunk_years: int = 1, pause_s: float = 2.0) -> pd.DataFrame:
    """Hourly ERA5 reanalysis at one point, indexed by UTC timestamp.

    Requested one year at a time. Open-Meteo's free tier prices a call by how
    much data it returns, and eight years of seven hourly variables in a single
    request is refused outright with a 429.
    """
    first, last = pd.Timestamp(start), pd.Timestamp(end)
    frames = []
    cursor = first
    while cursor <= last:
        stop = min(cursor + pd.DateOffset(years=chunk_years) - pd.Timedelta(days=1), last)
        frames.append(_fetch_chunk(lat, lon, cursor.strftime("%Y-%m-%d"),
                                   stop.strftime("%Y-%m-%d"), variables, timeout))
        cursor = stop + pd.Timedelta(days=1)
        if cursor <= last:
            time.sleep(pause_s)

    df = pd.concat(frames, ignore_index=True)
    df["timestamp_utc"] = pd.to_datetime(df.pop("time"), utc=True)
    df = df.rename(columns=VARIABLE_RENAME)
    df = df.drop_duplicates(subset="timestamp_utc", keep="last")
    return df.set_index("timestamp_utc").sort_index()


def fetch_all(start: str, end: str, pause_s: float = 1.0) -> dict[str, Path]:
    """Fetch met covariates for every configured station with known coordinates."""
    cfg = load_config("data")
    meteo_cfg = cfg["meteo"]
    if not meteo_cfg.get("enabled", False):
        log.warning(
            "meteo.enabled is false in configs/data.yaml — fetching anyway "
            "because you invoked this module directly, but the preprocessing "
            "pipeline will ignore the result until you flip that flag."
        )

    coords = coordinates()
    # §5: India's stations are not in the EEA registry, so their coordinates
    # come from the CPCB archive itself (see harmonize._write_cpcb_coordinates).
    # Merged only here, where the job is "fetch met data for every station we
    # have", never into `coordinates()` itself — that feeds EEA-only paths.
    if cfg.get("cpcb", {}).get("enabled", False):
        from src.data.station_metadata import cpcb_coordinates
        india = cpcb_coordinates()
        if india:
            log.info("including %d CPCB station(s) in the met fetch", len(india))
            coords = {**coords, **india}
    if not coords:
        raise RuntimeError(
            "no station coordinates available. Run "
            "`python -m src.data.station_metadata --refresh` first — met data is "
            "sampled at the EEA registry coordinates."
        )

    missing = [s.station_id for s in configured_stations() if s.station_id not in coords]
    if missing:
        log.warning("no coordinates for %d station(s): %s", len(missing), missing)

    out_dir = data_paths()["raw"] / "meteo"
    out_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}
    for i, (sid, (lat, lon)) in enumerate(sorted(coords.items())):
        dest = out_dir / f"{sid}.parquet"
        if dest.exists():
            log.info("[%d/%d] %s already fetched", i + 1, len(coords), sid)
            written[sid] = dest
            continue
        log.info("[%d/%d] %s (%.4f, %.4f) %s..%s", i + 1, len(coords), sid, lat, lon,
                 start, end)
        df = fetch_station_meteo(lat, lon, start, end, list(meteo_cfg["variables"]))
        df.to_parquet(dest)
        written[sid] = dest
        time.sleep(pause_s)   # be a good citizen on a free public API
    return written


def load_station_meteo(station_id: str) -> pd.DataFrame | None:
    """Cached met frame for one station, or None if it was never fetched.

    Filtered to the variables currently configured, not to whatever happens to
    be on disk. The config is what governs which variables the model uses, so a
    variable dropped there must stop being a feature without needing a
    re-download — `boundary_layer_height` is dropped for exactly this reason
    and its cached column would otherwise still be joined.
    """
    path = data_paths()["raw"] / "meteo" / f"{station_id}.parquet"
    if not path.exists():
        return None

    df = pd.read_parquet(path)
    configured = load_config("data").get("meteo", {}).get("variables", [])
    wanted = {VARIABLE_RENAME.get(v, v) for v in configured}
    # wind_direction is consumed as sin/cos downstream, so keep it under either
    # name if it was asked for.
    keep = [c for c in df.columns if c in wanted]
    dropped = [c for c in df.columns if c not in wanted]
    if dropped:
        log.info("%s | ignoring %d cached met column(s) not in the config: %s",
                 station_id, len(dropped), dropped)
    return df[keep]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--start", default=None, help="YYYY-MM-DD (default: eea.year_from)")
    ap.add_argument("--end", default=None, help="YYYY-MM-DD (default: eea.year_to)")
    args = ap.parse_args()

    cfg = load_config("data")["eea"]
    start = args.start or f"{cfg['year_from']}-01-01"
    end = args.end or f"{cfg['year_to']}-12-31"
    written = fetch_all(start, end)
    log.info("meteo written for %d station(s)", len(written))


if __name__ == "__main__":
    main()
