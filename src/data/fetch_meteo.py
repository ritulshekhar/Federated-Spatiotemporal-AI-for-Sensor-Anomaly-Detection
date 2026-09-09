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


def fetch_station_meteo(lat: float, lon: float, start: str, end: str,
                        variables: list[str], timeout: int = 120) -> pd.DataFrame:
    """Hourly ERA5 reanalysis at one point, indexed by UTC timestamp."""
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
    resp = requests.get(OPEN_METEO_ARCHIVE, params=params, timeout=timeout)
    resp.raise_for_status()
    payload = resp.json()
    if "hourly" not in payload:
        raise ValueError(f"unexpected Open-Meteo response: {payload}")

    hourly = payload["hourly"]
    df = pd.DataFrame(hourly)
    df["timestamp_utc"] = pd.to_datetime(df.pop("time"), utc=True)
    df = df.rename(columns=VARIABLE_RENAME)
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
    """Cached met frame for one station, or None if it was never fetched."""
    path = data_paths()["raw"] / "meteo" / f"{station_id}.parquet"
    if not path.exists():
        return None
    return pd.read_parquet(path)


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
