"""Acquire EEA Air Quality e-Reporting data for the Catalonia node set.

CLAUDE.md §3.1. Three acquisition routes, tried in this order:

1. ``airbase`` (``pip install -r requirements-optional.txt``) — programmatic
   download of the whole of Spain, filtered afterwards to the configured EoI
   codes. This is the documented primary route.
2. ``--from-dir <path>`` — point at an already-downloaded tree (the Zenodo
   record at https://zenodo.org/records/17085064, or a manual export from
   https://eeadmz1-downloads-webapp.azurewebsites.net/). Any mix of .csv,
   .parquet and .zip underneath is picked up.
3. ``--demo`` — a clearly-labelled *simulated* station set, region ``demo``, so
   the preprocessing pipeline and the baseline ladder can be exercised end to
   end without network access. Read the warning in src/data/demo_source.py
   before using any number that comes out of it.

LOCKED (§3.1): dataset E1a (verified) only. E2a carries provisional flags and
this module refuses to fetch it.
"""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

import pandas as pd

from src.data.station_metadata import (
    configured_stations,
    download_registry,
    verify_configured_stations,
)
from src.utils.logging_setup import get_logger
from src.utils.paths import data_paths, load_config

log = get_logger(__name__)

# The columns CLAUDE.md §3.1 documents. Anything we read must expose these
# (possibly under a legacy alias, see _COLUMN_ALIASES) or harmonization fails.
REQUIRED_COLUMNS = [
    "AirQualityStationEoICode",
    "Countrycode",
    "AirPollutant",
    "AveragingTime",
    "Concentration",
    "UnitOfMeasurement",
    "DatetimeBegin",
    "DatetimeEnd",
    "Validity",
    "Verification",
]

# EEA renamed several columns when it moved to the Parquet download service.
# Map the new names back onto the schema CLAUDE.md specifies.
_COLUMN_ALIASES: dict[str, str] = {
    "Samplingpoint": "SamplingPoint",
    "Pollutant": "AirPollutantCode",
    "Value": "Concentration",
    "Unit": "UnitOfMeasurement",
    "Start": "DatetimeBegin",
    "End": "DatetimeEnd",
    "Validity": "Validity",
    "Verification": "Verification",
    "AggType": "AveragingTime",
    "station_code": "AirQualityStationEoICode",
    "pollutant": "AirPollutant",
    "value": "Concentration",
    "datetime_begin": "DatetimeBegin",
    "datetime_end": "DatetimeEnd",
    "validity": "Validity",
    "verification": "Verification",
}

# CLAUDE.md §3.1 pollutant code table, used to recover AirPollutant when a
# source only carries the numeric/URI code.
POLLUTANT_BY_CODE: dict[int, str] = {
    1: "SO2", 5: "PM10", 6001: "PM2.5", 7: "O3", 8: "NO2", 10: "CO",
}


# ---------------------------------------------------------------------------
# Route 1 — airbase
# ---------------------------------------------------------------------------
def fetch_via_airbase(dest: Path, cfg: dict) -> Path | None:
    """Download via the ``airbase`` client. Returns the directory, or None."""
    try:
        import airbase  # type: ignore
    except ImportError:
        log.warning(
            "airbase not installed — skipping route 1. "
            "Install with: pip install -r requirements-optional.txt"
        )
        return None

    eea = cfg["eea"]
    if eea["dataset"] != "E1a":
        raise ValueError(
            f"configs/data.yaml eea.dataset is {eea['dataset']!r}; LOCKED to 'E1a' "
            "(CLAUDE.md §3.1 — E2a flags are provisional and must never be used)"
        )

    dest.mkdir(parents=True, exist_ok=True)
    log.info(
        "airbase: requesting country=%s pollutants=%s years=%s-%s (E1a, verified)",
        eea["country"], eea["pollutants"], eea["year_from"], eea["year_to"],
    )

    # airbase's public API changed between 0.x and 1.x. Try the documented 0.x
    # call first (that is what CLAUDE.md §3.1 records), then the 1.x entry point.
    try:
        client = airbase.AirbaseClient()
        request = client.request(
            country=[eea["country"]],
            pl=list(eea["pollutants"]),
            year_from=int(eea["year_from"]),
            year_to=int(eea["year_to"]),
            dir=str(dest),
            verbose=True,
        )
        request.download_to_directory(str(dest), skip_existing=True)
        return dest
    except (AttributeError, TypeError) as exc:
        log.info("airbase 0.x API unavailable (%s); trying the 1.x entry point", exc)

    try:
        airbase.download(  # type: ignore[attr-defined]
            airbase.Dataset.Historical,  # type: ignore[attr-defined]
            dest,
            countries=[eea["country"]],
            pollutants=set(eea["pollutants"]),
            overwrite=False,
        )
        return dest
    except Exception as exc:  # noqa: BLE001 — report and fall through to route 2
        log.error("airbase download failed: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Route 2 — read whatever is on disk
# ---------------------------------------------------------------------------
def read_local_tree(root: Path, station_ids: set[str] | None = None) -> pd.DataFrame:
    """Read every .csv/.parquet (including inside .zip) under ``root``.

    Filtering to ``station_ids`` happens per file so a full-Spain download never
    has to be held in memory at once.
    """
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"no such directory: {root}")

    files = [
        p for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in {".csv", ".parquet", ".zip"}
    ]
    if not files:
        raise FileNotFoundError(
            f"{root} contains no .csv/.parquet/.zip files. "
            "Download EEA data into it first (see this module's docstring)."
        )

    log.info("reading %d file(s) under %s", len(files), root)
    frames: list[pd.DataFrame] = []
    skipped = 0
    for path in files:
        try:
            for chunk in _read_any(path):
                chunk = _normalise_columns(chunk)
                if station_ids is not None and "AirQualityStationEoICode" in chunk:
                    chunk = chunk[chunk["AirQualityStationEoICode"].isin(station_ids)]
                if not chunk.empty:
                    frames.append(chunk)
        except Exception as exc:  # noqa: BLE001 — one bad file must not kill the run
            skipped += 1
            log.warning("skipping %s: %s", path.name, exc)

    if skipped:
        log.warning("%d/%d file(s) unreadable and skipped", skipped, len(files))
    if not frames:
        raise ValueError(
            f"read {len(files)} file(s) under {root} but none contained rows for the "
            f"configured stations. Check the EoI codes in configs/data.yaml — run "
            f"`python -m src.data.station_metadata --refresh --print`."
        )

    df = pd.concat(frames, ignore_index=True)
    log.info("loaded %s rows across %d station(s)",
             f"{len(df):,}", df["AirQualityStationEoICode"].nunique())
    return df


def _read_any(path: Path):
    """Yield DataFrames from a csv / parquet / zip-of-those."""
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        yield pd.read_parquet(path)
    elif suffix == ".csv":
        yield pd.read_csv(path, low_memory=False)
    elif suffix == ".zip":
        with zipfile.ZipFile(path) as zf:
            for member in zf.namelist():
                if member.lower().endswith(".csv"):
                    with zf.open(member) as fh:
                        yield pd.read_csv(fh, low_memory=False)
                elif member.lower().endswith(".parquet"):
                    with zf.open(member) as fh:
                        yield pd.read_parquet(fh)


def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename known aliases onto the CLAUDE.md §3.1 schema and derive what we can."""
    df = df.rename(columns={k: v for k, v in _COLUMN_ALIASES.items() if k in df.columns})

    # Recover AirPollutant from AirPollutantCode when only the code is present.
    if "AirPollutant" not in df.columns and "AirPollutantCode" in df.columns:
        code = (
            df["AirPollutantCode"].astype(str)
            .str.extract(r"(\d+)$", expand=False)
            .astype("Float64")
        )
        df["AirPollutant"] = code.map(
            lambda c: POLLUTANT_BY_CODE.get(int(c)) if pd.notna(c) else None
        )

    if "AveragingTime" not in df.columns:
        df["AveragingTime"] = "hour"
    if "Countrycode" not in df.columns and "AirQualityStationEoICode" in df.columns:
        df["Countrycode"] = df["AirQualityStationEoICode"].astype(str).str[:2]
    if "Verification" not in df.columns:
        # Absent in some exports. 3 = "not verified" is the conservative default:
        # §3.1 says consider excluding those rows from EVALUATION.
        df["Verification"] = 3

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"missing required column(s): {missing}")
    return df[REQUIRED_COLUMNS + [c for c in ("AirPollutantCode",) if c in df.columns]]


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def fetch(from_dir: str | None = None, demo: bool = False,
          use_airbase: bool = True) -> pd.DataFrame:
    """Return raw EEA-schema rows for the configured node set."""
    cfg = load_config("data")
    paths = data_paths()

    if demo:
        from src.data.demo_source import generate_demo_raw
        return generate_demo_raw(cfg)

    station_ids = {s.station_id for s in configured_stations()}

    if from_dir is not None:
        return read_local_tree(Path(from_dir), station_ids)

    raw_dir = paths["raw"] / "eea"
    if use_airbase and not any(raw_dir.rglob("*")):
        fetch_via_airbase(raw_dir, cfg)

    return read_local_tree(raw_dir, station_ids)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--from-dir", default=None,
                    help="read an already-downloaded EEA tree instead of fetching")
    ap.add_argument("--demo", action="store_true",
                    help="generate a SIMULATED station set (region='demo'); "
                         "never a source of reportable results")
    ap.add_argument("--no-airbase", action="store_true",
                    help="do not attempt the airbase download")
    ap.add_argument("--out", default=None,
                    help="parquet path to write (default: data/raw/eea_raw.parquet)")
    args = ap.parse_args()

    if not args.demo:
        try:
            download_registry(force=False)
            verify_configured_stations()
        except Exception as exc:  # noqa: BLE001 — offline is survivable here
            log.warning("could not verify station codes against the EEA registry: %s", exc)

    df = fetch(from_dir=args.from_dir, demo=args.demo, use_airbase=not args.no_airbase)
    out = Path(args.out) if args.out else data_paths()["raw"] / "eea_raw.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    log.info("wrote %s rows -> %s", f"{len(df):,}", out)


if __name__ == "__main__":
    main()
