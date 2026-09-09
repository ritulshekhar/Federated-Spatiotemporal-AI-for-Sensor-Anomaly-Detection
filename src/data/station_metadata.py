"""station_id -> (region, station_class, coordinates) lookup.

Two sources, in priority order:

1. ``data/raw/eea_station_metadata.csv`` — the authoritative EEA station
   registry, downloaded on demand. Use this.
2. ``configs/data.yaml`` ``stations:`` — a hand-written starting list whose EoI
   codes are explicitly marked UNVERIFIED in that file.

The registry download exists precisely so the hardcoded codes never silently
become "the truth". ``verify_configured_stations()`` diffs the two and is called
by the fetch entry point.

Note on coordinates: the server needs station lat/lon to build distance-based
graph edges (§7.2, Wang et al. 2020). These come from this *public registry*,
not from a node upload — see ``configs/federation.yaml`` privacy.never_transmitted.
"""

from __future__ import annotations

import argparse
import io
from dataclasses import dataclass, asdict
from pathlib import Path

import pandas as pd

from src.utils.logging_setup import get_logger
from src.utils.paths import data_paths, load_config

log = get_logger(__name__)

# EEA's pan-European station metadata export. Public, no key.
EEA_METADATA_URL = (
    "https://discomap.eea.europa.eu/map/fme/metadata/PanEuropean_metadata.csv"
)
_LOCAL_METADATA = "eea_station_metadata.csv"

# EEA "AirQualityStationArea" x "AirQualityStationType" -> our station_class
# vocabulary. Kept explicit rather than string-concatenated so an unexpected
# combination surfaces as "unknown" instead of inventing a class.
_CLASS_MAP: dict[tuple[str, str], str] = {
    ("urban", "traffic"): "urban_traffic",
    ("urban", "background"): "urban_background",
    ("urban", "industrial"): "urban_industrial",
    ("suburban", "traffic"): "suburban_traffic",
    ("suburban", "background"): "suburban_background",
    ("suburban", "industrial"): "suburban_industrial",
    ("rural", "background"): "rural_background",
    ("rural-nearcity", "background"): "rural_background",
    ("rural-regional", "background"): "rural_background",
    ("rural-remote", "background"): "rural_background",
    ("rural", "traffic"): "rural_traffic",
    ("rural", "industrial"): "rural_industrial",
}


@dataclass(frozen=True)
class Station:
    station_id: str          # AirQualityStationEoICode, e.g. "ES1438A"
    name: str
    region: str
    station_class: str
    latitude: float | None = None
    longitude: float | None = None
    altitude_m: float | None = None

    def as_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# EEA registry
# ---------------------------------------------------------------------------
def registry_path() -> Path:
    return data_paths()["raw"] / _LOCAL_METADATA


def download_registry(force: bool = False, timeout: int = 120) -> Path:
    """Fetch EEA's PanEuropean_metadata.csv into data/raw (cached)."""
    dest = registry_path()
    if dest.exists() and not force:
        log.info("station registry already present: %s", dest)
        return dest

    import requests

    log.info("downloading EEA station registry from %s", EEA_METADATA_URL)
    resp = requests.get(EEA_METADATA_URL, timeout=timeout)
    resp.raise_for_status()
    dest.write_bytes(resp.content)
    log.info("wrote %s (%.1f MB)", dest, len(resp.content) / 1e6)
    return dest


def load_registry(country: str = "ES") -> pd.DataFrame:
    """Load the cached registry, filtered to one country. Raises if absent."""
    path = registry_path()
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run: python -m src.data.station_metadata --refresh"
        )
    raw = path.read_bytes()
    # EEA ships this as tab-separated latin-1 in some vintages and comma-separated
    # utf-8 in others. Sniff rather than guess.
    for encoding in ("utf-8-sig", "latin-1"):
        for sep in ("\t", ","):
            try:
                df = pd.read_csv(io.BytesIO(raw), sep=sep, encoding=encoding,
                                 low_memory=False)
            except Exception:
                continue
            if "AirQualityStationEoICode" in df.columns:
                return df[df.get("Countrycode", country) == country].copy()
    raise ValueError(f"could not parse {path} as the EEA station registry")


def stations_in_city(city: str = "Barcelona", country: str = "ES") -> pd.DataFrame:
    """All registry rows whose municipality matches ``city`` (case-insensitive)."""
    df = load_registry(country=country)
    muni_col = next(
        (c for c in ("AirQualityStationNatCode", "Municipality", "City")
         if c in df.columns),
        None,
    )
    name_col = "AirQualityStationName" if "AirQualityStationName" in df.columns else None

    mask = pd.Series(False, index=df.index)
    for col in (muni_col, name_col):
        if col:
            mask |= df[col].astype(str).str.contains(city, case=False, na=False)
    keep = [
        c for c in (
            "AirQualityStationEoICode", "AirQualityStationName", "Municipality",
            "AirQualityStationArea", "AirQualityStationType",
            "Longitude", "Latitude", "Altitude", "AirPollutant",
        )
        if c in df.columns
    ]
    out = df.loc[mask, keep].drop_duplicates(subset=["AirQualityStationEoICode"])
    return out.sort_values("AirQualityStationEoICode").reset_index(drop=True)


def classify(area: object, stype: object) -> str:
    key = (str(area).strip().lower(), str(stype).strip().lower())
    return _CLASS_MAP.get(key, "unknown")


# ---------------------------------------------------------------------------
# The configured node set
# ---------------------------------------------------------------------------
def configured_stations() -> list[Station]:
    """The node set from configs/data.yaml, enriched from the registry if present."""
    cfg = load_config("data")
    region = cfg["eea"]["region"]
    entries = cfg["stations"]

    enrich: dict[str, dict] = {}
    try:
        reg = load_registry(country=cfg["eea"]["country"])
        reg = reg.drop_duplicates(subset=["AirQualityStationEoICode"])
        for _, row in reg.iterrows():
            enrich[str(row["AirQualityStationEoICode"])] = {
                "latitude": _as_float(row.get("Latitude")),
                "longitude": _as_float(row.get("Longitude")),
                "altitude_m": _as_float(row.get("Altitude")),
                "registry_class": classify(
                    row.get("AirQualityStationArea"), row.get("AirQualityStationType")
                ),
                "registry_name": row.get("AirQualityStationName"),
            }
    except FileNotFoundError:
        log.warning(
            "station registry not downloaded — coordinates unavailable and the "
            "configured EoI codes are UNVERIFIED. Run "
            "`python -m src.data.station_metadata --refresh` before trusting results."
        )

    out: list[Station] = []
    for e in entries:
        sid = e["eoi_code"]
        extra = enrich.get(sid, {})
        reg_class = extra.get("registry_class")
        if reg_class and reg_class != "unknown" and reg_class != e["station_class"]:
            log.warning(
                "station_class mismatch for %s: config says %r, EEA registry says %r "
                "— trusting the registry",
                sid, e["station_class"], reg_class,
            )
        out.append(
            Station(
                station_id=sid,
                name=e["name"],
                region=region,
                station_class=reg_class if reg_class and reg_class != "unknown"
                else e["station_class"],
                latitude=extra.get("latitude"),
                longitude=extra.get("longitude"),
                altitude_m=extra.get("altitude_m"),
            )
        )
    return out


def station_class_lookup() -> dict[str, str]:
    """``station_id -> station_class``, the dict harmonize.py expects (§5.1)."""
    return {s.station_id: s.station_class for s in configured_stations()}


def region_lookup() -> dict[str, str]:
    return {s.station_id: s.region for s in configured_stations()}


def coordinates() -> dict[str, tuple[float, float]]:
    """``station_id -> (lat, lon)`` for stations whose coordinates are known."""
    return {
        s.station_id: (s.latitude, s.longitude)
        for s in configured_stations()
        if s.latitude is not None and s.longitude is not None
    }


def verify_configured_stations() -> dict[str, list[str]]:
    """Diff configs/data.yaml against the EEA registry.

    Returns ``{"present": [...], "missing": [...]}``. ``missing`` codes do not
    exist in the registry at all — almost certainly transcription errors in the
    UNVERIFIED block of configs/data.yaml.
    """
    cfg = load_config("data")
    wanted = [e["eoi_code"] for e in cfg["stations"]]
    try:
        known = set(load_registry(country=cfg["eea"]["country"])
                    ["AirQualityStationEoICode"].astype(str))
    except FileNotFoundError:
        return {"present": [], "missing": [], "unchecked": wanted}

    present = [s for s in wanted if s in known]
    missing = [s for s in wanted if s not in known]
    if missing:
        log.error(
            "%d configured EoI code(s) are NOT in the EEA registry: %s. "
            "Fix configs/data.yaml — see stations_in_city('Barcelona') for candidates.",
            len(missing), ", ".join(missing),
        )
    return {"present": present, "missing": missing, "unchecked": []}


def _as_float(v: object) -> float | None:
    try:
        f = float(v)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return None if pd.isna(f) else f


# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--refresh", action="store_true",
                    help="download the EEA station registry into data/raw")
    ap.add_argument("--city", default="Barcelona",
                    help="list registry stations matching this municipality")
    ap.add_argument("--print", dest="do_print", action="store_true",
                    help="print candidate stations and verify the configured node set")
    args = ap.parse_args()

    if args.refresh:
        download_registry(force=True)

    if args.do_print or args.refresh:
        try:
            cand = stations_in_city(args.city)
        except FileNotFoundError as exc:
            log.error("%s", exc)
            return
        print(f"\n=== EEA registry stations matching {args.city!r} "
              f"({len(cand)} unique) ===")
        with pd.option_context("display.max_rows", 200, "display.width", 200):
            print(cand.to_string(index=False))

        status = verify_configured_stations()
        print("\n=== configs/data.yaml node set ===")
        print(f"  present in registry : {len(status['present'])} "
              f"-> {', '.join(status['present']) or '(none)'}")
        print(f"  NOT in registry     : {len(status['missing'])} "
              f"-> {', '.join(status['missing']) or '(none)'}")
        if status["missing"]:
            print("\n  ^ correct configs/data.yaml before running the pipeline.")


if __name__ == "__main__":
    main()
