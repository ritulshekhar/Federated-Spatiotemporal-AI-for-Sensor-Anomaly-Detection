"""Real sensor error, from low-cost sensors co-located with reference instruments.

Everything in this project's Objective 4 evaluation rests on *synthetic* faults,
because EEA destroys the reading whenever it flags one (§2.1a, CONTEXT.md entry
4). §2.1a names the fix: co-located reference-vs-low-cost-sensor data, as
Ferrer-Cid et al. (2022) used in Barcelona. This module builds it.

**Source.** The H2020 CAPTOR project (Zenodo 2564825, CC-BY-4.0) deployed MOX
ozone sensors across Catalonia in summer 2018, several of them deliberately
placed *at* official monitoring stations for calibration. Pairing node
coordinates against the EEA registry gives five genuine co-locations, four of
them within 150 metres:

    captor-18016  Ciutadella AQMS  ->  ES1679A   0.00 km   (already a node here)
    captor-18013  Manlleu AQMS     ->  ES1397A   0.02 km
    captor-18022  Tona AQMS        ->  ES1923A   0.08 km
    captor-18009  Vic AQMS         ->  ES1642A   0.13 km
    captor-18018  Badalona         ->  ES1892A   1.84 km

**What it gives that synthetic injection cannot.** The difference between the
low-cost sensor and the reference instrument at the same place and hour is a
*real* sensor error: real drift, real calibration offset, real noise, in a
reading that looks perfectly plausible. That is exactly the regime EEA's
`Validity` flags cannot describe, and the only way to check whether the
injected fault model resembles reality.

**Two cautions, both structural.** The error is continuous, not a binary flag,
so turning it into "fault" or "not fault" needs a stated threshold rather than
one inherited from anywhere. And a low-cost MOX sensor is a *different
instrument class* from the reference-grade analysers this project otherwise
models — its errors are larger and more frequent than a reference station's
would be. Treat it as evidence about fault *shape*, not about fault *rate*.
"""

from __future__ import annotations

import argparse
import glob
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.station_metadata import load_registry
from src.utils.logging_setup import get_logger
from src.utils.paths import data_paths

log = get_logger(__name__)

ZENODO_RECORD = "2564825"
ZENODO_API = f"https://zenodo.org/api/records/{ZENODO_RECORD}"

#: Verified by pairing CAPTOR node coordinates against the EEA registry.
COLOCATED: dict[int, tuple[str, str, float]] = {
    16: ("ES1679A", "Ciutadella AQMS", 0.00),
    13: ("ES1397A", "Manlleu AQMS", 0.02),
    22: ("ES1923A", "Tona AQMS", 0.08),
    9: ("ES1642A", "Vic AQMS", 0.13),
    18: ("ES1892A", "Badalona", 1.84),
}

#: CAPTOR timestamps are local Spanish civil time; summer 2018 is CEST.
CAPTOR_TZ = "Europe/Madrid"


def download(dest: Path | None = None) -> Path:
    """Fetch and unpack the CAPTOR 2018 release."""
    import requests

    dest = dest or (data_paths()["raw"] / "colocated")
    dest.mkdir(parents=True, exist_ok=True)
    meta = requests.get(ZENODO_API, timeout=60).json()
    entry = meta["files"][0]
    archive = dest / entry["key"]

    if not archive.exists():
        log.info("downloading %s (%.1f MB)", entry["key"], entry["size"] / 1e6)
        archive.write_bytes(requests.get(entry["links"]["self"], timeout=600).content)
    out = dest / "final2018"
    if not out.exists():
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(out)
    return out


def load_sensor(node_id: int, root: Path | None = None) -> pd.Series:
    """One CAPTOR node's ozone series, hourly means, indexed in UTC."""
    root = root or (data_paths()["raw"] / "colocated" / "final2018")
    matches = glob.glob(str(root / "**" / f"DataSet_Spain_*Node-ID{node_id:02d}_2018.csv"),
                        recursive=True)
    if not matches:
        raise FileNotFoundError(f"no CAPTOR file for node {node_id}")

    # latin-1: the unit header carries a literal micro sign.
    df = pd.read_csv(matches[0], sep=None, engine="python", encoding="latin-1")
    value_col = next(c for c in df.columns if "zone" in c.lower())

    local = pd.to_datetime(df["Start_local"], format="%d/%m/%Y %H:%M")
    # Ambiguous/nonexistent wall-clock times occur at the DST change; the
    # deployment is summer-only so this is defensive rather than load-bearing.
    utc = (local.dt.tz_localize(CAPTOR_TZ, ambiguous="NaT", nonexistent="NaT")
           .dt.tz_convert("UTC"))

    s = pd.Series(pd.to_numeric(df[value_col], errors="coerce").to_numpy(),
                  index=pd.DatetimeIndex(utc), name="sensor").dropna()
    # Half-hourly -> hourly mean, matching EEA's averaging period.
    return s.resample("1h").mean().dropna()


def load_reference(station_id: str) -> pd.Series:
    """EEA reference O3 for one station, hourly, UTC, valid readings only."""
    reg = load_registry("ES")
    reg["pol"] = reg["AirPollutantCode"].astype(str).str.extract(r"(\d+)$").astype("Int64")
    points = reg[(reg["pol"] == 7)
                 & (reg["AirQualityStationEoICode"] == station_id)]["SamplingPoint"]
    wanted = {str(p) for p in points}

    frames = []
    for path in glob.glob("data/raw/eea_o3/**/*.parquet", recursive=True):
        if Path(path).stem not in wanted:
            continue
        frames.append(pd.read_parquet(path))
    if not frames:
        raise FileNotFoundError(f"no downloaded O3 parquet for {station_id}")

    df = pd.concat(frames, ignore_index=True)
    df = df[df["Validity"] > 0]            # the sentinel rows carry no reading
    ts = pd.to_datetime(df["Start"], errors="coerce").dt.tz_localize("UTC")
    s = pd.Series(pd.to_numeric(df["Value"], errors="coerce").to_numpy(),
                  index=pd.DatetimeIndex(ts), name="reference").dropna()
    return s[~s.index.duplicated(keep="last")].sort_index()


def build_pairs() -> pd.DataFrame:
    """Sensor, reference and their difference for every co-located pair."""
    rows = []
    for node_id, (station_id, location, km) in COLOCATED.items():
        try:
            sensor = load_sensor(node_id)
            reference = load_reference(station_id)
        except FileNotFoundError as exc:
            log.warning("skipping node %d: %s", node_id, exc)
            continue

        joined = pd.concat([sensor, reference], axis=1, join="inner").dropna()
        if joined.empty:
            log.warning("node %d / %s: no overlapping hours", node_id, station_id)
            continue

        joined["error"] = joined["sensor"] - joined["reference"]
        joined["node_id"] = node_id
        joined["station_id"] = station_id
        joined["location"] = location
        joined["distance_km"] = km
        rows.append(joined.reset_index(names="timestamp_utc"))
        log.info("node %d (%s) vs %s: %s overlapping hours, mean error %+.1f ug/m3",
                 node_id, location, station_id, f"{len(joined):,}",
                 joined["error"].mean())

    if not rows:
        raise RuntimeError("no co-located pairs could be built")
    return pd.concat(rows, ignore_index=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--download", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.download:
        download()

    pairs = build_pairs()
    out = Path(args.out) if args.out else (data_paths()["harmonized"] / "colocated_o3.parquet")
    out.parent.mkdir(parents=True, exist_ok=True)
    pairs.to_parquet(out, index=False)
    log.info("wrote %s rows -> %s", f"{len(pairs):,}", out)

    print("\n=== real low-cost sensor error vs co-located reference ===")
    summary = pairs.groupby(["node_id", "location", "station_id"]).agg(
        hours=("error", "size"), mean_error=("error", "mean"),
        sd_error=("error", "std"), p05=("error", lambda s: s.quantile(0.05)),
        p95=("error", lambda s: s.quantile(0.95)),
        reference_mean=("reference", "mean"),
    )
    print(summary.round(2).to_string())


if __name__ == "__main__":
    main()
