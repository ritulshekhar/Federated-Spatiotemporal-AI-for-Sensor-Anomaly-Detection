"""Map each source into the common schema (CLAUDE.md §5.1).

Target schema — every downstream module depends on exactly these columns:

    station_id | region | timestamp_utc | pollutant | concentration |
    unit | label | label_source | station_class

plus two EEA-only passthroughs the evaluation layer needs:

    validity | verification

``region`` and ``label_source`` are never dropped downstream (§5.3). They are
what makes "recall on real EEA faults" and "recall on synthetic faults" two
honestly-separate numbers instead of one that overstates confidence.

Timezone handling happens *here*, not later (§5.2). EEA timestamps are CET/CEST;
CPCB's are IST. Deferring the conversion makes "the same hour" mean different
real moments in different regions and silently corrupts every cross-station
alignment.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.data.station_metadata import station_class_lookup
from src.utils.logging_setup import get_logger
from src.utils.paths import data_paths, load_config

log = get_logger(__name__)

SCHEMA = [
    "station_id", "region", "timestamp_utc", "pollutant", "concentration",
    "unit", "label", "label_source", "station_class",
]
EEA_EXTRA = ["validity", "verification"]

# CLAUDE.md §3.1 — the single most important two lines in the project.
FAULT_VALIDITY_CODES = (-1, -99)
VALID_VALIDITY_CODES = (1, 2, 3, 4)

# CLAUDE.md §3.1 unit trap: everything is ug/m3 except CO, which is mg/m3.
CANONICAL_UNITS = {
    "SO2": "ug/m3", "PM10": "ug/m3", "PM2.5": "ug/m3",
    "O3": "ug/m3", "NO2": "ug/m3", "CO": "mg/m3",
}
_UNIT_ALIASES = {
    "ug/m3": "ug/m3", "µg/m3": "ug/m3", "ug.m-3": "ug/m3", "µg.m-3": "ug/m3",
    "microgram per cubic metre": "ug/m3", "ug/m³": "ug/m3", "µg/m³": "ug/m3",
    "mg/m3": "mg/m3", "mg.m-3": "mg/m3", "mg/m³": "mg/m3",
    "milligram per cubic metre": "mg/m3",
}


def label_from_validity(v: int) -> str:
    """CLAUDE.md §3.1, verbatim. The primary evaluation ground truth (§2.1)."""
    return "fault" if v in FAULT_VALIDITY_CODES else "trustworthy"


# ---------------------------------------------------------------------------
def harmonize_eea(
    df: pd.DataFrame,
    station_class_map: dict[str, str] | None = None,
    region: str = "catalonia",
    label_source: str = "real",
) -> pd.DataFrame:
    """EEA raw schema -> common schema.

    ``label_source`` is a parameter rather than a constant only so the simulated
    demo source can pass ``"simulated"``. For genuine EEA downloads it is
    ``"real"`` and nothing else.
    """
    if station_class_map is None:
        station_class_map = station_class_lookup()

    ts = pd.to_datetime(df["DatetimeBegin"], utc=True, errors="coerce", format="mixed")
    n_bad_ts = int(ts.isna().sum())
    if n_bad_ts:
        log.warning("dropping %d row(s) with unparseable DatetimeBegin", n_bad_ts)

    validity = pd.to_numeric(df["Validity"], errors="coerce").astype("Int64")
    station_id = df["AirQualityStationEoICode"].astype(str)

    out = pd.DataFrame({
        "station_id": station_id,
        "region": region,
        "timestamp_utc": ts,
        "pollutant": df["AirPollutant"].astype(str).str.upper().str.replace(" ", ""),
        "concentration": pd.to_numeric(df["Concentration"], errors="coerce"),
        "unit": df["UnitOfMeasurement"].astype(str),
        # Rows whose Validity we could not parse get no label at all rather
        # than a defaulted "trustworthy" — an unparseable flag is missing
        # ground truth, not evidence of validity.
        "label": validity.map(
            lambda v: label_from_validity(int(v)) if pd.notna(v) else None
        ),
        "label_source": label_source,
        "station_class": station_id.map(station_class_map).fillna("unknown"),
        "validity": validity,
        "verification": pd.to_numeric(
            df.get("Verification", 3), errors="coerce"
        ).astype("Int64"),
    })

    out = out[ts.notna()].copy()
    out = _canonicalise_units(out)
    _report_unknown_validity(out)
    return _finalise(out)


def harmonize_cpcb(
    df: pd.DataFrame,
    station_class_map: dict[str, str] | None = None,
) -> pd.DataFrame:
    """CPCB raw schema -> common schema. Phase 3 only (CLAUDE.md §3.2, §5.1).

    !! UNVERIFIED !! CLAUDE.md §11 records this as an open risk: CPCB's export
    column names, units, and whether ``pollutant_avg`` is hourly or some other
    aggregation have NOT been checked against a real downloaded sample. Confirm
    before trusting this mapping.

    ``label`` is left None here on purpose. CPCB has no Validity equivalent, so
    every fault label on this data is synthetic and gets attached later by the
    rate-matched injector in ``src/preprocessing/fault_labels.py`` (§5.5) —
    never by inventing one at harmonization time.
    """
    if station_class_map is None:
        station_class_map = {}

    raise NotImplementedError(
        "harmonize_cpcb is a Phase 3 stretch goal and its column mapping is "
        "unverified against a real CPCB export (CLAUDE.md §11). Do not enable "
        "cpcb in configs/data.yaml until a real sample has been inspected. "
        "The intended mapping is documented in CLAUDE.md §5.1."
    )


# ---------------------------------------------------------------------------
def _canonicalise_units(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise unit strings and refuse to silently mix mg/m3 with ug/m3."""
    df = df.copy()
    df["unit"] = (
        df["unit"].astype(str).str.strip().str.lower()
        .map(lambda u: _UNIT_ALIASES.get(u, u))
    )
    expected = df["pollutant"].map(CANONICAL_UNITS)
    mismatch = expected.notna() & (df["unit"] != expected)
    if mismatch.any():
        pairs = (
            df.loc[mismatch, ["pollutant", "unit"]]
            .drop_duplicates().to_dict("records")
        )
        raise ValueError(
            "unit mismatch against the CLAUDE.md §3.1 table: "
            f"{pairs}. CO is mg/m3 and everything else is ug/m3 — this is the "
            "documented unit trap; resolve it explicitly rather than coercing."
        )
    return df


def _report_unknown_validity(df: pd.DataFrame) -> None:
    """Log any Validity code outside the documented vocabulary."""
    known = set(VALID_VALIDITY_CODES) | set(FAULT_VALIDITY_CODES)
    seen = set(df["validity"].dropna().astype(int).unique())
    unknown = seen - known
    if unknown:
        log.warning(
            "Validity codes outside the documented vocabulary: %s. They are "
            "treated as 'trustworthy' by label_from_validity — check "
            "dd.eionet.europa.eu/vocabulary/aq/observationvalidity before "
            "reporting anything that depends on them.",
            sorted(unknown),
        )


def _finalise(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(subset=["timestamp_utc", "pollutant"])
    # Exact duplicates happen where EEA re-submits a year. Keep the last.
    before = len(df)
    df = df.drop_duplicates(
        subset=["station_id", "pollutant", "timestamp_utc"], keep="last"
    )
    if before != len(df):
        log.info("dropped %s duplicate (station, pollutant, timestamp) row(s)",
                 f"{before - len(df):,}")
    df = df.sort_values(["station_id", "pollutant", "timestamp_utc"])
    return df[SCHEMA + EEA_EXTRA].reset_index(drop=True)


def real_fault_rate(df: pd.DataFrame) -> float:
    """Measured fault prevalence on real-labelled rows (CLAUDE.md §2.7 / §5.5).

    This is the number synthetic injection must match. Computed from
    ``label_source == "real"`` rows only — matching a rate that was itself
    partly synthetic would defeat the purpose.
    """
    real = df[df["label_source"] == "real"]
    if real.empty:
        raise ValueError(
            "no rows with label_source == 'real'; cannot measure a real fault "
            "rate. Synthetic injection must not fall back to an arbitrary "
            "default (CLAUDE.md §2.7)."
        )
    return float((real["label"] == "fault").mean())


# ---------------------------------------------------------------------------
def write_harmonized(df: pd.DataFrame, region: str) -> Path:
    """Per-region parquet under data/harmonized (§4)."""
    out = data_paths()["harmonized"] / f"{region}.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    log.info("wrote %s rows -> %s", f"{len(df):,}", out)
    return out


def load_harmonized(region: str) -> pd.DataFrame:
    path = data_paths()["harmonized"] / f"{region}.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run: python -m src.data.harmonize [--demo]"
        )
    return pd.read_parquet(path)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--from-dir", default=None,
                    help="read an already-downloaded EEA tree")
    ap.add_argument("--raw-parquet", default=None,
                    help="read the parquet written by src.data.fetch_eea")
    ap.add_argument("--demo", action="store_true",
                    help="use the SIMULATED source (region='demo', "
                         "label_source='simulated')")
    args = ap.parse_args()

    cfg = load_config("data")

    if args.raw_parquet:
        raw = pd.read_parquet(args.raw_parquet)
        region, label_source, class_map = cfg["eea"]["region"], "real", station_class_lookup()
    elif args.demo:
        from src.data.demo_source import (
            DEMO_LABEL_SOURCE, DEMO_REGION, demo_station_table, generate_demo_raw,
        )
        raw = generate_demo_raw(cfg)
        region, label_source = DEMO_REGION, DEMO_LABEL_SOURCE
        class_map = dict(
            zip(demo_station_table()["eoi_code"], demo_station_table()["station_class"])
        )
    else:
        from src.data.fetch_eea import fetch
        raw = fetch(from_dir=args.from_dir)
        region, label_source, class_map = cfg["eea"]["region"], "real", station_class_lookup()

    df = harmonize_eea(raw, class_map, region=region, label_source=label_source)
    write_harmonized(df, region)

    summary = (
        df.groupby(["region", "station_id", "station_class", "label_source"])
        .agg(rows=("concentration", "size"),
             fault_rate=("label", lambda s: (s == "fault").mean()),
             first=("timestamp_utc", "min"), last=("timestamp_utc", "max"))
        .reset_index()
    )
    print("\n=== harmonized summary (never pooled — §8.3) ===")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
