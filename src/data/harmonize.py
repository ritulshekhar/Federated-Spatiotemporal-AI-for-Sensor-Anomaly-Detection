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

# ---------------------------------------------------------------------------
# No-data sentinel
# ---------------------------------------------------------------------------
# MEASURED 2026-09-16, and it contradicts CLAUDE.md §3.1.
#
# §3.1 states: "Concentration is present even when Validity is negative — this
# is what makes real fault labels possible." In the Catalonia E1a delivery that
# is **false**. Every single Validity=-1 row, across all five pollutants
# (15,263 NO2 / 7,205 SO2 / 5,109 O3 / 103 PM10 / 103 PM2.5), carries exactly
# -999.0. Not one carries a real reading.
#
# Consequences, both handled here:
#   1. Left in place, -999 is not a concentration but a catastrophic outlier.
#      It dragged the NO2 scaler to mean -1.41 / std 180.03, against a true
#      30.60 / 22.12, corrupting every normalized feature, lag and rolling
#      statistic downstream.
#   2. It makes "fault detection" against these labels trivial and vacuous —
#      `value == -999` scores perfectly — while the research problem, a
#      plausible-looking but wrong reading, is not represented at all.
#
# So the sentinel is converted to missing here, which routes it into §6.2's
# imputation and marks it is_imputed, keeping it out of anomaly scoring. The
# Validity-derived *label* is preserved regardless: the row really was flagged
# faulty, we simply have no reading to adjudicate. See CONTEXT.md entry 4.
SENTINEL_THRESHOLD = -99.0

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
    out = _mask_sentinels(out)
    _report_unknown_validity(out)
    return _finalise(out)


def harmonize_cpcb(
    df: pd.DataFrame,
    station_class_map: dict[str, str] | None = None,
    station_prefix: str = "IN-OAQ-",
    *,
    resample_hourly: bool = True,
    min_samples_per_hour: int = 2,
) -> pd.DataFrame:
    """CPCB (via OpenAQ) raw schema -> common schema. Phase 3 (§3.2, §5.1).

    Schema VERIFIED 2026-09-16 against a real OpenAQ archive object, which
    closes the §11 open risk for this route. Observed columns:

        location_id | sensors_id | location | datetime | lat | lon |
        parameter | units | value

    Two differences from the sketch in CLAUDE.md §5.1, which assumed the
    ``data.gov.in`` export:

    * ``datetime`` is already ISO-8601 **with a UTC offset**, so it converts
      directly. The §5.1 sketch localises a naive timestamp to Asia/Kolkata
      first; doing that here would shift every reading by +5:30.
    * ``parameter`` is a lowercase name (``pm25``, ``no2``), not a numeric id.

    ``label`` is None on purpose and this is LOCKED (§2.6): CPCB has no
    Validity equivalent, so every fault label on this data is synthetic and is
    attached later by the rate-matched injector in
    ``src/preprocessing/fault_labels.py``, at the rate measured from real EEA
    data (§5.5) — never invented at harmonization time.
    """
    from src.data.fetch_cpcb import PARAMETER_MAP, REQUIRED_ARCHIVE_COLUMNS

    if station_class_map is None:
        station_class_map = {}

    missing = [c for c in REQUIRED_ARCHIVE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"OpenAQ archive schema drifted: missing {missing}, got "
            f"{list(df.columns)}. Re-run "
            "`python -m src.data.fetch_cpcb --inspect <location_id>` and update "
            "the mapping and REQUIRED_ARCHIVE_COLUMNS together (§11)."
        )

    ts = pd.to_datetime(df["datetime"], utc=True, errors="coerce", format="mixed")
    pollutant = df["parameter"].astype(str).str.strip().str.lower().map(PARAMETER_MAP)

    unmapped = sorted(set(df.loc[pollutant.isna(), "parameter"].astype(str).unique()))
    if unmapped:
        log.info("dropping %d OpenAQ parameter(s) outside this project's "
                 "pollutant set: %s", len(unmapped), unmapped[:8])

    station_id = station_prefix + df["location_id"].astype(str)

    out = pd.DataFrame({
        "station_id": station_id,
        "region": "india",
        "timestamp_utc": ts,
        "pollutant": pollutant,
        "concentration": pd.to_numeric(df["value"], errors="coerce"),
        "unit": df["units"].astype(str),
        # LOCKED (§2.6): no Validity equivalent exists, so there is no real
        # label to assign. Left missing rather than defaulted to "trustworthy".
        "label": None,
        "label_source": "synthetic",
        "station_class": station_id.map(station_class_map).fillna("unknown"),
        "validity": pd.Series(pd.NA, index=df.index, dtype="Int64"),
        "verification": pd.Series(pd.NA, index=df.index, dtype="Int64"),
    })

    out = out[ts.notna() & pollutant.notna()].copy()
    if out.empty:
        raise ValueError(
            "no CPCB rows survived harmonization — every timestamp or pollutant "
            "failed to parse. Inspect the raw export before continuing."
        )
    out = _convert_openaq_units(out)
    _check_basis_continuity(out)
    out = _canonicalise_units(out)
    if resample_hourly:
        out = _resample_cpcb_to_hourly(out, min_samples=min_samples_per_hour)
    elif "unit_basis" in out.columns:
        # Without the hourly grouping there is nowhere to resolve a station
        # that reports one gas through both a mass and a ppb sensor, so the
        # caller gets both rows and is told so rather than quietly handed a
        # frame with duplicate timestamps of differing provenance.
        if (out["unit_basis"] == _BASIS_CONVERTED).any():
            log.warning(
                "resample_hourly=False: mass and ppb-converted readings are "
                "both present and NOT de-duplicated. Resolve before any "
                "aggregation keyed on (station, pollutant, time)."
            )
        out = out.drop(columns=["unit_basis"])
    log.info("CPCB/OpenAQ: %s rows, %d station(s), pollutants %s",
             f"{len(out):,}", out["station_id"].nunique(),
             sorted(out["pollutant"].unique()))
    return _finalise(out)


def _resample_cpcb_to_hourly(df: pd.DataFrame,
                             min_samples: int = 2) -> pd.DataFrame:
    """Average OpenAQ's 15-minute samples into the hourly means EEA supplies.

    The archive is **quarter-hourly** — measured spacing is exactly 00:15:00
    across the Delhi stations — while every downstream stage assumes hourly.
    §6.1 builds an hourly index and checks contiguity, §6.4's lags are at t-1,
    t-24 and t-168 hours, and EEA's own rows are `AveragingTime = hour`. Feeding
    15-minute rows in would silently reindex four samples onto one hour and
    keep whichever landed last, discarding three quarters of the data and
    making India's effective sampling differ from Catalonia's.

    ``min_samples`` is why this is a mean rather than a resample-and-hope: an
    hour backed by a single 15-minute reading is noisier than one backed by
    four, and that extra noise would land in the residual the detector
    thresholds on. Requiring two is a compromise — it keeps hours with a
    dropped sample while rejecting hours that are really one instantaneous
    reading wearing an hour's clothes.
    """
    df = df.copy()
    df["timestamp_utc"] = df["timestamp_utc"].dt.floor("h")

    keys = ["station_id", "pollutant", "timestamp_utc"]

    # A station can report the same gas through a mass sensor AND a ppb one
    # (see `_convert_openaq_units`). Averaging across both would mix a measured
    # value with a temperature-assuming conversion of a different instrument's
    # reading, so keep only the best basis available for each hour and average
    # within it.
    if "unit_basis" in df.columns:
        best = df.groupby(keys, observed=True, dropna=False)["unit_basis"] \
                 .transform("min")
        mixed = int((df["unit_basis"] != best).sum())
        if mixed:
            log.info("dropping %s reading(s) from a lower-preference unit basis "
                     "where the same station-pollutant-hour also has a mass "
                     "sensor", f"{mixed:,}")
            df = df[df["unit_basis"] == best].copy()
        df = df.drop(columns=["unit_basis"])

    grouped = df.groupby(keys, observed=True, dropna=False)
    out = grouped.agg(
        concentration=("concentration", "mean"),
        n_samples=("concentration", "size"),
        unit=("unit", "first"),
        region=("region", "first"),
        label=("label", "first"),
        label_source=("label_source", "first"),
        station_class=("station_class", "first"),
        validity=("validity", "first"),
        verification=("verification", "first"),
    ).reset_index()

    thin = out["n_samples"] < min_samples
    if (n := int(thin.sum())):
        log.info("dropping %s hour(s) backed by fewer than %d sub-hourly "
                 "sample(s) of %s total", f"{n:,}", min_samples, f"{len(out):,}")
        out = out[~thin].copy()

    per_hour = float(out["n_samples"].mean()) if len(out) else float("nan")
    log.info("resampled %s quarter-hourly row(s) to %s hourly mean(s) "
             "(%.2f samples per hour)", f"{len(df):,}", f"{len(out):,}", per_hour)
    return out.drop(columns=["n_samples"])


#: ppb -> ug/m3 at 25 C and 1013.25 hPa, the convention EEA and US EPA report
#: against: ug/m3 = ppb * molar_mass / 24.45.
_PPB_TO_UGM3 = {
    "NO2": 46.0055 / 24.45,   # 1.8816
    "SO2": 64.066 / 24.45,    # 2.6203
    "O3": 48.00 / 24.45,      # 1.9632
}


#: Provenance of a reading's unit, and the preference order used when one
#: station reports the same pollutant through more than one sensor. Lower wins.
_BASIS_MASS, _BASIS_CONVERTED = 0, 1


def _check_basis_continuity(df: pd.DataFrame) -> None:
    """Refuse to let one station's series straddle a change of sensor basis.

    This is the failure that actually matters, and it is worth being precise
    about why, because the obvious worry is the wrong one.

    A *constant* unit error is nearly harmless here. §6.5 fits a z-score per
    station on that station's own training window, so multiplying a whole
    station-pollutant series by 1.88 is absorbed almost entirely by the
    scaler — it survives only in physical-unit reporting, not in the
    normalised residuals the detector thresholds on.

    A unit change *partway through one station's series* is a different animal.
    It is a genuine step discontinuity in the target, of a size no real
    instrument produces, landing in whichever split happens to contain it. In
    train it teaches the forecaster a phantom regime change; in test it is
    indistinguishable from a catastrophic sensor fault and would be "detected"
    as the project's headline result. Measured here, the NO2/SO2 basis changes
    between 2021 and 2025 — conveniently inside the archive's 2023-2024 hole,
    which is why a window on one side of it is clean.

    So this warns rather than silently proceeding, and names what to narrow.
    """
    if "unit_basis" not in df.columns:
        return
    per = (df.groupby(["station_id", "pollutant"], observed=True)["unit_basis"]
             .nunique())
    straddling = per[per > 1]
    if straddling.empty:
        return
    log.warning(
        "%d station-pollutant series straddle a change of unit basis: %s. "
        "That is a step discontinuity of ~1.9-2.6x in the middle of a series, "
        "which no instrument produces and which the detector would flag as an "
        "enormous fault. Narrow cpcb.year_from/year_to in configs/data.yaml to "
        "stay on one side of it (2025 onward is a single basis).",
        len(straddling),
        [f"{s}/{p}" for s, p in straddling.index[:8].tolist()],
    )


def _convert_openaq_units(df: pd.DataFrame) -> pd.DataFrame:
    """Bring OpenAQ readings onto the §3.1 canonical units, mass first.

    The §3.1 unit trap, in its most awkward form yet. A CPCB station exposes
    **two sensors for the same gas**, confirmed against the OpenAQ v3 metadata
    for location 5626:

        no2  units=ug/m3  displayName="NO2 mass"
        no2  units=ppb    displayName="NO2"
        so2  units=ug/m3  displayName="SO2 mass"
        so2  units=ppb    displayName="SO2"

    So a unit is a property of the *sensor*, not of the year — an earlier
    reading of this data as "OpenAQ switched units in 2025" was wrong about the
    mechanism. What actually varies is which of a station's sensors reported on
    a given day, and both can report the same hour.

    That makes `unit` load-bearing in a way it is easy to miss: any aggregation
    keyed on (station, pollutant, hour) that ignores it will happily average a
    ug/m3 value together with a ppb one and produce a number that means
    nothing. `_resample_cpcb_to_hourly` therefore keys on the basis column this
    function adds, and prefers mass.

    Handling, in preference order:

    * **Mass readings pass through** (ug/m3 for everything, mg/m3 for CO after
      a /1000). These need no assumption and are always preferred.
    * **ppb readings convert** with the standard molar factors, and are marked
      as converted so the resampler only falls back to them when a station has
      no mass sensor for that pollutant and hour. The factors assume 25 C and
      1013.25 hPa, which real Delhi air is not; preferring mass means that
      assumption is usually unused rather than merely documented.
    * **CO in ppb is dropped.** Its label is demonstrably wrong rather than
      inconvenient: the 2025 median is 1.08 "ppb", which as ppb is 1.24 ug/m3
      against 2010 ug/m3 for the same station in 2020, while as ppm it is
      1.24 mg/m3 — consistent. The archive means ppm and says ppb. CO is
      outside this project's target and feature set, so dropping it costs
      nothing, and guessing per row would not be free.

    One thing the mass preference does *not* fix: where a station has only a
    ppb sensor, the converted value stands unvalidated, and a sanity check is
    worth doing per pollutant. Converted SO2 came out at ~120 ug/m3 median
    against ~22 from the same station's mass sensor — far above plausible Delhi
    SO2 — which is the kind of discrepancy that motivated preferring mass in
    the first place.

    `_canonicalise_units` still runs afterwards and still raises on anything
    unresolved, so the EEA guard is untouched.
    """
    df = df.copy()
    unit = (df["unit"].astype(str).str.strip().str.lower()
            .map(lambda u: _UNIT_ALIASES.get(u, u)))
    conc = pd.to_numeric(df["concentration"], errors="coerce")
    df["unit_basis"] = _BASIS_MASS

    co_in_ug = (df["pollutant"] == "CO") & (unit == "ug/m3")
    if (n := int(co_in_ug.sum())):
        df.loc[co_in_ug, "concentration"] = conc[co_in_ug] / 1000.0
        df.loc[co_in_ug, "unit"] = "mg/m3"
        log.info("converted %s CO reading(s) from ug/m3 to mg/m3", f"{n:,}")

    for pol, factor in _PPB_TO_UGM3.items():
        sel = (df["pollutant"] == pol) & (unit == "ppb")
        if (n := int(sel.sum())):
            df.loc[sel, "concentration"] = conc[sel] * factor
            df.loc[sel, "unit"] = "ug/m3"
            df.loc[sel, "unit_basis"] = _BASIS_CONVERTED
            log.info("converted %s %s reading(s) from ppb to ug/m3 (x%.4f); "
                     "used only where the station has no mass sensor",
                     f"{n:,}", pol, factor)

    co_in_ppb = (df["pollutant"] == "CO") & (unit == "ppb")
    if (n := int(co_in_ppb.sum())):
        log.warning(
            "dropping %s CO reading(s) labelled ppb: the archive's values are "
            "~1000x too small for ppb and match ppm instead, so the unit label "
            "is wrong and the true one cannot be established per row. CO is "
            "outside this project's target and feature set, so this costs "
            "nothing.", f"{n:,}",
        )
        df = df[~co_in_ppb].copy()
    return df


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


def _mask_sentinels(df: pd.DataFrame) -> pd.DataFrame:
    """Turn EEA's -999 no-data sentinel into a genuine missing value.

    The ``label`` column is deliberately left untouched. The reading really was
    flagged faulty; we simply have no value to adjudicate, which is a different
    statement from "this reading was fine".
    """
    df = df.copy()
    sentinel = df["concentration"] <= SENTINEL_THRESHOLD
    n = int(sentinel.sum())
    if not n:
        return df

    df.loc[sentinel, "concentration"] = pd.NA
    labelled_fault = int((df.loc[sentinel, "label"] == "fault").sum())
    log.warning(
        "masked %s no-data sentinel(s) <= %.0f as missing (%.2f%% of rows); "
        "%d of them were Validity-flagged faults. Contradicts CLAUDE.md §3.1 — "
        "see the SENTINEL_THRESHOLD note.",
        f"{n:,}", SENTINEL_THRESHOLD, 100 * n / len(df), labelled_fault,
    )
    if labelled_fault and labelled_fault == n:
        log.warning(
            "EVERY sentinel row is a Validity fault, and every Validity fault is "
            "a sentinel. Against these labels 'fault detection' is just "
            "missing-data detection: the readings the project exists to "
            "adjudicate are absent, not wrong. Objective 4 cannot be evaluated "
            "on real EEA labels alone — see CONTEXT.md entry 4."
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


def filter_years(df: pd.DataFrame, year_from: int, year_to: int) -> pd.DataFrame:
    """Restrict to ``[year_from, year_to]`` inclusive, on the UTC timestamp.

    A post-download filter rather than a request parameter: airbase 1.x fetches
    a dataset's whole range and offers no year arguments, so the configured
    window is applied here instead.
    """
    year = df["timestamp_utc"].dt.year
    kept = df[(year >= year_from) & (year <= year_to)].copy()
    log.info(
        "year filter %d-%d: kept %s of %s rows (%.1f%%)",
        year_from, year_to, f"{len(kept):,}", f"{len(df):,}",
        100 * len(kept) / max(len(df), 1),
    )
    if kept.empty:
        raise ValueError(
            f"no rows fall in {year_from}-{year_to}; the data covers "
            f"{int(year.min())}-{int(year.max())}"
        )
    return kept


def label_coverage_by_year(df: pd.DataFrame) -> pd.DataFrame:
    """Per-year fault rate and row count — the check that caught the 2013-2016 gap.

    EEA's Catalonia delivery carries no negative ``Validity`` codes at all before
    2017: 477k rows, every one flagged valid. That is almost certainly a
    reporting-practice change rather than four fault-free years, and training on
    it would quietly teach the model that older data never fails. Run this
    before choosing a year window for any new region.
    """
    year = df["timestamp_utc"].dt.year
    out = df.assign(year=year).groupby("year").agg(
        rows=("label", "size"),
        n_fault=("label", lambda s: int((s == "fault").sum())),
        fault_rate=("label", lambda s: float((s == "fault").mean())),
    )
    return out.reset_index()


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


CPCB_COORDS_CSV = "cpcb_station_coords.csv"


def _write_cpcb_coordinates(raw: pd.DataFrame,
                            station_prefix: str = "IN-OAQ-") -> Path:
    """Record each CPCB station's coordinates from the raw archive rows.

    The OpenAQ archive carries ``lat``/``lon`` on every row while the §5.1
    common schema has no column for them, so they would otherwise be lost
    between harmonization and the stages that need them: ERA5 meteorology is
    sampled at station coordinates, and §7.2's graph is built from spatial
    proximity. Derived from the data rather than typed into config, so a
    station cannot end up with coordinates that disagree with its readings.

    The median is used rather than the first row because a handful of archive
    rows carry slightly different coordinates for the same station, presumably
    from re-siting or re-survey; the median is stable against a few outliers.
    """
    if not {"lat", "lon", "location_id"} <= set(raw.columns):
        log.warning("raw CPCB frame has no lat/lon columns; skipping the "
                    "coordinate export. Meteorology and the graph will be "
                    "unavailable for these stations.")
        return data_paths()["raw"] / CPCB_COORDS_CSV

    coords = (raw.assign(station_id=station_prefix
                         + raw["location_id"].astype(str))
                 .groupby("station_id")[["lat", "lon"]].median()
                 .reset_index()
                 .rename(columns={"lat": "latitude", "lon": "longitude"}))
    out = data_paths()["raw"] / CPCB_COORDS_CSV
    out.parent.mkdir(parents=True, exist_ok=True)
    coords.to_csv(out, index=False)
    log.info("wrote coordinates for %d CPCB station(s) -> %s", len(coords), out)
    return out


def cpcb_station_classes() -> dict[str, str]:
    """station_id -> station_class for the configured CPCB stations.

    INFERRED, and labelled as such in `configs/data.yaml`: CPCB publishes
    station types but OpenAQ does not carry them, so these are a reading of
    each station's name and siting. Any India result grouped by station_class
    inherits that caveat.
    """
    stations = (load_config("data").get("cpcb", {}) or {}).get("stations", [])
    return {s["id"]: s.get("station_class", "unknown") for s in stations}


def _main_cpcb(cfg: dict, args) -> None:
    """Harmonize the downloaded CPCB/OpenAQ parquets into region 'india'.

    Kept separate from the EEA path rather than folded into it, because the two
    sources agree on almost nothing: CPCB has no Validity, no Verification, no
    hourly averaging and a unit regime that changes mid-archive. Sharing one
    code path would mean a cascade of `if region == "india"`, and §2.6 wants the
    distinction visible rather than smoothed over.
    """
    raw_dir = data_paths()["raw"] / "cpcb"
    parquets = sorted(raw_dir.glob("openaq_*.parquet"))
    if not parquets:
        raise SystemExit(
            f"no CPCB parquets in {raw_dir}. Download first:\n"
            "  python -m src.data.fetch_cpcb --discover --city Delhi\n"
            "  python -m src.data.fetch_cpcb --download <ids...> "
            "--year-from 2025 --year-to 2026"
        )

    raw = pd.concat([pd.read_parquet(p) for p in parquets], ignore_index=True)
    log.info("read %s raw CPCB row(s) from %d parquet(s)",
             f"{len(raw):,}", len(parquets))

    cpcb_cfg = cfg.get("cpcb", {}) or {}
    lo = args.cpcb_year_from or cpcb_cfg.get("year_from")
    hi = args.cpcb_year_to or cpcb_cfg.get("year_to")

    # Filter BEFORE harmonizing, not after. The window exists to keep each
    # station on a single sensor basis (see `_check_basis_continuity`), so the
    # continuity check has to run on what will actually be used.
    if lo or hi:
        year = pd.to_datetime(raw["datetime"], utc=True, errors="coerce",
                              format="mixed").dt.year
        keep = year.between(int(lo or 0), int(hi or 9999))
        log.info("year filter %s-%s: keeping %s of %s raw row(s)",
                 lo, hi, f"{int(keep.sum()):,}", f"{len(raw):,}")
        raw = raw[keep].copy()
    if raw.empty:
        raise SystemExit(f"no CPCB rows in {lo}-{hi}")

    # Capture coordinates before harmonizing: the archive carries lat/lon per
    # row but the common schema has no place for them, and §6/§7 need them for
    # meteorology and for the graph. Taken from the data itself rather than
    # written into config by hand, so they cannot drift from the readings.
    _write_cpcb_coordinates(raw)

    df = harmonize_cpcb(raw, station_class_map=cpcb_station_classes())
    if df.empty:
        raise SystemExit("no CPCB rows survived harmonization")

    region = args.region_name or cpcb_cfg.get("region", "india")
    df["region"] = region

    # LOCKED (§2.6, §5.3): this data carries no real fault labels, so it must
    # never contribute to a real-fault metric. Assert rather than trust.
    if (df["label_source"] != "synthetic").any():
        raise ValueError(
            "CPCB rows must all carry label_source='synthetic' (§2.6). "
            f"Found: {sorted(df['label_source'].unique())}"
        )
    if df["label"].notna().any():
        raise ValueError(
            "CPCB rows must carry no real label — CPCB has no Validity "
            "equivalent, so any label here would be invented (§2.6, §3.2)."
        )

    write_harmonized(df, region)

    summary = (
        df.groupby(["region", "station_id", "station_class", "label_source"])
        .agg(rows=("concentration", "size"),
             first=("timestamp_utc", "min"), last=("timestamp_utc", "max"),
             pollutants=("pollutant", "nunique"))
        .reset_index()
    )
    print("\n=== CPCB/OpenAQ harmonized (§8.3: grouped, never pooled) ===")
    print(summary.to_string(index=False))
    print(f"\nregion={region}  rows={len(df):,}  "
          f"stations={df['station_id'].nunique()}")
    print("label_source is 'synthetic' for every row: CPCB has no Validity "
          "equivalent (§2.6). Faults are attached later by the rate-matched "
          "injector at the rate measured from EEA (§5.5).")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--from-dir", default=None,
                    help="read an already-downloaded EEA tree")
    ap.add_argument("--raw-parquet", default=None,
                    help="read the parquet written by src.data.fetch_eea")
    ap.add_argument("--demo", action="store_true",
                    help="use the SIMULATED source (region='demo', "
                         "label_source='simulated')")
    ap.add_argument("--region-name", default=None,
                    help="write under this region instead of eea.region — use "
                         "for an alternative node set so an existing build is "
                         "not overwritten (e.g. the 71-station scale-up)")
    ap.add_argument("--cpcb", action="store_true",
                    help="harmonize the CPCB/OpenAQ parquets under "
                         "data/raw/cpcb instead of EEA (§3.2, Phase 3). This is "
                         "a SECONDARY generalization set, never primary "
                         "training or evaluation data (§2.6).")
    ap.add_argument("--cpcb-year-from", type=int, default=None,
                    help="restrict CPCB to one unit regime. OpenAQ switched the "
                         "gases from ug/m3 to ppb partway through the archive, "
                         "and 2025 onward is a single consistent regime with no "
                         "conversion to take on trust.")
    ap.add_argument("--cpcb-year-to", type=int, default=None)
    args = ap.parse_args()

    cfg = load_config("data")

    if args.cpcb:
        _main_cpcb(cfg, args)
        return

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

    if args.region_name:
        region = args.region_name
    df = harmonize_eea(raw, class_map, region=region, label_source=label_source)

    if not args.demo:
        coverage = label_coverage_by_year(df)
        print("\n=== fault-label coverage by year (check before windowing) ===")
        print(coverage.to_string(index=False))
        unlabelled = coverage.loc[coverage["n_fault"] == 0, "year"].tolist()
        if unlabelled:
            log.warning(
                "year(s) %s contain rows but ZERO fault flags. Including them "
                "dilutes the measured fault rate and teaches the model that "
                "those periods never fail. Check eea.year_from in "
                "configs/data.yaml.", unlabelled,
            )
        eea = cfg["eea"]
        df = filter_years(df, int(eea["year_from"]), int(eea["year_to"]))
        log.info("measured real fault rate (the §5.5 target): %.5f",
                 real_fault_rate(df))

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
