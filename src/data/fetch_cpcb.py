"""CPCB (India) acquisition — Phase 3 generalization experiment.

CLAUDE.md §3.2, §5, §9 Phase 3, §11.

**Ordering note.** §9 puts this after Phase 1 and Phase 2 have produced real
reported numbers. It is implemented early at the user's explicit request. That
is fine as *acquisition* — it resolves the §11 open risk by replacing an
unverified schema guess with an inspected sample — but it does not license
training on this data, and none of the constraints below are relaxed:

* **LOCKED (§2.6):** India is a secondary generalization experiment, never
  primary training or evaluation data.
* **LOCKED (§2.6, §3.2):** CPCB has no `Validity`/`Verification` equivalent.
  AQICN's own notice is that the data are unvalidated at publication. Every
  fault label on this data is therefore synthetic, tagged
  ``label_source="synthetic"``, and never blended into a real-fault metric.
* **LOCKED (§5.5):** any synthetic injection rate must match the *measured* EEA
  real fault rate.

Why OpenAQ rather than CPCB's own portal
----------------------------------------
Of the routes CLAUDE.md §3.2 lists, only one yields multi-year *hourly*
per-pollutant data without scraping:

* ``app.cpcbccr.com`` — the official bulk export, but it is an undocumented
  internal API behind a session, capped at about a week per request. Hundreds
  of requests for a multi-year pull.
* ``data.gov.in`` — needs an API key, and serves AQI-level min/max/avg, which
  is coarser than the per-pollutant hourly structure this project needs.
* **OpenAQ** — ingests CPCB as its India source, so this *is* CPCB data, and it
  exposes the full history. Chosen.

OpenAQ splits into two halves, and only the first needs credentials:

1. **Station discovery** (v3 REST API, ``X-API-Key`` required, free from
   https://explore.openaq.org/register) — maps country/city to location ids.
2. **History** (public S3 archive, **no credentials**) — once a location id is
   known, its whole record is at
   ``records/csv.gz/locationid=<id>/year=<Y>/month=<M>/``.

So the key is needed once to learn the ids; the bulk download is keyless. Set
it as ``OPENAQ_API_KEY``, or pass ``api_key=``.

Schema verification (§11)
-------------------------
``harmonize_cpcb`` in ``src/data/harmonize.py`` must not be trusted against a
guessed schema. ``inspect_schema()`` below downloads one real file and reports
its actual columns and dtypes; ``REQUIRED_ARCHIVE_COLUMNS`` is then enforced at
read time so a drifted export fails loudly instead of silently mismapping.
"""

from __future__ import annotations

import argparse
import gzip
import io
import os
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd
import requests

from src.utils.logging_setup import get_logger
from src.utils.paths import REPO_ROOT, data_paths

log = get_logger(__name__)

OPENAQ_API = "https://api.openaq.org/v3"
OPENAQ_S3 = "https://openaq-data-archive.s3.amazonaws.com"
_S3_NS = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}

# Routes kept for the record (see the module docstring).
CPCB_BULK_PORTAL = "https://app.cpcbccr.com/ccr/#/caaqm-dashboard-all/caaqm-landing"
DATA_GOV_IN_API = "https://api.data.gov.in/resource/3b01bcb8-0b14-4abf-b6f2-c1bfd384ba69"

#: Columns the archive CSVs are expected to carry. Enforced, not assumed.
REQUIRED_ARCHIVE_COLUMNS = ["location_id", "parameter", "value", "datetime", "units"]

#: OpenAQ parameter name -> this project's pollutant vocabulary (§5.1).
PARAMETER_MAP = {
    "pm25": "PM2.5", "pm10": "PM10", "no2": "NO2",
    "o3": "O3", "so2": "SO2", "co": "CO",
}


def _from_dotenv(name: str) -> str | None:
    """Read one key from a gitignored ``.env`` at the repo root.

    Deliberately hand-rolled rather than adding a python-dotenv dependency for
    a single secret, and deliberately *not* exported into ``os.environ`` — the
    key is handed to the one request that needs it and nothing else.
    """
    path = REPO_ROOT / ".env"
    if not path.exists():
        return None
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        if k.strip() == name:
            return v.strip().strip('"').strip("'") or None
    return None


def _api_key(api_key: str | None = None) -> str:
    key = (api_key or os.environ.get("OPENAQ_API_KEY")
           or _from_dotenv("OPENAQ_API_KEY"))
    if not key:
        raise RuntimeError(
            "an OpenAQ API key is required for station discovery only. Register "
            "free at https://explore.openaq.org/register, then either put it in "
            "a gitignored .env at the repo root:\n"
            "  OPENAQ_API_KEY=<your key>\n"
            "or set it in the environment:\n"
            '  $env:OPENAQ_API_KEY = "<your key>"     (PowerShell)\n'
            "  export OPENAQ_API_KEY=<your key>       (bash)\n"
            "The bulk history download afterwards needs no credentials."
        )
    return key


# ---------------------------------------------------------------------------
# 1. Station discovery (needs the key)
# ---------------------------------------------------------------------------
def discover_locations(city: str | None = None, api_key: str | None = None,
                       limit: int = 100, max_pages: int = 10) -> pd.DataFrame:
    """Indian monitoring locations known to OpenAQ (CPCB-sourced)."""
    headers = {"X-API-Key": _api_key(api_key)}
    rows: list[dict] = []
    for page in range(1, max_pages + 1):
        params: dict[str, object] = {"iso": "IN", "limit": limit, "page": page}
        resp = requests.get(f"{OPENAQ_API}/locations", headers=headers,
                            params=params, timeout=60)
        if resp.status_code == 401:
            raise RuntimeError(
                "OpenAQ rejected the API key (401). Check OPENAQ_API_KEY."
            )
        resp.raise_for_status()
        results = resp.json().get("results", [])
        if not results:
            break
        for r in results:
            rows.append({
                "location_id": r.get("id"),
                "name": r.get("name"),
                "locality": r.get("locality"),
                "provider": (r.get("provider") or {}).get("name"),
                "latitude": (r.get("coordinates") or {}).get("latitude"),
                "longitude": (r.get("coordinates") or {}).get("longitude"),
                "first_seen": (r.get("datetimeFirst") or {}).get("utc"),
                "last_seen": (r.get("datetimeLast") or {}).get("utc"),
                "parameters": ",".join(
                    sorted({(s.get("parameter") or {}).get("name", "")
                            for s in (r.get("sensors") or [])} - {""})
                ),
            })
        if len(results) < limit:
            break

    df = pd.DataFrame(rows)
    if city and not df.empty:
        mask = (df["locality"].astype(str).str.contains(city, case=False, na=False)
                | df["name"].astype(str).str.contains(city, case=False, na=False))
        df = df[mask]
    log.info("OpenAQ: %d Indian location(s)%s", len(df),
             f" matching {city!r}" if city else "")
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# 2. History from the public S3 archive (no credentials)
# ---------------------------------------------------------------------------
def list_archive_keys(location_id: int, year_from: int | None = None,
                      year_to: int | None = None) -> list[str]:
    """Every archive object key for one location, optionally year-filtered."""
    prefix = f"records/csv.gz/locationid={location_id}/"
    keys, token = [], None
    while True:
        params = {"list-type": "2", "prefix": prefix, "max-keys": "1000"}
        if token:
            params["continuation-token"] = token
        resp = requests.get(OPENAQ_S3, params=params, timeout=60)
        resp.raise_for_status()
        root = ET.fromstring(resp.content)
        for c in root.findall("s3:Contents", _S3_NS):
            key = c.findtext("s3:Key", default="", namespaces=_S3_NS)
            if key.endswith(".csv.gz"):
                keys.append(key)
        if root.findtext("s3:IsTruncated", default="false",
                         namespaces=_S3_NS) != "true":
            break
        token = root.findtext("s3:NextContinuationToken", namespaces=_S3_NS)

    if year_from or year_to:
        lo, hi = year_from or 0, year_to or 9999
        keys = [k for k in keys
                if any(f"year={y}/" in k for y in range(lo, hi + 1))]
    return sorted(keys)


def read_archive_object(key: str, retries: int = 3,
                        backoff: float = 2.0) -> pd.DataFrame:
    """Download and parse one gzipped archive CSV.

    ``encoding="utf-8"`` is not optional. The archive stores units as "µg/m³",
    and letting gzip fall back to the platform default (cp1252 on Windows)
    mangles that into "Âµg/mÂ³", which then fails the unit check in
    ``harmonize._canonicalise_units`` — as it should.

    Retried because a full pull is thousands of sequential requests against a
    public bucket, and ``download_location`` treats a failed object as a skip.
    Without retries a transient blip becomes a silently missing day of data,
    which is the worst possible failure mode here: the gap is invisible
    downstream and looks like a sensor outage.
    """
    last: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(f"{OPENAQ_S3}/{key}", timeout=120)
            resp.raise_for_status()
            with gzip.open(io.BytesIO(resp.content), "rt", encoding="utf-8") as fh:
                return pd.read_csv(fh)
        except (requests.RequestException, OSError, EOFError) as exc:
            last = exc
            if attempt < retries:
                time.sleep(backoff * attempt)
    raise RuntimeError(f"{key}: failed after {retries} attempt(s)") from last


def inspect_schema(location_id: int, api_key: str | None = None) -> pd.DataFrame:
    """Download ONE real file and report its actual schema.

    This is the §11 mitigation: it replaces the guessed CPCB column mapping with
    an observed one. Run it before trusting ``harmonize_cpcb``.
    """
    keys = list_archive_keys(location_id)
    if not keys:
        raise FileNotFoundError(
            f"no archive objects for location {location_id}. Confirm the id "
            "with discover_locations()."
        )
    df = read_archive_object(keys[0])
    print(f"\n=== OpenAQ archive schema, location {location_id} ===")
    print(f"object   : {keys[0]}")
    print(f"objects  : {len(keys)} total for this location")
    print(f"columns  : {list(df.columns)}")
    print("\ndtypes:\n" + df.dtypes.to_string())
    print("\nfirst rows:\n" + df.head(5).to_string())
    missing = [c for c in REQUIRED_ARCHIVE_COLUMNS if c not in df.columns]
    print(f"\nrequired columns present: {not missing}"
          + (f" (missing {missing})" if missing else ""))
    return df


def download_location(location_id: int, year_from: int, year_to: int,
                      dest: Path | None = None) -> Path:
    """Fetch a location's full history into one parquet under data/raw/cpcb."""
    dest = dest or (data_paths()["raw"] / "cpcb")
    dest.mkdir(parents=True, exist_ok=True)
    out = dest / f"openaq_{location_id}_{year_from}_{year_to}.parquet"
    if out.exists():
        log.info("already downloaded: %s", out)
        return out

    keys = list_archive_keys(location_id, year_from, year_to)
    if not keys:
        raise FileNotFoundError(
            f"no archive objects for location {location_id} in "
            f"{year_from}-{year_to}"
        )
    log.info("location %d: downloading %d object(s)", location_id, len(keys))

    frames, failed = [], []
    for i, key in enumerate(keys, 1):
        try:
            frames.append(read_archive_object(key))
        except Exception as exc:  # noqa: BLE001 — one bad object must not stop the pull
            failed.append(key)
            log.warning("skipping %s: %s", key, exc)
        if i % 100 == 0:
            log.info("  %d/%d objects (%d failed)", i, len(keys), len(failed))

    if not frames:
        raise ValueError(f"every object failed for location {location_id}")

    # A skipped object is an invisible hole in the series: downstream it is
    # indistinguishable from a sensor outage, and §6.2 would impute straight
    # over it. A few are tolerable and logged; a lot means the pull is not
    # trustworthy and should fail rather than quietly produce a thin dataset.
    share = len(failed) / len(keys)
    if failed:
        log.warning("location %d: %d of %d object(s) failed (%.1f%%)",
                    location_id, len(failed), len(keys), 100 * share)
    if share > 0.05:
        raise RuntimeError(
            f"location {location_id}: {len(failed)} of {len(keys)} objects "
            f"({100 * share:.1f}%) failed to download, above the 5% tolerance. "
            "Re-run to resume — successful locations are cached as parquet."
        )
    df = pd.concat(frames, ignore_index=True)

    missing = [c for c in REQUIRED_ARCHIVE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"OpenAQ archive schema drifted: missing {missing}, got "
            f"{list(df.columns)}. Re-run inspect_schema() and update "
            "REQUIRED_ARCHIVE_COLUMNS and harmonize_cpcb together (§11)."
        )

    df.to_parquet(out, index=False)
    log.info("wrote %s rows -> %s", f"{len(df):,}", out)
    return out


def fetch(location_ids: list[int], year_from: int = 2019, year_to: int = 2024,
          api_key: str | None = None) -> pd.DataFrame:
    """Download several locations and return them concatenated (still raw).

    A location with nothing in the requested window is skipped rather than
    fatal. The OpenAQ archive's CPCB coverage is genuinely discontinuous —
    2023 and 2024 are absent for every Delhi station surveyed — so one empty
    station must not abandon the other six.
    """
    frames, skipped = [], []
    for lid in location_ids:
        try:
            frames.append(pd.read_parquet(
                download_location(lid, year_from, year_to)))
        except FileNotFoundError as exc:
            skipped.append(lid)
            log.warning("location %d skipped: %s", lid, exc)

    if not frames:
        raise FileNotFoundError(
            f"no location had archive objects in {year_from}-{year_to}. "
            "Survey coverage first — the archive has multi-year holes."
        )
    df = pd.concat(frames, ignore_index=True)
    log.info("CPCB/OpenAQ: %s raw rows across %d location(s)%s",
             f"{len(df):,}", len(location_ids) - len(skipped),
             f", {len(skipped)} skipped as empty: {skipped}" if skipped else "")
    return df


# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--discover", action="store_true",
                    help="list Indian locations (needs OPENAQ_API_KEY)")
    ap.add_argument("--city", default=None, help="filter discovery by city")
    ap.add_argument("--inspect", type=int, default=None, metavar="LOCATION_ID",
                    help="download one file and print its real schema (§11)")
    ap.add_argument("--download", type=int, nargs="*", default=None,
                    metavar="LOCATION_ID", help="download full history (no key)")
    ap.add_argument("--year-from", type=int, default=2019)
    ap.add_argument("--year-to", type=int, default=2024)
    args = ap.parse_args()

    if args.discover:
        df = discover_locations(city=args.city)
        if df.empty:
            print("no locations found")
        else:
            with pd.option_context("display.width", 220, "display.max_rows", 200):
                print(df.to_string(index=False))
            out = data_paths()["raw"] / "cpcb_openaq_locations.csv"
            out.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(out, index=False)
            print(f"\nwrote {out}")

    if args.inspect is not None:
        inspect_schema(args.inspect)

    if args.download:
        fetch(args.download, args.year_from, args.year_to)

    if not (args.discover or args.inspect is not None or args.download):
        ap.print_help()


if __name__ == "__main__":
    main()
