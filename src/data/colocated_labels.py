"""Real fault labels from co-located reference-vs-low-cost sensor pairs.

This is §2.1a's recommended stretch goal, and the only route in this project to
fault labels that are **real**. Every other fault number here is measured
against synthetic injection, because EEA destroys the readings its own
`Validity` flags point at (CONTEXT entry 4): a flagged row carries the -999
sentinel, so there is a label but no wrong reading to adjudicate.

A co-located pair solves exactly that. A low-cost MOX sensor sits beside a
reference-grade instrument measuring the same air, so when the cheap one is
wrong you know it is wrong **and by how much**. That is a plausible-but-wrong
reading with ground truth attached, which is what Objective 4 needs and has
never had.

Source: H2020 CAPTOR (Zenodo 2564825, CC-BY-4.0), Catalonia, summer 2018.

What a "fault" means here
-------------------------
A low-cost sensor is *always* somewhat wrong -- pooled MAE across these pairs
is 15.7-19.8 ug/m3 against a reference sigma of 31-45 -- so "fault" is not a
natural binary. Two choices make it one, and both are deliberate:

* **Rate-matched.** The threshold is set so the real fault rate equals the
  measured EEA rate of 2.805%, the same constant §2.7 locks the synthetic
  injector to. That lands at |error| > 1.337 sigma. Matching the rate is what
  makes real and synthetic results comparable at all; a different prevalence
  would move precision for reasons having nothing to do with the detector.
* **Expressed in the station's own sigma**, like the synthetic magnitude bands,
  so "a 2-sigma real fault" and "a 2-sigma injected fault" mean the same thing.

The headline caveat, measured rather than assumed: real errors are
**concentrated in the band the detector is worst at**. Median magnitude is
1.62 sigma and the maximum is 3.07; synthetic evaluation put recall at 0.211
in the 1-2 sigma band against 0.946 above 5 sigma. There are no easy faults in
this set.

Distance matters
----------------
Four pairs are within 150 m. ES1892A is 1.84 km away, and its 11.08% "fault"
rate against 0.77-5.32% for the others is the reason: at that separation the
difference between sensor and reference includes genuine spatial variation in
ozone, not just sensor error. It is excluded from the headline set and
reported separately rather than dropped silently.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from src.data.harmonize import write_harmonized
from src.utils.logging_setup import get_logger
from src.utils.paths import data_paths

log = get_logger(__name__)

#: Pairs within this distance are treated as genuinely co-located.
MAX_COLOCATED_KM = 0.5

#: §2.7 / §5.5 -- the measured EEA fault rate every label set is matched to.
TARGET_FAULT_RATE = 0.02805

REGION = "colocated"


def _rate_matched_threshold(z: pd.Series, target: float = TARGET_FAULT_RATE) -> float:
    """The |error|/sigma cut giving the measured real fault rate.

    Solved rather than assumed, so the real and synthetic label sets have the
    same prevalence and their precisions are comparable.
    """
    lo, hi = 0.05, 10.0
    for _ in range(80):
        mid = (lo + hi) / 2
        if float((z > mid).mean()) > target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def build(max_km: float = MAX_COLOCATED_KM,
          target_rate: float = TARGET_FAULT_RATE) -> pd.DataFrame:
    """Co-located pairs -> the §5.1 common schema, with REAL fault labels."""
    path = data_paths()["harmonized"] / "colocated_o3.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Build the pairs first:\n"
            "  python -m src.data.fetch_colocated"
        )
    pairs = pd.read_parquet(path)

    far = sorted(pairs.loc[pairs["distance_km"] > max_km, "station_id"].unique())
    if far:
        log.warning(
            "excluding %d pair(s) beyond %.1f km from the headline set: %s. At "
            "that separation the sensor-reference difference includes real "
            "spatial variation in ozone, not only sensor error.",
            len(far), max_km, far)
    near = pairs[pairs["distance_km"] <= max_km].copy()
    if near.empty:
        raise ValueError("no pairs within the co-location distance")

    sigma = near.groupby("station_id")["reference"].transform("std")
    z = (near["error"].abs() / sigma)
    k = _rate_matched_threshold(z, target_rate)
    near["is_fault"] = z > k
    near["magnitude_sigma"] = z

    achieved = float(near["is_fault"].mean())
    log.info("real fault threshold |error| > %.3f sigma -> rate %.4f "
             "(target %.4f, the measured EEA rate)", k, achieved, target_rate)
    log.info("real fault magnitudes: median %.2f sigma, max %.2f sigma -- the "
             "band synthetic evaluation found hardest",
             float(z[near["is_fault"]].median()), float(z.max()))

    out = pd.DataFrame({
        "station_id": near["station_id"].astype(str),
        "region": REGION,
        "timestamp_utc": pd.to_datetime(near["timestamp_utc"], utc=True),
        "pollutant": "O3",
        # The LOW-COST sensor's reading is what the station reports. The
        # reference exists only to label it; it is never a feature, because a
        # deployed node would not have one.
        "concentration": near["sensor"].astype(float),
        "unit": "ug/m3",
        "label": np.where(near["is_fault"], "fault", "trustworthy"),
        "label_source": "real",
        "station_class": "colocated_low_cost",
        "validity": pd.Series(pd.NA, index=near.index, dtype="Int64"),
        "verification": pd.Series(pd.NA, index=near.index, dtype="Int64"),
    })
    out = out.dropna(subset=["concentration"]).sort_values(
        ["station_id", "timestamp_utc"]).reset_index(drop=True)

    log.info("co-located: %s row(s), %d station(s), %s fault(s)",
             f"{len(out):,}", out["station_id"].nunique(),
             f"{int((out['label'] == 'fault').sum()):,}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--max-km", type=float, default=MAX_COLOCATED_KM)
    ap.add_argument("--target-rate", type=float, default=TARGET_FAULT_RATE)
    args = ap.parse_args()

    df = build(args.max_km, args.target_rate)
    write_harmonized(df, REGION)

    summary = (df.groupby(["region", "station_id", "station_class", "label_source"])
                 .agg(rows=("concentration", "size"),
                      faults=("label", lambda s: int((s == "fault").sum())),
                      first=("timestamp_utc", "min"),
                      last=("timestamp_utc", "max"))
                 .reset_index())
    summary["fault_rate"] = (summary["faults"] / summary["rows"]).round(4)
    print("\n=== co-located real-fault set (§8.3: grouped, never pooled) ===")
    print(summary.to_string(index=False))
    print("\nlabel_source is 'real' for every row: these are genuine "
          "plausible-but-wrong readings with a reference instrument as truth, "
          "which is what §2.1a asks for and no other data in this project has.")


if __name__ == "__main__":
    main()
