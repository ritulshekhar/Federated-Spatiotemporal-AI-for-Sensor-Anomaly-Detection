"""Detection rate stratified by how large the fault actually is.

A single F1 averages together faults that shift a reading by five standard
deviations and faults that shift it by half of one. Those are not the same
problem, and a detector that misses the second while catching the first is
doing its job. CONTEXT.md entry 7 measured why this matters here: four of six
subtypes produce readings that sit *inside* the normal spread, so the aggregate
is dominated by whether the injected faults happened to be large.

Magnitude is recovered exactly rather than estimated. The clean and injected
processed builds differ only by the injection, so

    magnitude(t) = | y_injected(t) - y_clean(t) |

is the amount the fault actually moved the reading, which is then expressed in
units of that station's own training-window standard deviation (§6.5) so
stations are comparable without sharing anything.

This is also the table that compares honestly against published work. Papers
that report 90%+ detection usually inject large, obvious faults; "recall 0.95
above 3σ, 0.21 below 1σ" says where this detector sits on the same axis instead
of arguing about a single summary number.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluation.rescore import calibrate, load_records
from src.evaluation.temporal_reference import station_series
from src.preprocessing.normalize import LocalScaler
from src.utils.logging_setup import get_logger
from src.utils.paths import station_dir

log = get_logger(__name__)

#: Bin edges in units of the station's own sigma. The first two bins are where
#: entry 7 showed the normal spread already lives.
DEFAULT_BINS = (0.0, 0.5, 1.0, 2.0, 5.0, np.inf)
BIN_LABELS = ("<0.5σ", "0.5–1σ", "1–2σ", "2–5σ", ">5σ")


def fault_magnitudes(station_id: str, clean_variant: str = "",
                     injected_variant: str = "injected") -> pd.Series:
    """Injected-minus-clean per timestamp, in units of the station's own sigma.

    The variants are parameters because the node set is no longer always the
    original seven: the 30-station build lives under `injected71`, and
    silently falling back to the 7-station names produced a magnitude table
    computed from whichever stations happened to have an old clean build -- a
    wrong answer that looked like a right one.
    """
    clean = station_series(station_id, variant=clean_variant)
    injected = station_series(station_id, variant=injected_variant)

    scaler = LocalScaler.load(
        station_dir(station_id, "processed", variant=injected_variant) / "scaler.json")
    target = json.loads(
        (station_dir(station_id, "processed", variant=injected_variant) / "manifest.json")
        .read_text(encoding="utf-8"))["target"]
    sigma = scaler.scale_of(target)

    common = clean.index.intersection(injected.index)
    if common.empty:
        raise ValueError(f"{station_id}: clean and injected builds share no timestamps")
    return (injected.reindex(common) - clean.reindex(common)).abs() / sigma


def attach_magnitudes(records: pd.DataFrame, clean_variant: str = "",
                      injected_variant: str = "injected") -> pd.DataFrame:
    """Add a ``magnitude_sigma`` column to records, for fault rows."""
    out = records.copy()
    out["magnitude_sigma"] = np.nan
    missing = []
    for station_id, g in out.groupby("station_id"):
        try:
            mags = fault_magnitudes(station_id, clean_variant, injected_variant)
        except (FileNotFoundError, ValueError) as exc:
            missing.append(station_id)
            log.warning("%s: no magnitudes (%s)", station_id, exc)
            continue
        out.loc[g.index, "magnitude_sigma"] = mags.reindex(
            pd.DatetimeIndex(g["t_target"])).to_numpy()

    # Partial coverage silently yields a magnitude table built from a subset of
    # the node set, which reads exactly like a complete one. Refuse instead.
    n_total = out["station_id"].nunique()
    if missing and len(missing) > n_total // 2:
        raise ValueError(
            f"{len(missing)} of {n_total} stations have no magnitudes "
            f"(clean_variant={clean_variant!r}, injected_variant="
            f"{injected_variant!r}). The resulting table would describe only "
            "the minority that do. Build the clean variant first, or pass the "
            "right --clean-variant/--injected-variant."
        )
    return out


def stratified_recall(records: pd.DataFrame, thresholds: dict,
                      horizon: int = 1, bins=DEFAULT_BINS,
                      by_subtype: bool = False) -> pd.DataFrame:
    """Detection rate per magnitude band, over genuine faults only."""
    df = records[(records["split"] == "test") & (~records["is_imputed"])
                 & (records["horizon"] == horizon)
                 & (records["label"] == "fault")].copy()
    df = df[df["magnitude_sigma"].notna() & np.isfinite(df["score"])]
    if df.empty:
        return pd.DataFrame()

    df["band"] = pd.cut(df["magnitude_sigma"], bins=list(bins),
                        labels=list(BIN_LABELS), right=False)
    df["thr"] = [thresholds.get((m, s), np.inf)
                 for m, s in zip(df["model"], df["station_id"])]
    df["detected"] = df["score"] > df["thr"]

    keys = ["model", "band"] + (["fault_type"] if by_subtype else [])
    out = (df.groupby(keys, observed=True)
             .agg(n_faults=("detected", "size"),
                  recall=("detected", "mean"),
                  median_sigma=("magnitude_sigma", "median"))
             .reset_index())
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_dir")
    ap.add_argument("--horizon", type=int, default=1)
    ap.add_argument("--threshold", default="val_fpr_extremes")
    ap.add_argument("--by-subtype", action="store_true")
    ap.add_argument("--clean-variant", default="")
    ap.add_argument("--injected-variant", default="injected")
    args = ap.parse_args()

    records = attach_magnitudes(load_records(args.run_dir),
                                args.clean_variant, args.injected_variant)
    thresholds = calibrate(records, args.threshold, target_fpr_extremes=0.10)

    table = stratified_recall(records, thresholds, horizon=args.horizon)
    print(f"\n=== detection rate by fault magnitude (h={args.horizon}) ===")
    print(table.to_string(index=False, float_format=lambda v: f"{v:7.3f}"))

    if args.by_subtype:
        sub = stratified_recall(records, thresholds, horizon=args.horizon,
                                by_subtype=True)
        print(f"\n=== by subtype and magnitude (h={args.horizon}) ===")
        piv = sub.pivot_table(index="fault_type", columns="band",
                              values="recall", observed=True)
        print(piv.round(3).to_string())

    covered = records["magnitude_sigma"].notna().mean()
    print(f"\nmagnitudes resolved for {covered:.1%} of records")


if __name__ == "__main__":
    main()
