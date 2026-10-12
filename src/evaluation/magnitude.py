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


def fault_magnitudes(station_id: str) -> pd.Series:
    """|injected − clean| per timestamp, in units of the station's own sigma."""
    clean = station_series(station_id, variant="")
    injected = station_series(station_id, variant="injected")

    scaler = LocalScaler.load(
        station_dir(station_id, "processed", variant="injected") / "scaler.json")
    target = json.loads(
        (station_dir(station_id, "processed", variant="injected") / "manifest.json")
        .read_text(encoding="utf-8"))["target"]
    sigma = scaler.scale_of(target)

    common = clean.index.intersection(injected.index)
    if common.empty:
        raise ValueError(f"{station_id}: clean and injected builds share no timestamps")
    return (injected.reindex(common) - clean.reindex(common)).abs() / sigma


def attach_magnitudes(records: pd.DataFrame) -> pd.DataFrame:
    """Add a ``magnitude_sigma`` column to records, for fault rows."""
    out = records.copy()
    out["magnitude_sigma"] = np.nan
    for station_id, g in out.groupby("station_id"):
        try:
            mags = fault_magnitudes(station_id)
        except (FileNotFoundError, ValueError) as exc:
            log.warning("%s: no magnitudes (%s)", station_id, exc)
            continue
        out.loc[g.index, "magnitude_sigma"] = mags.reindex(
            pd.DatetimeIndex(g["t_target"])).to_numpy()
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
    args = ap.parse_args()

    records = attach_magnitudes(load_records(args.run_dir))
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
