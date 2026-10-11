"""Diagnostic: is the four-of-six blindness caused by input-window contamination?

CONTEXT.md entry 6 measured that a residual detector catches spikes (0.59) and
noise (0.40) but is blind to drift, bias, stuck-at and dropout (0.000–0.052).
The proposed explanation was *contamination*: the faulty values sit inside the
model's own 24h input window, so the forecast follows the fault and the residual
never grows.

This module tests that explanation with a reference the fault cannot reach.

**Multi-lag median, not a single week-ago value.** The obvious test is
``|y(t) - y(t-168)|``, but that leans on a stable *weekly* cycle — sharp in
traffic-driven Barcelona, far weaker in Indian cities where biomass burning,
construction dust and winter inversions dominate day-to-day. Since India is the
generalization target (§5.8), the reference used here is

    median{ y(t-24), y(t-48), ..., y(t-168) }

which leans on the *diurnal* cycle instead — near-universal — and is robust to
one or two of those lags themselves falling inside a fault episode. It has no
learned parameters and is scaled per station, so it carries nothing
Barcelona-specific (§5.8 issue 2 does not apply to it).

**This is a diagnostic, not a proposed detector.** Any temporal reference
assumes periodicity that may not transfer. The structurally generic mechanisms
are *relative* comparisons — neighbouring stations (rung 6), or other pollutants
at the same station — because whatever the local pollution regime is, the
comparator shares it. What this run buys is advance knowledge of whether the
"use an uncontaminated reference" strategy works at all, before committing to
the graph layer.
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluation.metrics import average_precision
from src.evaluation.rescore import apply_score, calibrate, load_records
from src.evaluation.report import build_report, subtype_recall_table
from src.preprocessing.windowing import SPLITS, load_windows
from src.utils.logging_setup import get_logger
from src.utils.paths import station_dir

log = get_logger(__name__)

#: One week of daily lags. Daily, because the diurnal cycle is the periodicity
#: most likely to survive a move to another country.
DEFAULT_LAGS = (24, 48, 72, 96, 120, 144, 168)


def station_series(station_id: str, variant: str = "injected") -> pd.Series:
    """Reconstruct a station's target series *as the detector sees it*.

    Rebuilt from the window files rather than from the harmonized parquet on
    purpose: the windows carry the injected corruption, so the lagged reference
    can itself be faulty exactly as often as it would be in a live system.
    Using the clean parquet would quietly hand the diagnostic a perfect history
    and overstate the result.
    """
    d = station_dir(station_id, "processed", variant=variant)
    times, values = [], []
    for split in SPLITS:
        path = d / f"{split}.npz"
        if not path.exists():
            continue
        ws = load_windows(path)
        if len(ws) == 0:
            continue
        times.append(ws.t_target.ravel())
        values.append(ws.y_raw.ravel())

    if not times:
        raise FileNotFoundError(f"no window files for {station_id} (variant={variant!r})")

    s = pd.Series(np.concatenate(values), index=pd.DatetimeIndex(np.concatenate(times)))
    # Stride-1 windows overlap, so each hour appears up to `horizon` times with
    # the same value. Keep one.
    return s[~s.index.duplicated(keep="first")].sort_index()


def multilag_reference(series: pd.Series, timestamps: np.ndarray,
                       lags: tuple[int, ...] = DEFAULT_LAGS) -> np.ndarray:
    """``median{ y(t - lag) }`` for each timestamp. NaN where no lag resolves."""
    idx = pd.DatetimeIndex(timestamps)
    stacked = np.vstack([
        series.reindex(idx - pd.Timedelta(hours=lag)).to_numpy(dtype=float)
        for lag in lags
    ])
    # An all-NaN column just means no lag resolved (start of series, or a long
    # gap); NaN is the correct answer there, so the warning is noise.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="All-NaN slice encountered")
        return np.nanmedian(stacked, axis=0)


def add_reference_score(records: pd.DataFrame, variant: str = "injected",
                        lags: tuple[int, ...] = DEFAULT_LAGS) -> pd.DataFrame:
    """Attach a reference-based anomaly score to every record."""
    out = records.copy()
    out["reference"] = np.nan

    for station_id, g in out.groupby("station_id"):
        series = station_series(station_id, variant=variant)
        out.loc[g.index, "reference"] = multilag_reference(
            series, g["t_target"].to_numpy(), lags
        )

    scale = out.groupby("station_id")["y_true"].transform("std").clip(lower=1e-8)
    out["score"] = (out["y_true"] - out["reference"]).abs() / scale

    unresolved = out["reference"].isna().mean()
    if unresolved:
        log.info("%.2f%% of records had no resolvable lagged reference",
                 100 * unresolved)
    return out


def _auc_pr_by_station(records: pd.DataFrame, horizons: list[int]) -> pd.Series:
    """Threshold-free separability per station — what calibration cannot fix."""
    df = records[(records["split"] == "test") & (~records["is_imputed"])
                 & records["horizon"].isin(horizons)]
    rows = {}
    for (model, sid), g in df.groupby(["model", "station_id"]):
        g = g[np.isfinite(g["score"].to_numpy())]   # a lagged ref may not resolve
        if g.empty:
            continue
        y = (g["label"] == "fault").to_numpy()
        rows[(model, sid)] = average_precision(y, g["score"].to_numpy())
    return pd.Series(rows)


def run(run_dir: str | Path, variant: str = "injected",
        horizons: list[int] | None = None,
        threshold_method: str = "val_fpr_extremes") -> None:
    horizons = horizons or [1, 3, 6]
    records = load_records(run_dir)
    log.info("loaded %s records", f"{len(records):,}")

    # --- the model-residual detector, as the control -----------------------
    base = records.copy()
    base["score"] = apply_score(base, "absolute")
    base_th = calibrate(base, threshold_method)
    base_sub = subtype_recall_table(
        base[base["split"] == "test"], base_th, horizons=horizons)

    # --- the uncontaminated reference --------------------------------------
    ref = add_reference_score(records, variant=variant)
    ref_th = calibrate(ref, threshold_method)
    ref_sub = subtype_recall_table(
        ref[ref["split"] == "test"], ref_th, horizons=horizons)

    merged = base_sub.merge(
        ref_sub, on=["model", "region", "station_class", "label_source",
                     "horizon", "fault_type"],
        suffixes=("_residual", "_reference"),
    )
    merged["delta"] = merged["recall_reference"] - merged["recall_residual"]

    print("\n" + "=" * 86)
    print("  Recall by fault subtype: model residual vs uncontaminated reference")
    print("  (median multi-lag reference; threshold = " + threshold_method + ")")
    print("=" * 86)
    for model in sorted(merged["model"].unique()):
        m = merged[merged["model"] == model]
        piv = m.pivot_table(index="fault_type", columns="horizon",
                            values=["recall_residual", "recall_reference"])
        print(f"\n--- {model} ---")
        print(piv.round(3).to_string())

    print("\n=== aggregate anomaly metrics, reference score ===")
    report = build_report(name="reference_diagnostic", records=ref,
                          thresholds=ref_th, horizons=horizons)
    an = report.anomaly
    for h, g in an.groupby("horizon"):
        print(f"  h={h}: recall {g['recall'].median():.3f}  "
              f"precision {g['precision'].median():.3f}  "
              f"auc_pr {g['auc_pr'].median():.3f}  "
              f"fpr_on_extremes {g['fpr_on_extremes'].median():.3f} "
              f"(max {g['fpr_on_extremes'].max():.3f})")

    print("\n=== threshold-free separability (AUC-PR, median across stations) ===")
    b = _auc_pr_by_station(base, horizons).median()
    r = _auc_pr_by_station(ref, horizons).median()
    print(f"  model residual        {b:.3f}")
    print(f"  uncontaminated ref    {r:.3f}")
    print("\nRead recall beside fpr_on_extremes: a reference that flags every "
          "genuine episode\nas well as every fault has bought nothing (§8.2).")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_dir", help="an outputs/<run> directory with records.parquet")
    ap.add_argument("--variant", default="injected")
    ap.add_argument("--horizons", nargs="*", type=int, default=[1, 3, 6])
    ap.add_argument("--threshold", default="val_fpr_extremes")
    args = ap.parse_args()
    run(args.run_dir, args.variant, args.horizons, args.threshold)


if __name__ == "__main__":
    main()
