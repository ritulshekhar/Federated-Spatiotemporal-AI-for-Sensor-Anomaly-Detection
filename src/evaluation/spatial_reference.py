"""Can a neighbouring station see a fault the station itself cannot?

This is the assumption the whole project rests on (§1.2, §7.4), and it is
cheap to test before building the graph layer. Entry 7 established that drift,
bias, stuck-at and dropout fall inside a station's own normal spread at every
temporal timescale, and argued that a *neighbour* is the only reference that
does not move with the fault.

The test here is deliberately the **easy version**: raw neighbour readings,
centralised, no privacy constraint, no embedding compression. That is exactly
Ferrer-Cid et al. (2022), §1.2's centralised precedent. So whatever it scores
is a **ceiling** on what rung 6 can achieve from compressed embeddings — and
the gap between the two is the price of the privacy constraint, measured
rather than asserted.

Method. Each station's target series is z-scored on its **own** training window
(§6.5 — no shared scaler, so this comparison stays legitimate even though the
oracle itself ignores the privacy constraint). A station's spatial deviation is
then its distance from the median of the other stations at the same hour:

    score_s(t) = | z_s(t) - median_{r != s} z_r(t) |

Taking the median over neighbours, not the mean, matters: with seven stations
and ~2.8% fault rate, a neighbour is occasionally faulty itself, and a median
ignores it.

Accumulation windows are swept because entry 7 showed no single timescale
serves every subtype — spikes need one hour, bias about a day, and drift
episodes run 72h or longer by construction.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluation.rescore import apply_score, calibrate, load_records
from src.evaluation.report import build_report, subtype_recall_table
from src.evaluation.temporal_reference import station_series
from src.preprocessing.normalize import LocalScaler
from src.utils.logging_setup import get_logger
from src.utils.paths import station_dir

log = get_logger(__name__)

#: Entry 7: spikes are a one-hour event, bias emerges over a day, drift
#: episodes are >= 72h. One window cannot serve all three.
DEFAULT_WINDOWS = (1, 6, 24, 72, 168)


def z_scored_matrix(station_ids: list[str], variant: str = "injected") -> pd.DataFrame:
    """Station x time matrix of target values, each z-scored on its own train window."""
    cols = {}
    for sid in station_ids:
        d = station_dir(sid, "processed", variant=variant)
        target = json.loads((d / "manifest.json").read_text(encoding="utf-8"))["target"]
        scaler = LocalScaler.load(d / "scaler.json")
        series = station_series(sid, variant=variant)
        cols[sid] = (series - scaler.mean_[target]) / scaler.std_[target]
    matrix = pd.DataFrame(cols).sort_index()
    log.info("built %d x %d station-hour matrix", matrix.shape[1], matrix.shape[0])
    return matrix


def spatial_deviation(matrix: pd.DataFrame) -> pd.DataFrame:
    """Each station's distance from the median of its neighbours, per hour."""
    out = {}
    for sid in matrix.columns:
        neighbours = matrix.drop(columns=sid).median(axis=1, skipna=True)
        out[sid] = (matrix[sid] - neighbours).abs()
    return pd.DataFrame(out)


def accumulate(deviation: pd.DataFrame, window: int) -> pd.DataFrame:
    """Trailing mean of the deviation. ``window=1`` is the per-reading score."""
    if window <= 1:
        return deviation
    return deviation.rolling(window, min_periods=max(2, window // 4)).mean()


def attach(records: pd.DataFrame, score_frame: pd.DataFrame) -> pd.DataFrame:
    """Map a station x time score frame onto the records by (station, timestamp)."""
    long = (score_frame.stack().rename("score")
            .rename_axis(["t_target", "station_id"]).reset_index())
    out = records.drop(columns=["score"]).merge(
        long, on=["station_id", "t_target"], how="left")
    return out


def run(run_dir: str | Path, variant: str = "injected",
        windows: tuple[int, ...] = DEFAULT_WINDOWS,
        horizon: int = 1, threshold_method: str = "val_fpr_extremes") -> None:
    records = load_records(run_dir)
    records = records[records["horizon"] == horizon].copy()
    stations = sorted(records["station_id"].unique())
    log.info("%d stations, %s records at h=%d", len(stations),
             f"{len(records):,}", horizon)

    # --- control: the station's own model residual -------------------------
    base = records.copy()
    base["score"] = apply_score(base, "absolute")
    base_sub = subtype_recall_table(
        base[base["split"] == "test"], calibrate(base, threshold_method),
        horizons=[horizon])
    control = (base_sub.groupby("fault_type")["recall"].mean()
               if not base_sub.empty else pd.Series(dtype=float))

    matrix = z_scored_matrix(stations, variant=variant)
    deviation = spatial_deviation(matrix)

    per_window, aggregate = {}, {}
    for w in windows:
        scored = attach(records, accumulate(deviation, w))
        scored = scored[np.isfinite(scored["score"])]
        if scored.empty or scored["split"].nunique() < 2:
            log.warning("window %dh produced no usable scores", w)
            continue
        th = calibrate(scored, threshold_method)
        test = scored[scored["split"] == "test"]
        sub = subtype_recall_table(test, th, horizons=[horizon])
        if not sub.empty:
            per_window[f"{w}h"] = sub.groupby("fault_type")["recall"].mean()
        an = build_report(name=f"spatial_{w}h", records=scored,
                          thresholds=th, horizons=[horizon]).anomaly
        aggregate[f"{w}h"] = {
            "recall": an["recall"].median(),
            "precision": an["precision"].median(),
            "auc_pr": an["auc_pr"].median(),
            "fpr_on_extremes": an["fpr_on_extremes"].median(),
            "fpr_extremes_max": an["fpr_on_extremes"].max(),
        }

    table = pd.DataFrame(per_window)
    table.insert(0, "own_residual", control)

    print("\n" + "=" * 92)
    print("  CEILING TEST — recall by fault subtype, raw neighbour readings")
    print("  (centralised, no privacy constraint: an upper bound for rung 6)")
    print("  Columns are accumulation windows for the spatial deviation.")
    print("=" * 92)
    print(table.round(3).to_string())

    print("\n=== aggregate, spatial score (median across stations) ===")
    print(pd.DataFrame(aggregate).T.round(3).to_string())
    print("\nRead recall beside fpr_on_extremes: a neighbour comparison that "
          "flags every\nregional pollution episode has bought nothing (§8.2).")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_dir", help="an outputs/<run> directory with records.parquet")
    ap.add_argument("--variant", default="injected")
    ap.add_argument("--windows", nargs="*", type=int, default=list(DEFAULT_WINDOWS))
    ap.add_argument("--horizon", type=int, default=1)
    ap.add_argument("--threshold", default="val_fpr_extremes")
    args = ap.parse_args()
    run(args.run_dir, args.variant, tuple(args.windows), args.horizon, args.threshold)


if __name__ == "__main__":
    main()
