"""Re-score a finished run from its saved records — no retraining.

A ladder run writes ``records.parquet`` containing, for every predicted hour of
both the validation and test splits: the truth, the forecast, the recent local
volatility, the imputation mask, the label and the fault subtype. Everything an
anomaly score needs is therefore already on disk.

That matters because the detector is the part of this project that needs the
most iteration (§7.4 calls `spatial_agreement` "the actual novel contribution;
expect it to need the most iteration"), while the forecaster is the part that
takes ~50 minutes to fit. Separating the two means a new scoring rule or
threshold policy can be evaluated in about a second.

Thresholds are still calibrated on **validation only**, per (model, station),
exactly as in a live run — the split column is what makes that possible offline.

    python -m src.evaluation.rescore outputs/phase1_catalonia_injected_v2 --compare
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluation.metrics import calibrate_threshold
from src.evaluation.report import build_report
from src.utils.logging_setup import get_logger

log = get_logger(__name__)

SCORE_MODES = ("absolute", "volatility")
THRESHOLD_METHODS = ("quantile", "mad", "val_f1_sweep", "val_fpr_extremes")


def load_records(run_dir: str | Path) -> pd.DataFrame:
    path = Path(run_dir) / "records.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Only runs made after records persistence was "
            "added can be re-scored; re-run the ladder to produce it."
        )
    return pd.read_parquet(path)


def apply_score(records: pd.DataFrame, mode: str,
                volatility_floor_frac: float = 0.25) -> pd.Series:
    """Recompute the anomaly score under ``mode`` from stored columns."""
    resid = (records["y_pred"] - records["y_true"]).abs()

    if mode == "absolute":
        # Per-station scale, recovered from the station's own records rather
        # than from a shared statistic (§6.5 — scalers never leave the node).
        scale = records.groupby("station_id")["y_true"].transform("std")
        return resid / scale.clip(lower=1e-8)

    if mode == "volatility":
        vol = records["local_volatility"]
        if vol.isna().all():
            raise ValueError(
                "records carry no local_volatility; re-run the ladder to store it"
            )
        scale = records.groupby("station_id")["y_true"].transform("std")
        return resid / np.maximum(vol, volatility_floor_frac * scale)

    raise ValueError(f"unknown score mode {mode!r}; use one of {SCORE_MODES}")


def calibrate(records: pd.DataFrame, method: str, quantile: float = 0.98,
              target_fpr_extremes: float = 0.10) -> dict[tuple[str, str], float]:
    """Per (model, station) thresholds, calibrated on the VALIDATION rows only."""
    val = records[records["split"] == "val"]
    if val.empty:
        raise ValueError(
            "records contain no validation rows, so a threshold cannot be "
            "calibrated without touching test (§7.4)."
        )

    codes = np.where(val["label"] == "fault", 1,
                     np.where(val["label"] == "trustworthy", 0, -1))
    val = val.assign(_code=codes)

    out: dict[tuple[str, str], float] = {}
    for (model, station), g in val.groupby(["model", "station_id"]):
        out[(model, station)] = calibrate_threshold(
            g["score"].to_numpy(), method=method, quantile=quantile,
            mask=~g["is_imputed"].to_numpy(), labels=g["_code"].to_numpy(),
            is_extreme=g["is_extreme"].to_numpy(),
            target_fpr_extremes=target_fpr_extremes,
        )
    return out


def rescore(records: pd.DataFrame, mode: str, method: str,
            horizons: list[int], quantile: float = 0.98,
            target_fpr_extremes: float = 0.10) -> pd.DataFrame:
    """Score / calibrate / evaluate, returning the per-station anomaly table."""
    rec = records.copy()
    rec["score"] = apply_score(rec, mode)
    thresholds = calibrate(rec, method, quantile, target_fpr_extremes)
    report = build_report(
        name=f"rescore_{mode}_{method}", records=rec,
        thresholds=thresholds, horizons=horizons,
        meta={"score_mode": mode, "threshold_method": method},
    )
    return report.anomaly


def compare(records: pd.DataFrame, horizons: list[int]) -> pd.DataFrame:
    """Every score mode x threshold method, summarised per horizon.

    Recall and `fpr_on_extremes` are shown side by side on purpose: a
    configuration that lifts recall while doubling the false-positive rate on
    genuine pollution episodes is not an improvement (§8.2).
    """
    rows = []
    for mode in SCORE_MODES:
        for method in THRESHOLD_METHODS:
            try:
                an = rescore(records, mode, method, horizons)
            except Exception as exc:  # noqa: BLE001 — report and continue
                log.warning("%s/%s failed: %s", mode, method, exc)
                continue
            for h, g in an.groupby("horizon"):
                rows.append({
                    "score_mode": mode,
                    "threshold": method,
                    "horizon": int(h),
                    "recall": g["recall"].median(),
                    "precision": g["precision"].median(),
                    "f1": g["f1"].median(),
                    "auc_pr": g["auc_pr"].median(),
                    "fpr_on_extremes": g["fpr_on_extremes"].median(),
                    "fpr_extremes_max": g["fpr_on_extremes"].max(),
                })
    return pd.DataFrame(rows).sort_values(["horizon", "score_mode", "threshold"])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_dir", help="an outputs/<run> directory")
    ap.add_argument("--mode", default="volatility", choices=SCORE_MODES)
    ap.add_argument("--method", default="val_f1_sweep", choices=THRESHOLD_METHODS)
    ap.add_argument("--horizons", nargs="*", type=int, default=[1, 3, 6])
    ap.add_argument("--compare", action="store_true",
                    help="sweep every score mode x threshold method")
    args = ap.parse_args()

    records = load_records(args.run_dir)
    log.info("loaded %s records (%s)", f"{len(records):,}", args.run_dir)

    if args.compare:
        table = compare(records, args.horizons)
        print("\n=== score mode x threshold method "
              "(median across stations; AUC-PR is threshold-independent) ===")
        print(table.to_string(index=False, float_format=lambda v: f"{v:7.3f}"))
        print("\nRead recall and fpr_on_extremes together: buying recall with a "
              "worse false-positive rate on genuine episodes is not a win (§8.2).")
        return

    an = rescore(records, args.mode, args.method, args.horizons)
    cols = ["model", "station_class", "station_id", "horizon", "recall",
            "precision", "f1", "auc_pr", "fpr_on_extremes"]
    print(an[cols].to_string(index=False, float_format=lambda v: f"{v:7.3f}"))


if __name__ == "__main__":
    main()
