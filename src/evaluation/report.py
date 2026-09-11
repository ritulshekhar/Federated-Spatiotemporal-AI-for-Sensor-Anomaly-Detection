"""§8.3 — Reporting. Grouped by construction; pooling is not reachable.

The rule: *every* result table is grouped by ``(region, station_class,
label_source)`` at minimum, and no result is ever a single pooled number.

§8.3 says to build this module around that rule from the start "so it's
structurally impossible to accidentally produce a misleading pooled figure".
That is implemented literally:

* Every aggregation goes through ``_grouped``, which raises unless all of
  ``REQUIRED_GROUP_KEYS`` are present *and* included in the grouping.
* There is no function that returns a scalar metric across stations. The
  closest thing, ``spread_table``, returns count/min/median/max — a
  *distribution* over per-station values, never a mean standing in for them —
  and refuses to run unless the per-station table was produced first.

The unit of record everywhere below is a long *prediction record* frame, one
row per (model, station, horizon, forecast hour):

    model | region | station_id | station_class | label_source | horizon |
    t_target | y_true | y_pred | is_imputed | label | is_extreme | score

Building tables from records rather than from pre-aggregated numbers is what
keeps the grouping honest — you cannot un-pool a mean, so nothing is ever
pooled on the way in.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluation.metrics import (
    MAPE_FLOOR, anomaly_metrics, average_precision, forecast_metrics,
)
from src.utils.logging_setup import get_logger
from src.utils.paths import data_paths

log = get_logger(__name__)

#: Non-negotiable (§8.3). Every table carries all three.
REQUIRED_GROUP_KEYS = ("region", "station_class", "label_source")

RECORD_COLUMNS = [
    "model", "region", "station_id", "station_class", "label_source",
    "horizon", "t_target", "y_true", "y_pred", "is_imputed", "label",
    "is_extreme", "score",
]


class PoolingError(RuntimeError):
    """Raised on any attempt to aggregate without the §8.3 group keys."""


def _grouped(df: pd.DataFrame, by: list[str]) -> pd.core.groupby.DataFrameGroupBy:
    missing_cols = [k for k in REQUIRED_GROUP_KEYS if k not in df.columns]
    if missing_cols:
        raise PoolingError(
            f"prediction records are missing required group key(s) {missing_cols}. "
            "CLAUDE.md §5.3 forbids dropping region/label_source downstream and "
            "§8.3 requires every table to be grouped by them."
        )
    missing_keys = [k for k in REQUIRED_GROUP_KEYS if k not in by]
    if missing_keys:
        raise PoolingError(
            f"grouping by {by} would pool across {missing_keys}. §8.3: every "
            "result table is grouped by (region, station_class, label_source) at "
            "minimum. Never a single pooled number for anything."
        )
    return df.groupby(by, dropna=False, observed=True)


# ---------------------------------------------------------------------------
def make_records(
    *,
    model: str,
    station_id: str,
    region: str,
    station_class: str,
    label_source: str,
    t_target: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    is_imputed: np.ndarray,
    label: np.ndarray,
    is_extreme: np.ndarray,
    score: np.ndarray,
    horizons: list[int],
) -> pd.DataFrame:
    """Flatten one station's (n_windows, horizon) arrays into long records.

    ``horizons`` names the lead time of each column: ``[1, 2, ..., H]``, of
    which §8.2 reports 1/3/6.
    """
    n, h = np.asarray(y_true).shape
    if len(horizons) != h:
        raise ValueError(f"{len(horizons)} horizon labels for {h} columns")

    return pd.DataFrame({
        "model": model,
        "region": region,
        "station_id": station_id,
        "station_class": station_class,
        "label_source": label_source,
        "horizon": np.tile(np.asarray(horizons), n),
        "t_target": np.asarray(t_target).ravel(),
        "y_true": np.asarray(y_true, dtype=float).ravel(),
        "y_pred": np.asarray(y_pred, dtype=float).ravel(),
        "is_imputed": np.asarray(is_imputed, dtype=bool).ravel(),
        "label": np.asarray(label).ravel(),
        "is_extreme": np.asarray(is_extreme, dtype=bool).ravel(),
        "score": np.asarray(score, dtype=float).ravel(),
    })[RECORD_COLUMNS]


# ---------------------------------------------------------------------------
def forecast_table(records: pd.DataFrame,
                   horizons: list[int] | None = None,
                   exclude_imputed: bool = True) -> pd.DataFrame:
    """RMSE / MAE / MAPE per (model, region, station_class, label_source,
    station, horizon). §8.2's "per station, per horizon, per station class"."""
    df = records
    if horizons is not None:
        df = df[df["horizon"].isin(horizons)]

    by = ["model", "region", "station_class", "label_source", "station_id", "horizon"]
    rows = []
    for keys, g in _grouped(df, by):
        mask = ~g["is_imputed"].to_numpy() if exclude_imputed else None
        m = forecast_metrics(g["y_true"].to_numpy(), g["y_pred"].to_numpy(), mask)
        rows.append(dict(zip(by, keys)) | m.as_dict())

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(by).reset_index(drop=True)
    return out


def anomaly_table(records: pd.DataFrame,
                  thresholds: dict[tuple[str, str], float],
                  horizons: list[int] | None = None,
                  exclude_imputed: bool = True,
                  exclude_unverified: bool = False) -> pd.DataFrame:
    """Precision / recall / F1 / AUC-PR **and** the false-positive control.

    ``thresholds`` maps ``(model, station_id) -> threshold``, each calibrated on
    that station's own validation split under that model (§7.4).

    Keyed by *both* on purpose. A scalar threshold is the §5.8 miscalibration
    failure mode. But keying by station alone is wrong too: rungs produce
    residuals of different magnitudes — persistence is far worse than a trained
    GRU — so one station-level threshold applied across models would silently
    compare them at incomparable operating points.
    """
    df = records
    if horizons is not None:
        df = df[df["horizon"].isin(horizons)]
    if exclude_unverified and "verification" in df.columns:
        df = df[df["verification"] != 3]

    by = ["model", "region", "station_class", "label_source", "station_id", "horizon"]
    rows = []
    for keys, g in _grouped(df, by):
        keyed = dict(zip(by, keys))
        tkey = (keyed["model"], keyed["station_id"])
        if tkey not in thresholds:
            raise PoolingError(
                f"no calibrated threshold for {tkey}. Thresholds are per (model, "
                "station) and calibrated on validation (§7.4); falling back to a "
                "shared one is the §5.8 miscalibration failure mode."
            )
        mask = ~g["is_imputed"].to_numpy() if exclude_imputed else None
        m = anomaly_metrics(
            y_label=_label_codes(g["label"]),
            score=g["score"].to_numpy(),
            threshold=thresholds[tkey],
            is_extreme=g["is_extreme"].to_numpy(),
            mask=mask,
        )
        rows.append(keyed | {"threshold": thresholds[tkey]} | m.as_dict())

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(by).reset_index(drop=True)
    return out


def spread_table(per_station: pd.DataFrame, value_columns: list[str]) -> pd.DataFrame:
    """Distribution of a per-station metric across the stations in a group.

    Reports n / min / median / max. Deliberately **not** a mean: a mean would
    read as "the" number for a region, which is the pooled figure §8.3 exists
    to prevent, and would let a low-variance station drag the headline (§2.2).
    A spread makes the disagreement between stations visible instead.
    """
    if per_station.empty:
        return per_station
    missing = [c for c in value_columns if c not in per_station.columns]
    if missing:
        raise KeyError(f"columns not in the per-station table: {missing}")
    if "station_id" not in per_station.columns:
        raise PoolingError(
            "spread_table operates on a per-station table. Build one with "
            "forecast_table()/anomaly_table() first — this function summarises "
            "per-station values, it does not replace them."
        )

    by = ["model", "region", "station_class", "label_source"]
    if "horizon" in per_station.columns:
        by.append("horizon")

    agg = {"station_id": "nunique"}
    for c in value_columns:
        agg[c] = ["min", "median", "max"]

    out = _grouped(per_station, by).agg(agg)
    out.columns = ["n_stations"] + [f"{a}_{b}" for a, b in out.columns[1:]]
    return out.reset_index()


def _label_codes(labels: pd.Series) -> np.ndarray:
    """"fault"/"trustworthy"/anything else -> 1 / 0 / -1."""
    s = labels.astype("object")
    out = np.full(len(s), -1, dtype=np.int8)
    out[(s == "trustworthy").to_numpy()] = 0
    out[(s == "fault").to_numpy()] = 1
    return out


# ---------------------------------------------------------------------------
@dataclass
class Report:
    """One run's tables, written together so no table travels without context."""
    name: str
    forecast: pd.DataFrame
    anomaly: pd.DataFrame
    forecast_spread: pd.DataFrame
    anomaly_spread: pd.DataFrame
    meta: dict

    def write(self, directory: str | Path | None = None) -> Path:
        directory = Path(directory) if directory else data_paths()["outputs"] / self.name
        directory.mkdir(parents=True, exist_ok=True)
        for label, df in (
            ("forecast_per_station", self.forecast),
            ("anomaly_per_station", self.anomaly),
            ("forecast_spread", self.forecast_spread),
            ("anomaly_spread", self.anomaly_spread),
        ):
            df.to_csv(directory / f"{label}.csv", index=False)
        (directory / "meta.json").write_text(
            json.dumps(self.meta, indent=2, default=str), encoding="utf-8"
        )
        log.info("report written -> %s", directory)
        return directory

    def print_summary(self, horizons: list[int] | None = None) -> None:
        hs = horizons or sorted(self.forecast["horizon"].unique())
        fc = self.forecast[self.forecast["horizon"].isin(hs)]

        print(f"\n{'=' * 78}")
        print(f"  {self.name} — forecasting, per station per horizon (§8.2/§8.3)")
        print(f"  MAPE excludes |y| < {MAPE_FLOOR} ug/m3; imputed points excluded")
        print("=" * 78)
        cols = ["model", "region", "station_class", "station_id", "horizon",
                "n", "rmse", "mae", "mape", "bias"]
        with pd.option_context("display.width", 200, "display.max_rows", 400):
            print(fc[cols].to_string(index=False,
                                     float_format=lambda v: f"{v:8.3f}"))

        if not self.anomaly.empty:
            an = self.anomaly[self.anomaly["horizon"].isin(hs)]
            print(f"\n{'=' * 78}")
            print(f"  {self.name} — anomaly classification")
            print("  fpr_on_extremes is the Gupchup (2019) control: genuine high")
            print("  readings wrongly called faults. Recall alone means nothing.")
            print("=" * 78)
            acols = ["model", "region", "station_class", "station_id", "horizon",
                     "label_source", "n", "n_positive", "precision", "recall",
                     "f1", "auc_pr", "fpr_on_extremes", "n_extremes"]
            with pd.option_context("display.width", 220, "display.max_rows", 400):
                print(an[acols].to_string(index=False,
                                          float_format=lambda v: f"{v:7.3f}"))


def build_report(name: str, records: pd.DataFrame,
                 thresholds: dict[tuple[str, str], float],
                 horizons: list[int] | None = None, meta: dict | None = None) -> Report:
    """The single sanctioned way to turn prediction records into a report."""
    fc = forecast_table(records, horizons=horizons)
    an = anomaly_table(records, thresholds=thresholds, horizons=horizons)
    return Report(
        name=name,
        forecast=fc,
        anomaly=an,
        forecast_spread=spread_table(fc, ["rmse", "mae", "mape"]),
        anomaly_spread=spread_table(
            an, ["precision", "recall", "f1", "auc_pr", "fpr_on_extremes"]
        ) if not an.empty else an,
        meta=(meta or {}) | {
            "n_records": int(len(records)),
            "models": sorted(records["model"].unique().tolist()),
            "stations": sorted(records["station_id"].unique().tolist()),
            "regions": sorted(records["region"].unique().tolist()),
            "label_sources": sorted(records["label_source"].unique().tolist()),
            "horizons_reported": horizons,
        },
    )
