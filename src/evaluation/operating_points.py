"""Two operating points, because one threshold cannot serve both jobs.

The faults this system finds divide cleanly by what you would *do* about them:

* **Spike, noise, dropout** are incidents. Someone wants to know now, so a false
  alarm is expensive and the threshold must be tight.
* **Drift, bias, stuck-at** are calibration problems. Nobody needs paging at
  3am; a technician needs a monthly list of instruments worth checking. There a
  false alarm costs one cheap inspection, while a miss costs months of quietly
  wrong data — so a loose threshold is correct, and a precision of 1-in-20 is
  perfectly workable.

Collapsing both into a single F1 is what made rung 6 look worse than it is
(CONTEXT.md entries 10-11): the tight setting that maximises aggregate F1 is
exactly the setting that gives up drift and bias entirely.

Both tiers are computed offline from the per-detector scores stored in
`records.parquet`, so changing either costs seconds rather than a retrain.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluation.magnitude import attach_magnitudes, stratified_recall
from src.evaluation.metrics import calibrate_threshold
from src.evaluation.report import build_report, subtype_recall_table
from src.evaluation.rescore import load_records
from src.utils.logging_setup import get_logger
from src.utils.paths import load_config

log = get_logger(__name__)

#: Fallbacks if configs/model.yaml carries no operating_points block.
DEFAULT_TIERS = {"alert": 0.005, "review": 0.20}


def detector_columns(records: pd.DataFrame) -> list[str]:
    return [c for c in records.columns
            if c.startswith("det_") and records[c].notna().any()]


def score_at_budget(records: pd.DataFrame, total_budget: float) -> pd.DataFrame:
    """Re-threshold every detector at ``total_budget / k`` and recombine.

    Each detector's score becomes its margin over its own budgeted threshold,
    so the combined score crosses zero exactly when some detector fires — the
    same rule the live decision layer uses (§7.4).
    """
    dets = detector_columns(records)
    if not dets:
        raise ValueError(
            "records carry no det_* columns, so operating points cannot be "
            "recomputed offline. Re-run rung 6 to persist them."
        )
    per = total_budget / len(dets)

    codes = np.where(records["label"] == "fault", 1,
                     np.where(records["label"] == "trustworthy", 0, -1))
    work = records.assign(_code=codes)
    val = work[work["split"] == "val"]

    margins = []
    for det in dets:
        thresholds = {}
        for (model, station), g in val.groupby(["model", "station_id"]):
            values = g[det].to_numpy()
            if not np.isfinite(values).any():
                continue
            thresholds[(model, station)] = calibrate_threshold(
                values, method="val_fpr_overall", labels=g["_code"].to_numpy(),
                target_fpr_extremes=per)
        keyed = np.array([thresholds.get((m, s), np.inf)
                          for m, s in zip(work["model"], work["station_id"])])
        margins.append(work[det].to_numpy() - keyed)

    out = records.copy()
    out["score"] = np.nanmax(np.vstack(margins), axis=0)
    return out


def evaluate_tier(records: pd.DataFrame, budget: float, horizons: list[int]) -> dict:
    """Aggregate, per-subtype and per-magnitude metrics at one operating point."""
    scored = score_at_budget(records, budget)
    # The combined score is a margin, so the decision boundary is exactly zero.
    thresholds = {k: 0.0 for k in set(zip(scored["model"], scored["station_id"]))}

    report = build_report(f"tier_{budget}", scored, thresholds, horizons=horizons)
    test = scored[scored["split"] == "test"]
    return {
        "budget": budget,
        "aggregate": report.anomaly,
        "subtype": subtype_recall_table(test, thresholds, horizons=[horizons[0]]),
        "magnitude": stratified_recall(scored, thresholds, horizon=horizons[0]),
    }


def run(run_dir: str | Path, horizons: list[int] | None = None,
        tiers: dict[str, float] | None = None) -> None:
    horizons = horizons or [1, 3, 6]
    if tiers is None:
        cfg = load_config("model").get("decision_layer", {}).get("operating_points")
        tiers = ({name: float(v["target_fpr_overall"]) for name, v in cfg.items()}
                 if cfg else DEFAULT_TIERS)

    records = attach_magnitudes(load_records(run_dir))
    log.info("loaded %s records; tiers: %s", f"{len(records):,}", tiers)

    results = {name: evaluate_tier(records, budget, horizons)
               for name, budget in tiers.items()}

    print("\n" + "=" * 78)
    print("  TWO OPERATING POINTS")
    print("=" * 78)
    for name, res in results.items():
        a = res["aggregate"]
        print(f"\n--- {name.upper()} (per-detector budget {res['budget']}) ---")
        for h, g in a.groupby("horizon"):
            print(f"  h={h}: recall {g.recall.median():.3f}  "
                  f"precision {g.precision.median():.3f}  "
                  f"F1 {g.f1.median():.3f}  "
                  f"fpr_extremes {g.fpr_on_extremes.median():.3f}")

    print("\n=== recall by subtype at each operating point (h=%d) ===" % horizons[0])
    sub = pd.DataFrame({
        name: res["subtype"].groupby("fault_type")["recall"].mean()
        for name, res in results.items() if not res["subtype"].empty})
    print(sub.round(3).to_string())

    print("\n=== recall by fault magnitude at each operating point ===")
    mag = pd.DataFrame({
        name: res["magnitude"].set_index("band")["recall"]
        for name, res in results.items() if not res["magnitude"].empty})
    print(mag.round(3).to_string())

    print("\nAlert tier answers 'wake someone up'. Review tier answers 'what "
          "should a\ntechnician check this month', where a cheap inspection "
          "beats months of bad data.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_dir")
    ap.add_argument("--horizons", nargs="*", type=int, default=[1, 3, 6])
    args = ap.parse_args()
    run(args.run_dir, args.horizons)


if __name__ == "__main__":
    main()
