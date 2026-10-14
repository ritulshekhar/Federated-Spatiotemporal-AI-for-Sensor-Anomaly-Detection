"""Collect the dashboard's data bundle from saved run outputs.

Objective 7 asks for a dashboard "visualising station data, forecasts, and the
trustworthy/faulty decision". This builds the JSON it renders from, so the
dashboard never recomputes anything — every figure traces to a run under
`outputs/`, and regenerating is a second rather than a retrain.

Deliberately includes the uncomfortable numbers. A dashboard that showed only
the headline recall would misrepresent the project: §8.2 bans recall without
its false-positive control, the bootstrap shows which rung differences are
inside their confidence intervals, and the aggregate F1 hides a 10x spread
across fault sizes. All of it travels with the data.

Runs it reads:
    phase5_scale30        30 stations, corrected injector, meteo  -> headline
    phase4_drift_fixed    7 stations, same injector               -> n=7 vs n=30
    phase1_catalonia*     clean-data forecast error
    phase6_transfer_india Catalonia -> Delhi generalization (§5.8)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.station_metadata import configured_stations, coordinates
from src.utils.logging_setup import get_logger
from src.utils.paths import data_paths

log = get_logger(__name__)

OUT = Path(__file__).parent / "data.json"

#: The headline run. Everything that says "the system" means this one.
SCALE_RUN = "phase5_scale30"
SMALL_RUN = "phase4_drift_fixed"
TRANSFER_RUN = "phase6_transfer_india"


def _csv(run: str, name: str) -> pd.DataFrame:
    path = data_paths()["outputs"] / run / f"{name}.csv"
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def _json(run: str, name: str) -> dict:
    path = data_paths()["outputs"] / run / f"{name}.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def stations_block() -> list[dict]:
    """Station identity, class and coordinates for the map."""
    coords = coordinates()
    out = []
    for s in configured_stations():
        lat, lon = coords.get(s.station_id, (None, None))
        if lat is None:
            continue
        out.append({"id": s.station_id, "name": s.name,
                    "klass": s.station_class, "lat": lat, "lon": lon})
    return out


def forecasting_block() -> dict:
    """Clean-data forecast error, with and without meteorology."""
    out: dict = {}
    for label, run in (("without_met", "phase1_catalonia"),
                       ("with_met", "phase1_catalonia_meteo")):
        fc = _csv(run, "forecast_per_station")
        if fc.empty:
            continue
        agg = (fc.groupby(["model", "horizon"])["rmse"]
                 .agg(["median", "min", "max"]).reset_index())
        out[label] = agg.rename(columns={"median": "rmse"}).to_dict("records")

    # Privacy cost, from the headline run: the gap between the no-privacy
    # ceiling and the federated rungs, plus the local-only baseline that beats
    # both on forecasting.
    fc = _csv(SCALE_RUN, "forecast_per_station")
    if not fc.empty:
        out["scale_rmse"] = (fc.groupby(["model", "horizon"])["rmse"]
                             .median().round(2).reset_index().to_dict("records"))
    return out


def ladder_block() -> list[dict]:
    """Every rung, forecasting and detection side by side, from the scale run."""
    fc, an = _csv(SCALE_RUN, "forecast_per_station"), _csv(SCALE_RUN, "anomaly_per_station")
    if fc.empty:
        return []
    f = fc.groupby(["model", "horizon"])["rmse"].median().reset_index()
    a = (an.groupby(["model", "horizon"])[["recall", "precision", "f1", "auc_pr",
                                           "fpr_on_extremes"]].median().reset_index()
         if not an.empty else pd.DataFrame())
    merged = f.merge(a, on=["model", "horizon"], how="left") if not a.empty else f
    return merged.round(3).to_dict("records")


def subtype_block() -> list[dict]:
    """Recall per fault subtype at both node-set sizes — the project's result."""
    rows = []
    for tag, run in (("n7", SMALL_RUN), ("n30", SCALE_RUN)):
        sub = _csv(run, "subtype_recall")
        if sub.empty:
            continue
        g = (sub[sub["horizon"] == 1].groupby("fault_type")["recall"]
             .mean().round(3))
        for fault_type, recall in g.items():
            rows.append({"group": tag, "fault_type": fault_type,
                         "recall": float(recall)})
    return rows


def drift_block() -> dict:
    """§2.4a station-level drift identification, at both node-set sizes.

    Reported separately from the per-reading anomaly table on purpose: §2.4a
    permits one ordinal per station per upload, so this names *stations* whose
    calibration has diverged, never individual readings.
    """
    out: dict = {}
    for tag, run in (("n7", SMALL_RUN), ("n30", SCALE_RUN)):
        meta = _json(run, "meta")
        di = meta.get("drift_identification")
        if not di:
            for v in meta.values():
                if isinstance(v, dict) and "drift_identification" in v:
                    di = v["drift_identification"]
                    break
        if not di:
            continue
        tp, fp, fn = di["true_positive"], di["false_positive"], di["missed"]
        out[tag] = {
            "true_positive": tp, "false_positive": fp, "missed": fn,
            "precision": round(tp / (tp + fp), 3) if tp + fp else None,
            "recall": round(tp / (tp + fn), 3) if tp + fn else None,
            "buckets": di.get("buckets", {}),
        }
    return out


def transfer_block() -> dict:
    """§5.8 — Catalonia-trained model applied to Delhi without retraining."""
    fc = _csv(TRANSFER_RUN, "forecast_per_station")
    det = _csv(TRANSFER_RUN, "detection_by_calibration")
    spread = _csv(TRANSFER_RUN, "threshold_spread")
    if fc.empty:
        return {}
    out = {
        "forecast": (fc.groupby(["region", "horizon"])[["rmse", "mae", "mape"]]
                     .median().round(2).reset_index().to_dict("records")),
        "summary": _json(TRANSFER_RUN, "summary"),
    }
    if not det.empty:
        out["detection"] = det.round(3).to_dict("records")
    if not spread.empty:
        out["spread"] = spread.round(3).to_dict("records")
    return out


def uncertainty_block() -> dict:
    """Bootstrap intervals. Hand-entered from the uncertainty tool's output.

    These are the numbers that decide what the project may claim, so they are
    written down explicitly rather than re-derived by a fragile parse of a
    console table. Regenerate with:
        python -m src.evaluation.uncertainty outputs/phase5_scale30 \
               --horizon 3 --compare 6_proposed 5_fedavg_gru --level both
    """
    return {
        "level": "stations and time resampled, horizon 3, n=30",
        "vs_fedavg": [
            {"metric": "recall", "diff": 0.119, "lo": 0.051, "hi": 0.200, "sig": True},
            {"metric": "precision", "diff": -0.191, "lo": -0.288, "hi": -0.094, "sig": True},
            {"metric": "F1", "diff": 0.059, "lo": -0.007, "hi": 0.136, "sig": False},
            {"metric": "AUC-PR", "diff": 0.017, "lo": -0.051, "hi": 0.097, "sig": False},
            {"metric": "FPR on extremes", "diff": 0.028, "lo": -0.006, "hi": 0.056, "sig": False},
        ],
        "vs_centralised": [
            {"metric": "recall", "diff": 0.071, "lo": 0.015, "hi": 0.116, "sig": True},
            {"metric": "precision", "diff": -0.211, "lo": -0.269, "hi": -0.113, "sig": True},
            {"metric": "F1", "diff": 0.002, "lo": -0.034, "hi": 0.045, "sig": False},
            {"metric": "AUC-PR", "diff": -0.021, "lo": -0.086, "hi": 0.038, "sig": False},
            {"metric": "FPR on extremes", "diff": 0.044, "lo": -0.004, "hi": 0.064, "sig": False},
        ],
        "n7_note": ("At 7 stations nothing survived station resampling: recall "
                    "+0.041 [-0.071, 0.151], F1 +0.006 [-0.102, 0.113]."),
    }


def findings_block() -> list[dict]:
    """The measured claims, each with its caveat. Hand-written on purpose.

    These are conclusions, not aggregates — they cannot be derived from a CSV,
    and stating them without their caveats would be the main way this dashboard
    could mislead.
    """
    return [
        {
            "title": "Neighbours find faults, at a price, and both halves are real",
            "detail": "At 30 stations the proposed model detects more faults "
                      "than federated training without spatial reasoning "
                      "(recall +0.119) and false-alarms more often "
                      "(precision −0.191). Both intervals exclude zero under "
                      "station and time resampling.",
            "caveat": "F1 cannot separate them: +0.059 [−0.007, 0.136]. Which "
                      "side matters is an operating-point choice, not a result.",
            "tone": "good",
        },
        {
            "title": "Privacy costs nothing measurable in F1",
            "detail": "Against a centralised model with full raw access, the "
                      "federated model differs by +0.002 F1 "
                      "[−0.034, 0.045] — statistically indistinguishable, "
                      "while never seeing a raw neighbour reading.",
            "caveat": "It does trade precision for recall against that ceiling "
                      "too, and forecasting still pays 4–7% RMSE.",
            "tone": "good",
        },
        {
            "title": "Seven stations were too few to conclude anything",
            "detail": "The same comparison at 7 stations gave recall "
                      "+0.041 [−0.071, 0.151] — an interval straddling zero. "
                      "Nothing survived station-level resampling until the node "
                      "set reached 30.",
            "caveat": None,
            "tone": "warn",
        },
        {
            "title": "Federation does not help forecasting here",
            "detail": "Purely local per-station models beat the federated "
                      "global model at every horizon (13.70/16.18/17.04 against "
                      "13.80/16.55/17.31 RMSE). Across 30 heterogeneous "
                      "stations, one shared forecaster is worse than many local "
                      "ones.",
            "caveat": "Federation earns its place through the spatial half, "
                      "not through weight sharing.",
            "tone": "bad",
        },
        {
            "title": "Accuracy is the wrong metric here",
            "detail": "At a 2.8% fault rate a classifier answering “never "
                      "faulty” scores 0.972 accuracy — higher than this system. "
                      "Recall, precision and AUC-PR are what mean anything.",
            "caveat": None,
            "tone": "warn",
        },
        {
            "title": "Drift is detectable per station, not per reading",
            "detail": "A coarse ordinal level summary, uploaded once per "
                      "station per round, identifies drifting stations with "
                      "0.688 precision and 0.647 recall at 30 stations. "
                      "Per-reading drift recall remains ~0.01.",
            "caveat": "It failed completely at 7 stations: all seven sat in the "
                      "same bucket, because one city's stations co-move. The "
                      "detector needs dispersion among neighbours, not just "
                      "neighbours.",
            "tone": "good",
        },
        {
            "title": "EEA's fault labels contain no reading",
            "detail": "All 27,783 Validity-flagged rows carry the −999 no-data "
                      "sentinel, so the readings this project exists to "
                      "adjudicate are absent rather than wrong. Evaluation uses "
                      "rate-matched synthetic faults instead.",
            "caveat": "Injection rate (2.805%) and episode durations are taken "
                      "from the real data; only the corrupted value is "
                      "synthesised.",
            "tone": "warn",
        },
        {
            "title": "Transferring to Delhi breaks calibration, not forecasting",
            "detail": "A Catalonia-trained model applied to Delhi unchanged "
                      "holds its relative forecast error (MAPE 68.7% against "
                      "68.3% at 6h), but per-station thresholds span 27× there "
                      "versus 1.6× in Catalonia.",
            "caveat": "Delhi detects better with Catalonia's threshold "
                      "(precision 0.564) than its own (0.092) — its validation "
                      "window is too short to calibrate stably.",
            "tone": "bad",
        },
    ]


def build() -> dict:
    an = _csv(SCALE_RUN, "anomaly_per_station")
    n_stations = int(an["station_id"].nunique()) if not an.empty else 30
    # Which stations the headline run actually trained on, so the map can mark
    # them rather than the page hardcoding a list that goes stale every time
    # the node set changes.
    node_set = (sorted(an["station_id"].unique().tolist())
                if not an.empty else [])
    return {
        "node_set": node_set,
        "generated": pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d %H:%M UTC"),
        "stations": stations_block(),
        "forecasting": forecasting_block(),
        "ladder": ladder_block(),
        "subtypes": subtype_block(),
        "drift": drift_block(),
        "transfer": transfer_block(),
        "uncertainty": uncertainty_block(),
        "operating_points": {
            "alert": {"recall": 0.315, "precision": 0.480, "f1": 0.390,
                      "fpr_extremes": 0.091},
            "review": {"recall": 0.575, "precision": 0.066, "f1": 0.118,
                       "fpr_extremes": 0.610},
            "by_magnitude": [
                {"band": "<0.5σ", "alert": 0.139, "review": 0.288},
                {"band": "0.5–1σ", "alert": 0.134, "review": 0.378},
                {"band": "1–2σ", "alert": 0.130, "review": 0.642},
                {"band": "2–5σ", "alert": 0.583, "review": 0.923},
                {"band": ">5σ", "alert": 0.924, "review": 1.000},
            ],
            "by_subtype": [
                {"subtype": "spike", "alert": 0.662, "review": 0.946},
                {"subtype": "noise", "alert": 0.546, "review": 0.930},
                {"subtype": "stuck_at", "alert": 0.136, "review": 0.267},
                {"subtype": "dropout", "alert": 0.120, "review": 0.450},
                {"subtype": "bias", "alert": 0.006, "review": 0.494},
                {"subtype": "drift", "alert": 0.000, "review": 0.132},
            ],
            "note": ("Measured on the 7-station build; the operating-point "
                     "split has not been recomputed at 30 stations."),
        },
        "dataset": {
            "region": "Catalonia", "stations": n_stations,
            "years": "2017–2024", "fault_rate": 0.02805, "target": "NO2",
            "india_stations": 7, "india_years": "2025–2026",
            "device": "NVIDIA RTX 4060 (CUDA 13.0)",
        },
        "findings": findings_block(),
    }


def main() -> None:
    bundle = build()
    OUT.write_text(json.dumps(bundle, indent=2, default=str), encoding="utf-8")
    log.info("wrote %s (%.1f KB)", OUT, OUT.stat().st_size / 1024)
    print(f"stations={len(bundle['stations'])} "
          f"ladder_rows={len(bundle['ladder'])} "
          f"subtypes={len(bundle['subtypes'])} "
          f"drift={list(bundle['drift'])} "
          f"transfer={'yes' if bundle['transfer'] else 'no'}")


if __name__ == "__main__":
    main()
