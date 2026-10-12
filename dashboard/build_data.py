"""Collect the dashboard's data bundle from saved run outputs.

Objective 7 asks for a dashboard "visualising station data, forecasts, and the
trustworthy/faulty decision". This builds the JSON it renders from, so the
dashboard never recomputes anything — every figure traces to a run under
`outputs/`, and regenerating is a second rather than a retrain.

Deliberately includes the uncomfortable numbers. A dashboard that showed only
the headline recall would misrepresent the project: §8.2 bans recall without
its false-positive control, entry 16 showed most rung differences are inside
their confidence intervals, and entry 14 showed the aggregate F1 hides a
10x spread across fault sizes. All three travel with the data.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.station_metadata import configured_stations, coordinates
from src.utils.logging_setup import get_logger
from src.utils.paths import REPO_ROOT, data_paths

log = get_logger(__name__)

OUT = Path(__file__).parent / "data.json"


def _csv(run: str, name: str) -> pd.DataFrame:
    path = data_paths()["outputs"] / run / f"{name}.csv"
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def stations_block() -> list[dict]:
    """Station identity, class and coordinates for the map."""
    coords = coordinates()
    out = []
    for s in configured_stations():
        lat, lon = coords.get(s.station_id, (None, None))
        out.append({
            "id": s.station_id, "name": s.name, "klass": s.station_class,
            "lat": lat, "lon": lon,
        })
    return [s for s in out if s["lat"] is not None]


def forecasting_block() -> dict:
    """Clean-data forecast error, with and without meteorology."""
    out = {}
    for label, run in (("without_met", "phase1_catalonia"),
                       ("with_met", "phase1_catalonia_meteo")):
        fc = _csv(run, "forecast_per_station")
        if fc.empty:
            continue
        agg = (fc.groupby(["model", "horizon"])["rmse"]
                 .agg(["median", "min", "max"]).reset_index())
        out[label] = agg.rename(columns={"median": "rmse"}).to_dict("records")
    per_station = _csv("phase1_catalonia_meteo", "forecast_per_station")
    if not per_station.empty:
        out["per_station"] = (per_station[["model", "station_id", "station_class",
                                           "horizon", "rmse", "mae"]]
                              .round(3).to_dict("records"))
    return out


def ladder_block() -> list[dict]:
    """Every rung, forecasting and detection side by side."""
    rows = []
    for run in ("phase2_ladder", "phase2_rung4", "phase2_rung6_tuned",
                "phase2_rung6_split"):
        fc, an = _csv(run, "forecast_per_station"), _csv(run, "anomaly_per_station")
        if fc.empty:
            continue
        if run == "phase2_rung6_split":
            fc = fc.assign(model="6_proposed_split")
            an = an.assign(model="6_proposed_split") if not an.empty else an
        f = fc.groupby(["model", "horizon"])["rmse"].median().reset_index()
        a = (an.groupby(["model", "horizon"])[["recall", "precision", "f1", "auc_pr",
                                               "fpr_on_extremes"]].median().reset_index()
             if not an.empty else pd.DataFrame())
        merged = f.merge(a, on=["model", "horizon"], how="left") if not a.empty else f
        rows += merged.round(3).to_dict("records")
    return rows


def subtype_block() -> list[dict]:
    """Recall per fault subtype — the project's actual result."""
    rows = []
    for label, run in (("baselines", "phase2_ladder"),
                       ("proposed", "phase2_rung6_split")):
        sub = _csv(run, "subtype_recall")
        if sub.empty:
            continue
        g = (sub[sub["horizon"] == 1].groupby(["model", "fault_type"])["recall"]
             .mean().reset_index())
        g["group"] = label
        rows += g.round(3).to_dict("records")
    return rows


def findings_block() -> list[dict]:
    """The measured claims, each with its caveat. Hand-written on purpose.

    These are conclusions, not aggregates — they cannot be derived from a CSV,
    and stating them without their caveats would be the main way this dashboard
    could mislead.
    """
    return [
        {
            "title": "Privacy costs 3–8% of forecast accuracy",
            "detail": "Federated training reaches 15.10/19.38/20.69 RMSE at "
                      "1/3/6h against a centralised ST-GNN ceiling of "
                      "14.62/18.10/19.21 — the price of never pooling raw data.",
            "caveat": "Measured on one test window; not bootstrapped.",
            "tone": "good",
        },
        {
            "title": "Neighbours detect faults no local model can",
            "detail": "Stuck-at recall is 0.000 for every rung that looks only at "
                      "a station's own history — including the centralised model "
                      "with full raw access — and 0.134–0.220 for the proposed "
                      "model, which uses cross-station agreement from compressed "
                      "embeddings only.",
            "caveat": "Real under time resampling; not significant when stations "
                      "are also resampled, at n=7.",
            "tone": "good",
        },
        {
            "title": "Accuracy is the wrong metric here",
            "detail": "At a 2.8% fault rate a classifier answering “never faulty” "
                      "scores 0.977 accuracy — higher than this system's 0.976. "
                      "Recall, precision and AUC-PR are what mean anything.",
            "caveat": None,
            "tone": "warn",
        },
        {
            "title": "Drift remains unsolved",
            "detail": "Slow calibration drift is never detected at the alert "
                      "operating point. Its per-hour deviation is smaller than "
                      "normal variation, so no threshold reaches it.",
            "caveat": "Measured against an injector that drifted ~100× faster "
                      "than real co-located sensors do, so this is optimistic: "
                      "the detector missed drift far coarser than the real "
                      "thing. The injector is fixed; these numbers predate it.",
            "tone": "bad",
        },
        {
            "title": "EEA's fault labels contain no reading",
            "detail": "All 27,783 Validity-flagged rows carry the −999 no-data "
                      "sentinel, so the readings this project exists to "
                      "adjudicate are absent rather than wrong. Evaluation uses "
                      "rate-matched synthetic faults instead.",
            "caveat": "Injection rate and episode durations are taken from the "
                      "real data; only the corrupted value is synthesised.",
            "tone": "warn",
        },
    ]


def build() -> dict:
    bundle = {
        "generated": pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d %H:%M UTC"),
        "stations": stations_block(),
        "forecasting": forecasting_block(),
        "ladder": ladder_block(),
        "subtypes": subtype_block(),
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
        },
        "uncertainty": [
            {"metric": "F1", "diff": 0.006, "lo": -0.102, "hi": 0.113, "sig": False},
            {"metric": "AUC-PR", "diff": 0.016, "lo": -0.099, "hi": 0.126, "sig": False},
            {"metric": "recall", "diff": 0.041, "lo": -0.071, "hi": 0.151, "sig": False},
            {"metric": "precision", "diff": -0.166, "lo": -0.348, "hi": 0.014, "sig": False},
        ],
        "dataset": {
            "region": "Catalonia (Barcelona)", "stations": 7,
            "rows": 990598, "years": "2017–2024",
            "fault_rate": 0.02805, "target": "NO2",
        },
        "findings": findings_block(),
    }
    return bundle


def main() -> None:
    bundle = build()
    OUT.write_text(json.dumps(bundle, indent=2, default=str), encoding="utf-8")
    log.info("wrote %s (%.1f KB)", OUT, OUT.stat().st_size / 1024)
    print(f"stations={len(bundle['stations'])} ladder_rows={len(bundle['ladder'])} "
          f"subtypes={len(bundle['subtypes'])}")


if __name__ == "__main__":
    main()
