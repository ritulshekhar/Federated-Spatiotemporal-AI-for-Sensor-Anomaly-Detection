"""§5.8 — train on Catalonia, evaluate on India, report the threshold shift.

CLAUDE.md §5.8 does not ask whether the model transfers; it predicts that it
will not, for two compounding reasons, and asks for the *size of the
miscalibration* as the result:

  (1) **scale** — Indian pollutant swings are larger, so even correct forecasts
      produce bigger residuals. §6.5's per-station z-scoring is supposed to
      absorb this automatically.
  (2) **relationship** — the GRU learns Barcelona's traffic-and-wind driven
      weather-to-pollution mapping, which does not describe Delhi's biomass
      burning and construction dust. Normalisation cannot fix this one.

So the experiment is deliberately framed as a *generalization test*, not a
dataset merge (§2.6 keeps India secondary). Nothing is retrained on Indian
data. Three numbers come out:

  * forecast error on India under the Catalonia-trained model, against the
    same model on Catalonia — isolating (2), since (1) is normalised away;
  * detection quality on India using **Catalonia's** calibrated threshold —
    what naive transfer actually delivers;
  * detection quality on India using **India's own** validation-calibrated
    threshold, and the ratio between the two thresholds. That ratio is the
    headline: §5.8 calls the required shift "a legitimate, reportable finding,
    not a bug to hide".

Run:
    python -m src.evaluation.transfer
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from src.baselines.ladder import load_region, score_station
from src.evaluation.metrics import forecast_metrics
from src.models.node_gru import NodeTrainer
from src.utils.logging_setup import get_logger
from src.utils.paths import data_paths, load_config
from src.utils.seeding import seed_everything

log = get_logger(__name__)


def shared_features(source, target):
    """Feature columns common to BOTH regions, with per-station reorderings.

    The two regions were preprocessed independently and do not agree on
    feature width: Catalonia carries whichever exogenous pollutants survived
    §6's coverage gate per station, and so does Delhi. Intersecting rather
    than zero-filling, for the reason `align_features` gives — a zero would
    teach the model that "absent" means "none of this pollutant".
    """
    common = None
    for st in list(source) + list(target):
        cols = set(st.splits["train"].feature_columns)
        common = cols if common is None else (common & cols)
    if not common:
        raise ValueError("the two regions share no feature columns")

    order = [c for c in source[0].splits["train"].feature_columns if c in common]
    tgt = source[0].target
    if tgt not in order:
        raise ValueError(f"target {tgt!r} is not common to both regions")

    idx = {}
    for st in list(source) + list(target):
        cols = st.splits["train"].feature_columns
        idx[st.station_id] = np.array([cols.index(c) for c in order], dtype=int)
    log.info("%d feature column(s) shared by both regions: %s", len(order),
             ", ".join(order[:8]) + (" ..." if len(order) > 8 else ""))
    return order, idx


def run(source_region, target_region, source_variant, target_variant,
        horizon_report=(1, 3, 6), epochs=None):
    model_cfg = load_config("model")
    seed_everything(model_cfg.get("seed", 1337))

    src = load_region(source_region, variant=source_variant)
    tgt = load_region(target_region, variant=target_variant)
    log.info("source %s: %d station(s) | target %s: %d station(s)",
             source_region, len(src), target_region, len(tgt))

    order, idx = shared_features(src, tgt)
    ti = order.index(src[0].target)
    horizon = src[0].splits["train"].y.shape[1]

    # Train ONE model on the source region only, pooled across its stations —
    # rung 3's centralised setup. The transfer question is about the learned
    # relationship, not about federation, so the cheapest faithful source
    # model is the right one.
    cfg = dict(model_cfg["node_gru"])
    cfg["train"] = dict(cfg["train"])
    if epochs is not None:
        cfg["train"]["epochs"] = epochs

    Xtr = np.concatenate([st.splits["train"].X[:, :, idx[st.station_id]] for st in src])
    ytr = np.concatenate([st.splits["train"].y for st in src])
    Xva = np.concatenate([st.splits["val"].X[:, :, idx[st.station_id]] for st in src])
    yva = np.concatenate([st.splits["val"].y for st in src])
    log.info("training the source model on %s pooled window(s)", f"{len(Xtr):,}")

    trainer = NodeTrainer(n_features=len(order), horizon=horizon, cfg=cfg,
                          target_index=ti,
                          station_id="SOURCE-" + source_region,
                          seed=model_cfg.get("seed", 1337))
    trainer.fit(Xtr, ytr, X_val=Xva, y_val=yva)

    sc = model_cfg["decision_layer"].get("score", {})
    dec = model_cfg["decision_layer"]["residual_threshold"]
    budget = model_cfg["decision_layer"].get("detector_budget", {})
    score_kwargs = dict(
        threshold_method=dec["method"], threshold_quantile=dec["quantile"],
        target_fpr_extremes=dec.get("target_fpr_extremes", 0.10),
        label_source="auto", score_mode=sc.get("mode", "absolute"),
        volatility_floor=sc.get("volatility_floor", 0.25),
        budget_on=budget.get("on", "overall"),
        target_fpr_overall=float(budget.get("target_fpr_overall", 0.005)),
    )

    rows, records, thresholds = [], [], []
    for region, stations in ((source_region, src), (target_region, tgt)):
        for st in stations:
            sid = st.station_id
            pv = trainer.predict(st.splits["val"].X[:, :, idx[sid]])
            pt = trainer.predict(st.splits["test"].X[:, :, idx[sid]])

            # De-normalise through the station's own scaler, which is what
            # y_pred_raw exists for -- the scaler stays local, so this is the
            # only correct way back to physical units here.
            pt_raw = st.y_pred_raw(pt)
            for h in horizon_report:
                j = h - 1
                if j >= pt_raw.shape[1]:
                    continue
                m = forecast_metrics(st.splits["test"].y_raw[:, j], pt_raw[:, j])
                rows.append({"region": region, "station_id": sid,
                             "station_class": st.splits["train"].station_class,
                             "horizon": h,
                             **{k: round(float(v), 4)
                                for k, v in m.as_dict().items()}})

            rec, thr = score_station(st, "transfer_source_model", pv, pt,
                                     **score_kwargs)
            rec["region"] = region
            records.append(rec)
            thresholds.append({"region": region, "station_id": sid,
                               "station_class": st.splits["train"].station_class,
                               "threshold": float(thr)})

    forecast = pd.DataFrame(rows)
    recs = pd.concat(records, ignore_index=True)
    thr_df = pd.DataFrame(thresholds)

    by_region = thr_df.groupby("region")["threshold"].median()
    shift = (float(by_region.get(target_region, np.nan))
             / float(by_region.get(source_region, np.nan)))
    log.info("median calibrated threshold: %s=%.4f  %s=%.4f  -> shift x%.2f",
             source_region, by_region.get(source_region, np.nan),
             target_region, by_region.get(target_region, np.nan), shift)

    return {"forecast": forecast, "records": recs, "thresholds": thr_df,
            "threshold_by_region": by_region, "threshold_shift": shift,
            "shared_features": order}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", default="catalonia")
    ap.add_argument("--source-variant", default="injected")
    ap.add_argument("--target", default="india")
    ap.add_argument("--target-variant", default="injected_india")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--name", default="phase6_transfer_india")
    args = ap.parse_args()

    out = run(args.source, args.target, args.source_variant,
              args.target_variant, epochs=args.epochs)

    d = data_paths()["outputs"] / args.name
    d.mkdir(parents=True, exist_ok=True)
    out["forecast"].to_csv(d / "forecast_per_station.csv", index=False)
    out["thresholds"].to_csv(d / "thresholds_per_station.csv", index=False)
    out["records"].to_parquet(d / "records.parquet", index=False)

    # §8.3: grouped, never pooled.
    print("\n=== forecast error under the source-trained model (§5.8) ===")
    print(out["forecast"].groupby(["region", "horizon"])[["rmse", "mae", "mape"]]
          .median().round(3).to_string())
    print("\n=== calibrated residual threshold, per region ===")
    print(out["threshold_by_region"].round(4).to_string())
    print("\nthreshold shift (target / source): x%.2f" % out["threshold_shift"])

    (d / "summary.json").write_text(json.dumps({
        "source": args.source, "target": args.target,
        "threshold_by_region": {k: round(float(v), 6)
                                for k, v in out["threshold_by_region"].items()},
        "threshold_shift": round(float(out["threshold_shift"]), 4),
        "n_shared_features": len(out["shared_features"]),
        "shared_features": out["shared_features"],
    }, indent=2), encoding="utf-8")
    print("\nreport -> %s" % d)


if __name__ == "__main__":
    main()
