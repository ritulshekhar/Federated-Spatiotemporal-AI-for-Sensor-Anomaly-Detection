"""Runs §6.1 - §6.7 **independently per station**.

Stage order, and the one place it departs from the numbering in CLAUDE.md §6:

    6.1 indexing
    6.2 imputation           (mask built before filling)
    6.3 physical screening
    6.4 feature engineering
    --> split boundaries computed here (§6.6, first half)
    6.5 local normalization  (fit on the train window only — which is why the
                              boundary has to exist first)
    6.6 windowing            (second half)
    6.7 fault labelling      (real labels ride along; synthetic injection is
                              optional and rate-matched)

§6.5 cannot precede §6.6 in full because "fit on each station's own training
window" presupposes knowing where the training window ends. Splitting the
windowing stage in two is the minimal change that satisfies both constraints.
Nothing else about the order moves.

Stations are never concatenated. Each gets its own directory under
``data/processed/<station_id>/`` containing ``train.npz`` / ``val.npz`` /
``test.npz``, its own ``scaler.json`` (local only, never uploaded), and its own
``manifest.json``. The layout is the enforcement mechanism for §2.2: there is no
single file to accidentally pool.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.harmonize import load_harmonized
from src.preprocessing.fault_labels import (
    DRIFT_SIGMA_PER_72H, inject_faults, measured_rate_from,
)
from src.preprocessing.features import build_features
from src.preprocessing.imputation import impute_frame
from src.preprocessing.indexing import IMPUTED_SUFFIX, build_station_frame
from src.preprocessing.normalize import fit_local_scaler
from src.preprocessing.validity_screen import screen_frame
from src.preprocessing.windowing import (
    compute_split_boundaries, make_windows, save_windows,
)
from src.utils.logging_setup import get_logger
from src.utils.paths import load_config, station_dir, variant_root

log = get_logger(__name__)


def process_station(
    long_df: pd.DataFrame,
    station_id: str,
    cfg: dict,
    inject_synthetic: bool = False,
    synthetic_rate: float | None = None,
    meteo: pd.DataFrame | None = None,
    write: bool = True,
    variant: str = "",
) -> dict:
    """Full §6 pipeline for one station. Returns its manifest."""
    pre = cfg["preprocessing"]
    target = pre["target_pollutant"]
    pollutants = [target] + [p for p in pre["feature_pollutants"] if p != target]

    # -- §6.1 ---------------------------------------------------------------
    frame, coverage = build_station_frame(long_df, station_id, pollutants)
    present = frame.attrs["pollutants"]
    if target not in present:
        raise ValueError(
            f"{station_id}: target pollutant {target!r} not reported "
            f"(station has {present}). Exclude the station or change "
            "preprocessing.target_pollutant in configs/data.yaml."
        )

    # -- §6.2 ---------------------------------------------------------------
    imp = pre["imputation"]
    frame, imputation = impute_frame(
        frame, present,
        short_gap_hours=imp["short_gap_hours"],
        seasonal_period_hours=imp["seasonal_period_hours"],
        max_fill_hours=imp["max_fill_hours"],
    )

    # Quality gate: a station whose target series is mostly invented is not a
    # node, it is a liability. Reject before it reaches the federation.
    observed_fraction = 1.0 - float(frame[target + IMPUTED_SUFFIX].mean())
    min_observed = pre["quality"]["min_observed_fraction"]
    if observed_fraction < min_observed:
        raise ValueError(
            f"{station_id}: only {observed_fraction:.1%} of {target} was genuinely "
            f"observed (< {min_observed:.0%} required). Excluded from the node set."
        )

    # -- split boundaries (§6.6, first half — see the module docstring) ------
    win = pre["windowing"]
    boundaries = compute_split_boundaries(
        frame.index, **{k: float(v) for k, v in win["split"].items()}
    )
    log.info("%s | split: train<=%s  val<=%s  test<=%s", station_id,
             boundaries.train_end.date(), boundaries.val_end.date(),
             boundaries.last.date())

    # -- drop exogenous pollutants that barely exist during training ---------
    # A window needs every feature column finite, so one late-installed
    # instrument silently annihilates a station's entire history: ES1438A added
    # PM10 and PM2.5 in 2024 and lost all 48,892 of its training windows to
    # them. A pollutant that was not being measured during the training window
    # cannot be a training feature, so drop it rather than the station.
    present = _prune_sparse_features(
        frame, present, target, boundaries.train_end,
        min_coverage=float(pre["quality"]["min_feature_coverage"]),
        station_id=station_id,
    )

    # -- §6.3 ---------------------------------------------------------------
    scr = pre["physical_screen"]
    frame, screening = screen_frame(
        frame, present, ranges=scr["ranges"],
        flatline_run_length=scr["flatline_run_length"],
    )

    # -- §6.7 (synthetic half — before features, so lags see the faults) ------
    injection = None
    if inject_synthetic:
        if synthetic_rate is None:
            raise ValueError(
                "synthetic_rate is required. Measure it from the real EEA labels "
                "(§2.7/§5.5) — an arbitrary default lets the model learn "
                "'region -> fault likelihood' as a shortcut."
            )
        inj = pre.get("fault_injection", {})
        frame, rep = inject_faults(
            frame, target, target_rate=synthetic_rate,
            # Train stays clean (§6.7). Only val and test receive faults.
            eligible_after=boundaries.train_end,
            drift_sigma_per_72h=tuple(
                inj.get("drift_sigma_per_72h", DRIFT_SIGMA_PER_72H)
            ),
        )
        injection = rep.as_dict()

    # -- §6.4 ---------------------------------------------------------------
    feat = pre["features"]
    frame, feature_cols = build_features(
        frame, present,
        cyclical=feat["cyclical"], lags=feat["lags"],
        rolling_windows=feat["rolling_windows"], meteo=meteo,
    )
    raw_target = frame[target].copy()   # physical units, kept for §8.2 metrics

    # -- §6.5 ---------------------------------------------------------------
    scaler = fit_local_scaler(frame, feature_cols, boundaries.train_end, station_id)
    frame = scaler.transform(frame)

    # -- §6.6, second half ---------------------------------------------------
    windows = make_windows(
        frame, feature_cols, target, boundaries, raw_target,
        input_hours=win["input_hours"], horizon_hours=win["horizon_hours"],
        stride=win["stride"],
    )

    manifest = {
        "station_id": station_id,
        "region": frame.attrs.get("region"),
        "station_class": frame.attrs.get("station_class"),
        "label_source": frame.attrs.get("label_source"),
        "target": target,
        "pollutants_present": present,
        "n_features": len(feature_cols),
        "feature_columns": feature_cols,
        "observed_fraction": round(observed_fraction, 4),
        "split_boundaries": boundaries.as_dict(),
        "windows": {s: len(w) for s, w in windows.items()},
        "variant": variant or "clean",
        "fault_rate_by_split": {
            s: (None if np.isnan(w.fault_rate("real")) else round(w.fault_rate("real"), 6))
            for s, w in windows.items()
        },
        "synthetic_fault_rate_by_split": {
            s: (None if np.isnan(w.fault_rate("synthetic"))
                else round(w.fault_rate("synthetic"), 6))
            for s, w in windows.items()
        },
        "coverage": [c.as_dict() for c in coverage],
        "imputation": [i.as_dict() for i in imputation],
        "screening": [s.as_dict() for s in screening],
        "synthetic_injection": injection,
    }

    if write:
        out_dir = station_dir(station_id, "processed", variant=variant)
        for ws in windows.values():
            save_windows(ws, out_dir)
        scaler.save(out_dir / "scaler.json")   # LOCAL ONLY — never uploaded
        (out_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2, default=str), encoding="utf-8"
        )
        log.info("%s | wrote %s", station_id, out_dir)

    return manifest


def _prune_sparse_features(frame: pd.DataFrame, present: list[str], target: str,
                           train_end: pd.Timestamp, min_coverage: float,
                           station_id: str) -> list[str]:
    """Drop exogenous pollutants with too little coverage in the train window.

    The target is never dropped — a station that cannot supply its own target
    has already failed the check above and is excluded outright.
    """
    train_slice = frame.loc[frame.index <= train_end]
    if train_slice.empty:
        return present

    kept = [target]
    for pol in present:
        if pol == target:
            continue
        coverage = float(train_slice[pol].notna().mean())
        if coverage < min_coverage:
            log.warning(
                "%s | dropping %s as a feature: only %.1f%% coverage in the "
                "training window (needs %.0f%%). Its first observation is %s.",
                station_id, pol, 100 * coverage, 100 * min_coverage,
                frame[pol].first_valid_index(),
            )
            continue
        kept.append(pol)

    dropped = [p for p in present if p not in kept]
    if dropped:
        cols = [c for c in frame.columns
                if any(c == p or c.startswith(f"{p}__") for p in dropped)]
        frame.drop(columns=cols, inplace=True)
        frame.attrs["pollutants"] = kept
    return kept


def run(region: str, inject_synthetic: bool = False,
        stations: list[str] | None = None, write: bool = True,
        variant: str | None = None) -> pd.DataFrame:
    """Process every station in ``region``. Returns a per-station summary table."""
    cfg = load_config("data")
    long_df = load_harmonized(region)

    # Injected and clean builds must not overwrite each other: one is the
    # forecasting reference, the other the Objective 4 ground truth (§2.1a).
    if variant is None:
        variant = "injected" if inject_synthetic else ""

    all_ids = sorted(long_df["station_id"].unique())
    ids = [s for s in (stations or all_ids)]
    unknown = [s for s in ids if s not in all_ids]
    if unknown:
        raise ValueError(f"unknown station(s) for region {region!r}: {unknown}")

    # §5.5 — the rate synthetic injection must match, measured once, from
    # REAL-labelled rows only, across the region.
    synthetic_rate = None
    if inject_synthetic:
        real_rows = long_df[long_df["label_source"] == "real"]
        source = "real"
        if real_rows.empty:
            # The demo source is label_source="simulated". Its Validity flags
            # are generated, so matching against them is a self-consistency
            # check, not a real-rate match. Say so out loud.
            real_rows = long_df
            source = long_df["label_source"].iloc[0]
            log.warning(
                "no label_source='real' rows in region %r — matching the injection "
                "rate against label_source=%r instead. This is NOT a real-rate "
                "match (§5.5) and must not be reported as one.",
                region, source,
            )
        synthetic_rate = measured_rate_from(real_rows["label"])
        log.info("measured fault rate for region %r (label_source=%s): %.5f",
                 region, source, synthetic_rate)

    meteo_enabled = cfg.get("meteo", {}).get("enabled", False)

    rows, failures = [], []
    for sid in ids:
        meteo = None
        if meteo_enabled:
            from src.data.fetch_meteo import load_station_meteo
            meteo = load_station_meteo(sid)
            if meteo is None:
                log.warning("%s | meteo enabled but no cached met data; "
                            "run `python -m src.data.fetch_meteo`", sid)
        try:
            m = process_station(
                long_df, sid, cfg, inject_synthetic=inject_synthetic,
                synthetic_rate=synthetic_rate, meteo=meteo, write=write,
                variant=variant,
            )
        except Exception as exc:  # noqa: BLE001 — one bad station must not stop the run
            log.error("%s | EXCLUDED: %s", sid, exc)
            failures.append({"station_id": sid, "reason": str(exc)})
            continue
        rows.append({
            "station_id": m["station_id"],
            "region": m["region"],
            "station_class": m["station_class"],
            "label_source": m["label_source"],
            "observed_fraction": m["observed_fraction"],
            "n_features": m["n_features"],
            "train": m["windows"]["train"],
            "val": m["windows"]["val"],
            "test": m["windows"]["test"],
            "test_fault_rate": m["fault_rate_by_split"]["test"],
        })

    if write:
        run_manifest = {
            "region": region,
            "stations_processed": [r["station_id"] for r in rows],
            "stations_excluded": failures,
            "inject_synthetic": inject_synthetic,
            "synthetic_rate": synthetic_rate,
            "meteo_enabled": meteo_enabled,
            "config": cfg["preprocessing"],
        }
        run_manifest["variant"] = variant or "clean"
        path = variant_root("processed", variant) / f"run_manifest_{region}.json"
        path.write_text(json.dumps(run_manifest, indent=2, default=str),
                        encoding="utf-8")
        log.info("wrote %s", path)

    if not rows:
        raise RuntimeError(
            f"every station in region {region!r} was excluded; see the errors above"
        )
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--region", default="catalonia")
    ap.add_argument("--stations", nargs="*", default=None)
    ap.add_argument("--inject-synthetic", action="store_true",
                    help="also inject rate-matched synthetic faults (§6.7). "
                         "Supplementary stress test only — never the headline metric.")
    ap.add_argument("--dry-run", action="store_true", help="do not write outputs")
    ap.add_argument("--variant", default=None,
                    help="processed-build subdirectory (default: 'injected' when "
                         "--inject-synthetic is set, otherwise the clean root)")
    args = ap.parse_args()

    summary = run(args.region, inject_synthetic=args.inject_synthetic,
                  stations=args.stations, write=not args.dry_run,
                  variant=args.variant)
    print("\n=== preprocessing summary (per station — §8.3) ===")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
