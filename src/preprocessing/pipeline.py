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
    borrowed_durations: "np.ndarray | None" = None,
) -> dict:
    """Full §6 pipeline for one station. Returns its manifest.

    ``borrowed_durations`` supplies a real `Validity` episode-duration
    distribution for a region that has none of its own (§5.5's argument for the
    rate, applied to durations). Without it an Indian build invents episode
    lengths, which is strictly weaker evidence.
    """
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
            real_durations=borrowed_durations,
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


def _durations_from_reference_region(cfg: dict, ref_region: str,
                                     pollutant: str):
    """Real `Validity` episode durations from a reference region.

    §2.1a's second LOCKED constraint is that episode durations come from the
    real `Validity` episode-length distribution, because EEA's flags genuinely
    record when instruments failed and for how long even though the reading
    itself is destroyed. A region without Validity flags cannot honour that
    from its own data, so it borrows the distribution rather than inventing
    episode lengths — the same reasoning §5.5 applies to the rate.

    Returns None on any failure, so the caller falls back to the per-type
    ranges and warns, rather than this silently returning something empty.
    """
    import numpy as np

    from src.preprocessing.indexing import LABEL_SUFFIX

    try:
        ref_df = load_harmonized(ref_region)
    except Exception as exc:  # noqa: BLE001 — absence is a normal outcome
        log.warning("cannot borrow episode durations from %r: %s", ref_region, exc)
        return None

    ref_df = ref_df[(ref_df["pollutant"] == pollutant)
                    & (ref_df["label_source"] == "real")]
    if ref_df.empty:
        return None

    runs = []
    for _, g in ref_df.groupby("station_id", observed=True):
        g = g.sort_values("timestamp_utc")
        is_fault = (g["label"] == "fault").to_numpy()
        if not is_fault.any():
            continue
        padded = np.concatenate(([False], is_fault, [False]))
        edges = np.flatnonzero(padded[1:] != padded[:-1])
        runs.append((edges[1::2] - edges[0::2]).astype(int))
    if not runs:
        return None

    durations = np.concatenate(runs)
    log.info("borrowed %d real %s fault-episode duration(s) from region %r "
             "(median %.0fh, max %dh) — §2.1a", len(durations), pollutant,
             ref_region, float(np.median(durations)), int(durations.max()))
    return durations


def _rate_from_reference_region(cfg: dict, region: str) -> tuple[float, str] | None:
    """The EEA-measured fault rate, for a region that has no real labels.

    LOCKED by §5.5 and §2.7. CPCB has no Validity equivalent, so an Indian
    build cannot measure its own rate — and must not default to an arbitrary
    one, because a rate that differs between regions is a shortcut the model
    can learn ("region X is likelier to be faulty") in place of a fault
    signature. The rate therefore comes from the EEA region, unchanged.

    Returns None when the EEA region's harmonized build is absent or itself
    carries no real labels, so the caller can fall back and say so rather than
    this inventing a number.
    """
    ref = cfg.get("eea", {}).get("region")
    if not ref or ref == region:
        return None
    try:
        ref_df = load_harmonized(ref)
    except Exception as exc:  # noqa: BLE001 — absence is a normal outcome here
        log.warning("cannot import an injection rate from region %r: %s", ref, exc)
        return None
    real = ref_df[ref_df["label_source"] == "real"]["label"].dropna()
    if real.empty:
        return None
    return measured_rate_from(real), ref


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
    borrowed_durations = None
    pre_target = cfg["preprocessing"]["target_pollutant"]
    if inject_synthetic:
        real_rows = long_df[long_df["label_source"] == "real"]
        source = "real"
        if real_rows.empty or real_rows["label"].dropna().empty:
            # A region with no real labels at all. CPCB/India is the intended
            # case (§2.6, §3.2: no Validity equivalent exists), and §5.5 says
            # exactly what to do about it — take the rate from the EEA region
            # rather than inventing one, because a region-specific rate would
            # let the model learn "region -> fault likelihood" as a shortcut
            # instead of learning fault signatures (§2.7).
            borrowed = _rate_from_reference_region(cfg, region)
            if borrowed is not None:
                synthetic_rate, ref_region = borrowed
                # §2.1a wants episode durations drawn from real operator-recorded
                # instrument failures. India has no Validity flags, so borrow
                # that distribution from the same reference region the rate came
                # from — inventing episode lengths is strictly weaker evidence,
                # and the alternative here is not "measure it" but "make it up".
                borrowed_durations = _durations_from_reference_region(
                    cfg, ref_region, pre_target)
                log.info(
                    "region %r carries no real fault labels, so the injection "
                    "rate is IMPORTED from region %r: %.5f (§5.5). It is not "
                    "measured from %r and must not be described as such.",
                    region, ref_region, synthetic_rate, region)
            else:
                # The demo source is label_source="simulated". Its Validity
                # flags are generated, so matching against them is a
                # self-consistency check, not a real-rate match. Say so.
                real_rows = long_df
                source = long_df["label_source"].iloc[0]
                log.warning(
                    "no label_source='real' rows in region %r and no reference "
                    "region available — matching the injection rate against "
                    "label_source=%r instead. This is NOT a real-rate match "
                    "(§5.5) and must not be reported as one.", region, source)
                synthetic_rate = measured_rate_from(real_rows["label"])
        else:
            synthetic_rate = measured_rate_from(real_rows["label"])
        if synthetic_rate is not None and source == "real":
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
                variant=variant, borrowed_durations=borrowed_durations,
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
            # Both, always, never merged (§5.3). Reporting only the real rate
            # made an injected Indian build look faultless: CPCB has no
            # Validity equivalent, so its real rate is correctly None while its
            # synthetic rate is the whole point of the build.
            "test_fault_rate_real": m["fault_rate_by_split"]["test"],
            "test_fault_rate_synth": m["synthetic_fault_rate_by_split"]["test"],
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
