"""Confidence intervals, by moving-block bootstrap over the test window.

Every number reported so far comes from a single test window with no error
bar, and the per-station spread has been wide throughout — test fault rates
range 0.007 to 0.078 across the seven stations (entry 5). Some of what looks
like a difference between rungs may not survive resampling, and until it is
measured nobody can tell which.

**Why blocks.** Ordinary bootstrap resamples rows independently, which assumes
the rows are independent. Hourly air quality is strongly autocorrelated, and
faults arrive in episodes lasting hours to days, so independent resampling
would split episodes apart and produce intervals far too narrow. Contiguous
blocks (default one week) keep episodes intact.

**Why paired.** "Is rung 6 better than rung 5" is not answered by two separate
intervals that happen to overlap — the two models are evaluated on the *same*
hours, so their errors are correlated and the comparison should be too. Each
replicate here resamples one set of blocks and scores both models on it, then
the interval is taken on the *difference*. That is a much sharper test, and it
can show a significant difference even when the individual intervals overlap.

Two levels of resampling, because there are two sources of variation:

* **time** — would a different stretch of 2024 have given the same answer?
* **stations** — would a different seven stations have?

``level="both"`` resamples stations with replacement and then blocks within
each, which is the honest interval for a claim about the network rather than
about these particular seven sensors.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluation.metrics import anomaly_metrics
from src.evaluation.rescore import apply_score, calibrate, load_records
from src.utils.logging_setup import get_logger

log = get_logger(__name__)

DEFAULT_BLOCK_HOURS = 168       # one week
DEFAULT_N_BOOT = 400
METRICS = ("recall", "precision", "f1", "auc_pr", "fpr_on_extremes")


def _blocks(n: int, block: int) -> list[np.ndarray]:
    """Contiguous index blocks over a time-ordered series."""
    if n == 0:
        return []
    return [np.arange(s, min(s + block, n)) for s in range(0, n, block)]


def _score_subset(g: pd.DataFrame, threshold: float) -> dict:
    codes = np.where(g["label"] == "fault", 1,
                     np.where(g["label"] == "trustworthy", 0, -1))
    m = anomaly_metrics(
        y_label=codes, score=g["score"].to_numpy(), threshold=threshold,
        is_extreme=g["is_extreme"].to_numpy(),
        mask=~g["is_imputed"].to_numpy(),
    )
    return {k: getattr(m, k) for k in METRICS}


def _subtype_recall(g: pd.DataFrame, threshold: float) -> dict:
    """Recall per fault subtype on one resample. Precision has no subtype."""
    faults = g[(g["label"] == "fault") & (~g["is_imputed"])]
    if faults.empty or "fault_type" not in faults.columns:
        return {}
    detected = faults["score"] > threshold
    return {f"recall_{t}": float(detected[faults["fault_type"] == t].mean())
            for t in faults["fault_type"].dropna().unique()}


def bootstrap(records: pd.DataFrame, thresholds: dict, models: list[str],
              horizon: int = 1, n_boot: int = DEFAULT_N_BOOT,
              block_hours: int = DEFAULT_BLOCK_HOURS, level: str = "both",
              seed: int = 1337, by_subtype: bool = False) -> pd.DataFrame:
    """One row per (replicate, model): metrics on that resample.

    All models are scored on the *same* resampled blocks, which is what makes
    the paired comparison valid.
    """
    rng = np.random.default_rng(seed)
    df = records[(records["split"] == "test") & (records["horizon"] == horizon)
                 & records["model"].isin(models)].copy()
    if df.empty:
        raise ValueError(f"no test records at horizon {horizon} for {models}")

    stations = sorted(df["station_id"].unique())
    # Pre-index per (model, station), time-ordered, so replicates are cheap.
    per: dict[tuple[str, str], pd.DataFrame] = {}
    for (model, station), g in df.groupby(["model", "station_id"]):
        per[(model, station)] = g.sort_values("t_target").reset_index(drop=True)

    rows = []
    for b in range(n_boot):
        chosen = (list(rng.choice(stations, size=len(stations), replace=True))
                  if level in ("both", "stations") else stations)

        for model in models:
            parts = []
            for station in chosen:
                g = per.get((model, station))
                if g is None or g.empty:
                    continue
                if level in ("both", "time"):
                    blocks = _blocks(len(g), block_hours)
                    if not blocks:
                        continue
                    pick = rng.integers(0, len(blocks), size=len(blocks))
                    idx = np.concatenate([blocks[i] for i in pick])
                    parts.append(g.iloc[idx])
                else:
                    parts.append(g)
            if not parts:
                continue
            sample = pd.concat(parts, ignore_index=True)
            # One threshold per station; score each station against its own.
            if by_subtype:
                # Subtype recall is pooled across the resampled stations: a
                # single station can carry too few of one subtype for a
                # per-station median to mean anything.
                pooled = []
                for station, gs in sample.groupby("station_id"):
                    thr = thresholds.get((model, station))
                    if thr is None:
                        continue
                    gs = gs.assign(_detected=gs["score"] > thr)
                    pooled.append(gs[(gs["label"] == "fault") & (~gs["is_imputed"])])
                if not pooled:
                    continue
                allf = pd.concat(pooled, ignore_index=True)
                rec = {f"recall_{t}": float(allf.loc[allf["fault_type"] == t, "_detected"].mean())
                       for t in allf["fault_type"].dropna().unique()}
                rows.append({"replicate": b, "model": model, **rec})
                continue

            metrics = []
            for station, gs in sample.groupby("station_id"):
                thr = thresholds.get((model, station))
                if thr is None:
                    continue
                metrics.append(_score_subset(gs, thr))
            if not metrics:
                continue
            med = pd.DataFrame(metrics).median()
            rows.append({"replicate": b, "model": model, **med.to_dict()})

    return pd.DataFrame(rows)


def summarise(samples: pd.DataFrame, alpha: float = 0.05) -> pd.DataFrame:
    """Point estimate and percentile interval per model."""
    lo, hi = 100 * alpha / 2, 100 * (1 - alpha / 2)
    out = []
    for model, g in samples.groupby("model"):
        row = {"model": model, "n_replicates": len(g)}
        for m in METRICS:
            v = g[m].dropna().to_numpy()
            if v.size == 0:
                continue
            row[m] = np.median(v)
            row[f"{m}_lo"] = np.percentile(v, lo)
            row[f"{m}_hi"] = np.percentile(v, hi)
        out.append(row)
    return pd.DataFrame(out)


def paired_difference(samples: pd.DataFrame, model_a: str, model_b: str,
                      alpha: float = 0.05) -> pd.DataFrame:
    """Interval on (a − b) within each replicate. The test that matters.

    A difference whose interval excludes zero is one the data supports;
    overlapping individual intervals do not refute it, and non-overlapping ones
    are not required to establish it.
    """
    wide = samples.pivot(index="replicate", columns="model", values=list(METRICS))
    rows = []
    for m in METRICS:
        try:
            diff = (wide[(m, model_a)] - wide[(m, model_b)]).dropna().to_numpy()
        except KeyError:
            continue
        if diff.size == 0:
            continue
        lo = np.percentile(diff, 100 * alpha / 2)
        hi = np.percentile(diff, 100 * (1 - alpha / 2))
        rows.append({
            "metric": m, "difference": float(np.median(diff)),
            "lo": float(lo), "hi": float(hi),
            "excludes_zero": bool(lo > 0 or hi < 0),
            "p_better": float((diff > 0).mean()),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("runs", nargs="+", help="one or more outputs/<run> directories")
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--horizon", type=int, default=1)
    ap.add_argument("--n-boot", type=int, default=DEFAULT_N_BOOT)
    ap.add_argument("--block-hours", type=int, default=DEFAULT_BLOCK_HOURS)
    ap.add_argument("--level", default="both", choices=["both", "time", "stations"])
    ap.add_argument("--compare", nargs=2, metavar=("A", "B"), default=None)
    args = ap.parse_args()

    frames = []
    for run in args.runs:
        rec = load_records(run)
        if "det_residual" not in rec.columns:   # plain-score runs need scoring
            rec["score"] = apply_score(rec, "absolute")
        frames.append(rec)
    records = pd.concat(frames, ignore_index=True)

    models = args.models or sorted(records["model"].unique())
    thresholds = calibrate(records, "val_fpr_extremes", target_fpr_extremes=0.10)

    log.info("bootstrapping %d model(s), %d replicates, %dh blocks, level=%s",
             len(models), args.n_boot, args.block_hours, args.level)
    samples = bootstrap(records, thresholds, models, horizon=args.horizon,
                        n_boot=args.n_boot, block_hours=args.block_hours,
                        level=args.level)

    print(f"\n=== 95% intervals, h={args.horizon} "
          f"({args.n_boot} replicates, {args.block_hours}h blocks) ===")
    summary = summarise(samples)
    for _, r in summary.iterrows():
        print(f"\n  {r['model']}")
        for m in METRICS:
            if m in r and pd.notna(r[m]):
                print(f"    {m:16s} {r[m]:6.3f}  [{r[f'{m}_lo']:.3f}, {r[f'{m}_hi']:.3f}]")

    if args.compare:
        a, b = args.compare
        print(f"\n=== paired difference: {a} − {b} ===")
        diff = paired_difference(samples, a, b)
        print(diff.to_string(index=False, float_format=lambda v: f"{v:7.3f}"))
        print("\n'excludes_zero' true means the data supports a real difference.")


if __name__ == "__main__":
    main()
