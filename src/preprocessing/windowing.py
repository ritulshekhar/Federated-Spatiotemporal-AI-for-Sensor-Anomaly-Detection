"""§6.6 — Sliding windows and the chronological split.

24h input -> 6h forecast horizon, stride 1, split 70/15/15 **by date**.

Two rules that are easy to state and easy to violate:

1. **Never shuffle.** A random split leaks the future into the past: with
   stride-1 windows, a shuffled test window shares 23 of its 24 input hours
   with training windows either side of it. The reported error would be a
   measurement of memorisation.
2. **Never let a window straddle a split boundary.** Assigning by the target
   timestamp alone is not enough — a window targeting the first hour of the
   validation period still reads 24 hours of training data. Windows whose full
   span (inputs *and* targets) does not sit inside one split are dropped. That
   costs ~30 windows per boundary out of tens of thousands, which is a cheap
   price for a clean claim.

Split boundaries are computed *before* normalization (§6.5 fits on the training
window only, which means the boundary has to exist first). The §6 stage order
is otherwise unchanged; ``pipeline.py`` documents the interleave.

Every window carries its target-side metadata alongside it — the imputation
mask, the EEA Validity label, the physical screen flag, the Verification code —
because the decision layer (§7.4) and the evaluation layer (§8.2) both need to
know, per predicted hour, whether that hour was real, whether it was flagged,
and whether it had been QA-reviewed.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.preprocessing.indexing import (
    IMPUTED_SUFFIX, LABEL_SUFFIX, PHYSICAL_SUFFIX, VERIFICATION_SUFFIX,
)
from src.utils.logging_setup import get_logger

log = get_logger(__name__)

SPLITS = ("train", "val", "test")

# Label encoding used inside the numeric arrays.
LABEL_FAULT, LABEL_TRUSTWORTHY, LABEL_UNKNOWN = 1, 0, -1


@dataclass
class SplitBoundaries:
    train_end: pd.Timestamp
    val_end: pd.Timestamp
    first: pd.Timestamp
    last: pd.Timestamp

    def assign(self, start: pd.Timestamp, end: pd.Timestamp) -> str | None:
        """Which split a window spanning ``[start, end]`` belongs to, or None."""
        if end <= self.train_end:
            return "train"
        if start > self.train_end and end <= self.val_end:
            return "val"
        if start > self.val_end:
            return "test"
        return None   # straddles a boundary — dropped

    def as_dict(self) -> dict:
        return {k: str(v) for k, v in self.__dict__.items()}


@dataclass
class WindowSet:
    """Windows for one station, one split."""
    station_id: str
    region: str
    station_class: str
    label_source: str
    split: str
    target: str

    X: np.ndarray            # (n, input_hours, n_features)  normalized
    y: np.ndarray            # (n, horizon)                  normalized
    y_raw: np.ndarray        # (n, horizon)                  physical units
    t0: np.ndarray           # (n,)  last input timestamp, datetime64[ns], UTC
    t_target: np.ndarray     # (n, horizon)  datetime64[ns], UTC
    target_is_imputed: np.ndarray    # (n, horizon) bool
    target_label: np.ndarray         # (n, horizon) int8, see LABEL_* above
    target_physical_flag: np.ndarray # (n, horizon) bool
    target_verification: np.ndarray  # (n, horizon) int8
    feature_columns: list[str]

    def __len__(self) -> int:
        return int(self.X.shape[0])

    @property
    def horizon(self) -> int:
        return int(self.y.shape[1])

    def fault_rate(self) -> float:
        known = self.target_label != LABEL_UNKNOWN
        return float((self.target_label[known] == LABEL_FAULT).mean()) if known.any() else float("nan")


def compute_split_boundaries(
    index: pd.DatetimeIndex,
    train: float = 0.70,
    val: float = 0.15,
    test: float = 0.15,
) -> SplitBoundaries:
    """Chronological 70/15/15 boundaries over ``index``.

    Computed per station, and — when a second region is in play — per region
    (§5.6), never once globally. A single global date cut can put one region's
    monsoon in test while another's test covers an unrelated season.
    """
    total = train + val + test
    if not np.isclose(total, 1.0):
        raise ValueError(f"split fractions must sum to 1.0, got {total}")
    if len(index) < 3:
        raise ValueError("need at least 3 timestamps to split")

    idx = index.sort_values()
    n = len(idx)
    train_end = idx[int(n * train) - 1]
    val_end = idx[int(n * (train + val)) - 1]
    return SplitBoundaries(train_end=train_end, val_end=val_end,
                           first=idx[0], last=idx[-1])


def make_windows(
    frame: pd.DataFrame,
    feature_columns: list[str],
    target: str,
    boundaries: SplitBoundaries,
    raw_target: pd.Series,
    input_hours: int = 24,
    horizon_hours: int = 6,
    stride: int = 1,
) -> dict[str, WindowSet]:
    """Build stride-1 windows and partition them into train/val/test.

    ``frame`` must already be normalized; ``raw_target`` is the same target
    series in physical units, so forecasting error can be reported in ug/m3
    (§8.2) without inverting the scaler at metric time.
    """
    station_id = frame.attrs.get("station_id", "?")
    index = frame.index
    n = len(index)
    if n < input_hours + horizon_hours:
        raise ValueError(
            f"{station_id}: {n} hours is shorter than one window "
            f"({input_hours}+{horizon_hours})"
        )

    X_all = frame.loc[:, feature_columns].to_numpy(dtype=np.float32)
    y_all = frame[target].to_numpy(dtype=np.float32)
    y_raw_all = raw_target.reindex(index).to_numpy(dtype=np.float32)

    imputed = _column_or_default(frame, target + IMPUTED_SUFFIX, n, False, bool)
    physical = _column_or_default(frame, target + PHYSICAL_SUFFIX, n, False, bool)
    verification = _column_or_default(
        frame, target + VERIFICATION_SUFFIX, n, 3, np.int8
    )
    labels = _encode_labels(frame, target, n)

    starts = np.arange(0, n - input_hours - horizon_hours + 1, stride, dtype=int)

    # A window is usable only if no input feature and no target value is NaN.
    # NaNs at this point are the deliberate leftovers of §6.2 (gaps beyond the
    # fill limit) and of the head-of-series lag warm-up in §6.4.
    finite_rows = np.isfinite(X_all).all(axis=1) & np.isfinite(y_all)
    ok = _all_true_in_span(finite_rows, starts, input_hours + horizon_hours)
    n_dropped_nan = int((~ok).sum())
    starts = starts[ok]

    buckets: dict[str, list[int]] = {s: [] for s in SPLITS}
    n_straddle = 0
    for s in starts:
        span_start = index[s]
        span_end = index[s + input_hours + horizon_hours - 1]
        split = boundaries.assign(span_start, span_end)
        if split is None:
            n_straddle += 1
            continue
        buckets[split].append(int(s))

    out: dict[str, WindowSet] = {}
    for split, idxs in buckets.items():
        arr = np.asarray(idxs, dtype=int)
        out[split] = _gather(
            frame, station_id, split, target, arr, X_all, y_all, y_raw_all,
            index, imputed, labels, physical, verification,
            input_hours, horizon_hours, feature_columns,
        )

    log.info(
        "%s | windows train/val/test = %d/%d/%d | dropped %d (NaN span) + %d "
        "(straddled a split boundary)",
        station_id, len(out["train"]), len(out["val"]), len(out["test"]),
        n_dropped_nan, n_straddle,
    )
    return out


# ---------------------------------------------------------------------------
def _gather(frame, station_id, split, target, starts, X_all, y_all, y_raw_all,
            index, imputed, labels, physical, verification,
            input_hours, horizon_hours, feature_columns) -> WindowSet:
    n_feat = X_all.shape[1]
    if starts.size == 0:
        empty2 = np.empty((0, horizon_hours), dtype=np.float32)
        return WindowSet(
            station_id=station_id,
            region=frame.attrs.get("region", "unknown"),
            station_class=frame.attrs.get("station_class", "unknown"),
            label_source=frame.attrs.get("label_source", "unknown"),
            split=split, target=target,
            X=np.empty((0, input_hours, n_feat), dtype=np.float32),
            y=empty2, y_raw=empty2.copy(),
            t0=np.empty((0,), dtype="datetime64[ns]"),
            t_target=np.empty((0, horizon_hours), dtype="datetime64[ns]"),
            target_is_imputed=np.empty((0, horizon_hours), dtype=bool),
            target_label=np.empty((0, horizon_hours), dtype=np.int8),
            target_physical_flag=np.empty((0, horizon_hours), dtype=bool),
            target_verification=np.empty((0, horizon_hours), dtype=np.int8),
            feature_columns=list(feature_columns),
        )

    in_off = np.arange(input_hours)
    tgt_off = np.arange(input_hours, input_hours + horizon_hours)
    in_idx = starts[:, None] + in_off[None, :]
    tgt_idx = starts[:, None] + tgt_off[None, :]
    # numpy's datetime64 has no timezone concept. The index is already UTC
    # (§5.2 converts at harmonization time), so drop the tzinfo rather than let
    # .astype() do it with a warning. Every timestamp in a WindowSet is UTC.
    ts = index.tz_convert("UTC").tz_localize(None).to_numpy(dtype="datetime64[ns]")

    return WindowSet(
        station_id=station_id,
        region=frame.attrs.get("region", "unknown"),
        station_class=frame.attrs.get("station_class", "unknown"),
        label_source=frame.attrs.get("label_source", "unknown"),
        split=split, target=target,
        X=X_all[in_idx],
        y=y_all[tgt_idx],
        y_raw=y_raw_all[tgt_idx],
        t0=ts[starts + input_hours - 1],
        t_target=ts[tgt_idx],
        target_is_imputed=imputed[tgt_idx],
        target_label=labels[tgt_idx],
        target_physical_flag=physical[tgt_idx],
        target_verification=verification[tgt_idx],
        feature_columns=list(feature_columns),
    )


def _column_or_default(frame: pd.DataFrame, col: str, n: int, default, dtype):
    if col in frame.columns:
        return frame[col].fillna(default).to_numpy(dtype=dtype)
    return np.full(n, default, dtype=dtype)


def _encode_labels(frame: pd.DataFrame, target: str, n: int) -> np.ndarray:
    col = target + LABEL_SUFFIX
    if col not in frame.columns:
        return np.full(n, LABEL_UNKNOWN, dtype=np.int8)
    s = frame[col]
    out = np.full(n, LABEL_UNKNOWN, dtype=np.int8)
    out[(s == "trustworthy").to_numpy()] = LABEL_TRUSTWORTHY
    out[(s == "fault").to_numpy()] = LABEL_FAULT
    return out


def _all_true_in_span(mask: np.ndarray, starts: np.ndarray, span: int) -> np.ndarray:
    """Vectorised 'is ``mask`` all True over [s, s+span)' for every s in starts."""
    if starts.size == 0:
        return np.zeros(0, dtype=bool)
    cumsum = np.concatenate(([0], np.cumsum(mask.astype(np.int64))))
    return (cumsum[starts + span] - cumsum[starts]) == span


# ---------------------------------------------------------------------------
def save_windows(ws: WindowSet, directory: str | Path) -> Path:
    """Persist one split's windows for a station."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{ws.split}.npz"
    np.savez_compressed(
        path,
        X=ws.X, y=ws.y, y_raw=ws.y_raw,
        t0=ws.t0.astype("datetime64[ns]").astype(np.int64),
        t_target=ws.t_target.astype("datetime64[ns]").astype(np.int64),
        target_is_imputed=ws.target_is_imputed,
        target_label=ws.target_label,
        target_physical_flag=ws.target_physical_flag,
        target_verification=ws.target_verification,
        feature_columns=np.array(ws.feature_columns, dtype=object),
        meta=np.array([ws.station_id, ws.region, ws.station_class,
                       ws.label_source, ws.split, ws.target], dtype=object),
    )
    return path


def load_windows(path: str | Path) -> WindowSet:
    with np.load(path, allow_pickle=True) as z:
        station_id, region, station_class, label_source, split, target = z["meta"]
        return WindowSet(
            station_id=str(station_id), region=str(region),
            station_class=str(station_class), label_source=str(label_source),
            split=str(split), target=str(target),
            X=z["X"], y=z["y"], y_raw=z["y_raw"],
            t0=z["t0"].astype("datetime64[ns]"),
            t_target=z["t_target"].astype("datetime64[ns]"),
            target_is_imputed=z["target_is_imputed"],
            target_label=z["target_label"],
            target_physical_flag=z["target_physical_flag"],
            target_verification=z["target_verification"],
            feature_columns=[str(c) for c in z["feature_columns"]],
        )
