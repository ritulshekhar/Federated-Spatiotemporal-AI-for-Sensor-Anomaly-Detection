"""§6.5 — Local normalization.

Per-station z-score, fit on **that station's own training window only**, and
never shared.

Two separate reasons this is locked (§2.2, §6.5), worth keeping distinct
because they fail in different ways:

1. **Leakage.** Fitting on the whole series lets validation and test statistics
   into the training scaler. Standard, boring, fatal.
2. **Privacy.** A shared scaler *is itself a disclosure*. Mean and standard
   deviation are summary statistics of a station's raw distribution; publishing
   them to a server, or deriving one global scaler from every station, leaks
   exactly the thing §2.4 says never leaves the node — quietly, and in a form
   that looks like a preprocessing detail rather than a data transfer.

So ``LocalScaler`` carries an explicit ``NEVER_TRANSMIT`` marker and
``src/federated/client.py`` asserts against it on every outbound payload. The
marker is not decoration; it is the thing the assertion looks for.

A useful side effect (§5.4): per-station scaling absorbs the scale difference
between Barcelona and, later, Indian pollutant levels automatically, with no
special-casing for region. It does *not* fix the second half of the §5.8
failure mode — the learned weather-to-pollution relationship still does not
transfer — and nothing in this module claims otherwise.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.logging_setup import get_logger

log = get_logger(__name__)

#: Sentinel checked by the federated privacy guard. Do not rename without
#: updating ``src/federated/client.py`` and ``configs/federation.yaml``.
NEVER_TRANSMIT = "NEVER_TRANSMIT__local_normalization_statistics"

_EPS = 1e-8


@dataclass
class LocalScaler:
    """Per-station, per-column mean/std fitted on the training window only."""

    station_id: str
    columns: list[str] = field(default_factory=list)
    mean_: dict[str, float] = field(default_factory=dict)
    std_: dict[str, float] = field(default_factory=dict)
    n_fit_rows: int = 0
    fit_window: tuple[str, str] | None = None

    #: See module docstring. Read by the federated privacy guard.
    privacy_marker: str = NEVER_TRANSMIT

    # -- fit / transform ---------------------------------------------------
    def fit(self, df: pd.DataFrame, columns: list[str]) -> "LocalScaler":
        """Fit on ``df`` — which must already be restricted to the train split."""
        self.columns = list(columns)
        sub = df.loc[:, self.columns]
        self.mean_ = {c: float(np.nanmean(sub[c].to_numpy(dtype=float)))
                      for c in self.columns}
        self.std_ = {c: float(np.nanstd(sub[c].to_numpy(dtype=float)))
                     for c in self.columns}

        degenerate = [c for c, s in self.std_.items() if not np.isfinite(s) or s < _EPS]
        if degenerate:
            # A constant column carries no information; scaling it by ~0 would
            # amplify float noise into large spurious z-scores.
            log.warning("%s | %d constant/degenerate column(s) fixed to std=1: %s",
                        self.station_id, len(degenerate), degenerate[:6])
            for c in degenerate:
                self.std_[c] = 1.0

        all_nan = [c for c, m in self.mean_.items() if not np.isfinite(m)]
        if all_nan:
            log.warning("%s | %d all-NaN column(s) fixed to mean=0: %s",
                        self.station_id, len(all_nan), all_nan[:6])
            for c in all_nan:
                self.mean_[c] = 0.0

        self.n_fit_rows = int(len(sub))
        if len(df.index):
            self.fit_window = (str(df.index.min()), str(df.index.max()))
        log.info("%s | scaler fitted on %s train rows x %d column(s), window %s",
                 self.station_id, f"{self.n_fit_rows:,}", len(self.columns),
                 self.fit_window)
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        self._check_fitted()
        out = df.copy()
        for c in self.columns:
            if c in out.columns:
                out[c] = (out[c].astype(float) - self.mean_[c]) / self.std_[c]
        return out

    def fit_transform(self, df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
        return self.fit(df, columns).transform(df)

    def inverse_transform_column(self, values: np.ndarray, column: str) -> np.ndarray:
        """Back to physical units — needed for RMSE/MAE in ug/m3 (§8.2)."""
        self._check_fitted()
        if column not in self.mean_:
            raise KeyError(f"{column!r} was not fitted by this scaler")
        return np.asarray(values, dtype=float) * self.std_[column] + self.mean_[column]

    def scale_of(self, column: str) -> float:
        """The std used for ``column``; the factor an unnormalised residual needs."""
        self._check_fitted()
        return self.std_[column]

    def _check_fitted(self) -> None:
        if not self.columns:
            raise RuntimeError(
                f"LocalScaler for {self.station_id!r} is not fitted. Fit on the "
                "TRAIN split only (§6.5)."
            )

    # -- local persistence -------------------------------------------------
    def save(self, path: str | Path) -> Path:
        """Write to the node's own disk. This file must never be uploaded."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "_warning": (
                "LOCAL ONLY. These statistics summarise this station's raw "
                "distribution. Transmitting them violates the privacy contract "
                "in CLAUDE.md §2.4 / §6.5."
            ),
            "privacy_marker": self.privacy_marker,
            "station_id": self.station_id,
            "columns": self.columns,
            "mean": self.mean_,
            "std": self.std_,
            "n_fit_rows": self.n_fit_rows,
            "fit_window": self.fit_window,
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: str | Path) -> "LocalScaler":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            station_id=payload["station_id"],
            columns=payload["columns"],
            mean_=payload["mean"],
            std_=payload["std"],
            n_fit_rows=payload.get("n_fit_rows", 0),
            fit_window=tuple(payload["fit_window"]) if payload.get("fit_window") else None,
        )


def fit_local_scaler(frame: pd.DataFrame, feature_columns: list[str],
                     train_end: pd.Timestamp, station_id: str) -> LocalScaler:
    """Fit on rows strictly at or before ``train_end``. The only sanctioned path.

    Taking the boundary rather than a pre-sliced frame makes the contract hard
    to get wrong: there is no argument you can pass that fits on validation
    data.
    """
    train_slice = frame.loc[frame.index <= train_end]
    if train_slice.empty:
        raise ValueError(
            f"{station_id}: no rows at or before train_end={train_end}; "
            "the chronological split is misconfigured (§6.6)."
        )
    return LocalScaler(station_id=station_id).fit(train_slice, feature_columns)


def contains_normalization_statistics(payload: object) -> bool:
    """True if ``payload`` carries local normalization statistics.

    Used by the federated privacy guard. Recursive and deliberately
    over-eager — a false positive costs a developer thirty seconds, a false
    negative is a silent breach of the project's entire privacy claim.
    """
    if isinstance(payload, LocalScaler):
        return True
    if isinstance(payload, str):
        return payload == NEVER_TRANSMIT
    if isinstance(payload, dict):
        keys = {str(k).lower() for k in payload}
        if {"mean", "std"} <= keys or "privacy_marker" in keys:
            return True
        return any(contains_normalization_statistics(v) for v in payload.values())
    if isinstance(payload, (list, tuple, set)):
        return any(contains_normalization_statistics(v) for v in payload)
    return False


# ---------------------------------------------------------------------------
# §2.4a — the one summary a node may upload about its own level
# ---------------------------------------------------------------------------
#: Bucket edges in units of the station's own training standard deviation.
#: Five levels: well below / below / normal / above / well above. Coarse on
#: purpose — the project owner chose an ordinal over a continuous scalar
#: precisely so this cannot act as a fine-grained channel.
LEVEL_BUCKET_EDGES: tuple[float, ...] = (-1.0, -0.33, 0.33, 1.0)
LEVEL_BUCKET_NORMAL = 2          # the index meaning "no unusual level"
N_LEVEL_BUCKETS = len(LEVEL_BUCKET_EDGES) + 1


def level_bucket(normalised_target: "np.ndarray", window: int = 72) -> int:
    """Bucket a station's recent normalised level into one of five ordinals.

    LOCKED by §2.4a. The input is **already z-scored** against this station's
    own training window (§6.5), which is the whole reason this is safe to send:
    the output says "about one of my own standard deviations above my own
    training mean" and carries no information about what that is in physical
    units. The scaler stays on the node.

    ``window`` is 72 hours to match the drift episode floor in
    ``preprocessing/fault_labels.py`` — a drift episode shorter than that is
    indistinguishable from a bias step, so averaging over less would mostly
    measure weather.

    Returns ``LEVEL_BUCKET_NORMAL`` for an all-NaN or empty input rather than
    raising, because a station with no recent valid data is reporting "nothing
    unusual", not a fault. Guessing otherwise would inject false positives.
    """
    import numpy as np

    arr = np.asarray(normalised_target, dtype=float).ravel()
    if arr.size == 0:
        return LEVEL_BUCKET_NORMAL
    recent = arr[-window:] if arr.size > window else arr
    if not np.isfinite(recent).any():
        return LEVEL_BUCKET_NORMAL
    return int(np.searchsorted(LEVEL_BUCKET_EDGES,
                               float(np.nanmean(recent)), side="right"))
