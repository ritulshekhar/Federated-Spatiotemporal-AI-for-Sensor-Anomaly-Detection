"""§8.1 — The baseline ladder.

    1. Persistence (last observed value)                  [Phase 1]
    2. Local-only per-station GRU (no federation)         [Phase 1]
    3. Centralised GRU (pooled windows, no privacy)       [Phase 1]
    4. Centralised ST-GNN (accuracy ceiling)              [Phase 2]
    5. FedAvg + GRU (federated, no spatial reasoning)     [Phase 2]
    6. Proposed: federated GRU + server GNN + decision    [Phase 2]

Built in order, every rung before the next (§8.1). Rungs 1-3 must be working
and reported before the graph layer starts — the graph layer is the highest-risk
component, and locking the ladder first guarantees a defensible result even if
it underperforms.

Each rung produces forecasts **and** an anomaly decision, because the anomaly
classification is the point (§1) and a forecasting-only table would let the
project drift towards optimising the mechanism instead of the objective. The
Phase 1 rungs use the simplest possible decision rule — threshold the residual —
which is deliberately the naive detector Gupchup et al. (2019) showed
misclassifies real events as faults. Its ``fpr_on_extremes`` column is the
number rung 6 has to beat, and having it early makes that comparison concrete
rather than rhetorical.

**One note on "pooling" in rung 3.** §8.1 calls rung 3 "all stations pooled, no
privacy constraint". What is pooled is the *window set* — one shared set of
model weights trained on every station's windows. Normalization stays
per-station (§2.2/§6.5: a shared scaler is itself a privacy leak and a
distribution leak), and every metric stays per-station (§8.3). Rung 3 is the
no-privacy *training* reference, not permission to pool the reporting.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.evaluation.metrics import (
    calibrate_threshold, extreme_mask, residual_score, volatility_scaled_score,
)
from src.evaluation.report import build_report, make_records
from src.models.decision_layer import DecisionLayer
from src.preprocessing.normalize import LocalScaler
from src.preprocessing.windowing import SPLITS, WindowSet, load_windows
from src.utils.logging_setup import get_logger
from src.utils.paths import load_config, station_dir, variant_root

log = get_logger(__name__)


@dataclass
class StationData:
    """One station's window splits, its local scaler, and its manifest."""
    station_id: str
    splits: dict[str, WindowSet]
    scaler: LocalScaler
    manifest: dict

    @property
    def target(self) -> str:
        return self.manifest["target"]

    @property
    def target_index(self) -> int:
        return self.splits["train"].feature_columns.index(self.target)

    @property
    def target_scale(self) -> float:
        """Local std of the target, in physical units. Stays on the node."""
        return self.scaler.scale_of(self.target)

    def y_pred_raw(self, y_pred_norm: np.ndarray) -> np.ndarray:
        return self.scaler.inverse_transform_column(y_pred_norm, self.target)


def load_station(station_id: str, variant: str = "") -> StationData:
    d = station_dir(station_id, "processed", variant=variant)
    missing = [s for s in SPLITS if not (d / f"{s}.npz").exists()]
    if missing:
        raise FileNotFoundError(
            f"{station_id}: missing {missing} under {d}. Run "
            "`python -m src.preprocessing.pipeline --region <region>` first."
        )
    return StationData(
        station_id=station_id,
        splits={s: load_windows(d / f"{s}.npz") for s in SPLITS},
        scaler=LocalScaler.load(d / "scaler.json"),
        manifest=json.loads((d / "manifest.json").read_text(encoding="utf-8")),
    )


def load_region(region: str, variant: str = "",
                max_stations: int | None = None) -> list[StationData]:
    """Load a region's preprocessed stations.

    ``max_stations`` truncates the station LIST before any window array is
    read, which is the whole point: loading 69 stations and then discarding 39
    of them still pays the full peak memory, and that peak is what killed two
    ~18-hour runs.

    Stations are chosen by **training coverage**, most-covered first, and this
    is a requirement rather than a preference. Rungs 4 and 6 can only use
    window origins present at *every* participating station, so one station
    with a short or offset record empties the intersection for all of them.
    Measured on the Catalan set: picking by coverage retains 42,076 of 48,892
    common train origins at 30 stations (86%), while picking the 30
    alphabetically first retains **zero**: the intersection erodes to 8,982 and
    is then emptied outright by ES1347A, which contributes only 5,416 windows.
    The full 69 retain zero under any selection, which is why the spatial rungs
    cannot run on the whole network at all -- see CONTEXT.md.

    Ties break on station id, so the subset stays reproducible.
    """
    manifest_path = variant_root("processed", variant) / f"run_manifest_{region}.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"{manifest_path} not found. Run the preprocessing pipeline for "
            f"region {region!r}"
            + (f" with --inject-synthetic (variant {variant!r})" if variant else "")
            + " first."
        )
    ids = json.loads(manifest_path.read_text(encoding="utf-8"))["stations_processed"]
    if max_stations is not None and len(ids) > max_stations:
        ids = _best_covered(ids, variant, max_stations)
        log.info("loading only %d of the region's stations, chosen by training "
                 "coverage: %s", len(ids), ", ".join(ids[:6]) + " ...")
    return [load_station(sid, variant=variant) for sid in ids]


def _best_covered(ids: list[str], variant: str, keep: int) -> list[str]:
    """The ``keep`` stations with the most training windows.

    Reads only each station's ``t0`` array, not its window payloads, so the
    choice costs a few MB rather than the full load it exists to avoid.
    """
    counts = {}
    for sid in ids:
        path = station_dir(sid, "processed", variant=variant) / "train.npz"
        try:
            with np.load(path, allow_pickle=True) as z:
                counts[sid] = len(z["t0"])
        except Exception as exc:  # noqa: BLE001 -- a station we cannot size
            log.warning("cannot size %s, ranking it last: %s", sid, exc)
            counts[sid] = -1
    return sorted(sorted(ids), key=lambda s: -counts[s])[:keep]


# ---------------------------------------------------------------------------
# Shared scoring path — every rung goes through this, so no rung can quietly
# use a different threshold rule or skip the false-positive control.
# ---------------------------------------------------------------------------
def _decision_scores(layers, ws, pred_raw: np.ndarray, scale: float,
                     spatial_series: pd.Series | None, fit: bool,
                     labels: np.ndarray | None = None,
                     is_extreme: np.ndarray | None = None,
                     target_fpr_extremes: float = 0.10,
                     budget_on: str = "overall",
                     target_fpr_overall: float = 0.005) -> tuple[np.ndarray, dict]:
    """Run the decision-layer ensemble per horizon, returning (n_windows, horizon).

    Each horizon is a separate stride-1 time series, so its rolling statistics
    are computed on its own index and never blended with another horizon's.
    """
    residuals = np.abs(pred_raw - ws.y_raw) / max(scale, 1e-8)
    out = np.empty_like(residuals, dtype=float)
    # Per-detector scores travel into records.parquet so a different budget or
    # combination rule can be swept offline, without retraining anything.
    detectors = {k: np.full_like(residuals, np.nan, dtype=float)
                 for k in ("residual", "accumulated", "low_variance", "spatial",
                           "drift_cusum")}

    spatial_by_t0 = (spatial_series.reindex(pd.DatetimeIndex(ws.t0)).to_numpy()
                     if spatial_series is not None else None)

    for j, layer in enumerate(layers):
        index = pd.DatetimeIndex(ws.t_target[:, j])
        frame = pd.DataFrame(
            {"residual": residuals[:, j], "y_true": ws.y_raw[:, j]}, index=index)
        if spatial_by_t0 is not None:
            frame["spatial"] = spatial_by_t0
        ordered = frame.sort_index()
        if fit:
            order = np.argsort(index.to_numpy())
            layer.fit(ordered, labels=labels[:, j][order],
                      is_extreme=is_extreme[:, j][order],
                      target_fpr_extremes=target_fpr_extremes,
                      budget_on=budget_on,
                      target_fpr_overall=target_fpr_overall)
        scored = layer.score(ordered)
        out[:, j] = pd.Series(scored.combined, index=ordered.index).reindex(index).to_numpy()
        for name, values in (("residual", scored.residual),
                             ("accumulated", scored.accumulated),
                             ("low_variance", scored.low_variance),
                             ("spatial", scored.spatial),
                             ("drift_cusum", scored.drift_cusum)):
            if values is None:
                continue
            detectors[name][:, j] = (
                pd.Series(values, index=ordered.index).reindex(index).to_numpy())
    return out, detectors


def score_station(station: StationData, model_name: str,
                  pred_val_norm: np.ndarray, pred_test_norm: np.ndarray,
                  extreme_quantile: float = 0.95,
                  threshold_method: str = "quantile",
                  threshold_quantile: float = 0.98,
                  label_source: str = "auto",
                  score_mode: str = "volatility",
                  volatility_floor: float = 0.25,
                  target_fpr_extremes: float = 0.10,
                  spatial: dict[str, pd.Series] | None = None,
                  budget_on: str = "overall",
                  target_fpr_overall: float = 0.005,
                  ) -> tuple[pd.DataFrame, float]:
    """Turn one station's forecasts into prediction records and a threshold.

    The threshold is calibrated on that station's **validation** split, never on
    test, and never shared with another station (§5.8, §7.4).
    """
    val, test = station.splits["val"], station.splits["test"]
    scale = station.target_scale
    ti = station.target_index

    # Volatility of the target over each window's own 24h input, in physical
    # units. Known at forecast time, so using it leaks nothing.
    val_vol = val.X[:, :, ti].std(axis=1) * scale
    test_vol = test.X[:, :, ti].std(axis=1) * scale
    vol_floor = volatility_floor * scale

    val_pred_raw = station.y_pred_raw(pred_val_norm)
    test_pred_raw = station.y_pred_raw(pred_test_norm)

    # Needed before scoring: the decision layer calibrates each detector on the
    # validation labels, so they have to exist first.
    resolved_source = label_source
    if resolved_source == "auto":
        resolved_source = ("synthetic" if val.target_synthetic_label is not None
                           else "real")
    val_codes = val.labels_for(resolved_source)
    val_extreme = extreme_mask(
        val.y_raw, quantile=extreme_quantile,
        reference=station.splits["train"].y_raw.ravel(),
    )

    if spatial is not None:
        # Rung 6: the four-detector ensemble, fitted on validation and applied
        # to test. One DecisionLayer per horizon, because each horizon is its
        # own time-ordered series and the rolling statistics must not mix them.
        layers = [DecisionLayer() for _ in range(val.horizon)]
        val_score, val_detectors = _decision_scores(
            layers, val, val_pred_raw, scale, spatial.get("val"), fit=True,
            labels=val_codes, is_extreme=val_extreme,
            target_fpr_extremes=target_fpr_extremes,
            budget_on=budget_on, target_fpr_overall=target_fpr_overall)
        test_score, test_detectors = _decision_scores(
            layers, test, test_pred_raw, scale, spatial.get("test"), fit=False)
        detector_scores = {"val": val_detectors, "test": test_detectors}
    elif score_mode == "volatility":
        detector_scores = None
        val_score = volatility_scaled_score(
            val.y_raw, val_pred_raw, val_vol[:, None], vol_floor)
        test_score = volatility_scaled_score(
            test.y_raw, test_pred_raw, test_vol[:, None], vol_floor)
    elif score_mode == "absolute":
        detector_scores = None
        val_score = residual_score(val.y_raw, val_pred_raw, scale)
        test_score = residual_score(test.y_raw, test_pred_raw, scale)
    else:
        raise ValueError(f"unknown score_mode {score_mode!r}")

    threshold = calibrate_threshold(
        val_score, method=threshold_method, quantile=threshold_quantile,
        mask=~val.target_is_imputed, labels=val_codes, is_extreme=val_extreme,
        target_fpr_extremes=target_fpr_extremes,
    )

    # "Extreme" is defined on the TRAINING target, so the control does not move
    # with the test set it is meant to police.
    is_extreme = extreme_mask(
        test.y_raw, quantile=extreme_quantile,
        reference=station.splits["train"].y_raw.ravel(),
    )

    # One label source per run, not both at once. Forecast error does not depend
    # on which labels you score against, so emitting both would duplicate every
    # forecast row under two label_source values — identical numbers, twice.
    # Instead each processed variant carries one source: the injected variant is
    # scored "synthetic" (the §2.1a primary ground truth) and the clean variant
    # "real" (the data-availability baseline). Both are reported, from their own
    # run, and §8.3's grouping keeps them apart.
    source = resolved_source

    # Validation records are kept too, so a different scoring rule or threshold
    # method can be calibrated and evaluated offline from records.parquet
    # without retraining a single model.
    frames = []
    for split_name, ws, pred_raw, sc, vol, ext in (
        ("val", val, val_pred_raw, val_score, val_vol, val_extreme),
        ("test", test, test_pred_raw, test_score, test_vol, is_extreme),
    ):
        codes = ws.labels_for(source)
        rec = make_records(
            model=model_name,
            station_id=station.station_id,
            region=ws.region,
            station_class=ws.station_class,
            label_source=source,
            split=split_name,
            t_target=ws.t_target,
            y_true=ws.y_raw,
            y_pred=pred_raw,
            is_imputed=ws.target_is_imputed,
            label=np.where(codes == 1, "fault",
                           np.where(codes == 0, "trustworthy", "unknown")),
            fault_type=ws.target_fault_type,
            is_extreme=ext,
            local_volatility=vol,
            score=sc,
            horizons=list(range(1, ws.horizon + 1)),
        )
        if detector_scores is not None:
            for name, values in detector_scores[split_name].items():
                rec[f"det_{name}"] = values.ravel()
        frames.append(rec)

    return pd.concat(frames, ignore_index=True), float(threshold)


# ---------------------------------------------------------------------------
# Rung 1 — persistence
# ---------------------------------------------------------------------------
def rung1_persistence(stations: list[StationData], **score_kwargs):
    """ŷ(t+h) = y(t), for every h. The floor any model must clear."""
    records, thresholds = [], {}
    for st in stations:
        preds = {}
        for split in ("val", "test"):
            ws = st.splits[split]
            last = ws.X[:, -1, st.target_index]            # normalized space
            preds[split] = np.repeat(last[:, None], ws.horizon, axis=1)
        r, thr = score_station(st, "1_persistence", preds["val"], preds["test"],
                               **score_kwargs)
        records.append(r)
        thresholds[("1_persistence", st.station_id)] = thr
    return pd.concat(records, ignore_index=True), thresholds


# ---------------------------------------------------------------------------
# Rung 2 — local-only GRU
# ---------------------------------------------------------------------------
def rung2_local_gru(stations: list[StationData], model_cfg: dict,
                    seed: int = 1337, **score_kwargs):
    """One GRU per station, trained only on that station's own data."""
    from src.models.node_gru import NodeTrainer

    records, thresholds, histories = [], {}, {}
    for st in stations:
        train, val = st.splits["train"], st.splits["val"]
        trainer = NodeTrainer(
            n_features=train.X.shape[2], horizon=train.horizon,
            cfg=model_cfg["node_gru"], target_index=st.target_index,
            station_id=st.station_id, seed=seed,
        )
        log.info("%s | rung 2: training on %s windows (%d params)",
                 st.station_id, f"{len(train):,}", trainer.n_parameters())
        trainer.fit(train.X, train.y, val.X, val.y)
        histories[st.station_id] = trainer.history.as_dict()

        r, thr = score_station(st, "2_local_gru",
                               trainer.predict(val.X), trainer.predict(st.splits["test"].X),
                               **score_kwargs)
        records.append(r)
        thresholds[("2_local_gru", st.station_id)] = thr
    return pd.concat(records, ignore_index=True), thresholds, histories


# ---------------------------------------------------------------------------
# Rung 3 — centralised GRU
# ---------------------------------------------------------------------------
def rung3_centralised_gru(stations: list[StationData], model_cfg: dict,
                          seed: int = 1337, **score_kwargs):
    """One shared GRU over every station's windows. No privacy constraint.

    Upper reference for what federation costs in accuracy (§8.1 rung 3,
    compared against Khan et al. 2024's 94.5% vs 96.7% federated-vs-centralised
    benchmark at rung 5).
    """
    from src.models.node_gru import NodeTrainer

    common, per_station_idx = align_features(stations)
    log.info("rung 3: %d feature column(s) common to all %d station(s)",
             len(common), len(stations))

    X_tr = np.concatenate([st.splits["train"].X[:, :, per_station_idx[st.station_id]]
                           for st in stations], axis=0)
    y_tr = np.concatenate([st.splits["train"].y for st in stations], axis=0)
    X_va = np.concatenate([st.splits["val"].X[:, :, per_station_idx[st.station_id]]
                           for st in stations], axis=0)
    y_va = np.concatenate([st.splits["val"].y for st in stations], axis=0)

    target = stations[0].target
    trainer = NodeTrainer(
        n_features=len(common), horizon=stations[0].splits["train"].horizon,
        cfg=model_cfg["centralised_gru"], target_index=common.index(target),
        station_id="CENTRALISED", seed=seed,
    )
    log.info("rung 3: training on %s pooled windows (%d params)",
             f"{len(X_tr):,}", trainer.n_parameters())
    trainer.fit(X_tr, y_tr, X_va, y_va)

    records, thresholds = [], {}
    for st in stations:
        idx = per_station_idx[st.station_id]
        r, thr = score_station(
            st, "3_centralised_gru",
            trainer.predict(st.splits["val"].X[:, :, idx]),
            trainer.predict(st.splits["test"].X[:, :, idx]),
            **score_kwargs,
        )
        records.append(r)
        thresholds[("3_centralised_gru", st.station_id)] = thr
    return pd.concat(records, ignore_index=True), thresholds, trainer.history.as_dict()


def align_features(stations: list[StationData]) -> tuple[list[str], dict[str, np.ndarray]]:
    """Common feature columns across stations, plus per-station reorder indices.

    Real EEA stations do not all report the same pollutants, so their feature
    matrices differ in width and order. A shared model needs one column order;
    taking the intersection keeps that honest rather than zero-filling absent
    pollutants, which would teach the model that "missing" means "zero".
    """
    sets = [set(st.splits["train"].feature_columns) for st in stations]
    common_set = set.intersection(*sets)
    if not common_set:
        raise ValueError("stations share no feature columns")

    # Preserve the first station's ordering for determinism.
    common = [c for c in stations[0].splits["train"].feature_columns if c in common_set]
    target = stations[0].target
    if target not in common:
        raise ValueError(f"target {target!r} is not common to all stations")

    dropped = {st.station_id: sorted(set(st.splits["train"].feature_columns) - common_set)
               for st in stations}
    for sid, cols in dropped.items():
        if cols:
            log.info("rung 3: %s drops %d station-specific feature(s): %s",
                     sid, len(cols), cols[:6])

    idx = {
        st.station_id: np.array(
            [st.splits["train"].feature_columns.index(c) for c in common], dtype=int
        )
        for st in stations
    }
    return common, idx


# ---------------------------------------------------------------------------
# Rungs 4-6 — Phase 2
# ---------------------------------------------------------------------------
class AlignedWindows(torch.utils.data.Dataset):
    """Cross-station windows assembled per sample, never materialised whole.

    The dense form of this is (origins, stations, hours, features) float32,
    which at 69 stations is **6.64 GiB for the training split alone** -- a
    duplicate of per-station arrays already resident, which is what killed the
    first full-network run. Each station's windows stay where they are and a
    sample is gathered on access, so a batch of 128 costs ~19 MB and peak
    memory does not move.

    ``positions[i, s]`` is the row of station ``s``'s own array holding origin
    ``i``, or -1 when that station did not report the hour. ``mask`` says the
    same thing as a boolean, and the model needs it: features are z-scored, so
    the zero fill IS the station's own training mean, and an unmasked model
    reads "absent" as "perfectly average".
    """

    def __init__(self, stations, per_station_idx, split, positions, mask, common):
        self.X = [st.splits[split].X for st in stations]
        self.y = [st.splits[split].y for st in stations]
        self.cols = [per_station_idx[st.station_id] for st in stations]
        self.positions = positions
        self.mask = mask
        self.common = common
        self.n_stations = len(stations)
        self.L = stations[0].splits[split].X.shape[1]
        self.H = stations[0].splits[split].y.shape[1]
        self.n_feat = len(self.cols[0])

    def __len__(self):
        return len(self.common)

    def __getitem__(self, i):
        x = np.zeros((self.n_stations, self.L, self.n_feat), dtype=np.float32)
        y = np.zeros((self.n_stations, self.H), dtype=np.float32)
        m = self.mask[i]
        for s in range(self.n_stations):
            if m[s]:
                r = self.positions[i, s]
                x[s] = self.X[s][r][:, self.cols[s]]
                y[s] = self.y[s][r]
        return (torch.from_numpy(x), torch.from_numpy(y),
                torch.from_numpy(np.ascontiguousarray(m)))


def _aligned_windows(stations: list[StationData], per_station_idx: dict,
                     split: str, min_fraction: float = 0.6):
    """Lazy cross-station alignment: a Dataset, the origins, positions, mask.

    Originally this required a window origin present at *every* station, which
    is clean but does not survive a real network: across 69 Catalan stations
    **no hour at all** is reported by all of them, so rungs 4 and 6 simply
    could not run. The intersection holds 100% of origins at 15 stations, 86%
    at 30, 57% at 50, and zero at 69.

    An hour now qualifies when at least ``min_fraction`` of stations reported
    it, and the result is returned as an :class:`AlignedWindows` dataset rather
    than a dense array -- see its docstring for why.
    """
    t0s = [st.splits[split].t0 for st in stations]
    counts: dict = {}
    for t in t0s:
        for v in t:
            counts[v] = counts.get(v, 0) + 1

    need = max(2, int(np.ceil(min_fraction * len(stations))))
    common = np.array(sorted(v for v, c in counts.items() if c >= need))
    if common.size == 0:
        raise ValueError(
            f"no window origin in {split!r} is reported by at least {need} of "
            f"{len(stations)} stations; lower min_fraction or narrow the node set"
        )

    full = sum(1 for v, c in counts.items() if c == len(stations))
    log.info("%s: %s origin(s) reported by >=%d of %d stations (%s reported by "
             "all)", split, f"{common.size:,}", need, len(stations), f"{full:,}")

    positions = np.full((common.size, len(stations)), -1, dtype=np.int64)
    mask = np.zeros((common.size, len(stations)), dtype=bool)
    per_station_pos = {}
    for s, st in enumerate(stations):
        t0 = st.splits[split].t0
        order = np.argsort(t0)
        sorted_t0 = t0[order]
        slot = np.clip(np.searchsorted(sorted_t0, common), 0,
                       max(len(sorted_t0) - 1, 0))
        present = sorted_t0[slot] == common
        pos = order[slot]
        positions[:, s] = np.where(present, pos, -1)
        mask[:, s] = present
        per_station_pos[st.station_id] = positions[:, s]

    ds = AlignedWindows(stations, per_station_idx, split, positions, mask, common)
    return ds, common, per_station_pos, mask


def _graph_sample(dataset, max_origins: int = 2000) -> np.ndarray:
    """(origins, stations, features) for correlation-based graph building.

    Subsampled on purpose: the adjacency only needs enough origins to estimate
    inter-station correlation, and using every one would rebuild the dense
    tensor this Dataset exists to avoid.
    """
    n = len(dataset)
    step = max(1, n // max_origins)
    rows = [dataset[i][0].numpy().reshape(dataset.n_stations, -1)
            for i in range(0, n, step)]
    return np.stack(rows, axis=0)


def rung4_centralised_stgnn(stations: list[StationData], model_cfg: dict,
                            seed: int = 1337, **score_kwargs):
    """§8.1 rung 4 — the accuracy ceiling, with no privacy constraint at all.

    Raw windows from every station, held centrally and trained jointly. The gap
    between this and rung 5/6 is what privacy costs.
    """
    import torch

    from src.data.station_metadata import coordinates
    from src.models.server_gnn import build_adjacency
    from src.models.st_gnn import SpatioTemporalGNN, predict, train_stgnn

    torch.manual_seed(seed)
    common_features, per_station_idx = align_features(stations)
    target = stations[0].target
    horizon = stations[0].splits["train"].horizon
    station_ids = [st.station_id for st in stations]

    data = {s: _aligned_windows(stations, per_station_idx, s)
            for s in ("train", "val", "test")}
    log.info("rung 4: %d common feature(s), %s/%s/%s aligned windows",
             len(common_features), f"{len(data['train'][0]):,}",
             f"{len(data['val'][0]):,}", f"{len(data['test'][0]):,}")
    # data[split] = (dataset, common_t0, positions_per_station, mask)

    gcfg = model_cfg["server_gnn"]
    adjacency = build_adjacency(
        station_ids, coordinates(), mode=gcfg["graph"]["mode"],
        distance_km_threshold=gcfg["graph"]["distance_km_threshold"],
        # build_adjacency averages the trailing axis per station, so flatten
        # each station's (hours x features) window into one vector.
        # Correlation-mode graphs need station vectors; building them from a
        # sample of origins keeps this off the 6.6 GiB dense path.
        hidden_states=(None if gcfg["graph"]["mode"] == "distance" else
                       _graph_sample(data["train"][0])),
        correlation_threshold=gcfg["graph"]["correlation_threshold"],
        self_loops=True,      # forecasting: a station's own history is allowed
    )

    model = SpatioTemporalGNN(
        n_features=len(common_features), adjacency=adjacency,
        hidden_size=gcfg["hidden_size"], horizon=horizon,
        num_layers=gcfg["num_layers"], dropout=gcfg["dropout"],
        target_index=common_features.index(target),
    )
    history = train_stgnn(
        model, data["train"][0], data["val"][0],
        **{k: v for k, v in model_cfg.get("centralised_gru", {}).get("train", {}).items()
           if k in ("epochs", "lr", "batch_size", "grad_clip")},
    )

    preds = {s: predict(model, data[s][0]) for s in ("val", "test")}

    records, thresholds = [], {}
    for i, st in enumerate(stations):
        # Scatter the aligned predictions back into each station's own window
        # order; windows the intersection dropped stay NaN and are excluded.
        full = {}
        for split in ("val", "test"):
            ws = st.splits[split]
            arr = np.full((len(ws), horizon), np.nan, dtype=np.float32)
            # A station absent for an aligned hour has position -1 and no
            # prediction to scatter; leaving those NaN keeps them out of
            # scoring rather than writing a zero-filled forecast back.
            pos = data[split][2][st.station_id]
            present = data[split][3][:, i]
            if present.any():
                arr[pos[present]] = preds[split][present, i, :]
            full[split] = arr
        r, thr = score_station(st, "4_centralised_stgnn", full["val"], full["test"],
                               **score_kwargs)
        records.append(r)
        thresholds[("4_centralised_stgnn", st.station_id)] = thr

    return (pd.concat(records, ignore_index=True), thresholds,
            history.as_dict() | {"graph_edges": int(adjacency.sum()),
                                 "aligned_windows": {s: int(len(data[s][0]))
                                                     for s in data}})


def rung5_fedavg_gru(stations: list[StationData], model_cfg: dict,
                     fed_cfg: dict, seed: int = 1337, **score_kwargs):
    """§8.1 rung 5 — FedAvg over per-station GRUs. No spatial reasoning yet.

    Federation without the graph layer, so the difference rung 6 makes is
    attributable to the graph layer rather than to federation. Compare against
    Khan et al. (2024): 94.5% federated vs 96.7% centralised.

    Weight averaging requires one architecture, so clients share the feature
    columns common to every station — the same constraint rung 3 has, and a
    real cost of weight sharing worth reporting rather than hiding.
    """
    from src.federated.client import FederatedClient
    from src.federated.server import run_federated_training

    common, per_station_idx = align_features(stations)
    target = stations[0].target
    horizon = stations[0].splits["train"].horizon
    log.info("rung 5: %d feature column(s) common to all %d client(s)",
             len(common), len(stations))

    # One config for every client: identical architecture is what makes the
    # weights averageable at all.
    client_cfg = dict(model_cfg["node_gru"])
    client_cfg["train"] = dict(client_cfg.get("train", {}))
    client_cfg["train"].update(fed_cfg.get("train", {}))

    clients = [
        FederatedClient(
            station=st, feature_index=per_station_idx[st.station_id],
            n_features=len(common), target_index=common.index(target),
            horizon=horizon, model_cfg=client_cfg, seed=seed,
        )
        for st in stations
    ]

    global_weights, history = run_federated_training(
        clients,
        rounds=int(fed_cfg.get("rounds", 60)),
        local_epochs=int(fed_cfg.get("local_epochs", 2)),
        scheme=fed_cfg.get("aggregation", {}).get("scheme", "region_first"),
        patience=int(fed_cfg.get("early_stopping_patience", 8)),
    )

    records, thresholds = [], {}
    for st, client in zip(stations, clients):
        r, thr = score_station(
            st, "5_fedavg_gru",
            client.predict(global_weights, "val"),
            client.predict(global_weights, "test"),
            **score_kwargs,
        )
        records.append(r)
        thresholds[("5_fedavg_gru", st.station_id)] = thr

    return pd.concat(records, ignore_index=True), thresholds, history.as_dict()


def _aligned_hidden_states(clients, global_weights, split: str,
                           min_fraction: float = 0.6):
    """(n_windows, n_stations, hidden), the origins, and a presence mask.

    Previously the server could only reason about hours *every* node reported,
    which is impossible past about 50 stations -- across the full 69-station
    Catalan set the intersection is empty, so rung 6 could not run at all. An
    hour now qualifies when ``min_fraction`` of stations reported it, and the
    mask says which. Absent stations get a zero hidden state, which the
    reconstructor must not treat as a real encoding.
    """
    per_station, counts = {}, {}
    for c in clients:
        ws = c.station.splits[split]
        c.trainer.set_weights(global_weights)
        per_station[c.station_id] = (ws.t0, c.trainer.embed(c.X(split)))
        for v in ws.t0:
            counts[v] = counts.get(v, 0) + 1

    need = max(2, int(np.ceil(min_fraction * len(clients))))
    common = np.array(sorted(v for v, n in counts.items() if n >= need))
    if common.size == 0:
        raise ValueError(
            f"no window origin in {split!r} is reported by at least {need} of "
            f"{len(clients)} stations"
        )

    stacked, mask = [], []
    hidden = next(iter(per_station.values()))[1].shape[1]
    for c in clients:
        t0, emb = per_station[c.station_id]
        order = np.argsort(t0)
        sorted_t0 = t0[order]
        slot = np.clip(np.searchsorted(sorted_t0, common), 0,
                       max(len(sorted_t0) - 1, 0))
        present = sorted_t0[slot] == common
        out = np.zeros((common.size, hidden), dtype=emb.dtype)
        if present.any():
            out[present] = emb[order[slot][present]]
        stacked.append(out)
        mask.append(present)

    full = sum(1 for v, n in counts.items() if n == len(clients))
    log.info("%s hidden states: %s origin(s) at >=%d of %d stations (%s at all)",
             split, f"{common.size:,}", need, len(clients), f"{full:,}")
    return np.stack(stacked, axis=1), common, np.stack(mask, axis=1)


def _make_embedder(reconstructor_holder: dict, gcfg: dict, station_ids: list[str],
                   coords: dict, gnn_epochs: int):
    """Build the server-side pass used by `run_split_learning`.

    Each round: collect hidden states on the common timeline, refresh the
    neighbour-reconstruction model, and scatter its embeddings back into every
    station's own window order. Windows outside the common timeline receive
    zeros, which the node's fusion layer reads as "no neighbourhood context".
    """
    import torch

    from src.models.server_gnn import NeighbourReconstructor, build_adjacency, train_reconstructor

    def embed_fn(clients, global_weights):
        hidden, positions, bytes_up = {}, {}, 0
        for split in ("train", "val", "test"):
            H, common, present = _aligned_hidden_states(
                clients, global_weights, split)
            hidden[split] = H
            bytes_up += H.nbytes
            positions[split] = {}
            for i, c in enumerate(clients):
                t0 = c.station.splits[split].t0
                order = np.argsort(t0)
                slot = np.clip(np.searchsorted(t0[order], common), 0,
                               max(len(t0) - 1, 0))
                # -1 marks an hour this station did not report, so nothing is
                # scattered back to it.
                positions[split][c.station_id] = np.where(
                    present[:, i], order[slot], -1)

        if reconstructor_holder.get("model") is None:
            adjacency = build_adjacency(
                station_ids, coords, mode=gcfg["graph"]["mode"],
                distance_km_threshold=gcfg["graph"]["distance_km_threshold"],
                hidden_states=hidden["train"],
                correlation_threshold=gcfg["graph"]["correlation_threshold"],
            )
            reconstructor_holder["model"] = NeighbourReconstructor(
                hidden_size=hidden["train"].shape[2], adjacency=adjacency,
                embed_size=gcfg["hidden_size"], num_layers=gcfg["num_layers"],
                dropout=gcfg["dropout"])
            reconstructor_holder["adjacency"] = adjacency

        model = reconstructor_holder["model"]
        train_reconstructor(model, hidden["train"], hidden["val"],
                            epochs=gnn_epochs, patience=gnn_epochs)

        out = {}
        for split, H in hidden.items():
            with torch.no_grad():
                _, emb = model(torch.from_numpy(np.ascontiguousarray(H, dtype=np.float32)))
            emb = emb.cpu().numpy()
            out[split] = {}
            for i, c in enumerate(clients):
                full = np.zeros((len(c.station.splits[split]), emb.shape[2]),
                                dtype=np.float32)
                full[positions[split][c.station_id]] = emb[:, i, :]
                out[split][c.station_id] = full
        return out, bytes_up

    return embed_fn


def rung6_proposed(stations: list[StationData], model_cfg: dict, fed_cfg: dict,
                   seed: int = 1337, split_learning: bool = False,
                   **score_kwargs):
    """§8.1 rung 6 — the proposed model: federated GRU + server GNN + decision layer.

    §11 flags `spatial_agreement()` from compressed embeddings as untested;
    entry 8 measured the raw-neighbour ceiling it has to reach for (drift
    0.149, bias 0.261). Whatever the gap turns out to be *is* the result.
    """
    from src.federated.client import FederatedClient
    from src.federated.server import run_federated_training, run_split_learning
    from src.models.server_gnn import (
        NeighbourReconstructor, build_adjacency, spatial_disagreement,
        train_reconstructor,
    )
    from src.data.station_metadata import coordinates

    common, per_station_idx = align_features(stations)
    target = stations[0].target
    horizon = stations[0].splits["train"].horizon
    gcfg = model_cfg["server_gnn"]
    sl_cfg = fed_cfg.get("split_learning", {})
    emb_size = int(gcfg["hidden_size"]) if split_learning else 0

    client_cfg = dict(model_cfg["node_gru"])
    client_cfg["train"] = dict(client_cfg.get("train", {}))
    client_cfg["train"].update(fed_cfg.get("train", {}))
    clients = [
        FederatedClient(
            station=st, feature_index=per_station_idx[st.station_id],
            n_features=len(common), target_index=common.index(target),
            horizon=horizon, model_cfg=client_cfg, seed=seed,
            embedding_size=emb_size,
        )
        for st in stations
    ]

    holder: dict = {}
    if split_learning:
        # §7.2's downlink: embeddings go back to the nodes each round.
        global_weights, fed_history = run_split_learning(
            clients,
            embed_fn=_make_embedder(holder, gcfg, [c.station_id for c in clients],
                                    coordinates(),
                                    int(sl_cfg.get("gnn_epochs_per_round", 3))),
            rounds=int(fed_cfg.get("rounds", 60)),
            local_epochs=int(fed_cfg.get("local_epochs", 2)),
            scheme=fed_cfg.get("aggregation", {}).get("scheme", "region_first"),
            patience=int(fed_cfg.get("early_stopping_patience", 8)),
        )
    else:
        # --- the federated forecaster, identical to rung 5 ------------------
        global_weights, fed_history = run_federated_training(
            clients,
            rounds=int(fed_cfg.get("rounds", 60)),
            local_epochs=int(fed_cfg.get("local_epochs", 2)),
            scheme=fed_cfg.get("aggregation", {}).get("scheme", "region_first"),
            patience=int(fed_cfg.get("early_stopping_patience", 8)),
        )

    # --- the server tier: hidden states only (§7.2) -------------------------
    H = {s: _aligned_hidden_states(clients, global_weights, s)
         for s in ("train", "val", "test")}
    station_ids = [c.station_id for c in clients]

    if holder.get("model") is not None:
        reconstructor = holder["model"]
        adjacency = holder["adjacency"]
        gnn_history = train_reconstructor(reconstructor, H["train"][0], H["val"][0])
    else:
        adjacency = build_adjacency(
            station_ids, coordinates(), mode=gcfg["graph"]["mode"],
            distance_km_threshold=gcfg["graph"]["distance_km_threshold"],
            hidden_states=H["train"][0],
            correlation_threshold=gcfg["graph"]["correlation_threshold"],
        )
        reconstructor = NeighbourReconstructor(
            hidden_size=H["train"][0].shape[2], adjacency=adjacency,
            embed_size=gcfg["hidden_size"], num_layers=gcfg["num_layers"],
            dropout=gcfg["dropout"],
        )
        gnn_history = train_reconstructor(reconstructor, H["train"][0], H["val"][0])

    metric = model_cfg["decision_layer"]["spatial_agreement"]["metric"]

    def _disagree(split):
        d = spatial_disagreement(reconstructor, H[split][0], metric=metric)
        # A station absent for an hour has a zero-filled hidden state, which
        # would reconstruct as a confident, meaningless disagreement. NaN keeps
        # it out of scoring entirely rather than inventing a score for it.
        return np.where(H[split][2], d, np.nan)

    disagreement = {split: (_disagree(split), H[split][1])
                    for split in ("val", "test")}

    # Hidden-state uploads are real communication cost (§8.2).
    hidden_mb = sum(H[s][0].nbytes for s in ("val", "test")) / 1e6

    records, thresholds = [], {}
    for i, (st, client) in enumerate(zip(stations, clients)):
        spatial = {
            split: pd.Series(disagreement[split][0][:, i],
                             index=pd.DatetimeIndex(disagreement[split][1]))
            for split in ("val", "test")
        }
        r, thr = score_station(
            st, "6_proposed",
            client.predict(global_weights, "val"),
            client.predict(global_weights, "test"),
            spatial=spatial, **score_kwargs,
        )
        records.append(r)
        thresholds[("6_proposed", st.station_id)] = thr

    history = {
        "federated": fed_history.as_dict(),
        "server_gnn": gnn_history.as_dict(),
        "graph_edges": int(adjacency.sum()),
        "hidden_state_upload_mb": round(hidden_mb, 3),
        "attention": reconstructor.attention().detach().cpu().numpy().round(3).tolist(),
        "station_order": station_ids,
        "drift_identification": _drift_identification(
            stations, clients, adjacency, station_ids),
    }
    return pd.concat(records, ignore_index=True), thresholds, history



def _drift_identification(stations, clients, adjacency, station_ids) -> dict:
    """§2.4a: station-level drift identification from uploaded level buckets.

    Separate from the per-reading anomaly table on purpose. §2.4a permits one
    ordinal per station per upload, so this names *stations* whose calibration
    has diverged from their neighbourhood, not readings. Reporting it as
    per-reading drift recall would overclaim what the channel can carry.

    Ground truth is "did this station's test split receive any injected drift
    episode", which is what a station-level claim can honestly be scored
    against.
    """
    from src.models.server_gnn import flag_drifting_stations, level_disagreement

    buckets = {st.station_id: client.level_bucket("test")
               for st, client in zip(stations, clients)}
    gaps = level_disagreement(buckets, adjacency, station_ids)
    flags = flag_drifting_stations(buckets, adjacency, station_ids)

    truth = {}
    for st in stations:
        ft = st.splits["test"].target_fault_type
        truth[st.station_id] = bool(
            ft is not None and (np.asarray(ft) == "drift").any())

    tp = sum(1 for s in station_ids if flags.get(s) and truth.get(s))
    fp = sum(1 for s in station_ids if flags.get(s) and not truth.get(s))
    fn = sum(1 for s in station_ids if not flags.get(s) and truth.get(s))
    undecidable = sum(1 for s in station_ids if not np.isfinite(gaps.get(s, np.nan)))

    log.info("drift identification (§2.4a, station level): %d flagged, "
             "%d true positive, %d false positive, %d missed, %d undecidable "
             "for want of neighbours", tp + fp, tp, fp, fn, undecidable)
    return {
        "buckets": {s: (None if buckets.get(s) is None else int(buckets[s]))
                    for s in station_ids},
        "disagreement_steps": {s: (None if not np.isfinite(gaps.get(s, np.nan))
                                   else round(float(gaps[s]), 3))
                               for s in station_ids},
        "flagged": {s: bool(flags.get(s, False)) for s in station_ids},
        "had_injected_drift": {s: bool(truth.get(s, False)) for s in station_ids},
        "true_positive": tp, "false_positive": fp, "missed": fn,
        "undecidable_no_neighbours": undecidable,
        "scope": ("station-level identification, not per-reading recall; "
                  "§2.4a permits one ordinal per station per upload"),
    }


# ---------------------------------------------------------------------------
def run_phase1(region: str, rungs: tuple[int, ...] = (1, 2, 3),
               out_name: str | None = None, variant: str = "",
               label_source: str = "auto",
               score_mode: str | None = None,
               split_learning: bool | None = None,
               max_stations: int | None = None) -> Path:
    """Run the Phase 1 rungs over one region and write one grouped report."""
    data_cfg = load_config("data")
    model_cfg = load_config("model")
    horizons = data_cfg["preprocessing"]["windowing"]["horizons_reported"]
    dec = model_cfg["decision_layer"]["residual_threshold"]
    seed = model_cfg.get("seed", 1337)

    stations = load_region(region, variant=variant, max_stations=max_stations)
    log.info("loaded %d station(s) for region %r (variant %r)", len(stations),
             region, variant or "clean")

    # (the node set was already narrowed in load_region, before any
    # window array was read — see its docstring)

    sc = model_cfg["decision_layer"].get("score", {})
    score_kwargs = {
        "threshold_method": dec["method"],
        "threshold_quantile": dec["quantile"],
        "target_fpr_extremes": dec.get("target_fpr_extremes", 0.10),
        "label_source": label_source,
        "score_mode": score_mode or sc.get("mode", "volatility"),
        "volatility_floor": sc.get("volatility_floor", 0.25),
    }
    budget = model_cfg["decision_layer"].get("detector_budget", {})
    score_kwargs["budget_on"] = budget.get("on", "overall")
    score_kwargs["target_fpr_overall"] = float(budget.get("target_fpr_overall", 0.005))
    log.info("scoring: mode=%s threshold=%s", score_kwargs["score_mode"],
             score_kwargs["threshold_method"])

    all_records, all_thresholds, extras = [], {}, {}

    if 1 in rungs:
        rec, thr = rung1_persistence(stations, **score_kwargs)
        all_records.append(rec)
        all_thresholds.update(thr)

    if 2 in rungs:
        rec, thr, hist = rung2_local_gru(stations, model_cfg, seed=seed, **score_kwargs)
        all_records.append(rec)
        all_thresholds.update(thr)
        extras["rung2_history"] = hist

    if 3 in rungs:
        rec, thr, hist = rung3_centralised_gru(stations, model_cfg, seed=seed,
                                               **score_kwargs)
        all_records.append(rec)
        all_thresholds.update(thr)
        extras["rung3_history"] = hist

    if 4 in rungs:
        rec, thr, hist = rung4_centralised_stgnn(stations, model_cfg, seed=seed,
                                                 **score_kwargs)
        all_records.append(rec)
        all_thresholds.update(thr)
        extras["rung4_history"] = hist

    if 5 in rungs:
        rec, thr, hist = rung5_fedavg_gru(
            stations, model_cfg, load_config("federation"), seed=seed,
            **score_kwargs)
        all_records.append(rec)
        all_thresholds.update(thr)
        extras["rung5_history"] = hist

    if 6 in rungs:
        fed_cfg = load_config("federation")
        use_split = (split_learning if split_learning is not None
                     else bool(fed_cfg.get("split_learning", {}).get("enabled", False)))
        rec, thr, hist = rung6_proposed(
            stations, model_cfg, fed_cfg, seed=seed,
            split_learning=use_split, **score_kwargs)
        all_records.append(rec)
        all_thresholds.update(thr)
        extras["rung6_history"] = hist

    if not all_records:
        raise ValueError(f"no rungs selected from {rungs}")

    records = pd.concat(all_records, ignore_index=True)

    report = build_report(
        name=out_name or f"phase1_{region}",
        records=records,
        thresholds=all_thresholds,
        horizons=horizons,
        meta={
            "region": region,
            "variant": variant or "clean",
            "label_source": label_source,
            "rungs": list(rungs),
            "seed": seed,
            "n_stations": len(stations),
            "target": stations[0].target,
            "threshold": score_kwargs,
            "extreme_quantile": 0.95,
            "thresholds_per_model_station": {
                f"{m}|{s}": v for (m, s), v in all_thresholds.items()
            },
            **{k: v for k, v in extras.items()},
        },
    )
    report.print_summary(horizons=horizons)
    return report.write()


def main() -> None:
    import argparse

    from src.utils.seeding import seed_everything

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--region", default="catalonia")
    ap.add_argument("--rungs", nargs="*", type=int, default=[1, 2, 3],
                    help="which ladder rungs to run (Phase 1 supports 1 2 3)")
    ap.add_argument("--name", default=None, help="report directory name")
    ap.add_argument("--variant", default="",
                    help="processed build to score, e.g. 'injected' (§2.1a)")
    ap.add_argument("--label-source", default="auto",
                    choices=["auto", "real", "synthetic"],
                    help="ground truth for the anomaly table; 'auto' picks "
                         "synthetic when the build has it")
    ap.add_argument("--score-mode", default=None,
                    choices=["absolute", "volatility"],
                    help="anomaly score form (default: configs/model.yaml)")
    ap.add_argument("--max-stations", type=int, default=None,
                    help="train on at most this many stations from the "
                         "preprocessed build, chosen by sorted station id. Lets "
                         "a smaller node set reuse an existing build instead of "
                         "rebuilding it.")
    ap.add_argument("--no-split-learning", action="store_true",
                    help="rung 6 without the §7.2 embedding downlink. Per-window "
                         "hidden states cost ~216 MB/round at 7 clients and scale "
                         "with the node set, so large runs want this off.")
    args = ap.parse_args()

    unsupported = [r for r in args.rungs if r not in (1, 2, 3, 4, 5, 6)]
    if unsupported:
        raise SystemExit(f"unknown rung(s) {unsupported}; §8.1 defines 1-6.")

    seed_everything(load_config("model").get("seed", 1337))
    out = run_phase1(args.region, rungs=tuple(args.rungs), out_name=args.name,
                     variant=args.variant, label_source=args.label_source,
                     score_mode=args.score_mode,
                     split_learning=False if args.no_split_learning else None,
                     max_stations=args.max_stations)
    print(f"\nreport -> {out}")


if __name__ == "__main__":
    main()
