"""§7.2, §7.3 — the server side: FedAvg aggregation and the training loop.

Rung 5 only. The server here aggregates GRU weights and nothing else; the
graph network over uploaded hidden states is rung 6 and lives in
`src/models/server_gnn.py`. Keeping them apart means rung 5 measures exactly
what §8.1 says it should — *federation without spatial reasoning* — so the
difference rung 6 makes is attributable.

The server never sees raw data. It holds station coordinates for later graph
construction, but those come from the public EEA registry, not from an upload
(`configs/federation.yaml` lists them under `never_transmitted` for precisely
this reason).

§8.2 asks for communication cost per round and rounds to convergence, so both
are measured here rather than estimated afterwards.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from src.federated.aggregation_weights import (
    aggregation_weights, cluster_assignments, describe,
)
from src.federated.client import FederatedClient, Upload
from src.utils.logging_setup import get_logger

log = get_logger(__name__)


@dataclass
class RoundRecord:
    round: int
    mean_local_train_loss: float
    mean_val_loss: float
    uplink_bytes: int
    downlink_bytes: int

    @property
    def total_mb(self) -> float:
        return (self.uplink_bytes + self.downlink_bytes) / 1e6


@dataclass
class FederatedHistory:
    rounds: list[RoundRecord] = field(default_factory=list)
    best_round: int = -1
    best_val: float = float("inf")
    stopped_early: bool = False

    def as_dict(self) -> dict:
        return {
            "n_rounds": len(self.rounds),
            "best_round": self.best_round,
            "best_val_loss": self.best_val,
            "stopped_early": self.stopped_early,
            "rounds_to_convergence": self.best_round + 1,
            "total_communication_mb": round(
                sum(r.total_mb for r in self.rounds), 3),
            "mb_per_round": round(
                float(np.mean([r.total_mb for r in self.rounds])), 4
            ) if self.rounds else 0.0,
            "val_loss_by_round": [round(r.mean_val_loss, 6) for r in self.rounds],
        }


def resolve_clusters(clients: list[FederatedClient], scheme: str,
                     clusters: dict[str, int] | None,
                     ) -> dict[str, int] | None:
    """Settle the community assignment once, before the first round.

    Only ``clustered`` needs one. When the caller does not supply it, it is
    built from the public EEA station-registry coordinates — which are not
    observations and so cost nothing against §2.4. Hidden-state embeddings
    would sharpen it (see ``cluster_assignments``) but do not exist yet at this
    point in the run, and clustering on round-1 states of an untrained encoder
    would be noise dressed up as structure.
    """
    if scheme != "clustered" or clusters is not None:
        return clusters
    from src.data.station_metadata import coordinates
    from src.utils.paths import load_config

    cfg = (load_config("federation").get("aggregation", {})
           .get("clustered", {}) or {})
    return cluster_assignments(
        clients, coords=coordinates(),
        n_clusters=cfg.get("n_clusters"),
        spatial_weight=float(cfg.get("spatial_weight", 0.5)),
    )


class FederatedServer:
    """FedAvg over per-node GRU weights (§7.2)."""

    def __init__(self, initial_weights: dict[str, torch.Tensor],
                 scheme: str = "region_first",
                 clusters: dict[str, int] | None = None):
        self.global_weights = {k: v.detach().clone()
                               for k, v in initial_weights.items()}
        self.scheme = scheme
        # Assigned once, before round 1, and held fixed. Re-clustering every
        # round would make the weighting a moving target and the rounds
        # incomparable.
        self.clusters = clusters
        self.history = FederatedHistory()

    def aggregate(self, uploads: list[Upload]) -> dict[str, torch.Tensor]:
        """Weighted average of the uploaded weights (§5.7 for the weights)."""
        if not uploads:
            raise ValueError("a round produced no uploads")
        weights = aggregation_weights(uploads, self.scheme, clusters=self.clusters)

        new_global: dict[str, torch.Tensor] = {}
        for key in self.global_weights:
            stacked = None
            for w, upload in zip(weights, uploads):
                tensor = upload.gru_weight_update[key].to(torch.float64) * w
                stacked = tensor if stacked is None else stacked + tensor
            new_global[key] = stacked.to(self.global_weights[key].dtype)

        self.global_weights = new_global
        return self.global_weights

    def broadcast_bytes(self, n_clients: int) -> int:
        per_client = sum(v.numel() * v.element_size()
                         for v in self.global_weights.values())
        return per_client * n_clients


def run_split_learning(
    clients: list[FederatedClient],
    embed_fn,
    rounds: int = 60,
    local_epochs: int = 2,
    scheme: str = "region_first",
    patience: int = 8,
    clusters: dict[str, int] | None = None,
) -> tuple[dict[str, torch.Tensor], FederatedHistory]:
    """§7.3 in full: weights up, hidden states up, embeddings **back down**.

    The loop rung 5 leaves out. After aggregating weights the server runs its
    graph network over the uploaded hidden states and sends each station a
    spatial summary, which the node fuses into its encoder state before
    decoding. Nodes therefore forecast with neighbourhood context they never
    saw raw.

    ``embed_fn(clients, global_weights)`` does the server-side work and returns
    ``{split: {station_id: embeddings}}`` plus the uplink bytes those hidden
    states cost. It is injected rather than imported so this loop stays
    agnostic about which spatial model is in use.

    **Simplification, stated rather than buried:** gradients do not cross the
    server boundary. CNFGNN (Meng et al. 2021) backpropagates through the
    server GNN into the node encoders; here the GNN is trained on its own
    objective and the embedding is treated as a fixed input by the node. That
    is weaker, and it is the honest description of what this implements.
    """
    if not clients:
        raise ValueError("no clients")

    server = FederatedServer(clients[0].trainer.get_weights(), scheme=scheme,
                             clusters=resolve_clusters(clients, scheme, clusters))
    best_weights = {k: v.clone() for k, v in server.global_weights.items()}
    stale = 0

    log.info("split learning: %d clients, <=%d rounds x %d local epochs, "
             "weighting=%s", len(clients), rounds, local_epochs, scheme)

    for rnd in range(rounds):
        uploads = [c.local_train(server.global_weights, local_epochs)
                   for c in clients]
        uplink = sum(u.payload_bytes() for u in uploads)
        downlink = server.broadcast_bytes(len(clients))
        server.aggregate(uploads)

        # The server pass: hidden states up, embeddings down.
        embeddings, hidden_bytes = embed_fn(clients, server.global_weights)
        for split, per_station in embeddings.items():
            for client in clients:
                if client.station_id in per_station:
                    client.receive_embeddings(split, per_station[client.station_id])
        uplink += hidden_bytes
        downlink += sum(e.nbytes for per in embeddings.values() for e in per.values())

        val = float(np.nanmean([c.evaluate(server.global_weights, "val")
                                for c in clients]))
        train = float(np.nanmean([u.local_train_loss for u in uploads]))

        server.history.rounds.append(RoundRecord(
            round=rnd, mean_local_train_loss=train, mean_val_loss=val,
            uplink_bytes=uplink, downlink_bytes=downlink))

        if val < server.history.best_val - 1e-6:
            server.history.best_val = val
            server.history.best_round = rnd
            best_weights = {k: v.detach().clone()
                            for k, v in server.global_weights.items()}
            stale = 0
        else:
            stale += 1

        log.info("round %3d | train %.5f | val %.5f | %.2f MB | best@%d",
                 rnd, train, val, server.history.rounds[-1].total_mb,
                 server.history.best_round)

        if stale >= patience:
            server.history.stopped_early = True
            log.info("early stop at round %d (best %d, val %.5f)", rnd,
                     server.history.best_round, server.history.best_val)
            break

    h = server.history
    log.info("split learning done: %d rounds, converged at %d, %.1f MB total",
             len(h.rounds), h.best_round, sum(r.total_mb for r in h.rounds))
    return best_weights, h


def run_federated_training(
    clients: list[FederatedClient],
    rounds: int = 60,
    local_epochs: int = 2,
    scheme: str = "region_first",
    patience: int = 8,
    log_every: int = 1,
    clusters: dict[str, int] | None = None,
) -> tuple[dict[str, torch.Tensor], FederatedHistory]:
    """The §7.3 loop. Returns the best global weights and the run history.

    "Best" is by mean validation loss across clients, and the round that
    achieved it is what §8.2 means by rounds to convergence. Early stopping is
    on the same quantity — without it a fixed 60 rounds spends most of its
    compute after the model has stopped improving.
    """
    if not clients:
        raise ValueError("no clients")

    server = FederatedServer(clients[0].trainer.get_weights(), scheme=scheme,
                             clusters=resolve_clusters(clients, scheme, clusters))
    best_weights = {k: v.clone() for k, v in server.global_weights.items()}
    stale = 0

    log.info("federated training: %d clients, <=%d rounds x %d local epochs, "
             "weighting=%s", len(clients), rounds, local_epochs, scheme)

    for rnd in range(rounds):
        uploads = [c.local_train(server.global_weights, local_epochs)
                   for c in clients]
        if rnd == 0:
            log.info("client weights -> %s",
                     describe(uploads, scheme, clusters=server.clusters))

        uplink = sum(u.payload_bytes() for u in uploads)
        downlink = server.broadcast_bytes(len(clients))
        server.aggregate(uploads)

        val = float(np.nanmean([c.evaluate(server.global_weights, "val")
                                for c in clients]))
        train = float(np.nanmean([u.local_train_loss for u in uploads]))

        server.history.rounds.append(RoundRecord(
            round=rnd, mean_local_train_loss=train, mean_val_loss=val,
            uplink_bytes=uplink, downlink_bytes=downlink,
        ))

        if val < server.history.best_val - 1e-6:
            server.history.best_val = val
            server.history.best_round = rnd
            best_weights = {k: v.detach().clone()
                            for k, v in server.global_weights.items()}
            stale = 0
        else:
            stale += 1

        if log_every and rnd % log_every == 0:
            log.info("round %3d | train %.5f | val %.5f | %.2f MB | best@%d",
                     rnd, train, val, server.history.rounds[-1].total_mb,
                     server.history.best_round)

        if stale >= patience:
            server.history.stopped_early = True
            log.info("early stop at round %d (best round %d, val %.5f)",
                     rnd, server.history.best_round, server.history.best_val)
            break

    h = server.history
    log.info("done: %d rounds, converged at %d, %.1f MB total (%.3f MB/round)",
             len(h.rounds), h.best_round, sum(r.total_mb for r in h.rounds),
             float(np.mean([r.total_mb for r in h.rounds])))
    return best_weights, h
