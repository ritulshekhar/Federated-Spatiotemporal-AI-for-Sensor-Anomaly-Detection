"""§7.1, §7.3 — the node side of a federated round.

A client holds one station's data, trains locally from the broadcast global
weights, and returns an ``Upload``. What may be in that upload is not a matter
of convention: ``configs/federation.yaml`` lists it, and ``check_upload``
enforces the list on every single round.

The privacy contract (§2.4) in one line: **weight updates and encoder hidden
states leave the node; raw readings and local normalization statistics never
do.** The scaler is the subtle one — mean and standard deviation summarise a
station's raw distribution, so shipping them would be a disclosure dressed up
as a preprocessing detail. ``normalize.contains_normalization_statistics``
exists for exactly this check and is deliberately over-eager.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from src.models.node_gru import NodeTrainer
from src.preprocessing.normalize import contains_normalization_statistics
from src.utils.logging_setup import get_logger
from src.utils.paths import load_config

log = get_logger(__name__)


class PrivacyViolation(RuntimeError):
    """Raised when an outbound payload contains something §2.4 forbids."""


@dataclass
class Upload:
    """Everything one client sends the server in one round. Nothing else."""
    station_id: str
    region: str
    n_samples: int
    gru_weight_update: dict[str, torch.Tensor]
    #: Only populated once the server GNN exists (rung 6). Rung 5 has no
    #: spatial model, so sending hidden states would be comms cost for nothing.
    encoder_hidden_state: np.ndarray | None = None
    local_train_loss: float = float("nan")

    def payload_bytes(self) -> int:
        """Uncompressed upload size — the §8.2 communication-cost metric."""
        total = sum(v.numel() * v.element_size()
                    for v in self.gru_weight_update.values())
        if self.encoder_hidden_state is not None:
            total += int(self.encoder_hidden_state.nbytes)
        return total


def check_upload(upload: Upload, cfg: dict | None = None) -> Upload:
    """Enforce `configs/federation.yaml` `privacy:` on an outbound payload.

    Called on every upload, every round. A guard that runs only in tests is not
    a guard.
    """
    cfg = cfg or load_config("federation")
    privacy = cfg.get("privacy", {})
    allowed = set(privacy.get("transmitted_from_node", []))

    present = {f.name for f in upload.__dataclass_fields__.values()}
    # Identity and bookkeeping fields are not payload.
    payload_fields = present - {"station_id", "region", "local_train_loss"}
    unexpected = payload_fields - allowed
    if unexpected:
        raise PrivacyViolation(
            f"{upload.station_id}: upload carries field(s) {sorted(unexpected)} "
            f"which configs/federation.yaml does not permit "
            f"(allowed: {sorted(allowed)}). See CLAUDE.md §2.4."
        )

    for name in payload_fields:
        value = getattr(upload, name)
        if contains_normalization_statistics(value):
            raise PrivacyViolation(
                f"{upload.station_id}: field {name!r} contains local "
                "normalization statistics. A shared scaler is itself a "
                "disclosure — it encodes the station's raw distribution "
                "(CLAUDE.md §2.4 / §6.5)."
            )
    return upload


class FederatedClient:
    """One station as an AIoT edge node (§1.1 objective 1)."""

    def __init__(self, station, feature_index: np.ndarray, n_features: int,
                 target_index: int, horizon: int, model_cfg: dict,
                 seed: int = 1337, embedding_size: int = 0):
        self.station = station
        self.station_id = station.station_id
        self.region = station.splits["train"].region
        self._idx = feature_index          # common-feature reordering (§ rung 3)
        self.embedding_size = embedding_size
        #: Embeddings the server sent down (§7.2), per split. None until the
        #: first round has completed a server pass.
        self.embeddings: dict[str, np.ndarray] = {}

        self.trainer = NodeTrainer(
            n_features=n_features, horizon=horizon, cfg=model_cfg,
            target_index=target_index, station_id=self.station_id, seed=seed,
            embedding_size=embedding_size,
        )
        self.trainer.cfg.verbose_every = 0   # the server logs per round instead

    # -- data views --------------------------------------------------------
    def X(self, split: str) -> np.ndarray:
        return self.station.splits[split].X[:, :, self._idx]

    def y(self, split: str) -> np.ndarray:
        return self.station.splits[split].y

    @property
    def n_samples(self) -> int:
        return len(self.station.splits["train"])

    # -- a federated round -------------------------------------------------
    def local_train(self, global_weights: dict[str, torch.Tensor],
                    epochs: int) -> Upload:
        """Start from the global weights, train locally, hand back an upload.

        Raw readings never appear in the return value — only what the station
        learned from them.
        """
        self.trainer.set_weights(global_weights)
        history = self.trainer.fit(self.X("train"), self.y("train"), epochs=epochs,
                                   E_train=self.embeddings.get("train"))
        loss = history.train_loss[-1] if history.train_loss else float("nan")

        return check_upload(Upload(
            station_id=self.station_id,
            region=self.region,
            n_samples=self.n_samples,
            gru_weight_update=self.trainer.get_weights(),
            local_train_loss=float(loss),
        ))

    def evaluate(self, global_weights: dict[str, torch.Tensor],
                 split: str = "val") -> float:
        """Local loss under the current global model — reported, never uploaded raw."""
        self.trainer.set_weights(global_weights)
        pred = self.trainer.predict(self.X(split), E=self.embeddings.get(split))
        y = self.y(split)
        if len(pred) == 0:
            return float("nan")
        return float(np.mean((pred - y) ** 2))

    def predict(self, global_weights: dict[str, torch.Tensor],
                split: str) -> np.ndarray:
        self.trainer.set_weights(global_weights)
        return self.trainer.predict(self.X(split), E=self.embeddings.get(split))

    def receive_embeddings(self, split: str, embeddings: np.ndarray) -> None:
        """Accept the server's spatial summary for this split (§7.2 downlink)."""
        expected = len(self.station.splits[split])
        if embeddings.shape[0] != expected:
            raise ValueError(
                f"{self.station_id}/{split}: got {embeddings.shape[0]} embeddings "
                f"for {expected} windows"
            )
        self.embeddings[split] = np.ascontiguousarray(embeddings, dtype=np.float32)
