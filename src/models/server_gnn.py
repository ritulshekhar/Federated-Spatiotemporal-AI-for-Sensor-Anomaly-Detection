"""§7.2 — server-side graph network over uploaded hidden states.

The server sees encoder hidden states and nothing else: never raw readings,
never the node's scaler, never the node's full model (split learning, Meng et
al. 2021 / CNFGNN). Station coordinates come from the public EEA registry, not
from an upload — `configs/federation.yaml` lists them under
`never_transmitted` for exactly that reason.

**How spatial disagreement is obtained.** Each node's hidden state is
reconstructed *from its neighbours' hidden states only*, and the reconstruction
error is the disagreement. A station behaving like its neighbours is easy to
predict from them; a station whose sensor has drifted is not. This is the
embedding-space analogue of entry 8's "distance from the neighbour median",
which recovered drift (0.004 → 0.149) and bias (0.024 → 0.261) from raw
readings. Whether it survives compression into a 64-dim hidden state is the
§11 open risk, and the point of the experiment.

**Self-information is structurally excluded, not merely discouraged.** The
adjacency has a zero diagonal and exactly **one** aggregation hop. Two hops
would walk a path i → j → i and hand the node its own state back through the
back door, quietly turning the reconstruction into an autoencoder and the
disagreement into noise. Depth lives in the MLP after aggregation, where it
cannot create such a path.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn

from src.utils.logging_setup import get_logger

log = get_logger(__name__)

EARTH_RADIUS_KM = 6371.0


def haversine_km(a: tuple[float, float], b: tuple[float, float]) -> float:
    """Great-circle distance between two (lat, lon) pairs."""
    lat1, lon1, lat2, lon2 = map(np.radians, (a[0], a[1], b[0], b[1]))
    dlat, dlon = lat2 - lat1, lon2 - lon1
    h = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return float(2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(h)))


def build_adjacency(station_ids: list[str],
                    coordinates: dict[str, tuple[float, float]],
                    mode: str = "hybrid",
                    distance_km_threshold: float = 12.0,
                    hidden_states: np.ndarray | None = None,
                    correlation_threshold: float = 0.5) -> np.ndarray:
    """Binary adjacency, zero diagonal.

    ``distance`` uses physical proximity, which Wang et al. (2020) justifies for
    air quality. ``correlation`` derives edges from similarity of the uploaded
    hidden states — computed server-side from embeddings, never from raw series
    (§2.4). ``hybrid`` takes the union, so a physically distant but
    behaviourally similar pair still exchanges information.
    """
    n = len(station_ids)
    adj = np.zeros((n, n), dtype=float)

    if mode in ("distance", "hybrid"):
        missing = [s for s in station_ids if s not in coordinates]
        if missing and mode == "distance":
            raise ValueError(f"no coordinates for {missing}; run station_metadata --refresh")
        for i, si in enumerate(station_ids):
            for j, sj in enumerate(station_ids):
                if i == j or si not in coordinates or sj not in coordinates:
                    continue
                if haversine_km(coordinates[si], coordinates[sj]) <= distance_km_threshold:
                    adj[i, j] = 1.0

    if mode in ("correlation", "hybrid"):
        if hidden_states is None:
            if mode == "correlation":
                raise ValueError("correlation mode needs uploaded hidden states")
        else:
            flat = hidden_states.reshape(hidden_states.shape[0], n, -1).mean(axis=2)
            corr = np.corrcoef(flat.T)
            np.fill_diagonal(corr, 0.0)
            adj = np.maximum(adj, (corr >= correlation_threshold).astype(float))

    np.fill_diagonal(adj, 0.0)       # LOCKED: a node never informs itself
    isolated = [station_ids[i] for i in range(n) if adj[i].sum() == 0]
    if isolated:
        log.warning("%d station(s) have no neighbours and cannot be scored "
                    "spatially: %s", len(isolated), isolated)
    log.info("graph: %d nodes, %d directed edges (mode=%s)",
             n, int(adj.sum()), mode)
    return adj


class NeighbourReconstructor(nn.Module):
    """Predict each node's hidden state from its neighbours' alone.

    The edge weights are learned (masked to the adjacency, diagonal forced to
    -inf before the softmax), so the model discovers which stations actually
    inform which — an artefact worth reporting in its own right.
    """

    def __init__(self, hidden_size: int, adjacency: np.ndarray,
                 embed_size: int = 64, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        adj = torch.tensor(adjacency, dtype=torch.float32)
        self.register_buffer("mask", adj)
        self.edge_logits = nn.Parameter(torch.zeros_like(adj))

        layers: list[nn.Module] = []
        in_dim = hidden_size
        for _ in range(max(num_layers, 1)):
            layers += [nn.Linear(in_dim, embed_size), nn.ReLU(), nn.Dropout(dropout)]
            in_dim = embed_size
        self.mlp = nn.Sequential(*layers)
        self.out = nn.Linear(embed_size, hidden_size)

    def attention(self) -> torch.Tensor:
        """Row-normalised neighbour weights; no self-loop can survive the mask."""
        logits = self.edge_logits.masked_fill(self.mask == 0, float("-inf"))
        attn = torch.softmax(logits, dim=1)
        return torch.nan_to_num(attn, nan=0.0)   # isolated rows become all-zero

    def forward(self, H: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """``H`` is (batch, n_stations, hidden). Returns (reconstruction, embedding)."""
        attn = self.attention()
        pooled = torch.einsum("ij,bjk->bik", attn, H)   # one hop, neighbours only
        embedding = self.mlp(pooled)
        return self.out(embedding), embedding


@dataclass
class GNNHistory:
    train_loss: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    best_epoch: int = -1
    best_val: float = float("inf")

    def as_dict(self) -> dict:
        return {"epochs": len(self.train_loss), "best_epoch": self.best_epoch,
                "best_val_loss": self.best_val}


def train_reconstructor(model: NeighbourReconstructor, H_train: np.ndarray,
                        H_val: np.ndarray, epochs: int = 40, lr: float = 1e-3,
                        batch_size: int = 256,
                        patience: int = 6) -> GNNHistory:
    """Fit on TRAIN hidden states only; early stop on validation."""
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    history = GNNHistory()
    best_state, stale = None, 0

    Xtr = torch.from_numpy(np.ascontiguousarray(H_train, dtype=np.float32))
    Xva = torch.from_numpy(np.ascontiguousarray(H_val, dtype=np.float32))
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(Xtr), batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        model.train()
        total, n = 0.0, 0
        for (batch,) in loader:
            opt.zero_grad()
            recon, _ = model(batch)
            loss = loss_fn(recon, batch)
            loss.backward()
            opt.step()
            total += float(loss.item()) * len(batch)
            n += len(batch)
        history.train_loss.append(total / max(n, 1))

        model.eval()
        with torch.no_grad():
            recon, _ = model(Xva)
            val = float(loss_fn(recon, Xva).item())
        history.val_loss.append(val)

        if val < history.best_val - 1e-7:
            history.best_val, history.best_epoch = val, epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    log.info("server GNN: %d epochs, best %d, val %.5f",
             len(history.train_loss), history.best_epoch, history.best_val)
    return history


def spatial_disagreement(model: NeighbourReconstructor, H: np.ndarray,
                         metric: str = "cosine") -> np.ndarray:
    """Per (window, station) disagreement with the neighbourhood.

    ``cosine``   1 - cosine similarity between the node's own hidden state and
                 the one its neighbours predict for it. Scale-free, which
                 matters because hidden-state magnitude varies by station.
    ``l2``       plain reconstruction error.
    """
    model.eval()
    with torch.no_grad():
        X = torch.from_numpy(np.ascontiguousarray(H, dtype=np.float32))
        recon, _ = model(X)
        if metric == "cosine":
            sim = torch.nn.functional.cosine_similarity(recon, X, dim=-1)
            return (1.0 - sim).cpu().numpy()
        if metric == "l2":
            return torch.linalg.norm(recon - X, dim=-1).cpu().numpy()
    raise ValueError(f"unknown metric {metric!r}; use 'cosine' or 'l2'")


def spatial_agreement(embedding: np.ndarray, neighbour_embeddings: np.ndarray,
                      metric: str = "cosine", min_neighbours: int = 2) -> float:
    """§7.4's named function: how well one node agrees with its neighbours.

    Kept with the signature §7.4 specifies. High agreement means an anomalous
    reading is probably a real event the whole neighbourhood is seeing; low
    agreement means the station is on its own, which is what a sensor fault
    looks like.
    """
    neighbour_embeddings = np.atleast_2d(neighbour_embeddings)
    if len(neighbour_embeddings) < min_neighbours:
        return float("nan")      # not enough neighbours to make a call
    centre = neighbour_embeddings.mean(axis=0)
    if metric == "cosine":
        denom = np.linalg.norm(embedding) * np.linalg.norm(centre)
        return float(np.dot(embedding, centre) / denom) if denom > 1e-12 else float("nan")
    if metric == "l2":
        return float(-np.linalg.norm(embedding - centre))
    raise ValueError(f"unknown metric {metric!r}")
