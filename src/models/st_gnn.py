"""§8.1 rung 4 — centralised spatio-temporal GNN: the accuracy ceiling.

What forecasting accuracy looks like with **no privacy constraint at all**.
Every station's raw windows are held centrally, a shared GRU encodes each one,
a graph layer mixes the encoded states across stations, and a shared decoder
rolls out the horizon. Nothing here respects §2.4 — that is the point. Rung 4
exists to say how much the privacy constraint costs, by showing what is
available without it.

Two differences from the rung 6 server GNN, both deliberate:

* **Self-loops are on.** A station's own history is its best single predictor
  of its own future, so excluding it would cripple the ceiling. Rung 6's
  reconstruction task is the opposite case — there, a node that sees its own
  state reconstructs it perfectly and the disagreement signal collapses, which
  is why `build_adjacency` defaults to a zero diagonal and this module opts in.
* **Multi-hop is fine.** The i → j → i path that would leak self-information
  into a reconstruction is simply useful context when the task is forecasting.

Structurally it follows Wang et al. (2020) for physically-informed edges and
Meng et al. (2021) for the encode → graph → decode arrangement, minus the
federated split that makes CNFGNN what it is.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn

from src.utils.device import resolve_device
from src.utils.logging_setup import get_logger

log = get_logger(__name__)


def normalise_adjacency(adjacency: np.ndarray) -> torch.Tensor:
    """Row-normalise so message passing averages rather than sums.

    Without this a well-connected station receives a larger-magnitude message
    purely for having more neighbours, which the network then has to spend
    capacity undoing.
    """
    adj = torch.tensor(adjacency, dtype=torch.float32)
    degree = adj.sum(dim=1, keepdim=True).clamp(min=1.0)
    return adj / degree


class SpatioTemporalGNN(nn.Module):
    """Shared GRU encoder → graph mixing across stations → shared GRU decoder."""

    def __init__(self, n_features: int, adjacency: np.ndarray,
                 hidden_size: int = 64, horizon: int = 6, num_layers: int = 2,
                 dropout: float = 0.1, target_index: int = 0):
        super().__init__()
        self.horizon = horizon
        self.hidden_size = hidden_size
        self.target_index = target_index
        self.register_buffer("adj", normalise_adjacency(adjacency))

        self.encoder = nn.GRU(n_features, hidden_size, batch_first=True)
        self.graph = nn.ModuleList(
            [nn.Linear(hidden_size, hidden_size) for _ in range(max(num_layers, 1))])
        self.dropout = nn.Dropout(dropout)
        self.decoder = nn.GRU(1, hidden_size, batch_first=True)
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, X: torch.Tensor,
                mask: torch.Tensor | None = None) -> torch.Tensor:
        """``X`` is (batch, n_stations, input_hours, n_features) -> (batch, n_stations, horizon).

        ``mask`` is (batch, n_stations) and marks which stations actually
        reported that hour. Absent stations arrive zero-filled, and zero is NOT
        neutral: the features are z-scored, so an unmasked model reads an
        absent station as one sitting exactly at its own average. Their hidden
        states are therefore zeroed before mixing and the adjacency is
        renormalised per sample, so every station averages over the neighbours
        that were actually present rather than over a constant.
        """
        B, N, L, F = X.shape

        _, h = self.encoder(X.reshape(B * N, L, F))
        h = h[-1].reshape(B, N, self.hidden_size)

        if mask is not None:
            m = mask.to(h.dtype)
            h = h * m.unsqueeze(-1)
            # Row-renormalise the adjacency against who is present, so an hour
            # with half the network missing is not quietly scaled down.
            w = torch.einsum("ij,bj->bi", self.adj, m).clamp_min(1e-6)

        # Spatial mixing. Residual connections so a station can keep its own
        # representation when its neighbours have nothing to add.
        for layer in self.graph:
            message = torch.einsum("ij,bjk->bik", self.adj, h)
            if mask is not None:
                message = message / w.unsqueeze(-1)
            h = h + self.dropout(torch.relu(layer(message)))
            if mask is not None:
                h = h * m.unsqueeze(-1)

        # Decode each station from its fused state, starting at its own last
        # observed target.
        prev = X[:, :, -1, self.target_index].reshape(B * N, 1, 1)
        state = h.reshape(1, B * N, self.hidden_size).contiguous()

        outputs = []
        for _ in range(self.horizon):
            out, state = self.decoder(prev, state)
            step = self.head(self.dropout(out[:, -1]))
            outputs.append(step)
            prev = step.reshape(-1, 1, 1)

        return torch.cat(outputs, dim=1).reshape(B, N, self.horizon)


@dataclass
class STGNNHistory:
    train_loss: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    best_epoch: int = -1
    best_val: float = float("inf")
    stopped_early: bool = False

    def as_dict(self) -> dict:
        return {"epochs": len(self.train_loss), "best_epoch": self.best_epoch,
                "best_val_loss": self.best_val,
                "stopped_early": self.stopped_early}


def train_stgnn(model: SpatioTemporalGNN, train_set, val_set,
                epochs: int = 40, lr: float = 1e-3, batch_size: int = 128,
                patience: int = 6, grad_clip: float = 1.0,
                device: str | None = "auto",
                num_workers: int = 0) -> STGNNHistory:
    """Joint training over every station at once, the centralised privilege.

    Takes torch Datasets rather than dense arrays. The dense cross-station
    tensor is (origins, stations, hours, features), which at 69 stations is
    6.64 GiB for training alone and duplicates per-station arrays that are
    already resident; assembling each batch instead keeps peak memory flat.
    Each item is ``(x, y, mask)``, the mask marking stations that reported.
    """
    dev = resolve_device(device)
    model.to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    history = STGNNHistory()
    best_state, stale = None, 0

    def loader(ds, shuffle):
        return torch.utils.data.DataLoader(
            ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    train_loader = loader(train_set, True)
    val_loader = loader(val_set, False)

    def masked_mse(pred, target, m):
        """Mean squared error over present (sample, station) pairs only.

        Scoring absent stations against their zero fill would train the model to
        predict each station's own mean whenever it is missing, which is both
        wrong and trivially easy -- it would drag the reported loss down while
        making the forecasts worse."""
        w = m.unsqueeze(-1).to(pred.dtype)
        denom = w.sum() * pred.shape[-1] / max(pred.shape[-1], 1)
        return ((pred - target) ** 2 * w).sum() / denom.clamp_min(1.0)

    for epoch in range(epochs):
        model.train()
        total, n = 0.0, 0
        for xb, yb, mb in train_loader:
            xb, yb, mb = (xb.to(dev, non_blocking=True),
                          yb.to(dev, non_blocking=True),
                          mb.to(dev, non_blocking=True))
            opt.zero_grad()
            loss = masked_mse(model(xb, mb), yb, mb)
            loss.backward()
            if grad_clip:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()
            total += float(loss.item()) * len(xb)
            n += len(xb)
        history.train_loss.append(total / max(n, 1))

        model.eval()
        vtotal, vn = 0.0, 0
        with torch.no_grad():
            for xb, yb, mb in val_loader:
                xb, yb, mb = (xb.to(dev, non_blocking=True),
                              yb.to(dev, non_blocking=True),
                              mb.to(dev, non_blocking=True))
                vtotal += float(masked_mse(model(xb, mb), yb, mb).item()) * len(xb)
                vn += len(xb)
        val = vtotal / max(vn, 1)
        history.val_loss.append(val)

        if val < history.best_val - 1e-6:
            history.best_val, history.best_epoch = val, epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                history.stopped_early = True
                break

        if epoch % 5 == 0:
            log.info("ST-GNN epoch %3d | train %.5f | val %.5f | best@%d",
                     epoch, history.train_loss[-1], val, history.best_epoch)

    if best_state is not None:
        model.load_state_dict(best_state)
    log.info("ST-GNN: %d epochs, best %d, val %.5f", len(history.train_loss),
             history.best_epoch, history.best_val)
    return history


def predict(model: SpatioTemporalGNN, dataset,
            batch_size: int = 256) -> np.ndarray:
    """Forecasts for every origin in ``dataset``, in its own order."""
    model.eval()
    # Follow the model rather than re-resolving: after training it already
    # lives wherever it was trained, and moving it back would be silent waste.
    dev = next(model.parameters()).device
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         shuffle=False)
    out = []
    with torch.no_grad():
        for xb, _, mb in loader:
            out.append(model(xb.to(dev), mb.to(dev)).cpu().numpy())
    return (np.concatenate(out, axis=0) if out
            else np.empty((0, 0, model.horizon), dtype=np.float32))
