"""§7.1 — Node tier: per-station GRU encoder-decoder.

Input:  24h window of pollutant + meteorological features.
Output: forecast for t+1..t+6 (1h / 3h / 6h are the reported horizons, §8.2).

The encoder's final hidden state is exposed separately via ``encode()`` because
it is the *only* representation of local data that ever leaves the node (§7.1,
§7.2). Everything else — raw readings, the local scaler, the residuals used by
the decision layer — stays put. Keeping ``encode`` a first-class method rather
than reaching into ``forward``'s internals means the Phase 2 split-learning
server (Meng et al. 2021 / CNFGNN) has exactly one documented seam to attach to.

A real decoder GRU is used rather than a dense multi-output head. It costs
little and it is what makes the architecture an encoder-decoder in the sense
§7.1 specifies: the hidden state is a summary of the input window that a
separate recurrent process consumes, which is the same object the server graph
network is supposed to reason over.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn

from src.utils.logging_setup import get_logger

log = get_logger(__name__)


class GRUEncoderDecoder(nn.Module):
    """Encoder GRU over the input window; decoder GRU rolls out the horizon."""

    def __init__(self, n_features: int, hidden_size: int = 64, num_layers: int = 1,
                 horizon: int = 6, dropout: float = 0.1):
        super().__init__()
        self.n_features = n_features
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.horizon = horizon

        self.encoder = nn.GRU(
            input_size=n_features, hidden_size=hidden_size, num_layers=num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0.0,
        )
        # Decoder input is the previous step's target value (teacher-forced during
        # training, self-fed at inference).
        self.decoder = nn.GRU(
            input_size=1, hidden_size=hidden_size, num_layers=num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_size, 1)

    # -- the privacy seam --------------------------------------------------
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Final-layer encoder hidden state, ``(batch, hidden_size)``.

        This is the payload the node uploads for server-side spatial modelling
        (§7.1). It is a compressed summary of the input window, never the window.
        """
        _, h = self.encoder(x)
        return h[-1]

    # -- forecasting -------------------------------------------------------
    def forward(self, x: torch.Tensor, y_teacher: torch.Tensor | None = None,
                teacher_forcing: float = 0.0) -> torch.Tensor:
        """``x`` is ``(batch, input_hours, n_features)``; returns ``(batch, horizon)``."""
        enc_out, h = self.encoder(x)

        # Decoding starts from the last observed target. The target is feature
        # column 0 by the convention that build_features puts the target
        # pollutant first (see NodeTrainer.target_index).
        prev = x[:, -1:, self.target_index].reshape(x.size(0), 1, 1)

        outputs = []
        for step in range(self.horizon):
            out, h = self.decoder(prev, h)
            pred = self.head(self.dropout(out[:, -1]))     # (batch, 1)
            outputs.append(pred)
            if (y_teacher is not None and teacher_forcing > 0.0
                    and float(torch.rand(1)) < teacher_forcing):
                prev = y_teacher[:, step].reshape(-1, 1, 1)
            else:
                prev = pred.reshape(-1, 1, 1)
        return torch.cat(outputs, dim=1)

    target_index: int = 0   # overridden by NodeTrainer when the target moves


@dataclass
class TrainConfig:
    epochs: int = 40
    batch_size: int = 128
    lr: float = 1e-3
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    early_stopping_patience: int = 6
    teacher_forcing_start: float = 0.5
    teacher_forcing_end: float = 0.0
    scheduler_factor: float = 0.5
    scheduler_patience: int = 3
    device: str = "cpu"
    verbose_every: int = 5


@dataclass
class TrainHistory:
    train_loss: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    best_epoch: int = -1
    best_val: float = float("inf")
    stopped_early: bool = False

    def as_dict(self) -> dict:
        return self.__dict__.copy()


class NodeTrainer:
    """Local training for one station. Also the unit of a FedAvg round (§7.3)."""

    def __init__(self, n_features: int, horizon: int, cfg: dict,
                 target_index: int = 0, station_id: str = "?", seed: int = 1337):
        torch.manual_seed(seed)
        self.station_id = station_id
        self.model = GRUEncoderDecoder(
            n_features=n_features,
            hidden_size=cfg.get("hidden_size", 64),
            num_layers=cfg.get("num_layers", 1),
            horizon=horizon,
            dropout=cfg.get("dropout", 0.1),
        )
        self.model.target_index = target_index
        tcfg = cfg.get("train", {})
        self.cfg = TrainConfig(**{k: v for k, v in tcfg.items()
                                  if k in TrainConfig.__dataclass_fields__})
        self.device = torch.device(self.cfg.device)
        self.model.to(self.device)
        self.history = TrainHistory()

    # -- training ----------------------------------------------------------
    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            X_val: np.ndarray | None = None, y_val: np.ndarray | None = None,
            epochs: int | None = None) -> TrainHistory:
        epochs = epochs if epochs is not None else self.cfg.epochs
        loader = self._loader(X_train, y_train, shuffle=True)
        # Shuffling WINDOWS during SGD is not the §6.6 violation — the split was
        # already made chronologically, so no future window is in this batch's
        # dataset. Only the split may never be shuffled.

        opt = torch.optim.Adam(self.model.parameters(), lr=self.cfg.lr,
                               weight_decay=self.cfg.weight_decay)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, factor=self.cfg.scheduler_factor, patience=self.cfg.scheduler_patience,
        )
        loss_fn = nn.MSELoss()

        best_state, patience = None, 0
        for epoch in range(epochs):
            tf = self._teacher_forcing(epoch, epochs)
            self.model.train()
            total, n_batches = 0.0, 0
            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                opt.zero_grad()
                pred = self.model(xb, y_teacher=yb, teacher_forcing=tf)
                loss = loss_fn(pred, yb)
                loss.backward()
                if self.cfg.grad_clip:
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
                opt.step()
                total += float(loss.item())
                n_batches += 1
            train_loss = total / max(n_batches, 1)
            self.history.train_loss.append(train_loss)

            if X_val is not None and len(X_val):
                val_loss = self._evaluate(X_val, y_val, loss_fn)
                self.history.val_loss.append(val_loss)
                sched.step(val_loss)
                if val_loss < self.history.best_val - 1e-6:
                    self.history.best_val = val_loss
                    self.history.best_epoch = epoch
                    best_state = {k: v.detach().clone()
                                  for k, v in self.model.state_dict().items()}
                    patience = 0
                else:
                    patience += 1
                    if patience >= self.cfg.early_stopping_patience:
                        self.history.stopped_early = True
                        log.info("%s | early stop at epoch %d (best %d, val %.5f)",
                                 self.station_id, epoch, self.history.best_epoch,
                                 self.history.best_val)
                        break
                if self.cfg.verbose_every and epoch % self.cfg.verbose_every == 0:
                    log.info("%s | epoch %3d | train %.5f | val %.5f | tf %.2f",
                             self.station_id, epoch, train_loss, val_loss, tf)
            elif self.cfg.verbose_every and epoch % self.cfg.verbose_every == 0:
                log.info("%s | epoch %3d | train %.5f", self.station_id, epoch,
                         train_loss)

        if best_state is not None:
            self.model.load_state_dict(best_state)
        return self.history

    def _teacher_forcing(self, epoch: int, epochs: int) -> float:
        a, b = self.cfg.teacher_forcing_start, self.cfg.teacher_forcing_end
        return a + (b - a) * (epoch / max(epochs - 1, 1))

    def _evaluate(self, X: np.ndarray, y: np.ndarray, loss_fn) -> float:
        self.model.eval()
        total, n = 0.0, 0
        with torch.no_grad():
            for xb, yb in self._loader(X, y, shuffle=False):
                xb, yb = xb.to(self.device), yb.to(self.device)
                total += float(loss_fn(self.model(xb), yb).item()) * len(xb)
                n += len(xb)
        return total / max(n, 1)

    # -- inference ---------------------------------------------------------
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Normalized-space forecasts, ``(n, horizon)``."""
        self.model.eval()
        out = []
        with torch.no_grad():
            for (xb,) in self._loader(X, None, shuffle=False):
                out.append(self.model(xb.to(self.device)).cpu().numpy())
        return (np.concatenate(out, axis=0) if out
                else np.empty((0, self.model.horizon), dtype=np.float32))

    def embed(self, X: np.ndarray) -> np.ndarray:
        """Encoder hidden states, ``(n, hidden_size)`` — the Phase 2 upload."""
        self.model.eval()
        out = []
        with torch.no_grad():
            for (xb,) in self._loader(X, None, shuffle=False):
                out.append(self.model.encode(xb.to(self.device)).cpu().numpy())
        return (np.concatenate(out, axis=0) if out
                else np.empty((0, self.model.hidden_size), dtype=np.float32))

    # -- federated plumbing (used in Phase 2) -------------------------------
    def get_weights(self) -> dict[str, torch.Tensor]:
        return {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}

    def set_weights(self, weights: dict[str, torch.Tensor]) -> None:
        self.model.load_state_dict(weights)

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.model.parameters())

    def payload_bytes(self) -> int:
        """Uncompressed size of one weight upload — the §8.2 comms metric."""
        return sum(v.numel() * v.element_size() for v in self.get_weights().values())

    # -- helpers -----------------------------------------------------------
    def _loader(self, X: np.ndarray, y: np.ndarray | None, shuffle: bool):
        xt = torch.from_numpy(np.ascontiguousarray(X, dtype=np.float32))
        if y is None:
            ds = torch.utils.data.TensorDataset(xt)
        else:
            yt = torch.from_numpy(np.ascontiguousarray(y, dtype=np.float32))
            ds = torch.utils.data.TensorDataset(xt, yt)
        return torch.utils.data.DataLoader(
            ds, batch_size=self.cfg.batch_size, shuffle=shuffle, drop_last=False,
        )
