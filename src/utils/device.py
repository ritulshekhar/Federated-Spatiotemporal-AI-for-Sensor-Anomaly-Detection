"""Where tensors live, decided in one place.

Every model in this project trained on CPU until 2026-10-13, not by choice but
because the installed torch was a CPU-only build on a machine with an idle
RTX 4060. The wheel is now `torch+cu130`, so the choice is real and belongs in
config rather than scattered across three training loops.

Measured on an RTX 4060 Laptop against the CPU build it replaced: the node GRU
is ~5.4x faster and the centralised ST-GNN ~3.9x. The server-side reconstructor
is marginally *slower* on GPU (0.23s against 0.18s for three epochs) because it
is tiny enough that kernel-launch overhead dominates -- left on the shared
default anyway, since a few hundred milliseconds either way is noise next to
the GRU training that surrounds it.

The default is ``auto``: use CUDA when it is genuinely available, otherwise
CPU. That keeps the project runnable on a machine without a GPU — the CI-style
test suite included — without anyone editing config, and it means a missing
driver degrades to "slower" rather than "crashes".
"""

from __future__ import annotations

import torch

from src.utils.logging_setup import get_logger

log = get_logger(__name__)

_WARNED: set[str] = set()


def resolve_device(spec: str | None = "auto") -> torch.device:
    """Turn a config string into a real device.

    ``auto`` picks CUDA when available. An explicit ``cuda`` that cannot be
    satisfied falls back to CPU with a warning rather than raising: a run that
    quietly takes longer is better than a run that dies at hour three because
    a driver update landed.
    """
    spec = (spec or "auto").strip().lower()

    if spec == "auto":
        spec = "cuda" if torch.cuda.is_available() else "cpu"

    if spec.startswith("cuda") and not torch.cuda.is_available():
        if "cuda-missing" not in _WARNED:
            log.warning(
                "device=%r requested but torch reports no CUDA (build %s). "
                "Falling back to CPU — this will be substantially slower. "
                "Install a CUDA wheel if a GPU is present.",
                spec, torch.version.cuda or "cpu-only",
            )
            _WARNED.add("cuda-missing")
        spec = "cpu"

    device = torch.device(spec)
    if device.type == "cuda" and "cuda-ok" not in _WARNED:
        free, total = torch.cuda.mem_get_info()
        log.info("using %s (%.1f of %.1f GB VRAM free, CUDA %s)",
                 torch.cuda.get_device_name(device), free / 2**30,
                 total / 2**30, torch.version.cuda)
        _WARNED.add("cuda-ok")
    return device
