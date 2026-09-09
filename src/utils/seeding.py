"""Deterministic seeding for reproducible baseline numbers."""

from __future__ import annotations

import os
import random

import numpy as np


def seed_everything(seed: int = 1337) -> int:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
    except ImportError:  # torch is optional for the data-only entry points
        return seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(False)  # cuDNN GRU has no deterministic path
    return seed
