"""Repository path resolution and config loading.

Small enough to be boring on purpose: every other module imports paths from
here rather than computing them relative to its own ``__file__``.
"""

from __future__ import annotations

import functools
import os
from pathlib import Path
from typing import Any

import yaml

# src/utils/paths.py -> src/utils -> src -> repo root
REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "configs"


def resolve(path_like: str | os.PathLike[str]) -> Path:
    """Resolve a possibly-relative config path against the repository root."""
    p = Path(path_like)
    return p if p.is_absolute() else REPO_ROOT / p


@functools.lru_cache(maxsize=None)
def load_config(name: str) -> dict[str, Any]:
    """Load ``configs/<name>.yaml``. Cached — configs are read-only at runtime."""
    stem = name[:-5] if name.endswith(".yaml") else name
    path = CONFIG_DIR / f"{stem}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"config not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def data_paths() -> dict[str, Path]:
    """The four data directories from ``configs/data.yaml``, created on demand."""
    cfg = load_config("data")["paths"]
    out = {key: resolve(value) for key, value in cfg.items()}
    for p in out.values():
        p.mkdir(parents=True, exist_ok=True)
    return out


def station_dir(station_id: str, kind: str = "processed") -> Path:
    """Per-station output directory. Stations are never written to one shared file.

    CLAUDE.md §2.2 / §6: the pipeline runs independently per station and the
    on-disk layout mirrors that, so a "just concatenate the folder" shortcut is
    never the path of least resistance.
    """
    d = data_paths()[kind] / station_id
    d.mkdir(parents=True, exist_ok=True)
    return d
