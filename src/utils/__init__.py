"""Utility helpers. Torch-dependent helpers are imported lazily so callers that
only need logging / seeding / config don't pull torch in."""
from __future__ import annotations

from .config import dump_config, load_config, merge_overrides
from .logger import get_logger, log_to_file
from .seed import set_seed


def __getattr__(name):
    if name in {"save_checkpoint", "load_checkpoint"}:
        from .checkpoint import load_checkpoint, save_checkpoint

        return {"save_checkpoint": save_checkpoint, "load_checkpoint": load_checkpoint}[name]
    raise AttributeError(name)
