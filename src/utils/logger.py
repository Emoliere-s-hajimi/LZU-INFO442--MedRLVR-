"""Lightweight logging — stdout + optional file sink, shared across stages."""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional


_FMT = "%(asctime)s | %(levelname).1s | %(name)s | %(message)s"
_DATE = "%H:%M:%S"


def get_logger(name: str = "tiantanmed", level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(level)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(_FMT, datefmt=_DATE))
    logger.addHandler(handler)
    logger.propagate = False
    return logger


def log_to_file(logger: logging.Logger, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(path)
    handler.setFormatter(logging.Formatter(_FMT, datefmt=_DATE))
    logger.addHandler(handler)
