"""Utility helpers for reproducibility, logging, and filesystem operations."""

from __future__ import annotations

import logging
import os
import random
from pathlib import Path

import numpy as np


def setup_logger(name: str = "tsa_capstone", level: int = logging.INFO) -> logging.Logger:
    """Create and configure a project logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s | %(name)s | %(levelname)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


def set_global_seed(seed: int) -> None:
    """Set random seeds for reproducible experiments."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def ensure_dir(path: str | Path) -> Path:
    """Ensure that a directory exists."""
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    return target
