"""General utilities: logging, reproducibility, IO."""
from __future__ import annotations

import logging
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import yaml


def setup_logger(log_path: str | Path | None = None,
                 level: int = logging.INFO) -> logging.Logger:
    """Configure a root logger that writes to console and (optionally) a file."""
    logger = logging.getLogger("pm25_hybrid")
    logger.setLevel(level)
    logger.handlers.clear()

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    if log_path is not None:
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_path)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger


def set_global_seed(seed: int = 42) -> None:
    """Set seeds across libraries for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load a YAML configuration file.

    Relative paths are resolved against the project root (the directory that
    contains main.py), not the shell's current working directory.  After
    loading, every value inside the top-level ``paths:`` block that is a
    relative path is made absolute using the same project root, so callers
    can open files directly without further resolution.
    """
    p = Path(config_path)
    # Project root = two levels up from this file  (src/utils.py → src/ → project/)
    project_root = Path(__file__).resolve().parent.parent
    if not p.is_absolute():
        p = project_root / p
    with open(p, "r") as f:
        cfg = yaml.safe_load(f)

    # Resolve every string value in paths: against the project root.
    for key, val in cfg.get("paths", {}).items():
        if isinstance(val, str) and not Path(val).is_absolute():
            cfg["paths"][key] = str((project_root / val).resolve())

    return cfg


def ensure_dir(path: str | Path) -> Path:
    """Create directory if missing and return as Path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p
