"""Shared helpers and constants for baseline training and evaluation."""

from __future__ import annotations

import json
from pathlib import Path


DEFAULT_BUDGETS = (0.0, 5.0, 10.0, 20.0, 35.0)
DEFAULT_PENALTY_COEFF_GRID = (0.0, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0)
DEFAULT_PPO_KWARGS = {
    "policy": "MlpPolicy",
    "learning_rate": 3e-4,
    "n_steps": 2048,
    "batch_size": 64,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "verbose": 0,
}


def ensure_directory(path):
    """Create a directory path if it does not already exist."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def float_token(value):
    """Convert a numeric value into a path-friendly token."""
    text = "{:g}".format(float(value))
    return text.replace("-", "neg").replace(".", "p")


def write_json(path, payload):
    """Write JSON with stable formatting."""
    path = Path(path)
    ensure_directory(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def read_json(path):
    """Read JSON from disk."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)
