"""Loads DocSense settings from config/*.yaml, per CLAUDE.md: all settings live
in config files so ablation variants are config changes, not code changes."""

from pathlib import Path
from typing import Any

import yaml

CONFIG_DIR = Path(__file__).resolve().parents[2] / "config"


def load_config(name: str) -> dict[str, Any]:
    """Load config/<name>.yaml and return it as a dict.

    Args:
        name: config file stem, e.g. "baseline" for config/baseline.yaml.

    Raises:
        FileNotFoundError: if config/<name>.yaml does not exist.
    """
    path = CONFIG_DIR / f"{name}.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"No such config: {path}")
    with path.open() as f:
        return yaml.safe_load(f)
