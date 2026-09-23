"""
src/config.py

Single point of access to config/config.yaml so that constants like
INITIAL_CAPITAL and TRANSACTION_COST are never hard-coded throughout
the project.
"""

import os
import yaml

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CONFIG_PATH = os.path.join(PROJECT_ROOT, "config", "config.yaml")

_config_cache = None


def load_config(path: str = CONFIG_PATH) -> dict:
    """Load and cache config.yaml. Re-reads if a different path is given."""
    global _config_cache
    if path == CONFIG_PATH and _config_cache is not None:
        return _config_cache

    if not os.path.isfile(path):
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if path == CONFIG_PATH:
        _config_cache = cfg
    return cfg
