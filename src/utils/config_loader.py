"""
MANGLAR — src/utils/config_loader.py
Load and merge YAML configuration files.
"""

import yaml
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIGS_DIR = REPO_ROOT / "configs"


def load_config(filename: str = "base_config.yaml") -> dict:
    """Load a YAML config file from the configs/ directory."""
    path = CONFIGS_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path) as f:
        return yaml.safe_load(f)


def load_all_configs() -> dict:
    """Load base_config and gee_config, merge into a single dict."""
    base = load_config("base_config.yaml")
    gee = load_config("gee_config.yaml")
    return {**base, "gee": gee}


if __name__ == "__main__":
    cfg = load_all_configs()
    print(f"Project: {cfg['project']['name']} v{cfg['project']['version']}")
    print(f"Zones: {list(cfg['zones'].keys())}")
    print(f"Time range: {cfg['time']['start_year']}–{cfg['time']['end_year']}")
