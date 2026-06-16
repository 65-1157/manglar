"""
MANGLAR — src/pipelines/pipeline2_zone2/extract.py

Zone 2 (contrast zone) — Sentinel-2 pixel time series extraction.

Thin wrapper around src.utils.pixel_extraction, identical pattern to
Zone 1 and Zone 3. Zone 2 is the "partial pipeline" per the project
brief: S2 only, no SAR, no GFW layer - this module handles the S2
extraction; ANA discharge and NDWI-based water quality proxies are
computed separately downstream.

USAGE IN COLAB: see pipeline1_zone1/extract.py docstring - identical
pattern, with ZONE="zone2".
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from src.utils.config_loader import load_config  # noqa: E402
from src.utils.pixel_extraction import (  # noqa: E402
    Timer, extract_pixel_timeseries, merge_checkpoints,
)

CFG = load_config("base_config.yaml")

ZONE        = "zone2"
PREFIX      = "zone2_s2"
MASK_PREFIX = "zone2_gmw_mask"
INDICES     = ["NDVI", "EVI", "CIre", "NDWI"]
START_YEAR  = CFG["time"]["start_year"]
END_YEAR    = CFG["time"]["end_year"]


def run_extraction(raw_dir, external_dir=None, processed_dir=None):
    external_dir = external_dir or raw_dir
    processed_dir = processed_dir or (
        REPO_ROOT / "data" / "processed" / ZONE
    )

    timer = Timer()
    return extract_pixel_timeseries(
        zone=ZONE,
        raw_dir=raw_dir,
        external_dir=external_dir,
        processed_dir=processed_dir,
        prefix=PREFIX,
        mask_prefix=MASK_PREFIX,
        indices=INDICES,
        start_year=START_YEAR,
        end_year=END_YEAR,
        timer=timer,
    )


def run_merge(processed_dir):
    timer = Timer()
    return merge_checkpoints(
        zone=ZONE,
        processed_dir=processed_dir,
        start_year=START_YEAR,
        end_year=END_YEAR,
        timer=timer,
    )


if __name__ == "__main__":
    result = run_extraction(
        raw_dir=REPO_ROOT / "data" / "raw" / "gee_exports",
        external_dir=REPO_ROOT / "data" / "external",
        processed_dir=REPO_ROOT / "data" / "processed" / ZONE,
    )
    print(result)
