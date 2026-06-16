"""MANGLAR — src/pipelines/pipeline1_zone1/extract.py"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from src.utils.config_loader import load_config
from src.utils.pixel_extraction import Timer, extract_pixel_timeseries, merge_checkpoints

CFG = load_config("base_config.yaml")
ZONE, PREFIX, MASK_PREFIX = "zone1", "zone1_s2", "zone1_gmw_mask"
INDICES = ["NDVI", "EVI", "CIre", "NDWI"]
START_YEAR, END_YEAR = CFG["time"]["start_year"], CFG["time"]["end_year"]


def run_extraction(raw_dir, external_dir=None, processed_dir=None):
    external_dir = external_dir or raw_dir
    processed_dir = processed_dir or (REPO_ROOT / "data" / "processed" / ZONE)
    return extract_pixel_timeseries(
        zone=ZONE, raw_dir=raw_dir, external_dir=external_dir,
        processed_dir=processed_dir, prefix=PREFIX, mask_prefix=MASK_PREFIX,
        indices=INDICES, start_year=START_YEAR, end_year=END_YEAR, timer=Timer(),
    )


def run_merge(processed_dir):
    return merge_checkpoints(
        zone=ZONE, processed_dir=processed_dir,
        start_year=START_YEAR, end_year=END_YEAR, timer=Timer(),
    )
