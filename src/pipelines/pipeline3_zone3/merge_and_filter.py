"""
MANGLAR — src/pipelines/pipeline3_zone3/merge_and_filter.py

Pipeline 3 — Zone 3 (calibration zone) ingestion.

Reads raw GEE exports directly from a Google Drive folder (no local
download needed), mosaics tiles, applies the GMW mangrove mask, and
writes a per-pixel monthly time series table back to Drive.

  1. MOSAIC   - stitches multiple GEE export tiles per year into one
                continuous multi-band raster (in memory).

  2. MASK     - applies the GMW mangrove extent mask (found by pattern,
                handles single or multi-tile mask exports). If absent,
                proceeds without masking and warns.

  3. RESHAPE  - converts 8 yearly multi-band rasters (96 months x 4
                indices) into a single pixel x time table. NoData
                (-9999) -> NaN.

Output (written to processed_dir):
  zone3_pixel_timeseries.parquet
  zone3_pixel_timeseries.npz

---------------------------------------------------------------------
USAGE IN GOOGLE COLAB
---------------------------------------------------------------------
    from google.colab import drive
    drive.mount('/content/drive')

    import sys
    sys.path.insert(0, '/content/drive/MyDrive/manglar')  # repo clone

    from src.pipelines.pipeline3_zone3.merge_and_filter import run

    run(
        raw_dir       = '/content/drive/MyDrive/MANGLAR_GEE_EXPORTS',
        external_dir  = '/content/drive/MyDrive/MANGLAR_GEE_EXPORTS',
        processed_dir = '/content/drive/MyDrive/manglar_processed/zone3',
    )

---------------------------------------------------------------------
USAGE LOCALLY (defaults to repo's data/ structure)
---------------------------------------------------------------------
    python src/pipelines/pipeline3_zone3/merge_and_filter.py
---------------------------------------------------------------------
"""

import re
import sys
import glob
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.merge import merge as rio_merge
from rasterio.warp import transform as rio_transform

# ---- Repo paths (for local defaults and config) -------------------
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from src.utils.config_loader import load_config  # noqa: E402

CFG = load_config("base_config.yaml")

START_YEAR = CFG["time"]["start_year"]   # 2017
END_YEAR   = CFG["time"]["end_year"]     # 2024
INDICES    = ["NDVI", "EVI", "CIre", "NDWI"]
NODATA_VAL = -9999.0


# ============================================================
# 1. MOSAIC - stitch GEE export tiles for one year
# ============================================================
def find_tiles(raw_dir: Path, prefix: str) -> list[Path]:
    """Find all GEE export tiles matching a prefix.

    GEE splits large exports into tiles named like:
        zone3_s2_2018-0000000000-0000000000.tif
        zone3_s2_2018-0000000000-0000004864.tif
        ...
    A single (untiled) export is matched as 'prefix.tif'.
    """
    tiles = sorted(glob.glob(str(raw_dir / f"{prefix}-*.tif")))

    if not tiles:
        single = raw_dir / f"{prefix}.tif"
        if single.exists():
            tiles = [str(single)]

    return [Path(t) for t in tiles]


def mosaic_tiles(tiles: list[Path]):
    """Mosaic a list of tile paths into one array.

    Returns:
        data       - array of shape (bands, height, width)
        transform  - rasterio Affine transform of the mosaic
        crs        - CRS string
        band_names - list of band descriptions
    """
    if not tiles:
        raise FileNotFoundError("No tiles provided to mosaic_tiles().")

    srcs = [rasterio.open(t) for t in tiles]
    mosaic, out_transform = rio_merge(srcs)
    crs = srcs[0].crs.to_string()
    band_names = [srcs[0].descriptions[i] or f"band_{i+1}"
                   for i in range(srcs[0].count)]

    for s in srcs:
        s.close()

    return mosaic, out_transform, crs, band_names


# ============================================================
# 2. MASK - apply GMW mangrove extent mask
# ============================================================
def load_gmw_mask(external_dir: Path, transform, crs: str, shape):
    """Find, mosaic (if needed), and reproject the GMW mask to match
    the Sentinel-2 mosaic grid.

    Searches for files matching 'zone3_gmw_mask*.tif' (handles single
    or multi-tile exports, same naming convention as the S2 exports).

    Returns a boolean array of shape (height, width) where True = mangrove.
    Returns None if no matching file is found (mask skipped).
    """
    mask_tiles = find_tiles(external_dir, "zone3_gmw_mask")
    if not mask_tiles:
        mask_tiles = [Path(p) for p in
                       sorted(glob.glob(str(external_dir / "zone3_gmw_mask*.tif")))]

    if not mask_tiles:
        warnings.warn(
            f"No GMW mask files found in {external_dir} "
            f"(expected 'zone3_gmw_mask*.tif'). "
            f"Proceeding WITHOUT mangrove masking - all pixels retained."
        )
        return None

    print(f"  Found {len(mask_tiles)} GMW mask tile(s): "
          f"{[t.name for t in mask_tiles]}")

    gmw_mosaic, gmw_transform, gmw_crs, _ = mosaic_tiles(mask_tiles)

    from rasterio.warp import reproject, Resampling

    mask_data = np.zeros(shape, dtype=np.uint8)
    reproject(
        source=gmw_mosaic[0],
        destination=mask_data,
        src_transform=gmw_transform,
        src_crs=gmw_crs,
        dst_transform=transform,
        dst_crs=crs,
        resampling=Resampling.nearest,
    )

    mask = mask_data > 0
    n_mangrove = mask.sum()
    print(f"  GMW mask applied - {n_mangrove:,} mangrove pixels "
          f"of {mask.size:,} total ({100*n_mangrove/mask.size:.1f}%)")
    return mask


# ============================================================
# 3. RESHAPE - build per-pixel monthly time series table
# ============================================================
def parse_band_name(name: str):
    """Parse a band name like 'NDVI_2018_07' -> ('NDVI', 2018, 7)."""
    m = re.match(r"(NDVI|EVI|CIre|NDWI)_(\d{4})_(\d{2})", name)
    if not m:
        return None
    index, year, month = m.groups()
    return index, int(year), int(month)


def build_pixel_timeseries(raw_dir: Path, external_dir: Path) -> pd.DataFrame:
    """Mosaic all years, apply mask, and build the pixel x time table."""

    all_band_data = {}   # (index, year, month) -> 2D array
    ref_transform = None
    ref_crs = None
    ref_shape = None

    for year in range(START_YEAR, END_YEAR + 1):
        print(f"\nProcessing year {year}...")
        tiles = find_tiles(raw_dir, f"zone3_s2_{year}")
        if not tiles:
            raise FileNotFoundError(
                f"No GEE export tiles found for year {year} in {raw_dir}. "
                f"Expected files matching 'zone3_s2_{year}-*.tif' "
                f"or 'zone3_s2_{year}.tif'."
            )
        print(f"  Found {len(tiles)} tile(s): {[t.name for t in tiles]}")

        mosaic, transform, crs, band_names = mosaic_tiles(tiles)
        print(f"  Mosaic shape: {mosaic.shape}")

        if ref_transform is None:
            ref_transform = transform
            ref_crs = crs
            ref_shape = mosaic.shape[1:]
        elif mosaic.shape[1:] != ref_shape:
            raise ValueError(
                f"Year {year} mosaic shape {mosaic.shape[1:]} does not "
                f"match reference shape {ref_shape}. Years must share "
                f"the same export grid - re-export with identical bbox/scale."
            )

        for i, name in enumerate(band_names):
            parsed = parse_band_name(name)
            if parsed is None:
                warnings.warn(f"Could not parse band name '{name}' - skipping")
                continue
            index, b_year, b_month = parsed
            band = mosaic[i].astype(np.float32)
            band[band == NODATA_VAL] = np.nan
            all_band_data[(index, b_year, b_month)] = band

    # ---- Apply GMW mask ----
    mask = load_gmw_mask(external_dir, ref_transform, ref_crs, ref_shape)
    if mask is None:
        mask = np.ones(ref_shape, dtype=bool)

    rows, cols = np.where(mask)
    n_pixels = len(rows)
    print(f"\nTotal pixels retained: {n_pixels:,}")

    # ---- Compute lon/lat for each retained pixel ----
    xs, ys = rasterio.transform.xy(ref_transform, rows, cols)
    lons, lats = np.array(xs), np.array(ys)

    if ref_crs != "EPSG:4326":
        lons, lats = rio_transform(ref_crs, "EPSG:4326", lons, lats)
        lons, lats = np.array(lons), np.array(lats)

    # ---- Assemble table ----
    data = {"pixel_id": np.arange(n_pixels), "lon": lons, "lat": lats}

    for year in range(START_YEAR, END_YEAR + 1):
        for month in range(1, 13):
            for index in INDICES:
                key = (index, year, month)
                col_name = f"{index}_{year}_{month:02d}"
                if key in all_band_data:
                    data[col_name] = all_band_data[key][rows, cols]
                else:
                    warnings.warn(f"Missing band {col_name} - filled with NaN")
                    data[col_name] = np.full(n_pixels, np.nan, dtype=np.float32)

    df = pd.DataFrame(data)
    return df


# ============================================================
# Main entry point
# ============================================================
def run(raw_dir, external_dir=None, processed_dir=None):
    """Run the full Zone 3 merge-and-filter pipeline.

    Args:
        raw_dir:       Folder containing 'zone3_s2_YYYY-*.tif' tiles
                        (e.g. your Drive MANGLAR_GEE_EXPORTS folder).
        external_dir:  Folder containing 'zone3_gmw_mask*.tif'.
                        Defaults to raw_dir if not given.
        processed_dir: Output folder for the parquet/npz results.
                        Defaults to REPO_ROOT/data/processed/zone3.

    Returns:
        The resulting pixel x time series DataFrame.
    """
    raw_dir = Path(raw_dir)
    external_dir = Path(external_dir) if external_dir else raw_dir
    processed_dir = Path(processed_dir) if processed_dir else (
        REPO_ROOT / "data" / "processed" / "zone3"
    )
    processed_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("MANGLAR - Pipeline 3 (Zone 3) - Merge and Filter")
    print("=" * 60)
    print(f"Years: {START_YEAR}-{END_YEAR}")
    print(f"Raw export dir: {raw_dir}")
    print(f"GMW mask dir:   {external_dir}")
    print(f"Output dir:     {processed_dir}")

    df = build_pixel_timeseries(raw_dir, external_dir)

    parquet_path = processed_dir / "zone3_pixel_timeseries.parquet"
    npz_path = processed_dir / "zone3_pixel_timeseries.npz"

    df.to_parquet(parquet_path, index=False)
    print(f"\nSaved: {parquet_path} ({parquet_path.stat().st_size / 1e6:.1f} MB)")

    meta_cols = ["pixel_id", "lon", "lat"]
    ts_cols = [c for c in df.columns if c not in meta_cols]
    np.savez_compressed(
        npz_path,
        pixel_id=df["pixel_id"].values,
        lon=df["lon"].values,
        lat=df["lat"].values,
        timeseries=df[ts_cols].values.astype(np.float32),
        column_names=np.array(ts_cols),
    )
    print(f"Saved: {npz_path} ({npz_path.stat().st_size / 1e6:.1f} MB)")

    print("\nDone. Output columns:")
    print(f"  Metadata: {meta_cols}")
    print(f"  Time series columns: {len(ts_cols)} "
          f"({len(INDICES)} indices x {END_YEAR - START_YEAR + 1} years x 12 months)")
    print(f"  Total rows (mangrove pixels): {len(df):,}")

    return df


if __name__ == "__main__":
    run(
        raw_dir=REPO_ROOT / "data" / "raw" / "gee_exports",
        external_dir=REPO_ROOT / "data" / "external",
        processed_dir=REPO_ROOT / "data" / "processed" / "zone3",
    )
