"""
MANGLAR — src/pipelines/pipeline3_zone3/merge_and_filter.py

Pipeline 3 - Zone 3 (calibration zone) ingestion.
Memory-efficient version: processes one band at a time to avoid
loading full 48-band x full-grid arrays into RAM.

Output (written to processed_dir):
  zone3_pixel_timeseries.parquet
  zone3_pixel_timeseries.npz

USAGE IN COLAB:
    from google.colab import drive
    drive.mount('/content/drive')

    import sys
    sys.path.insert(0, '/content/drive/MyDrive/manglar')

    from src.pipelines.pipeline3_zone3.merge_and_filter import run

    df = run(
        raw_dir       = '/content/drive/MyDrive/MANGLAR_GEE_EXPORTS',
        external_dir  = '/content/drive/MyDrive/MANGLAR_GEE_EXPORTS',
        processed_dir = '/content/drive/MyDrive/manglar_processed/zone3',
    )

USAGE LOCALLY:
    python src/pipelines/pipeline3_zone3/merge_and_filter.py
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
from rasterio.warp import transform as rio_transform, reproject, Resampling

# ---- Repo paths -----------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from src.utils.config_loader import load_config  # noqa: E402

CFG = load_config("base_config.yaml")

START_YEAR = CFG["time"]["start_year"]   # 2017
END_YEAR   = CFG["time"]["end_year"]     # 2024
INDICES    = ["NDVI", "EVI", "CIre", "NDWI"]
NODATA_VAL = -9999.0


# ============================================================
# Tile discovery and mosaicking
# ============================================================
def find_tiles(raw_dir, prefix):
    """Find all GEE export tiles matching a prefix.

    GEE splits large exports into tiles named like:
        zone3_s2_2018-0000000000-0000000000.tif
        zone3_s2_2018-0000000000-0000004864.tif
    A single (untiled) export is matched as 'prefix.tif'.
    """
    raw_dir = Path(raw_dir)
    tiles = sorted(glob.glob(str(raw_dir / f"{prefix}-*.tif")))
    if not tiles:
        single = raw_dir / f"{prefix}.tif"
        if single.exists():
            tiles = [str(single)]
    return [Path(t) for t in tiles]


def mosaic_tiles(tiles, indexes=None):
    """Mosaic a list of tile paths into one array.

    Args:
        tiles:   list of Path objects
        indexes: optional list of band indexes (1-based) to mosaic.
                 If None, mosaics ALL bands (memory-heavy — avoid for
                 multi-band yearly stacks).

    Returns:
        data       - array of shape (bands, height, width)
        transform  - rasterio Affine transform of the mosaic
        crs        - CRS string
    """
    if not tiles:
        raise FileNotFoundError("No tiles provided to mosaic_tiles().")

    srcs = [rasterio.open(t) for t in tiles]
    if indexes is not None:
        mosaic, out_transform = rio_merge(srcs, indexes=indexes)
    else:
        mosaic, out_transform = rio_merge(srcs)
    crs = srcs[0].crs.to_string()

    for s in srcs:
        s.close()

    return mosaic, out_transform, crs


# ============================================================
# Band name parsing
# ============================================================
def parse_band_name(name):
    """Parse a band name like 'NDVI_2018_07' -> ('NDVI', 2018, 7)."""
    m = re.match(r"(NDVI|EVI|CIre|NDWI)_(\d{4})_(\d{2})", name)
    if not m:
        return None
    index, year, month = m.groups()
    return index, int(year), int(month)


# ============================================================
# Reference grid (geometry only, cheap)
# ============================================================
def get_reference_grid(raw_dir, start_year):
    """Determine the mosaic grid (transform, crs, shape, band layout)
    from the first year's tiles, mosaicking only band 1 to avoid
    loading the full 48-band stack into memory.
    """
    tiles = find_tiles(raw_dir, f"zone3_s2_{start_year}")
    if not tiles:
        raise FileNotFoundError(
            f"No tiles found for reference year {start_year} in {raw_dir}. "
            f"Expected files matching 'zone3_s2_{start_year}-*.tif'."
        )

    srcs = [rasterio.open(t) for t in tiles]
    n_bands = srcs[0].count
    band_names = [srcs[0].descriptions[i] or f"band_{i+1}"
                   for i in range(n_bands)]
    for s in srcs:
        s.close()

    sample, out_transform, crs = mosaic_tiles(tiles, indexes=[1])
    shape = sample.shape[1:]  # (height, width)
    del sample

    return out_transform, crs, shape, n_bands, band_names


# ============================================================
# Mangrove mask
# ============================================================
def build_mangrove_mask(external_dir, transform, crs, shape):
    """Build the boolean mangrove mask on the reference grid.

    Returns:
        mask: bool array (height, width)
        rows, cols: flat pixel indices where mask is True
    """
    external_dir = Path(external_dir)
    mask_tiles = find_tiles(external_dir, "zone3_gmw_mask")
    if not mask_tiles:
        mask_tiles = [Path(p) for p in
                       sorted(glob.glob(str(external_dir / "zone3_gmw_mask*.tif")))]

    if not mask_tiles:
        warnings.warn(
            f"No GMW mask files found in {external_dir} "
            f"(expected 'zone3_gmw_mask*.tif'). "
            f"Proceeding WITHOUT masking — ALL pixels retained. "
            f"This may exceed memory for large grids."
        )
        mask = np.ones(shape, dtype=bool)
    else:
        print(f"  Found {len(mask_tiles)} GMW mask tile(s): "
              f"{[t.name for t in mask_tiles]}")
        gmw_mosaic, gmw_transform, gmw_crs = mosaic_tiles(mask_tiles)

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
        del gmw_mosaic, mask_data

    rows, cols = np.where(mask)
    n_pixels = len(rows)
    print(f"  Mangrove pixels: {n_pixels:,} of {mask.size:,} "
          f"({100 * n_pixels / mask.size:.2f}%)")

    if n_pixels > 5_000_000:
        warnings.warn(
            f"{n_pixels:,} pixels retained — output table will have "
            f"{n_pixels:,} rows x ~387 columns. Check available memory "
            f"for the final DataFrame/parquet write."
        )
    if n_pixels == 0:
        raise ValueError(
            "Mangrove mask matched ZERO pixels. Check that the GMW mask "
            "export covers the same area/CRS as the Sentinel-2 exports."
        )

    return mask, rows, cols


# ============================================================
# Main extraction — band-by-band, memory efficient
# ============================================================
def build_pixel_timeseries(raw_dir, external_dir):
    """Build the pixel x time series table, processing one band
    at a time to keep peak memory low.
    """
    raw_dir = Path(raw_dir)
    external_dir = Path(external_dir)

    print("Determining reference grid from first year...")
    ref_transform, ref_crs, ref_shape, n_bands, _ = \
        get_reference_grid(raw_dir, START_YEAR)
    print(f"  Grid shape: {ref_shape}, bands per year: {n_bands}")

    print("\nBuilding mangrove mask...")
    mask, rows, cols = build_mangrove_mask(
        external_dir, ref_transform, ref_crs, ref_shape
    )
    n_pixels = len(rows)
    del mask  # only rows/cols needed from here on

    print("\nComputing pixel coordinates...")
    xs, ys = rasterio.transform.xy(ref_transform, rows, cols)
    lons, lats = np.array(xs), np.array(ys)
    if ref_crs != "EPSG:4326":
        lons, lats = rio_transform(ref_crs, "EPSG:4326", lons, lats)
        lons, lats = np.array(lons), np.array(lats)

    data = {"pixel_id": np.arange(n_pixels), "lon": lons, "lat": lats}

    for year in range(START_YEAR, END_YEAR + 1):
        print(f"\nProcessing year {year}...")
        tiles = find_tiles(raw_dir, f"zone3_s2_{year}")
        if not tiles:
            raise FileNotFoundError(
                f"No tiles found for year {year} in {raw_dir}. "
                f"Expected files matching 'zone3_s2_{year}-*.tif'."
            )

        srcs = [rasterio.open(t) for t in tiles]
        band_names = [srcs[0].descriptions[i] or f"band_{i+1}"
                       for i in range(srcs[0].count)]
        year_n_bands = srcs[0].count
        for s in srcs:
            s.close()

        if year_n_bands != n_bands:
            raise ValueError(
                f"Year {year} has {year_n_bands} bands, "
                f"expected {n_bands} (from {START_YEAR}). "
                f"Years must share the same export structure."
            )

        for band_idx in range(1, n_bands + 1):
            name = band_names[band_idx - 1]
            parsed = parse_band_name(name)
            if parsed is None:
                warnings.warn(f"Could not parse band '{name}' in year "
                               f"{year} — skipping")
                continue
            index, b_year, b_month = parsed

            band_mosaic, _, _ = mosaic_tiles(tiles, indexes=[band_idx])
            band_arr = band_mosaic[0].astype(np.float32)
            del band_mosaic

            if band_arr.shape != ref_shape:
                raise ValueError(
                    f"Year {year} band {band_idx} ('{name}') shape "
                    f"{band_arr.shape} != reference shape {ref_shape}. "
                    f"Re-export with identical bbox/scale across years."
                )

            band_arr[band_arr == NODATA_VAL] = np.nan

            col_name = f"{index}_{b_year}_{b_month:02d}"
            data[col_name] = band_arr[rows, cols]
            del band_arr

            if band_idx % 12 == 0:
                print(f"    ...{band_idx}/{n_bands} bands done "
                      f"(through {index}_{b_year}_{b_month:02d})")

    df = pd.DataFrame(data)
    return df


# ============================================================
# Entry point
# ============================================================
def run(raw_dir, external_dir=None, processed_dir=None):
    """Run the full Zone 3 merge-and-filter pipeline.

    Args:
        raw_dir:       Folder containing 'zone3_s2_YYYY-*.tif' tiles.
        external_dir:  Folder containing 'zone3_gmw_mask*.tif'.
                        Defaults to raw_dir.
        processed_dir: Output folder for parquet/npz.
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
    print("(memory-efficient, band-by-band)")
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

    print("\nDone. Output summary:")
    print(f"  Metadata columns: {meta_cols}")
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
