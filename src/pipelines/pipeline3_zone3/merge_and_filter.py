"""
MANGLAR — src/pipelines/pipeline3_zone3/merge_and_filter.py

Pipeline 3 - Zone 3 (calibration zone) ingestion.
Disk-backed version: each band's masked values are written to a small
.npy file on local disk as it's processed, then assembled into the
final table via a memory-mapped array. RAM never holds more than one
band's data plus the small per-pixel metadata - independent of how
many mangrove pixels exist.

Output (written to processed_dir):
  zone3_pixel_timeseries.parquet   (chunked write, memory-safe)

USAGE IN COLAB:
    from google.colab import drive
    drive.mount('/content/drive')

    import sys
    sys.path.insert(0, '/content/drive/MyDrive/manglar')

    from src.pipelines.pipeline3_zone3.merge_and_filter import run

    result = run(
        raw_dir       = '/content/drive/MyDrive/MANGLAR_GEE_EXPORTS',
        external_dir  = '/content/drive/MyDrive/MANGLAR_GEE_EXPORTS',
        processed_dir = '/content/drive/MyDrive/manglar_processed/zone3',
        scratch_dir   = '/content/manglar_scratch',
    )

USAGE LOCALLY:
    python src/pipelines/pipeline3_zone3/merge_and_filter.py
"""

import re
import sys
import glob
import shutil
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

START_YEAR = CFG["time"]["start_year"]
END_YEAR   = CFG["time"]["end_year"]
INDICES    = ["NDVI", "EVI", "CIre", "NDWI"]
NODATA_VAL = -9999.0


# ============================================================
# Tile discovery and mosaicking
# ============================================================
def find_tiles(raw_dir, prefix):
    raw_dir = Path(raw_dir)
    tiles = sorted(glob.glob(str(raw_dir / f"{prefix}-*.tif")))
    if not tiles:
        single = raw_dir / f"{prefix}.tif"
        if single.exists():
            tiles = [str(single)]
    return [Path(t) for t in tiles]


def mosaic_tiles(tiles, indexes=None):
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
    m = re.match(r"(NDVI|EVI|CIre|NDWI)_(\d{4})_(\d{2})", name)
    if not m:
        return None
    index, year, month = m.groups()
    return index, int(year), int(month)


# ============================================================
# Reference grid (geometry only, cheap)
# ============================================================
def get_reference_grid(raw_dir, start_year):
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
    shape = sample.shape[1:]
    del sample

    return out_transform, crs, shape, n_bands, band_names


# ============================================================
# Mangrove mask
# ============================================================
def build_mangrove_mask(external_dir, transform, crs, shape):
    external_dir = Path(external_dir)
    mask_tiles = find_tiles(external_dir, "zone3_gmw_mask")
    if not mask_tiles:
        mask_tiles = [Path(p) for p in
                       sorted(glob.glob(str(external_dir / "zone3_gmw_mask*.tif")))]

    if not mask_tiles:
        warnings.warn(
            f"No GMW mask files found in {external_dir} "
            f"(expected 'zone3_gmw_mask*.tif'). "
            f"Proceeding WITHOUT masking - ALL pixels retained. "
            f"This will likely exceed memory for large grids."
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

    if n_pixels == 0:
        raise ValueError(
            "Mangrove mask matched ZERO pixels. Check that the GMW mask "
            "export covers the same area/CRS as the Sentinel-2 exports."
        )

    n_cols = len(INDICES) * (END_YEAR - START_YEAR + 1) * 12
    est_gb = n_pixels * n_cols * 4 / 1e9
    print(f"  Estimated time-series table size: "
          f"{n_pixels:,} rows x {n_cols} cols "
          f"~= {est_gb:.2f} GB (float32)")
    print(f"  Parquet output is written in 500,000-row chunks - "
          f"memory-safe regardless of total size.")

    return mask, rows, cols


# ============================================================
# Main extraction - band-by-band, disk-backed
# ============================================================
def build_pixel_timeseries(raw_dir, external_dir, scratch_dir):
    raw_dir = Path(raw_dir)
    external_dir = Path(external_dir)
    scratch_dir = Path(scratch_dir)

    if scratch_dir.exists():
        shutil.rmtree(scratch_dir)
    scratch_dir.mkdir(parents=True, exist_ok=True)

    print("Determining reference grid from first year...")
    ref_transform, ref_crs, ref_shape, n_bands, _ = \
        get_reference_grid(raw_dir, START_YEAR)
    print(f"  Grid shape: {ref_shape}, bands per year: {n_bands}")

    print("\nBuilding mangrove mask...")
    mask, rows, cols = build_mangrove_mask(
        external_dir, ref_transform, ref_crs, ref_shape
    )
    n_pixels = len(rows)
    del mask

    print("\nComputing pixel coordinates...")
    xs, ys = rasterio.transform.xy(ref_transform, rows, cols)
    lons, lats = np.array(xs), np.array(ys)
    if ref_crs != "EPSG:4326":
        lons, lats = rio_transform(ref_crs, "EPSG:4326", lons, lats)
        lons, lats = np.array(lons), np.array(lats)

    np.save(scratch_dir / "lon.npy", lons)
    np.save(scratch_dir / "lat.npy", lats)

    column_order = []

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
                f"expected {n_bands} (from {START_YEAR})."
            )

        for band_idx in range(1, n_bands + 1):
            name = band_names[band_idx - 1]
            parsed = parse_band_name(name)
            if parsed is None:
                warnings.warn(f"Could not parse band '{name}' in year "
                               f"{year} - skipping")
                continue
            index, b_year, b_month = parsed

            band_mosaic, _, _ = mosaic_tiles(tiles, indexes=[band_idx])
            band_arr = band_mosaic[0].astype(np.float32)
            del band_mosaic

            if band_arr.shape != ref_shape:
                raise ValueError(
                    f"Year {year} band {band_idx} ('{name}') shape "
                    f"{band_arr.shape} != reference shape {ref_shape}."
                )

            band_arr[band_arr == NODATA_VAL] = np.nan
            values = band_arr[rows, cols]
            del band_arr

            col_name = f"{index}_{b_year}_{b_month:02d}"
            npy_path = scratch_dir / f"{col_name}.npy"
            np.save(npy_path, values)
            column_order.append((col_name, npy_path))
            del values

            if band_idx % 12 == 0:
                print(f"    ...{band_idx}/{n_bands} bands done "
                      f"(through {col_name})")

    n_cols = len(column_order)
    print(f"\nAssembling final table: {n_pixels:,} rows x {n_cols} cols")
    memmap_path = scratch_dir / "timeseries_memmap.npy"
    ts_memmap = np.lib.format.open_memmap(
        memmap_path, mode="w+", dtype=np.float32, shape=(n_pixels, n_cols)
    )
    ts_col_names = []
    for i, (col_name, npy_path) in enumerate(column_order):
        ts_memmap[:, i] = np.load(npy_path)
        ts_col_names.append(col_name)
    ts_memmap.flush()

    pixel_id = np.arange(n_pixels)
    lons = np.load(scratch_dir / "lon.npy")
    lats = np.load(scratch_dir / "lat.npy")

    return pixel_id, lons, lats, ts_memmap, ts_col_names, scratch_dir


# ============================================================
# Entry point
# ============================================================
def run(raw_dir, external_dir=None, processed_dir=None, scratch_dir=None):
    raw_dir = Path(raw_dir)
    external_dir = Path(external_dir) if external_dir else raw_dir
    processed_dir = Path(processed_dir) if processed_dir else (
        REPO_ROOT / "data" / "processed" / "zone3"
    )
    scratch_dir = Path(scratch_dir) if scratch_dir else (
        REPO_ROOT / ".scratch" / "zone3"
    )
    processed_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("MANGLAR - Pipeline 3 (Zone 3) - Merge and Filter")
    print("(disk-backed, band-by-band, parquet-only output)")
    print("=" * 60)
    print(f"Years: {START_YEAR}-{END_YEAR}")
    print(f"Raw export dir: {raw_dir}")
    print(f"GMW mask dir:   {external_dir}")
    print(f"Output dir:     {processed_dir}")
    print(f"Scratch dir:    {scratch_dir}  (local disk)")

    pixel_id, lons, lats, ts_memmap, ts_col_names, scratch_dir = \
        build_pixel_timeseries(raw_dir, external_dir, scratch_dir)

    n_pixels, n_cols = ts_memmap.shape

    parquet_path = processed_dir / "zone3_pixel_timeseries.parquet"
    print(f"\nWriting {parquet_path} (chunked)...")

    import pyarrow as pa
    import pyarrow.parquet as pq

    chunk_size = 500_000
    writer = None
    for start in range(0, n_pixels, chunk_size):
        end = min(start + chunk_size, n_pixels)
        chunk_dict = {
            "pixel_id": pixel_id[start:end],
            "lon": lons[start:end],
            "lat": lats[start:end],
        }
        for i, col_name in enumerate(ts_col_names):
            chunk_dict[col_name] = ts_memmap[start:end, i]

        table = pa.Table.from_pydict(chunk_dict)
        if writer is None:
            writer = pq.ParquetWriter(parquet_path, table.schema)
        writer.write_table(table)
        print(f"  Wrote rows {start:,}-{end:,}")

    if writer is not None:
        writer.close()
    print(f"Saved: {parquet_path} ({parquet_path.stat().st_size / 1e6:.1f} MB)")

    print(f"\nCleaning up scratch dir {scratch_dir} ...")
    shutil.rmtree(scratch_dir, ignore_errors=True)

    print("\nDone. Output summary:")
    print(f"  Metadata columns: ['pixel_id', 'lon', 'lat']")
    print(f"  Time series columns: {n_cols} "
          f"({len(INDICES)} indices x {END_YEAR - START_YEAR + 1} years x 12 months)")
    print(f"  Total rows (mangrove pixels): {n_pixels:,}")

    return {
        "n_pixels": n_pixels,
        "n_cols": n_cols,
        "parquet_path": str(parquet_path),
        "column_names": ts_col_names,
    }


if __name__ == "__main__":
    run(
        raw_dir=REPO_ROOT / "data" / "raw" / "gee_exports",
        external_dir=REPO_ROOT / "data" / "external",
        processed_dir=REPO_ROOT / "data" / "processed" / "zone3",
        scratch_dir=REPO_ROOT / ".scratch" / "zone3",
    )
