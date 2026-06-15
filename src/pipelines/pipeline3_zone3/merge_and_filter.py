"""
MANGLAR — src/pipelines/pipeline3_zone3/merge_and_filter.py

Pipeline 3 - Zone 3 (calibration zone) ingestion.
Optimized version:
  - Each year's GEE tiles are copied from Drive to LOCAL disk ONCE,
    then all 48 bands are mosaicked from local copies (avoids
    hundreds of slow Drive-mounted file opens).
  - Each year's result is checkpointed as a small parquet on Drive
    immediately after processing. On restart, completed years are
    skipped automatically.
  - Final step merges all yearly parquets (on pixel_id) into the
    single zone3_pixel_timeseries.parquet.

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

If the runtime crashes/disconnects, just re-run the same cell after
reconnecting — completed years are skipped automatically.
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
def find_tiles(folder, prefix):
    folder = Path(folder)
    tiles = sorted(glob.glob(str(folder / f"{prefix}-*.tif")))
    if not tiles:
        single = folder / f"{prefix}.tif"
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
            f"No tiles found for reference year {start_year} in {raw_dir}."
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
            f"No GMW mask files found in {external_dir}. "
            f"Proceeding WITHOUT masking - ALL pixels retained."
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
        raise ValueError("Mangrove mask matched ZERO pixels.")

    return mask, rows, cols


# ============================================================
# Per-year processing — local tile copy, then band-by-band
# ============================================================
def process_year(year, raw_dir, local_dir, ref_shape, n_bands,
                  rows, cols, checkpoint_dir):
    """Copy this year's tiles to local disk, mosaic each band,
    extract masked values, save year checkpoint parquet to Drive.

    Skips entirely if the checkpoint already exists (resume support).
    """
    checkpoint_path = checkpoint_dir / f"zone3_{year}.parquet"
    if checkpoint_path.exists():
        print(f"  Year {year}: checkpoint already exists, skipping.")
        return

    print(f"\nProcessing year {year}...")

    # ---- Copy this year's tiles to local disk ----
    drive_tiles = find_tiles(raw_dir, f"zone3_s2_{year}")
    if not drive_tiles:
        raise FileNotFoundError(
            f"No tiles found for year {year} in {raw_dir}."
        )

    local_year_dir = local_dir / f"year_{year}"
    local_year_dir.mkdir(parents=True, exist_ok=True)

    print(f"  Copying {len(drive_tiles)} tile(s) to local disk...")
    local_tiles = []
    for t in drive_tiles:
        dst = local_year_dir / t.name
        shutil.copy2(t, dst)
        local_tiles.append(dst)
    print(f"  Copy done.")

    # ---- Get band names from local copy ----
    srcs = [rasterio.open(t) for t in local_tiles]
    band_names = [srcs[0].descriptions[i] or f"band_{i+1}"
                   for i in range(srcs[0].count)]
    year_n_bands = srcs[0].count
    for s in srcs:
        s.close()

    if year_n_bands != n_bands:
        raise ValueError(
            f"Year {year} has {year_n_bands} bands, expected {n_bands}."
        )

    # ---- Process each band from LOCAL tiles ----
    year_data = {}
    for band_idx in range(1, n_bands + 1):
        name = band_names[band_idx - 1]
        parsed = parse_band_name(name)
        if parsed is None:
            warnings.warn(f"Could not parse band '{name}' - skipping")
            continue
        index, b_year, b_month = parsed

        band_mosaic, _, _ = mosaic_tiles(local_tiles, indexes=[band_idx])
        band_arr = band_mosaic[0].astype(np.float32)
        del band_mosaic

        if band_arr.shape != ref_shape:
            raise ValueError(
                f"Year {year} band {band_idx} shape {band_arr.shape} "
                f"!= reference shape {ref_shape}."
            )

        band_arr[band_arr == NODATA_VAL] = np.nan
        col_name = f"{index}_{b_year}_{b_month:02d}"
        year_data[col_name] = band_arr[rows, cols]
        del band_arr

        if band_idx % 12 == 0:
            print(f"    ...{band_idx}/{n_bands} bands done "
                  f"(through {col_name})")

    # ---- Save year checkpoint to Drive ----
    df_year = pd.DataFrame(year_data)
    df_year.insert(0, "pixel_id", np.arange(len(df_year)))
    df_year.to_parquet(checkpoint_path, index=False)
    print(f"  Checkpoint saved: {checkpoint_path.name} "
          f"({checkpoint_path.stat().st_size / 1e6:.1f} MB)")

    # ---- Clean up local copies for this year ----
    shutil.rmtree(local_year_dir, ignore_errors=True)


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
    scratch_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_dir = processed_dir / "year_checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("MANGLAR - Pipeline 3 (Zone 3) - Merge and Filter")
    print("(local tile cache + per-year checkpoints, resumable)")
    print("=" * 60)
    print(f"Years: {START_YEAR}-{END_YEAR}")
    print(f"Raw export dir: {raw_dir}")
    print(f"GMW mask dir:   {external_dir}")
    print(f"Output dir:     {processed_dir}")
    print(f"Checkpoints:    {checkpoint_dir}")
    print(f"Local scratch:  {scratch_dir}")

    print("\nDetermining reference grid from first year...")
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

    # Save coords checkpoint (small, needed for final merge)
    coords_path = checkpoint_dir / "zone3_coords.parquet"
    pd.DataFrame({
        "pixel_id": np.arange(n_pixels), "lon": lons, "lat": lats
    }).to_parquet(coords_path, index=False)

    # ---- Process each year (skips if checkpoint exists) ----
    for year in range(START_YEAR, END_YEAR + 1):
        process_year(year, raw_dir, scratch_dir, ref_shape, n_bands,
                      rows, cols, checkpoint_dir)

    # ---- Final merge: coords + all yearly checkpoints ----
    print("\nMerging all year checkpoints into final table...")
    final_path = processed_dir / "zone3_pixel_timeseries.parquet"

    import pyarrow.parquet as pq
    import pyarrow as pa

    coords_table = pq.read_table(coords_path)
    chunk_size = 500_000
    n_rows = coords_table.num_rows

    year_files = [checkpoint_dir / f"zone3_{y}.parquet"
                   for y in range(START_YEAR, END_YEAR + 1)]
    for yf in year_files:
        if not yf.exists():
            raise FileNotFoundError(f"Missing checkpoint: {yf}")

    writer = None
    for start in range(0, n_rows, chunk_size):
        end = min(start + chunk_size, n_rows)
        chunk = coords_table.slice(start, end - start).to_pandas()

        for yf in year_files:
            yt = pq.read_table(yf, columns=None).slice(start, end - start)
            ydf = yt.to_pandas().drop(columns=["pixel_id"])
            chunk = pd.concat([chunk.reset_index(drop=True),
                                ydf.reset_index(drop=True)], axis=1)

        table = pa.Table.from_pandas(chunk, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(final_path, table.schema)
        writer.write_table(table)
        print(f"  Merged rows {start:,}-{end:,}")

    if writer is not None:
        writer.close()

    print(f"\nSaved: {final_path} ({final_path.stat().st_size / 1e6:.1f} MB)")

    n_cols = sum(48 for _ in year_files)
    print("\nDone. Output summary:")
    print(f"  Total rows (mangrove pixels): {n_rows:,}")
    print(f"  Time series columns: {n_cols}")
    print(f"  Final file: {final_path}")

    return {"n_pixels": n_rows, "n_cols": n_cols,
            "parquet_path": str(final_path)}


if __name__ == "__main__":
    run(
        raw_dir=REPO_ROOT / "data" / "raw" / "gee_exports",
        external_dir=REPO_ROOT / "data" / "external",
        processed_dir=REPO_ROOT / "data" / "processed" / "zone3",
        scratch_dir=REPO_ROOT / ".scratch" / "zone3",
    )
