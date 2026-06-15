"""
MANGLAR — src/pipelines/pipeline3_zone3/merge_and_filter.py

Pipeline 3 - Zone 3 (calibration zone) ingestion. (v5)

ARCHITECTURE:
  v3 mosaicked all 48 bands at once (~17 GB/year) -> OOM.
  v4 read each tile once, all bands, full tile in RAM (~2.85 GB/tile)
     PLUS a local-disk copy step whose Drive-FUSE page cache pushed
     total RAM past the limit before the read even started -> OOM.

  v5 removes the local-copy step entirely (it solved a latency problem
  that benchmarking showed was NOT the bottleneck) and reads each tile
  directly from Drive in small row-blocks (windowed reads). Memory per
  block = block_rows x tile_width x n_bands x 4 bytes, independent of
  total tile/grid size - bounded and predictable regardless of zone
  size, which matters for Zone 1 later too.

  Retains: per-year checkpoints (resumable on crash) and time.time()
  instrumentation at every milestone.

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
    )

If the runtime crashes/disconnects, re-run the same cell - completed
years are skipped automatically.
"""

import re
import sys
import glob
import gc
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window
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

# Row-block size for windowed reads. With n_bands=48 and a typical
# tile width of ~5000 px, 200 rows x 5000 x 48 x 4 bytes ~= 192 MB
# per block - safely bounded regardless of total grid size.
BLOCK_ROWS = 200


# ============================================================
# Timing helper
# ============================================================
class Timer:
    """Tracks elapsed time since start and since the last milestone."""

    def __init__(self):
        self.t0 = time.time()
        self.last = self.t0

    def mark(self, label):
        now = time.time()
        since_last = now - self.last
        since_start = now - self.t0
        print(f"  [{label}] "
              f"+{since_last:6.1f}s since last  |  "
              f"{since_start/60:6.1f} min total elapsed")
        self.last = now


# ============================================================
# Tile discovery
# ============================================================
def find_tiles(folder, prefix):
    folder = Path(folder)
    tiles = sorted(glob.glob(str(folder / f"{prefix}-*.tif")))
    if not tiles:
        single = folder / f"{prefix}.tif"
        if single.exists():
            tiles = [str(single)]
    return [Path(t) for t in tiles]


def mosaic_band1(tiles):
    """Cheap one-time merge of band 1 only, used to determine the
    reference grid geometry (transform, crs, shape).
    """
    if not tiles:
        raise FileNotFoundError("No tiles provided to mosaic_band1().")
    srcs = [rasterio.open(t) for t in tiles]
    mosaic, out_transform = rio_merge(srcs, indexes=[1])
    crs = srcs[0].crs.to_string()
    for s in srcs:
        s.close()
    return mosaic, out_transform, crs


def parse_band_name(name):
    """Parse a band name containing 'NDVI_2018_07' etc. Uses search
    (not match) to tolerate any GEE-added prefixes like '0_NDVI_2018_07'.
    """
    m = re.search(r"(NDVI|EVI|CIre|NDWI)_(\d{4})_(\d{2})", name)
    if not m:
        return None
    index, year, month = m.groups()
    return index, int(year), int(month)


# ============================================================
# Reference grid (geometry only, one cheap merge of band 1)
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

    sample, out_transform, crs = mosaic_band1(tiles)
    shape = sample.shape[1:]  # (height, width)
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
        gmw_mosaic, gmw_transform, gmw_crs = mosaic_band1(mask_tiles)
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

    return rows, cols


# ============================================================
# Tile pixel-offset computation
# ============================================================
def tile_offset(tile_transform, ref_transform, tol=1e-6):
    """Compute this tile's (row_offset, col_offset) within the global
    reference grid, assuming identical pixel size/CRS (pure
    translation, no resampling needed).
    """
    px_w = ref_transform.a
    px_h = ref_transform.e  # negative

    d_col = (tile_transform.c - ref_transform.c) / px_w
    d_row = (tile_transform.f - ref_transform.f) / px_h

    col_off = round(d_col)
    row_off = round(d_row)

    if abs(d_col - col_off) > tol or abs(d_row - row_off) > tol:
        raise ValueError(
            f"Tile offset is not an integer number of pixels "
            f"(d_col={d_col}, d_row={d_row}). Tile grid does not align "
            f"with the reference grid - resampling would be required."
        )
    return row_off, col_off


# ============================================================
# Per-year processing — windowed reads directly from Drive
# ============================================================
def process_year(year, raw_dir, ref_transform, ref_shape,
                  n_bands, ref_band_names, rows, cols,
                  checkpoint_dir, timer):
    checkpoint_path = checkpoint_dir / f"zone3_{year}.parquet"
    if checkpoint_path.exists():
        print(f"  Year {year}: checkpoint already exists, skipping.")
        timer.mark(f"year {year} (skipped)")
        return

    print(f"\nProcessing year {year}...")

    tiles = find_tiles(raw_dir, f"zone3_s2_{year}")
    if not tiles:
        raise FileNotFoundError(f"No tiles found for year {year} in {raw_dir}.")

    # Column names for this year, derived from the reference band
    # name pattern (index + month), with this year's number substituted.
    col_names = []
    for name in ref_band_names:
        parsed = parse_band_name(name)
        if parsed is None:
            col_names.append(None)
            continue
        index, _, month = parsed
        col_names.append(f"{index}_{year}_{month:02d}")

    n_pixels = len(rows)
    year_data = {
        cn: np.full(n_pixels, np.nan, dtype=np.float32)
        for cn in col_names if cn is not None
    }

    for tile_idx, tile_path in enumerate(tiles, start=1):
        with rasterio.open(tile_path) as src:
            if src.count != n_bands:
                raise ValueError(
                    f"Year {year} tile {tile_path.name} has {src.count} "
                    f"bands, expected {n_bands}."
                )

            row_off, col_off = tile_offset(src.transform, ref_transform)
            tile_h, tile_w = src.height, src.width

            local_rows = rows - row_off
            local_cols = cols - col_off
            in_tile = (
                (local_rows >= 0) & (local_rows < tile_h) &
                (local_cols >= 0) & (local_cols < tile_w)
            )
            n_in_tile = int(in_tile.sum())
            if n_in_tile == 0:
                print(f"    Tile {tile_idx}/{len(tiles)} "
                      f"({tile_path.name}): 0 mangrove pixels, skipping.")
                continue

            global_idx = np.where(in_tile)[0]
            lr = local_rows[in_tile]
            lc = local_cols[in_tile]

            # ---- Read this tile in row-blocks (bounded memory) ----
            n_blocks = (tile_h + BLOCK_ROWS - 1) // BLOCK_ROWS
            for block_i, block_start in enumerate(range(0, tile_h, BLOCK_ROWS)):
                block_h = min(BLOCK_ROWS, tile_h - block_start)
                in_block = (lr >= block_start) & (lr < block_start + block_h)
                if not in_block.any():
                    continue

                window = Window(col_off=0, row_off=block_start,
                                 width=tile_w, height=block_h)
                block_data = src.read(window=window).astype(np.float32)
                block_data[block_data == NODATA_VAL] = np.nan
                # shape: (n_bands, block_h, tile_w)

                block_lr = lr[in_block] - block_start
                block_lc = lc[in_block]
                block_global_idx = global_idx[in_block]

                for band_idx in range(n_bands):
                    cn = col_names[band_idx]
                    if cn is None:
                        continue
                    year_data[cn][block_global_idx] = \
                        block_data[band_idx, block_lr, block_lc]

                del block_data
                gc.collect()

            print(f"    Tile {tile_idx}/{len(tiles)} "
                  f"({tile_path.name}): {n_in_tile:,} px, "
                  f"{n_blocks} row-blocks done")

        timer.mark(f"year {year}: tile {tile_idx}/{len(tiles)} done")

    df_year = pd.DataFrame(year_data)
    df_year.insert(0, "pixel_id", np.arange(n_pixels))
    df_year.to_parquet(checkpoint_path, index=False)
    timer.mark(f"year {year}: checkpoint saved "
               f"({checkpoint_path.stat().st_size/1e6:.1f} MB)")

    del year_data, df_year
    gc.collect()


# ============================================================
# Entry point
# ============================================================
def run(raw_dir, external_dir=None, processed_dir=None, scratch_dir=None):
    """scratch_dir is accepted for backward compatibility with earlier
    call sites but is no longer used (no local tile copy in v5)."""
    timer = Timer()

    raw_dir = Path(raw_dir)
    external_dir = Path(external_dir) if external_dir else raw_dir
    processed_dir = Path(processed_dir) if processed_dir else (
        REPO_ROOT / "data" / "processed" / "zone3"
    )
    processed_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_dir = processed_dir / "year_checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("MANGLAR - Pipeline 3 (Zone 3) - Merge and Filter")
    print("(v5: windowed reads direct from Drive, bounded memory)")
    print("=" * 60)
    print(f"Years: {START_YEAR}-{END_YEAR}")
    print(f"Raw export dir: {raw_dir}")
    print(f"GMW mask dir:   {external_dir}")
    print(f"Output dir:     {processed_dir}")
    print(f"Checkpoints:    {checkpoint_dir}")
    print(f"Block size:     {BLOCK_ROWS} rows/read")

    print("\nDetermining reference grid from first year...")
    ref_transform, ref_crs, ref_shape, n_bands, ref_band_names = \
        get_reference_grid(raw_dir, START_YEAR)
    print(f"  Grid shape: {ref_shape}, bands per year: {n_bands}")
    timer.mark("reference grid determined")

    print("\nBuilding mangrove mask...")
    rows, cols = build_mangrove_mask(external_dir, ref_transform, ref_crs, ref_shape)
    n_pixels = len(rows)
    timer.mark("mangrove mask computed")

    print("\nComputing pixel coordinates...")
    xs, ys = rasterio.transform.xy(ref_transform, rows, cols)
    lons, lats = np.array(xs), np.array(ys)
    if ref_crs != "EPSG:4326":
        lons, lats = rio_transform(ref_crs, "EPSG:4326", lons, lats)
        lons, lats = np.array(lons), np.array(lats)

    coords_path = checkpoint_dir / "zone3_coords.parquet"
    pd.DataFrame({
        "pixel_id": np.arange(n_pixels), "lon": lons, "lat": lats
    }).to_parquet(coords_path, index=False)
    timer.mark("pixel coordinates computed + saved")

    # ---- Process each year ----
    for year in range(START_YEAR, END_YEAR + 1):
        process_year(year, raw_dir, ref_transform, ref_shape,
                      n_bands, ref_band_names, rows, cols,
                      checkpoint_dir, timer)

    # ---- Final merge ----
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
            yt = pq.read_table(yf).slice(start, end - start)
            ydf = yt.to_pandas().drop(columns=["pixel_id"])
            chunk = pd.concat([chunk.reset_index(drop=True),
                                ydf.reset_index(drop=True)], axis=1)

        table = pa.Table.from_pandas(chunk, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(final_path, table.schema)
        writer.write_table(table)
        print(f"  Merged rows {start:,}-{end:,}")
        del chunk, table
        gc.collect()

    if writer is not None:
        writer.close()
    timer.mark("final parquet merged + saved")

    print(f"\nSaved: {final_path} ({final_path.stat().st_size / 1e6:.1f} MB)")

    n_cols = 48 * len(year_files)
    print("\nDone. Output summary:")
    print(f"  Total rows (mangrove pixels): {n_rows:,}")
    print(f"  Time series columns: {n_cols}")
    print(f"  Final file: {final_path}")
    timer.mark("FINISHED")

    return {"n_pixels": n_rows, "n_cols": n_cols,
            "parquet_path": str(final_path)}


if __name__ == "__main__":
    run(
        raw_dir=REPO_ROOT / "data" / "raw" / "gee_exports",
        external_dir=REPO_ROOT / "data" / "external",
        processed_dir=REPO_ROOT / "data" / "processed" / "zone3",
    )
