"""
MANGLAR — src/utils/pixel_extraction.py  (v7)

Shared, zone-agnostic pixel-timeseries extraction pipeline.

v7 CHANGE vs v6: the year-level merge step (after all tiles for a
year are spilled to disk) no longer allocates a single full-zone
array. v6 still did:
    year_array = np.full((n_pixels_total, n_bands), ...)
which for Zone 1 (28,349,239 pixels x 48 bands x 4 bytes = 5.44 GB)
silently OOM-killed the process at merge time (no Python traceback -
OS-level kill). v7 writes the year's parquet checkpoint in ROW-CHUNKS
(YEAR_MERGE_CHUNK_SIZE = 2,000,000 rows), re-reading the small tile
spill files once per chunk. Peak RAM during merge is now
chunk_size x n_bands x 4 bytes (~384 MB), independent of total zone
pixel count.

Used by:
  src/pipelines/pipeline1_zone1/extract.py  (zone="zone1")
  src/pipelines/pipeline2_zone2/extract.py  (zone="zone2")
  src/pipelines/pipeline3_zone3/extract.py  (zone="zone3")
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

NODATA_VAL = -9999.0
BLOCK_ROWS = 200
YEAR_MERGE_CHUNK_SIZE = 2_000_000  # rows per chunk during year-level merge


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
        return since_last


def find_tiles(folder, prefix):
    folder = Path(folder)
    tiles = sorted(glob.glob(str(folder / f"{prefix}-*.tif")))
    if not tiles:
        single = folder / f"{prefix}.tif"
        if single.exists():
            tiles = [str(single)]
    return [Path(t) for t in tiles]


def mosaic_band1(tiles):
    if not tiles:
        raise FileNotFoundError("No tiles provided to mosaic_band1().")
    srcs = [rasterio.open(t) for t in tiles]
    mosaic, out_transform = rio_merge(srcs, indexes=[1])
    crs = srcs[0].crs.to_string()
    for s in srcs:
        s.close()
    return mosaic, out_transform, crs


def make_band_parser(indices):
    pattern = re.compile(
        r"(" + "|".join(re.escape(i) for i in indices) + r")_(\d{4})_(\d{2})"
    )

    def parse(name):
        m = pattern.search(name)
        if not m:
            return None
        index, year, month = m.groups()
        return index, int(year), int(month)

    return parse


def get_reference_grid(raw_dir, zone, prefix, start_year):
    tiles = find_tiles(raw_dir, f"{prefix}_{start_year}")
    if not tiles:
        raise FileNotFoundError(
            f"[{zone}] No tiles found for reference year {start_year} "
            f"in {raw_dir} (expected '{prefix}_{start_year}-*.tif')."
        )
    srcs = [rasterio.open(t) for t in tiles]
    n_bands = srcs[0].count
    band_names = [srcs[0].descriptions[i] or f"band_{i+1}"
                   for i in range(n_bands)]
    for s in srcs:
        s.close()

    sample, out_transform, crs = mosaic_band1(tiles)
    shape = sample.shape[1:]
    del sample

    return out_transform, crs, shape, n_bands, band_names


def build_mangrove_mask(external_dir, zone, mask_prefix, transform, crs, shape):
    external_dir = Path(external_dir)
    mask_tiles = find_tiles(external_dir, mask_prefix)
    if not mask_tiles:
        mask_tiles = [Path(p) for p in
                       sorted(glob.glob(str(external_dir / f"{mask_prefix}*.tif")))]

    if not mask_tiles:
        warnings.warn(
            f"[{zone}] No GMW mask files found in {external_dir} "
            f"(expected '{mask_prefix}*.tif'). "
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
    print(f"  [{zone}] Mangrove pixels: {n_pixels:,} of {mask.size:,} "
          f"({100 * n_pixels / mask.size:.2f}%)")

    if n_pixels == 0:
        raise ValueError(f"[{zone}] Mangrove mask matched ZERO pixels.")

    return rows, cols


def tile_offset(tile_transform, ref_transform, tol=1e-6):
    px_w = ref_transform.a
    px_h = ref_transform.e

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


def merge_tile_spills_to_parquet(tile_spill_files, n_pixels_total, n_bands,
                                   valid_col_names, valid_col_idx,
                                   checkpoint_path,
                                   chunk_size=YEAR_MERGE_CHUNK_SIZE):
    """Write a year's checkpoint parquet in row-chunks, never holding
    a full (n_pixels_total x n_bands) array in memory at once.

    For each chunk of rows [start:end), loads each tile spill file's
    (global_idx, values) and writes whichever rows of THIS CHUNK that
    tile contributes to. Re-reads the (small) spill files once per
    chunk - trades a little redundant disk I/O for bounded RAM.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    print(f"  Merging {len(tile_spill_files)} tile spill files "
          f"({chunk_size:,} rows/chunk)...")

    writer = None
    for chunk_start in range(0, n_pixels_total, chunk_size):
        chunk_end = min(chunk_start + chunk_size, n_pixels_total)
        chunk_n = chunk_end - chunk_start

        chunk_array = np.full((chunk_n, n_bands), np.nan, dtype=np.float32)

        for idx_path, val_path in tile_spill_files:
            global_idx = np.load(idx_path)
            in_chunk = (global_idx >= chunk_start) & (global_idx < chunk_end)
            if not in_chunk.any():
                del global_idx, in_chunk
                continue

            tile_values = np.load(val_path)
            local_idx_in_chunk = global_idx[in_chunk] - chunk_start
            chunk_array[local_idx_in_chunk, :] = tile_values[in_chunk, :]

            del global_idx, tile_values, in_chunk, local_idx_in_chunk

        df_chunk = pd.DataFrame(
            chunk_array[:, valid_col_idx], columns=valid_col_names
        )
        df_chunk.insert(0, "pixel_id", np.arange(chunk_start, chunk_end))

        table = pa.Table.from_pandas(df_chunk, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(checkpoint_path, table.schema)
        writer.write_table(table)

        print(f"    Chunk rows {chunk_start:,}-{chunk_end:,} written")

        del chunk_array, df_chunk, table
        gc.collect()

    if writer is not None:
        writer.close()

    for idx_path, val_path in tile_spill_files:
        idx_path.unlink(missing_ok=True)
        val_path.unlink(missing_ok=True)


def process_year(year, raw_dir, prefix, ref_transform, n_bands,
                  ref_band_names, parse_band_name, rows, cols,
                  checkpoint_dir, zone, timer, scratch_dir=None):
    """v7: tile-level disk spill + chunked year-merge. Peak RAM during
    BOTH tile reading and year-merge is bounded by a fixed chunk/tile
    size, never by total zone pixel count.
    """
    checkpoint_path = checkpoint_dir / f"{zone}_{year}.parquet"
    if checkpoint_path.exists():
        print(f"  [{zone}] Year {year}: checkpoint exists, skipping.")
        timer.mark(f"{zone} year {year} (skipped)")
        return

    print(f"\n[{zone}] Processing year {year}...")

    tiles = find_tiles(raw_dir, f"{prefix}_{year}")
    if not tiles:
        raise FileNotFoundError(
            f"[{zone}] No tiles found for year {year} in {raw_dir} "
            f"(expected '{prefix}_{year}-*.tif')."
        )

    col_names = []
    for name in ref_band_names:
        parsed = parse_band_name(name)
        if parsed is None:
            col_names.append(None)
            continue
        index, _, month = parsed
        col_names.append(f"{index}_{year}_{month:02d}")

    n_pixels_total = len(rows)

    spill_dir = Path(scratch_dir) if scratch_dir else checkpoint_dir
    year_spill_dir = spill_dir / f"_spill_{zone}_{year}"
    year_spill_dir.mkdir(parents=True, exist_ok=True)

    tile_spill_files = []

    for tile_idx, tile_path in enumerate(tiles, start=1):
        with rasterio.open(tile_path) as src:
            if src.count != n_bands:
                raise ValueError(
                    f"[{zone}] Year {year} tile {tile_path.name} has "
                    f"{src.count} bands, expected {n_bands}."
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
                      f"({tile_path.name}): 0 mask pixels, skipping.")
                continue

            global_idx = np.where(in_tile)[0]
            lr = local_rows[in_tile]
            lc = local_cols[in_tile]

            tile_values = np.full((n_in_tile, n_bands), np.nan, dtype=np.float32)

            n_blocks = (tile_h + BLOCK_ROWS - 1) // BLOCK_ROWS
            for block_start in range(0, tile_h, BLOCK_ROWS):
                block_h = min(BLOCK_ROWS, tile_h - block_start)
                in_block = (lr >= block_start) & (lr < block_start + block_h)
                if not in_block.any():
                    continue

                window = Window(col_off=0, row_off=block_start,
                                 width=tile_w, height=block_h)
                block_data = src.read(window=window).astype(np.float32)
                block_data[block_data == NODATA_VAL] = np.nan

                block_lr = lr[in_block] - block_start
                block_lc = lc[in_block]
                block_local_idx = np.where(in_block)[0]

                for band_idx in range(n_bands):
                    if col_names[band_idx] is None:
                        continue
                    tile_values[block_local_idx, band_idx] = \
                        block_data[band_idx, block_lr, block_lc]

                del block_data
                gc.collect()

            idx_path = year_spill_dir / f"tile{tile_idx}_idx.npy"
            val_path = year_spill_dir / f"tile{tile_idx}_val.npy"
            np.save(idx_path, global_idx)
            np.save(val_path, tile_values)
            tile_spill_files.append((idx_path, val_path))

            del tile_values, global_idx, lr, lc, local_rows, local_cols, in_tile
            gc.collect()

            print(f"    Tile {tile_idx}/{len(tiles)} "
                  f"({tile_path.name}): {n_in_tile:,} px, "
                  f"{n_blocks} row-blocks done, spilled to disk")

        timer.mark(f"{zone} year {year}: tile {tile_idx}/{len(tiles)} done")

    # ---- v7: chunked merge, never allocates full-zone array ----
    valid_col_names = [cn for cn in col_names if cn is not None]
    valid_col_idx = [i for i, cn in enumerate(col_names) if cn is not None]

    merge_tile_spills_to_parquet(
        tile_spill_files, n_pixels_total, n_bands,
        valid_col_names, valid_col_idx, checkpoint_path
    )

    year_spill_dir.rmdir()
    gc.collect()

    timer.mark(f"{zone} year {year}: checkpoint saved "
               f"({checkpoint_path.stat().st_size/1e6:.1f} MB)")


def extract_pixel_timeseries(zone, raw_dir, external_dir, processed_dir,
                              prefix, mask_prefix, indices,
                              start_year, end_year, timer=None,
                              scratch_dir=None):
    if timer is None:
        timer = Timer()

    raw_dir = Path(raw_dir)
    external_dir = Path(external_dir)
    processed_dir = Path(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_dir = processed_dir / "year_checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"MANGLAR - [{zone}] Pixel time series extraction")
    print("(v7: tile spill + chunked year-merge, bounded memory)")
    print("=" * 60)
    print(f"Years: {start_year}-{end_year}")
    print(f"Raw export dir: {raw_dir}")
    print(f"Mask dir:       {external_dir}")
    print(f"Output dir:     {processed_dir}")
    print(f"Block size:     {BLOCK_ROWS} rows/read")
    print(f"Year-merge chunk size: {YEAR_MERGE_CHUNK_SIZE:,} rows")
    if scratch_dir:
        print(f"Scratch dir:    {scratch_dir} (local spill)")

    print(f"\n[{zone}] Determining reference grid from {start_year}...")
    ref_transform, ref_crs, ref_shape, n_bands, ref_band_names = \
        get_reference_grid(raw_dir, zone, prefix, start_year)
    print(f"  Grid shape: {ref_shape}, bands per year: {n_bands}")
    timer.mark(f"{zone}: reference grid determined")

    print(f"\n[{zone}] Building mangrove mask...")
    rows, cols = build_mangrove_mask(
        external_dir, zone, mask_prefix, ref_transform, ref_crs, ref_shape
    )
    n_pixels = len(rows)
    timer.mark(f"{zone}: mask computed")

    print(f"\n[{zone}] Computing pixel coordinates...")
    xs, ys = rasterio.transform.xy(ref_transform, rows, cols)
    lons, lats = np.array(xs), np.array(ys)
    if ref_crs != "EPSG:4326":
        lons, lats = rio_transform(ref_crs, "EPSG:4326", lons, lats)
        lons, lats = np.array(lons), np.array(lats)

    coords_path = checkpoint_dir / f"{zone}_coords.parquet"
    pd.DataFrame({
        "pixel_id": np.arange(n_pixels), "lon": lons, "lat": lats
    }).to_parquet(coords_path, index=False)
    timer.mark(f"{zone}: pixel coordinates computed + saved")

    parse_band_name = make_band_parser(indices)

    for year in range(start_year, end_year + 1):
        process_year(year, raw_dir, prefix, ref_transform, n_bands,
                      ref_band_names, parse_band_name, rows, cols,
                      checkpoint_dir, zone, timer, scratch_dir=scratch_dir)

    year_files = [checkpoint_dir / f"{zone}_{y}.parquet"
                   for y in range(start_year, end_year + 1)]

    return {
        "n_pixels": n_pixels,
        "checkpoint_dir": checkpoint_dir,
        "coords_path": coords_path,
        "year_files": year_files,
    }


def merge_checkpoints(zone, processed_dir, start_year, end_year,
                       chunk_size=500_000, timer=None):
    if timer is None:
        timer = Timer()

    import pyarrow as pa
    import pyarrow.parquet as pq

    processed_dir = Path(processed_dir)
    checkpoint_dir = processed_dir / "year_checkpoints"
    final_path = processed_dir / f"{zone}_pixel_timeseries.parquet"

    coords_path = checkpoint_dir / f"{zone}_coords.parquet"
    year_files = [checkpoint_dir / f"{zone}_{y}.parquet"
                   for y in range(start_year, end_year + 1)]

    for p in [coords_path] + year_files:
        if not p.exists():
            raise FileNotFoundError(f"[{zone}] Missing checkpoint: {p}")

    print("=" * 60)
    print(f"MANGLAR - [{zone}] Merge checkpoints")
    print("=" * 60)

    coords_pf = pq.ParquetFile(coords_path)
    year_pfs = [pq.ParquetFile(yf) for yf in year_files]

    n_rows = coords_pf.metadata.num_rows
    print(f"Total rows: {n_rows:,}")
    print(f"Year files: {[yf.name for yf in year_files]}")

    coords_batches = coords_pf.iter_batches(batch_size=chunk_size)
    year_batches = [pf.iter_batches(batch_size=chunk_size) for pf in year_pfs]

    writer = None
    rows_done = 0

    while True:
        try:
            coords_batch = next(coords_batches)
        except StopIteration:
            break

        dfs = [coords_batch.to_pandas()]
        for pf_iter in year_batches:
            yb = next(pf_iter)
            dfs.append(yb.to_pandas().drop(columns=["pixel_id"]))

        chunk_df = pd.concat(dfs, axis=1)
        table = pa.Table.from_pandas(chunk_df, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(final_path, table.schema)
        writer.write_table(table)

        rows_done += len(chunk_df)
        timer.mark(f"{zone}: merged rows {rows_done:,}/{n_rows:,}")

        del dfs, chunk_df, table
        gc.collect()

    if writer is not None:
        writer.close()

    size_mb = final_path.stat().st_size / 1e6
    print(f"\n[{zone}] Saved: {final_path} ({size_mb:.1f} MB)")

    n_cols = sum(len(pf.schema_arrow.names) - 1 for pf in year_pfs)
    timer.mark(f"{zone}: FINISHED")

    return {"n_pixels": n_rows, "n_cols": n_cols,
            "parquet_path": str(final_path), "size_mb": size_mb}
