"""
MANGLAR — standalone merge step for Pipeline 3 (Zone 3).

Run this in a FRESH Colab session (after the per-year processing has
completed and all 8 zone3_{year}.parquet checkpoints + zone3_coords.parquet
exist in Drive). Starting fresh clears the RAM baseline accumulated
during the per-year loop.

EFFICIENCY FIX vs the inline version in merge_and_filter.py:
  The previous version called pq.read_table(yf) ONCE PER CHUNK PER
  YEAR FILE - i.e. 27 chunks x 8 files = 216 full re-reads of ~491MB
  files (~106 GB of redundant I/O), plus O(n) pd.concat copies per
  chunk.

  This version opens each year file ONCE via ParquetFile.iter_batches(),
  so each file is streamed exactly once, in lockstep with the others,
  and each chunk is built with a SINGLE pd.concat call.

USAGE IN COLAB (fresh runtime):
    from google.colab import drive
    drive.mount('/content/drive')
    !pip install pyarrow --quiet

    import sys
    sys.path.insert(0, '/content/drive/MyDrive/manglar')

    from src.pipelines.pipeline3_zone3.merge_checkpoints import merge_checkpoints

    result = merge_checkpoints(
        processed_dir='/content/drive/MyDrive/manglar_processed/zone3',
    )
    print(result)
"""

import gc
import time
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

START_YEAR = 2017
END_YEAR = 2024
CHUNK_SIZE = 500_000


def merge_checkpoints(processed_dir):
    t0 = time.time()
    processed_dir = Path(processed_dir)
    checkpoint_dir = processed_dir / "year_checkpoints"
    final_path = processed_dir / "zone3_pixel_timeseries.parquet"

    coords_path = checkpoint_dir / "zone3_coords.parquet"
    year_files = [checkpoint_dir / f"zone3_{y}.parquet"
                   for y in range(START_YEAR, END_YEAR + 1)]

    for p in [coords_path] + year_files:
        if not p.exists():
            raise FileNotFoundError(f"Missing checkpoint: {p}")

    print("=" * 60)
    print("MANGLAR - Pipeline 3 (Zone 3) - Merge checkpoints")
    print("=" * 60)

    coords_pf = pq.ParquetFile(coords_path)
    year_pfs = [pq.ParquetFile(yf) for yf in year_files]

    n_rows = coords_pf.metadata.num_rows
    print(f"Total rows: {n_rows:,}")
    print(f"Year files: {[yf.name for yf in year_files]}")

    coords_batches = coords_pf.iter_batches(batch_size=CHUNK_SIZE)
    year_batches = [pf.iter_batches(batch_size=CHUNK_SIZE) for pf in year_pfs]

    writer = None
    rows_done = 0
    chunk_idx = 0

    while True:
        try:
            coords_batch = next(coords_batches)
        except StopIteration:
            break

        chunk_idx += 1
        dfs = [coords_batch.to_pandas()]

        for pf_iter in year_batches:
            yb = next(pf_iter)
            ydf = yb.to_pandas().drop(columns=["pixel_id"])
            dfs.append(ydf)

        chunk_df = pd.concat(dfs, axis=1)

        table = pa.Table.from_pandas(chunk_df, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(final_path, table.schema)
        writer.write_table(table)

        rows_done += len(chunk_df)
        elapsed = time.time() - t0
        print(f"  Chunk {chunk_idx}: rows {rows_done:,}/{n_rows:,} "
              f"| {elapsed/60:.1f} min elapsed")

        del dfs, chunk_df, table
        gc.collect()

    if writer is not None:
        writer.close()

    size_mb = final_path.stat().st_size / 1e6
    elapsed = time.time() - t0
    print(f"\nSaved: {final_path} ({size_mb:.1f} MB)")
    print(f"Total merge time: {elapsed/60:.1f} min")

    n_cols = sum(len(pf.schema_arrow.names) - 1 for pf in year_pfs)
    return {
        "n_pixels": n_rows,
        "n_cols": n_cols,
        "parquet_path": str(final_path),
        "size_mb": size_mb,
    }


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python merge_checkpoints.py <processed_dir>")
        sys.exit(1)
    result = merge_checkpoints(sys.argv[1])
    print(result)
