"""
MANGLAR — src/qa/data_quality_audit.py

Standardized data-quality audit, applied identically to all three
zones (zone1, zone2, zone3). Produces:

  1. A structured JSON summary per zone (machine-readable, for
     building paper tables programmatically and for audit trail).
  2. A histogram figure per zone (dpi=600, publication-ready).
  3. A combined 3-panel comparison figure across all zones (dpi=600).
  4. A markdown summary table comparing all zones side by side.

USAGE IN COLAB:
    import sys
    sys.path.insert(0, '/content/drive/MyDrive/manglar')

    from src.qa.data_quality_audit import audit_zone, compare_zones

    s3 = audit_zone('zone3', '.../zone3_pixel_timeseries.parquet', qa_dir)
    s1 = audit_zone('zone1', '.../zone1_pixel_timeseries.parquet', qa_dir)
    s2 = audit_zone('zone2', '.../zone2_pixel_timeseries.parquet', qa_dir)
    compare_zones([s1, s2, s3], qa_dir)
"""

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DPI = 600
SAMPLE_SIZE = 5000
RANDOM_SEED = 42
NDVI_LOW_THRESHOLD = 0.3
HIST_BINS = 100
HIST_XLIM = (-1.0, 1.0)

ZONE_COLORS = {
    "zone1": "#1f77b4",
    "zone2": "#2ca02c",
    "zone3": "#7f7f7f",
}
ZONE_LABELS = {
    "zone1": "Zone 1 - Primary (Reentrancias, fishing pressure)",
    "zone2": "Zone 2 - Contrast (Itapecuru estuary, riverine pressure)",
    "zone3": "Zone 3 - Calibration (near-pristine reference)",
}


def _load_ndvi_sample(parquet_path, sample_size=SAMPLE_SIZE, seed=RANDOM_SEED):
    pf = pq.ParquetFile(parquet_path)
    all_cols = pf.schema_arrow.names
    ndvi_cols = sorted([c for c in all_cols if c.startswith("NDVI_")])
    needed_cols = ["pixel_id", "lon", "lat"] + ndvi_cols

    batch = next(pf.iter_batches(batch_size=sample_size, columns=needed_cols))
    df = batch.to_pandas()

    return df, ndvi_cols, pf.metadata.num_rows, len(all_cols)


def audit_zone(zone, parquet_path, output_dir, ndvi_low_threshold=NDVI_LOW_THRESHOLD):
    t0 = time.time()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[{zone}] Loading NDVI sample from {parquet_path} ...")
    df, ndvi_cols, total_rows, total_cols = _load_ndvi_sample(parquet_path)

    ndvi_vals = df[ndvi_cols].values.astype(float)
    nan_mask = np.isnan(ndvi_vals)
    nan_frac_per_pixel = nan_mask.mean(axis=1)
    valid = ndvi_vals[~nan_mask]

    summary = {
        "zone": zone,
        "parquet_path": str(parquet_path),
        "audit_timestamp_utc": pd.Timestamp.utcnow().isoformat(),
        "total_rows_full_dataset": int(total_rows),
        "total_cols_full_dataset": int(total_cols),
        "sample_size_rows": int(len(df)),
        "sample_random_seed": RANDOM_SEED,
        "n_ndvi_columns": len(ndvi_cols),
        "nan_fraction_mean": float(nan_frac_per_pixel.mean()),
        "nan_fraction_median": float(np.median(nan_frac_per_pixel)),
        "nan_fraction_std": float(nan_frac_per_pixel.std()),
        "valid_value_count": int(valid.size),
        "valid_value_min": float(valid.min()) if valid.size else None,
        "valid_value_max": float(valid.max()) if valid.size else None,
        "valid_value_mean": float(valid.mean()) if valid.size else None,
        "valid_value_median": float(np.median(valid)) if valid.size else None,
        "valid_value_std": float(valid.std()) if valid.size else None,
        "pct_negative": float((valid < 0.0).mean()) if valid.size else None,
        "pct_below_low_threshold": float((valid < ndvi_low_threshold).mean()) if valid.size else None,
        "ndvi_low_threshold_used": ndvi_low_threshold,
    }

    print(f"  NaN fraction (mean/median): "
          f"{summary['nan_fraction_mean']:.2%} / {summary['nan_fraction_median']:.2%}")
    print(f"  Valid range: [{summary['valid_value_min']:.3f}, "
          f"{summary['valid_value_max']:.3f}]")
    print(f"  % negative: {summary['pct_negative']:.2%}  "
          f"| % below {ndvi_low_threshold}: {summary['pct_below_low_threshold']:.2%}")

    json_path = output_dir / f"{zone}_quality_summary.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    summary["json_path"] = str(json_path)
    print(f"  Saved summary: {json_path}")

    fig, ax = plt.subplots(figsize=(10, 5))
    color = ZONE_COLORS.get(zone, "#1f77b4")
    label = ZONE_LABELS.get(zone, zone)

    ax.hist(valid, bins=HIST_BINS, range=HIST_XLIM, color=color, alpha=0.85)
    ax.axvline(0, color="red", linestyle="--", linewidth=1, label="NDVI = 0")
    ax.axvline(ndvi_low_threshold, color="darkorange", linestyle="--",
               linewidth=1, label=f"Low-NDVI threshold ({ndvi_low_threshold})")
    ax.set_xlim(HIST_XLIM)
    ax.set_xlabel("NDVI")
    ax.set_ylabel("Pixel-month count (sample)")
    ax.set_title(f"{label}\nValid NDVI distribution "
                 f"(n={summary['sample_size_rows']:,} sampled pixels, "
                 f"NaN={summary['nan_fraction_mean']:.1%})")
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()

    fig_path = output_dir / f"{zone}_ndvi_distribution.png"
    fig.savefig(fig_path, dpi=DPI)
    plt.close(fig)
    summary["figure_path"] = str(fig_path)
    print(f"  Saved figure ({DPI} dpi): {fig_path}")

    elapsed = time.time() - t0
    print(f"  [{zone}] Audit complete in {elapsed:.1f}s")

    return summary


def compare_zones(summaries, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    n = len(summaries)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5), sharey=True)
    if n == 1:
        axes = [axes]

    for ax, s in zip(axes, summaries):
        zone = s["zone"]
        df, ndvi_cols, _, _ = _load_ndvi_sample(s["parquet_path"])
        ndvi_vals = df[ndvi_cols].values.astype(float)
        valid = ndvi_vals[~np.isnan(ndvi_vals)]

        color = ZONE_COLORS.get(zone, "#1f77b4")
        ax.hist(valid, bins=HIST_BINS, range=HIST_XLIM, color=color, alpha=0.85)
        ax.axvline(0, color="red", linestyle="--", linewidth=1)
        ax.axvline(s["ndvi_low_threshold_used"], color="darkorange",
                   linestyle="--", linewidth=1)
        ax.set_xlim(HIST_XLIM)
        ax.set_xlabel("NDVI")
        ax.set_title(f"{zone.upper()}\nNaN={s['nan_fraction_mean']:.1%}, "
                     f"neg={s['pct_negative']:.1%}", fontsize=10)

    axes[0].set_ylabel("Pixel-month count (sample)")
    fig.suptitle("Cross-zone NDVI distribution comparison "
                 "(standardized audit, identical binning)", fontsize=12)
    fig.tight_layout()

    combined_fig_path = output_dir / "cross_zone_ndvi_comparison.png"
    fig.savefig(combined_fig_path, dpi=DPI)
    plt.close(fig)
    print(f"Saved combined comparison figure ({DPI} dpi): {combined_fig_path}")

    rows = []
    for s in summaries:
        rows.append({
            "Zone": s["zone"],
            "Total pixels": f"{s['total_rows_full_dataset']:,}",
            "NaN frac (mean)": f"{s['nan_fraction_mean']:.2%}",
            "NaN frac (median)": f"{s['nan_fraction_median']:.2%}",
            "Valid range": f"[{s['valid_value_min']:.3f}, {s['valid_value_max']:.3f}]",
            "Valid mean": f"{s['valid_value_mean']:.3f}",
            "% negative": f"{s['pct_negative']:.2%}",
            f"% below {s['ndvi_low_threshold_used']}": f"{s['pct_below_low_threshold']:.2%}",
        })

    table_df = pd.DataFrame(rows)
    table_path = output_dir / "cross_zone_quality_table.md"
    with open(table_path, "w") as f:
        f.write("# MANGLAR - Cross-zone data quality audit\n\n")
        f.write(f"Generated: {pd.Timestamp.utcnow().isoformat()}\n\n")
        f.write(table_df.to_markdown(index=False))
        f.write("\n")

    print(f"Saved comparison table: {table_path}")
    print("\n" + table_df.to_string(index=False))

    return str(combined_fig_path), str(table_path)
