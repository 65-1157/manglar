# MANGLAR

**Spatial co-occurrence of mangrove canopy degradation and coastal fishing pressure in the Reentrâncias Maranhenses, Maranhão, Brazil**

A zero-budget academic research project combining Global Fishing Watch AIS-derived fishing effort with Sentinel-2 per-pixel canopy trend surfaces in a multi-zone spatial lag regression framework.

Target journal: **IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing (JSTARS)**
Submission target: **August 2026**

---

## Research question

Do patterns of mangrove canopy degradation in the Reentrâncias Maranhenses spatially co-occur with patterns of adjacent coastal fishing pressure, after controlling for riverine, urban, and road-access stressors?

---

## Study zones

| Zone | Location | Role | Pipelines |
|------|----------|------|-----------|
| Zone 1 | Reentrâncias Maranhenses (~1°S–2°30'S, 44°W–46°W) | Primary study zone | Full (S2 + S1 + GFW + regression) |
| Zone 2 | Itapecuru River estuary (~2°42'S, 43°54'W) | Contrast zone (riverine pressure) | Partial (S2 only, no GFW) |
| Zone 3 | Baixada Maranhense | Calibration zone (near-pristine reference) | Calibration only (no regression) |

---

## Data sources

| Dataset | Source | Format | Used in |
|---------|--------|--------|---------|
| Sentinel-2 Level-2A | Google Earth Engine | GeoTIFF | All zones |
| Sentinel-1 GRD IW | Google Earth Engine | GeoTIFF | Zone 1 only |
| Global Fishing Watch annual fishing effort | globalfishingwatch.org / GEE | GeoTIFF raster | Zone 1 only |
| MapBiomas annual land cover | Google Earth Engine | GeoTIFF | Zone 1, Zone 2 |
| Global Mangrove Watch extent | JAXA / GEE | GeoTIFF | All zones (masking) |
| ANA HidroWeb river discharge | snirh.gov.br/hidroweb | CSV | Zone 1, Zone 2 |
| OpenStreetMap road network | Geofabrik | GeoJSON/Shapefile | Zone 1 |
| GEDI canopy height footprints | NASA Earthdata / GEE | GeoTIFF | Zone 1 (supplementary) |

---

## Model comparison (ablation study)

| Method | Type | Role |
|--------|------|------|
| STL decomposition | Statistical baseline | Comparison reference |
| N-BEATS | DL basis expansion (no labels needed) | Primary DL method |
| Presto (fine-tuned on Zone 3) | Remote sensing foundation model | Foundation model method |

---

## Repository structure

```
manglar/
├── src/
│   ├── gee/                  # Google Earth Engine JavaScript scripts
│   ├── pipelines/            # Per-zone processing pipelines
│   │   ├── pipeline1_zone1/
│   │   ├── pipeline2_zone2/
│   │   └── pipeline3_zone3/
│   ├── models/               # DL model implementations
│   │   ├── stl_baseline/
│   │   ├── nbeats/
│   │   └── presto/
│   ├── regression/           # Spatial lag and GWR models
│   └── utils/                # Shared utilities
├── notebooks/                # Colab-ready numbered notebooks
│   ├── 01_exploration/
│   ├── 02_pipeline1/
│   ├── 03_pipeline2/
│   ├── 04_pipeline3/
│   ├── 05_models/
│   ├── 06_regression/
│   └── 07_figures/
├── data/
│   ├── raw/                  # Source data (not committed — see .gitignore)
│   ├── processed/            # Derived products by zone
│   └── external/             # Reference shapefiles, zone boundaries
├── configs/                  # YAML configuration files
├── tests/                    # Unit and integration tests
├── outputs/                  # Maps, figures, tables, exports
├── paper/                    # Manuscript drafts, submission files
└── docs/                     # Extended documentation
```

---

## Setup

```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/manglar.git
cd manglar

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate        # Linux/macOS
# .venv\Scripts\activate         # Windows

# Install dependencies
pip install -r requirements.txt

# Authenticate Google Earth Engine (first time only)
earthengine authenticate
```

---

## Compute environment

- **GEE processing**: Google Earth Engine free tier (JavaScript API)
- **Model training and inference**: Google Colab free tier (NVIDIA T4, ~15 GB VRAM)
- **Spatial regression**: Local CPU or Colab CPU runtime
- **Required VRAM by stage**: STL=0 GB, N-BEATS=~1 GB, Presto fine-tune=~3 GB

All stages fit within the Colab free tier memory ceiling. See `docs/compute_guide.md` for session management protocols.

---

## Reproducibility

All random seeds are set in `configs/base_config.yaml`. GEE export parameters are versioned in `configs/gee_config.yaml`. Model checkpoints are saved to `outputs/exports/` and tracked in the run log at `docs/run_log.md`.

---

## Citation

*Manuscript in preparation. Citation will be added upon acceptance.*

---

## License

Code: MIT License — see `LICENSE`
Data: Subject to original source licenses (GEE, GFW, JAXA, ANA). No raw data is committed to this repository.

---

## Contact

*[Author contact to be added before repository goes public]*
