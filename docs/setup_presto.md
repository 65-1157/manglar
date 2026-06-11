# MANGLAR — Presto Setup Guide

Presto is not on PyPI. Install from source before running any Presto notebooks.

## Installation

```bash
# From repo root, with venv active
pip install git+https://github.com/nasaharvest/presto.git
```

## Download pretrained weights

```bash
# From repo root
mkdir -p data/external
wget https://github.com/nasaharvest/presto/raw/main/data/default_models.pt \
     -O data/external/presto_default.pt
```

## Verify installation

```python
from presto import Presto
model = Presto.load_pretrained()
print(f"Presto loaded — encoder params: {sum(p.numel() for p in model.encoder.parameters()):,}")
# Expected output: ~500K parameters
```

## Input format

Presto expects pixel-timeseries of shape `(batch, time, channels)` where:
- `batch` = number of pixels
- `time` = number of monthly timesteps (e.g., 84 for 2017–2023)
- `channels` = spectral bands + indices (Sentinel-2 bands + derived indices)

See `notebooks/05_models/presto_finetune_zone3.ipynb` for full implementation.

## GEE integration

Presto embeddings can also be generated directly in GEE via the `presto` GEE community package. See `src/gee/presto_gee_embeddings.js` for the GEE script (to be implemented in Phase 2).
