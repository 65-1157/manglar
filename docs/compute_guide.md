# MANGLAR — Compute Guide

## Hardware budget summary

| Stage | Runtime | Peak VRAM | Peak RAM | Notes |
|-------|---------|-----------|----------|-------|
| GEE compositing | GEE servers | n/a | n/a | Async — no session needed |
| Pixel extraction | CPU | 0 | ~4–8 GB | Load tiles sequentially |
| STL decomposition | CPU | 0 | ~2–4 GB | statsmodels |
| N-BEATS train+infer | T4 GPU | ~1 GB | ~2 GB | Batch 4096 pixels |
| Presto fine-tune (Zone 3) | T4 GPU | ~3 GB | ~3 GB | FP16 mixed precision |
| Presto inference (Zone 1) | T4 GPU | ~2 GB | ~2 GB | Feature extraction mode |
| Spatial lag regression | CPU | 0 | ~4–6 GB | pysal/spreg |
| GWR (if used) | CPU | 0 | ~6–10 GB | Tile Zone 1 if OOM |

**All stages fit within Colab free T4 (15 GB VRAM, 12 GB RAM).**

## Session protocol

Colab free tier sessions disconnect after 90 minutes idle and hard-terminate at 12 hours.

1. GEE exports run asynchronously on GEE servers — trigger and close Colab
2. Save all outputs to Google Drive immediately after each stage
3. Never rely on Colab `/content/` between sessions — it resets
4. Model checkpoints → `outputs/exports/checkpoints/` in Drive
5. Processed rasters → `data/processed/` in Drive

## Google Drive mount

```python
from google.colab import drive
drive.mount('/content/drive')
REPO_ROOT = '/content/drive/MyDrive/manglar'
```

## Enabling GPU in Colab

Runtime → Change runtime type → T4 GPU

## FP16 mixed precision (Presto fine-tuning)

```python
from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()
with autocast():
    output = model(batch)
    loss = criterion(output, target)
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```
