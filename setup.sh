#!/usr/bin/env bash
# ============================================================
# MANGLAR — setup.sh
# Run once on any new machine or fresh Colab session.
# Usage:
#   Local:  bash setup.sh
#   Colab:  !bash setup.sh --colab
# ============================================================

set -e  # Exit on first error

COLAB=false
for arg in "$@"; do
  if [ "$arg" = "--colab" ]; then
    COLAB=true
  fi
done

echo "============================================================"
echo "  MANGLAR setup — $(date)"
echo "  Colab mode: $COLAB"
echo "============================================================"

# ---- 1. Python virtual environment (skip on Colab) ---------
if [ "$COLAB" = false ]; then
  echo "[1/6] Creating virtual environment..."
  python -m venv .venv
  source .venv/bin/activate
  echo "      Activated: $(which python)"
else
  echo "[1/6] Colab mode — skipping venv creation."
fi

# ---- 2. Core dependencies ----------------------------------
echo "[2/6] Installing Python dependencies..."
pip install --upgrade pip --quiet

if [ "$COLAB" = true ]; then
  # Install CPU torch separately to avoid Colab version conflicts
  pip install -r requirements.txt --quiet
else
  pip install -r requirements.txt --quiet
fi

# ---- 3. Presto (from source) --------------------------------
echo "[3/6] Installing Presto from source..."
pip install git+https://github.com/nasaharvest/presto.git --quiet

# ---- 4. Download Presto pretrained weights ------------------
echo "[4/6] Downloading Presto pretrained weights..."
mkdir -p data/external
WEIGHTS_URL="https://github.com/nasaharvest/presto/raw/main/data/default_models.pt"
WEIGHTS_PATH="data/external/presto_default.pt"
if [ ! -f "$WEIGHTS_PATH" ]; then
  curl -L "$WEIGHTS_URL" -o "$WEIGHTS_PATH" --silent
  echo "      Saved to $WEIGHTS_PATH"
else
  echo "      Already exists — skipping download."
fi

# ---- 5. GEE authentication ---------------------------------
echo "[5/6] Google Earth Engine authentication..."
echo "      Run 'earthengine authenticate' manually if not yet done."
echo "      For Colab: use ee.Authenticate() in the notebook."

# ---- 6. Verify installation --------------------------------
echo "[6/6] Verifying installation..."
python - <<'EOF'
import importlib, sys
required = [
    "ee", "geemap", "geopandas", "rasterio",
    "statsmodels", "pysal", "torch", "yaml",
    "matplotlib", "numpy", "pandas"
]
failed = []
for pkg in required:
    try:
        importlib.import_module(pkg)
    except ImportError:
        failed.append(pkg)

if failed:
    print(f"  MISSING: {failed}")
    sys.exit(1)
else:
    import torch
    print(f"  All packages OK.")
    print(f"  PyTorch: {torch.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

try:
    from presto import Presto
    m = Presto.load_pretrained()
    params = sum(p.numel() for p in m.encoder.parameters())
    print(f"  Presto encoder: {params:,} parameters")
except Exception as e:
    print(f"  Presto load warning: {e}")
EOF

echo ""
echo "============================================================"
echo "  Setup complete. Activate environment with:"
echo "    source .venv/bin/activate"
echo "  Then start with:"
echo "    notebooks/01_exploration/"
echo "============================================================"
