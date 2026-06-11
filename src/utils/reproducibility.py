"""
MANGLAR — src/utils/reproducibility.py
Set all random seeds for full reproducibility.
Call set_all_seeds() at the top of every training script.
"""

import random
import numpy as np


def set_all_seeds(seed: int = 42) -> None:
    """Set seeds for random, numpy, and torch (if available)."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
    print(f"[reproducibility] All seeds set to {seed}")
