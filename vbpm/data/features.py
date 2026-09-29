"""Feature-cache helper: atomic writes of per-song frontend arrays."""
from __future__ import annotations

import numpy as np


def atomic_save_npy(cache_path, array):
    """Write-then-rename: a reader racing a writer never sees a half-written array."""
    partial = cache_path.with_suffix(".npy.partial")
    with open(partial, "wb") as fh:
        np.save(fh, array.astype(np.float32))
    partial.replace(cache_path)
