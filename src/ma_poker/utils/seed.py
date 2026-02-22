from __future__ import annotations

import os
import random
from typing import Optional

import numpy as np


def set_global_seed(seed: Optional[int]) -> None:
    """Best-effort global seeding (python/random/numpy/env vars).

    Note: The underlying poker env may have its own RNG; we pass seed where possible too.
    """
    if seed is None:
        return
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

