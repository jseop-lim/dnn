import os

if os.getenv("DEVICE", "cpu") == "gpu":
    import cupy as np
else:
    import numpy as np


__all__ = ["np"]
