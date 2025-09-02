"""Top-level package for changepoint detection (optimized build).

Re-exports the primary public API of the detector.
"""

from .detect import detect_single_cp, detect_on_array, detect_on_df_window
from .utils import infer_dt

__all__ = [
    "detect_single_cp",
    "detect_on_array",
    "detect_on_df_window",
    "infer_dt",
]
