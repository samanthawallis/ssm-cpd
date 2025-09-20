
import numpy as np
from scipy.special import expit


def severity_score_from_gap(gap: float) -> float:
    """Pure Wood et al.-style mapping: v = sigmoid(ΔNLML)."""
    return float(expit(float(gap)))