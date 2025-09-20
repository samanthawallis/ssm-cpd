
import numpy as np
from scipy.special import expit

def severity_score_from_gap(gap: float) -> float:
    """Calculates a severity"""
    return float(expit(float(gap)))