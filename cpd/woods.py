import numpy as np
from scipy.special import expit

def woods_score_from_gap(gap: float) -> float:
    """Pure Wood et al.-style mapping: v = sigmoid(ΔNLML)."""
    return float(expit(float(gap)))

def woods_score_with_window_prior(gap: float, pi_window: float = 0.2) -> float:
    logit_pi = np.log(pi_window) - np.log(1.0 - pi_window)
    return float(expit(float(gap) + logit_pi))

def calibrated_woods_score(gap: float, nc_threshold: float, scale: float = 1.0) -> float:
    return float(expit((float(gap) - float(nc_threshold)) / float(scale)))
