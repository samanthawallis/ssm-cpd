
import numpy as np

def default_grids_for_L_m32(L):
    ells = np.array([L/512, L/256, L/128, L/64, L/32, L/16, L/8], float)
    s2s  = (np.array([0.5, 1.0, 2.0, 4.0, 8.0])**2).astype(float)
    return ells, s2s

def default_grids_for_L_hybrid(L):
    ell_abs  = np.array([0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0], float)
    ell_frac = np.array([L/512, L/256, L/128, L/64, L/32, L/16], float)
    ells = np.unique(np.r_[ell_abs, ell_frac])
    s2s  = (np.array([0.5, 1.0, 2.0, 4.0, 8.0])**2).astype(float)
    return ells, s2s

def grids_fast_ou(L):
    return np.array([0.5, 1.0, 2.0, 4.0]), np.array([1.0, 4.0])

def grids_fast_m32(L):
    return np.array([L/256, L/128, L/64, L/32], float), np.array([1.0, 4.0])

def grids_auto_lite(L):
    return np.array([0.5, 2.0, 8.0, L/256, L/64, L/16], float), np.array([1.0, 4.0])
