import numpy as np, pandas as pd, time
from sklearn.preprocessing import StandardScaler
from cpd import detect_single_cp
from cpd.grids import grids_fast_ou

# Globals set once per worker by init_pool
RETURNS = None
DATES = None
L = None
DT = None

def init_pool(returns: np.ndarray, dates: np.ndarray, lookback_L: int, dt: float):
    global RETURNS, DATES, L, DT
    RETURNS = returns
    DATES = dates
    L = int(lookback_L)
    DT = float(dt)

def process_end(window_end: int, mode="ou"):
    """Replicates your per-window logic for a given end index."""
    start = window_end - (L + 1)
    win_vals = RETURNS[start:window_end]                    # includes L+1 points, matching your iloc slice
    Y = StandardScaler().fit_transform(win_vals.reshape(-1,1)).ravel()
    y_win = Y[-L:]                                          # last L points
    time_index = window_end - 1
    window_date = pd.to_datetime(DATES[time_index]).strftime("%Y-%m-%d %H:%M:%S")

    t0 = time.perf_counter()
    res = detect_single_cp(
        y_win, 
        DT, 
        mode=mode, 
        zscore=False,
    )
    runtime = time.perf_counter() - t0

    return {
        "date": window_date,
        "t": int(time_index),
        "cp_location": int(res["location_k"]),
        "cp_location_norm":  float(res["location_k"] / L),
        "cp_score": float(1/(1 + np.exp(-res["severity_log"]))),
        "gap": float(res["severity_log"]),
        "runtime": float(runtime),
    }