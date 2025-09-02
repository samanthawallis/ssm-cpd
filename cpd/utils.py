import numpy as np
import pandas as pd
import scipy.linalg

RIDGE = 1e-12

def _sym(A):
    A = 0.5 * (A + A.T)
    return A + RIDGE * np.eye(A.shape[0])

def _logdet_psd(A):
    L = np.linalg.cholesky(_sym(A))
    return 2.0 * np.sum(np.log(np.diag(L)))

def _inv_pd(A):
    return np.linalg.inv(_sym(A))

def _prior_P0(F, Q, stable_dim=2, bias_var=1e6):
    n = F.shape[0]
    rho = max(abs(np.linalg.eigvals(F)))
    if rho < 1.0 - 1e-10:
        return _sym(scipy.linalg.solve_discrete_lyapunov(F, Q))
    # assume last (n-stable_dim) dims are unit-root(s)
    Fst = F[:stable_dim, :stable_dim]; Qst = Q[:stable_dim, :stable_dim]
    Pst = _sym(scipy.linalg.solve_discrete_lyapunov(Fst, Qst))
    P0 = np.zeros((n, n))
    P0[:stable_dim, :stable_dim] = Pst
    P0[stable_dim:, stable_dim:] = np.eye(n-stable_dim) * bias_var
    return _sym(P0)

def infer_dt(df: pd.DataFrame) -> float:
    if "X" in df and df["X"].diff().dropna().nunique() == 1:
        return float(df["X"].iloc[1] - df["X"].iloc[0])
    return float((pd.to_datetime(df["date"].iloc[1]) - pd.to_datetime(df["date"].iloc[0])).total_seconds())
