
import numpy as np
import pandas as pd
import time
from scipy.special import logsumexp, expit
from sklearn.preprocessing import StandardScaler

from .ssm import build_ssm
from .utils import _sym, _logdet_psd, _inv_pd, _prior_P0

jitter = 1e-12

try:
    from numba import njit, prange
    NUMBA_OK = True
except Exception:
    NUMBA_OK = False

def _prob_from_gap(gap, prior_cp):
    if prior_cp is None:
        return float(expit(gap))
    logit_prior = np.log(prior_cp) - np.log(1.0 - prior_cp)
    return float(expit(gap + logit_prior))

if NUMBA_OK:
    @njit(fastmath=True)
    def _sym_njit(A):
        A = 0.5 * (A + A.T)
        return A + jitter * np.eye(A.shape[0])

    @njit(fastmath=True)
    def _logdet_psd_njit(A):
        L = np.linalg.cholesky(_sym_njit(A))
        s = 0.0
        n = L.shape[0]
        for i in range(n):
            s += np.log(L[i, i])
        return 2.0 * s

    @njit(fastmath=True)
    def _inv_pd_njit(A):
        return np.linalg.inv(_sym_njit(A))

    @njit(fastmath=True)
    def _kalman_prefix_stats_njit(y, F, Q, H, R, P0):
        y = y.reshape(-1)
        n = y.shape[0]; nx = F.shape[0]
        P = P0.copy()
        x = np.zeros((nx,), dtype=np.float64)
        ll = np.empty(n, dtype=np.float64)
        m_pred = np.empty((n, nx), dtype=np.float64)
        P_pred = np.empty((n, nx, nx), dtype=np.float64)
        I = np.eye(nx)

        for t in range(n):
            m_pred[t] = x
            P_pred[t] = P
            S = (H @ P @ H.T + R)[0,0]
            if S < jitter:
                S = jitter
            K = (P @ H.T) / S
            K = K.reshape(nx)
            innov = y[t] - (H @ x)[0]
            ll[t] = -0.5 * (np.log(2*np.pi*S) + (innov*innov)/S)
            x = x + K * innov
            # NEW (fast):  P ← P − K (H P)
            HP = (H @ P).ravel()             # [n]
            P  = P - np.outer(K, HP)         # rank-1 update
            P  = _sym_njit(P)                # keep PSD & symmetric
            x = F @ x
            P = _sym_njit(F @ P @ F.T + Q)
        for i in range(1, n):
            ll[i] = ll[i-1] + ll[i]
        return ll, m_pred, P_pred

    @njit(fastmath=True)
    def _suffix_messages_backward_njit(y, F, Q, H, R):
        y = y.reshape(-1)
        L = y.shape[0]; n = F.shape[0]
        invR = 1.0 / max(R[0,0], jitter); Qi = _inv_pd_njit(Q)
        J = np.zeros((L, n, n), dtype=np.float64)
        h = np.zeros((L, n), dtype=np.float64)
        c = np.zeros((L,), dtype=np.float64)
        J_next = np.zeros((n, n), dtype=np.float64)
        h_next = np.zeros((n,), dtype=np.float64)
        c_next = 0.0
        for t in range(L-1, -1, -1):
            Lmb = _sym_njit(Qi + J_next)
            Lmb_inv = _inv_pd_njit(Lmb)
            J_pred = _sym_njit(F.T @ (Qi - Qi @ Lmb_inv @ Qi) @ F)
            h_pred = F.T @ Qi @ Lmb_inv @ h_next
            c_pred = (c_next - 0.5*_logdet_psd_njit(Q) - 0.5*_logdet_psd_njit(Lmb) + 0.5*h_next.T @ Lmb_inv @ h_next)
            y_t = y[t]
            J_like = (H.T * invR) @ H
            h_like = (H.T * invR).flatten() * y_t
            c_like = -0.5*(y_t*y_t*invR + np.log(2*np.pi*max(R[0,0], jitter)))
            J_t = _sym_njit(J_pred + J_like)
            h_t = h_pred + h_like
            c_t = c_pred + c_like
            J[t] = J_t; h[t] = h_t; c[t] = c_t
            J_next = J_t; h_next = h_t; c_next = c_t
        return J, h, c

    @njit(fastmath=True)
    def _suffix_ll_from_messages_njit(J, h, c, P0):
        P0i = _inv_pd_njit(P0); logdetP0 = _logdet_psd_njit(P0)
        L = J.shape[0]
        ll_suff = np.empty(L, dtype=np.float64)
        for k in range(L):
            Sk = _sym_njit(P0i + J[k]); Sk_inv = _inv_pd_njit(Sk)
            ll_suff[k] = c[k] - 0.5*logdetP0 - 0.5*_logdet_psd_njit(Sk) + 0.5*h[k].T @ Sk_inv @ h[k]
        return ll_suff

def kalman_prefix_stats(y, F, Q, H, R, *, stable_dim_if_needed=2, bias_var=1e6):
    y = np.asarray(y, float).reshape(-1)
    n = len(y); nx = F.shape[0]
    P = _prior_P0(F, Q, stable_dim=stable_dim_if_needed, bias_var=bias_var)
    x = np.zeros((nx,), dtype=float)
    ll = np.empty(n, dtype=float)
    m_pred = np.empty((n, nx), dtype=float)
    P_pred = np.empty((n, nx, nx), dtype=float)
    I = np.eye(nx)
    for t in range(n):
        m_pred[t] = x; P_pred[t] = P
        S = (H @ P @ H.T + R).item(); S = float(max(S, jitter))
        K = (P @ H.T / S).reshape(nx)
        innov = y[t] - (H @ x).item()
        ll[t] = -0.5 * (np.log(2*np.pi*S) + (innov**2)/S)
        x = x + K*innov
        P = _sym((I - K[:,None] @ H) @ P)
        x = F @ x
        P = _sym(F @ P @ F.T + Q)
    return np.cumsum(ll), m_pred, P_pred

def suffix_messages_backward(y, F, Q, H, R):
    y = np.asarray(y, float).reshape(-1)
    L = len(y); n = F.shape[0]
    invR = 1.0 / max(R.item(), jitter); Qi = _inv_pd(Q)
    J = np.zeros((L, n, n), dtype=float)
    h = np.zeros((L, n), dtype=float)
    c = np.zeros((L,), dtype=float)
    J_next = np.zeros((n, n), dtype=float); h_next = np.zeros((n,), dtype=float); c_next = 0.0
    for t in range(L-1, -1, -1):
        Lmb = _sym(Qi + J_next); Lmb_inv = _inv_pd(Lmb)
        J_pred = _sym(F.T @ (Qi - Qi @ Lmb_inv @ Qi) @ F)
        h_pred = F.T @ Qi @ Lmb_inv @ h_next
        c_pred = (c_next - 0.5*_logdet_psd(Q) - 0.5*_logdet_psd(Lmb) + 0.5*h_next.T @ Lmb_inv @ h_next)
        y_t = y[t]
        J_like = (H.T * invR) @ H
        h_like = (H.T * invR).flatten() * y_t
        c_like = -0.5*(y_t*y_t*invR + np.log(2*np.pi*max(R.item(), jitter)))
        J_t = _sym(J_pred + J_like); h_t = h_pred + h_like; c_t = c_pred + c_like
        J[t], h[t], c[t] = J_t, h_t, c_t
        J_next, h_next, c_next = J_t, h_t, c_t
    return J, h, c

def suffix_ll_from_messages(J, h, c, F, Q, *, stable_dim_if_needed=2, bias_var=1e6):
    P0 = _prior_P0(F, Q, stable_dim=stable_dim_if_needed, bias_var=bias_var)
    P0i = _inv_pd(P0); logdetP0 = _logdet_psd(P0)
    L = J.shape[0]
    ll_suff = np.empty(L, dtype=float)
    for k in range(L):
        Sk = _sym(P0i + J[k]); Sk_inv = _inv_pd(Sk)
        ll_suff[k] = c[k] - 0.5*logdetP0 - 0.5*_logdet_psd(Sk) + 0.5*h[k].T @ Sk_inv @ h[k]
    return ll_suff

def _kalman_prefix_stats_fast(y, F, Q, H, R, *, stable_dim_if_needed=2, bias_var=1e6):
    if NUMBA_OK:
        P0 = _prior_P0(F, Q, stable_dim=stable_dim_if_needed, bias_var=bias_var)
        return _kalman_prefix_stats_njit(y.astype(np.float64), F, Q, H, R, P0)
    else:
        return kalman_prefix_stats(y, F, Q, H, R, stable_dim_if_needed=stable_dim_if_needed, bias_var=bias_var)

def _suffix_ll_fast(y, F, Q, H, R, *, stable_dim_if_needed=2, bias_var=1e6):
    if NUMBA_OK:
        J, h, c = _suffix_messages_backward_njit(y.astype(np.float64), F, Q, H, R)
        P0 = _prior_P0(F, Q, stable_dim=stable_dim_if_needed, bias_var=bias_var)
        return _suffix_ll_from_messages_njit(J, h, c, P0)
    else:
        J, h, c = suffix_messages_backward(y, F, Q, H, R)
        return suffix_ll_from_messages(J, h, c, F, Q, stable_dim_if_needed=stable_dim_if_needed, bias_var=bias_var)


_SSM_CACHE = {}

def _get_ssm_cached(fam, ell, s2, noise_var, dt, *, q_const, q_matern_const, mean_mode="bias"):
    key = (fam, float(ell), float(s2), float(noise_var), float(dt), float(q_const), float(q_matern_const), mean_mode)
    hit = _SSM_CACHE.get(key)
    if hit is not None:
        return hit
    F,Q,H,R = build_ssm(fam, ell, s2, noise_var, dt,
                        mean_mode=mean_mode, q_const=q_const, q_matern_const=q_matern_const)
    stable_dim = (1 if fam == "m12" else 2)
    F = np.ascontiguousarray(F); Q = np.ascontiguousarray(Q)
    H = np.ascontiguousarray(H); R = np.ascontiguousarray(R)
    _SSM_CACHE[key] = (F,Q,H,R,stable_dim)
    return _SSM_CACHE[key]

def _two_theta_fast_hybrid(y, dt, ells, s2s, noise_var, *,
                           families=("m12","m32"), min_seg_frac=0.1,
                           q_const=1e-5, q_matern_const=4.0, bias_prior_var=1e6,
                           k_select="marginal", gap_mode="ml_at_k",
                           prior_cp=None, zscore=True):
    y = np.asarray(y, float).reshape(-1)
    if zscore: y = (y - y.mean())/(y.std()+1e-12)
    L = len(y)
    min_seg = max(1, int(min_seg_frac * L))
    ks = np.arange(min_seg, L - min_seg + 1)

    cfgs = [(fam, float(ell), float(s2)) for fam in families for ell in ells for s2 in s2s]
    S = len(cfgs)

    pref = np.empty((S, L)); stat = np.empty(S)
    for i,(fam,ell,s2) in enumerate(cfgs):
        F,Q,H,R,stable_dim = _get_ssm_cached(fam, ell, s2, noise_var, dt,
                                             q_const=q_const, q_matern_const=q_matern_const, mean_mode="bias")
        ll_pref,_,_ = _kalman_prefix_stats_fast(y, F,Q,H,R,
                                                stable_dim_if_needed=stable_dim, bias_var=bias_prior_var)
        pref[i] = ll_pref; stat[i] = ll_pref[-1]

    suff = np.empty((S, L))
    for j,(fam,ell,s2) in enumerate(cfgs):
        F,Q,H,R,stable_dim = _get_ssm_cached(fam, ell, s2, noise_var, dt,
                                             q_const=q_const, q_matern_const=q_matern_const, mean_mode="bias")
        suff[j] = _suffix_ll_fast(y, F,Q,H,R, stable_dim_if_needed=stable_dim, bias_var=bias_prior_var)

    if k_select == "marginal":
        a = logsumexp(pref[:, ks-1], axis=0) - np.log(S)
        b = logsumexp(suff[:, ks],   axis=0) - np.log(S)
        cp_curve = a + b
        k_idx = int(np.argmax(cp_curve))
    elif k_select == "ml":
        cp_curve = pref[:, ks-1].max(axis=0) + suff[:, ks].max(axis=0)
        k_idx = int(np.argmax(cp_curve))
    else:
        raise ValueError("k_select must be 'marginal' or 'ml'")

    if gap_mode == "ml_at_k":
        best_ll_cp   = float(pref[:, ks[k_idx]-1].max() + suff[:, ks[k_idx]].max())
        best_ll_stat = float(stat.max())
    else:
        ll_cp_k_marg = (logsumexp(pref[:, ks-1], axis=0) - np.log(S)) + (logsumexp(suff[:, ks], axis=0) - np.log(S))
        best_ll_cp   = float(ll_cp_k_marg[k_idx])
        best_ll_stat = float(logsumexp(stat) - np.log(S))

    gap = best_ll_cp - best_ll_stat
    post_k = np.exp(cp_curve - logsumexp(cp_curve))
    prob = _prob_from_gap(gap, prior_cp)

    iL = int(np.argmax(pref[:, ks[k_idx]-1])); iR = int(np.argmax(suff[:, ks[k_idx]]))
    famL, ellL, s2L = cfgs[iL]; famR, ellR, s2R = cfgs[iR]

    return {
        "severity_log": gap,
        "severity_prob": float(prob),
        "location_k": int(ks[k_idx]),
        "location_conf": float(post_k[k_idx]),
        "post_k": post_k,
        "winner_left":  {"family": famL, "ell": ellL, "sigma2": s2L},
        "winner_right": {"family": famR, "ell": ellR, "sigma2": s2R},
    }

def detect_single_cp(
    y, dt, mode="auto", *,
    prior_cp=None,
    min_seg_frac=0.10,
    noise_var=5e-3,
    q_const=1e-5,
    q_matern_const=4.0,
    k_select="marginal",
    gap_mode="ml_at_k",
    zscore=True,
    custom_grids=None,
):
    y = np.asarray(y, float).reshape(-1)
    L = len(y)
    if mode not in {"auto","ou","m32"}:
        raise ValueError("mode must be 'auto', 'ou', or 'm32'")
    if custom_grids is None:
        from .grids import default_grids_for_L_m32, default_grids_for_L_hybrid
        if mode == "m32":
            ells_steps, s2s = default_grids_for_L_m32(L); families = ("m32",)
        elif mode == "ou":
            ells_steps, s2s = default_grids_for_L_hybrid(L); families = ("m12",)
        else:
            ells_steps, s2s = default_grids_for_L_hybrid(L); families = ("m12","m32")
    else:
        ells_steps, s2s = custom_grids
        families = ("m12","m32") if mode=="auto" else (("m12",) if mode=="ou" else ("m32",))

    dt = float(dt)
    ells_time = np.asarray(ells_steps, float) * dt

    t0 = time.perf_counter()
    res = _two_theta_fast_hybrid(
        y, dt, ells_time, s2s, noise_var,
        families=families, min_seg_frac=min_seg_frac,
        q_const=q_const, q_matern_const=q_matern_const,
        prior_cp=prior_cp, k_select=k_select, gap_mode=gap_mode,
        bias_prior_var=1e6, zscore=zscore
    )
    res["meta"] = {
        "mode": mode, "L": L, "noise_var": noise_var, "min_seg_frac": min_seg_frac,
        "q_const": q_const, "q_matern_const": q_matern_const,
        "S_models": len(families)*len(ells_time)*len(s2s),
        "runtime_sec": time.perf_counter()-t0
    }
    return res