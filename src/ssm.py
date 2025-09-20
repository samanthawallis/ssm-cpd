
import numpy as np
import scipy.linalg

jitter = 1e-12

def _sym(A):
    A = 0.5*(A + A.T)
    return A + jitter*np.eye(A.shape[0])

def _logdet_psd(A):
    L = np.linalg.cholesky(_sym(A))
    return 2.0*np.sum(np.log(np.diag(L)))

def _inv_pd(A):
    return np.linalg.inv(_sym(A))

def ou_matern12_ssm(ell, sigma2, noise_var, dt, *, mean_mode="none", q_const=1e-5):
    ell = float(ell); dt = float(dt)
    phi = float(np.exp(-dt/max(ell, 1e-8)))
    Q1  = float(sigma2) * (1.0 - phi**2)
    F1  = np.array([[phi]], dtype=float)
    Qb  = np.array([[Q1]], dtype=float)
    if mean_mode == "bias":
        F = np.block([[F1, np.zeros((1,1))],
                      [np.zeros((1,1)), np.array([[1.0]])]])
        Q = np.block([[Qb, np.zeros((1,1))],
                      [np.zeros((1,1)), np.array([[q_const*dt]])]])
        H = np.array([[1.0, 1.0]], dtype=float)
    else:
        F, Q = F1, Qb
        H = np.array([[1.0]], dtype=float)
    R = np.array([[float(noise_var)]], dtype=float)
    return F, Q, H, R

def matern32_ssm(ell, sigma2, noise_var, dt, *, mean_mode="none", q_const=1e-5, q_matern_const=4.0):
    lam = np.sqrt(3.0) / float(ell)
    F_ct = np.array([[0.0, 1.0],
                     [-(lam**2), -2.0*lam]], dtype=float)
    L_ct = np.array([[0.0],[1.0]], dtype=float)
    q  = float(q_matern_const) * float(sigma2) * (lam**3)
    Qc = L_ct @ L_ct.T * q
    M = np.block([[F_ct, Qc],
                  [np.zeros_like(F_ct), -F_ct.T]]) * float(dt)
    E = scipy.linalg.expm(M)
    F2 = E[:2, :2]
    E12 = E[:2, 2:]
    Q2 = E12 @ F2.T
    if mean_mode == "bias":
        F = np.block([[F2, np.zeros((2,1))],
                      [np.zeros((1,2)), np.array([[1.0]])]])
        Q = np.block([[Q2, np.zeros((2,1))],
                      [np.zeros((1,2)), np.array([[q_const*float(dt)]])]])
        H = np.array([[1.0, 0.0, 1.0]], dtype=float)
    else:
        F, Q = F2, Q2
        H = np.array([[1.0, 0.0]], dtype=float)
    R = np.array([[float(noise_var)]], dtype=float)
    return F, Q, H, R

def build_ssm(family, ell, s2, noise_var, dt, *, mean_mode, q_const, q_matern_const):
    if family == "m12":
        return ou_matern12_ssm(ell, s2, noise_var, dt, mean_mode=mean_mode, q_const=q_const)
    elif family == "m32":
        return matern32_ssm(ell, s2, noise_var, dt, mean_mode=mean_mode,
                            q_const=q_const, q_matern_const=q_matern_const)
    else:
        raise ValueError("family must be 'm12' or 'm32'")
