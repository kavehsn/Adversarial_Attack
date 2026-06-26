"""
Local geometry of the attackability framework

All functions operate on numpy arrays; conversion from torch tensors is left
to the caller. The generalised eigenproblem (A, B + nu I) is solved via the
symmetric-definite path in scipy.linalg.

Quantities computed:

    A = J_M^T J_M, B = J_P^T J_P                            (pullback metrics)
    lambda*(x)     = top generalised eigenvalue of (A, B + nu I)
    Sigma_w(x)     = w^T J_M (B + nu I)^{-1} J_M^T w         (readout sensitivity)
    gamma_w(x)     = | w^T e_M(x) + b |                      (geometric margin)
    Z_w(x)         = gamma_w(x) / sqrt(Sigma_w(x))          (adjusted margin)

    u*_flip(eta;x) = -sign(s0) * eta * (B + nu I)^{-1} J_M^T w / sqrt(Sigma_w(x))

Regularisation caveat (revision item 5): lambda* and S are chart-invariant only
for B > 0 (Proposition prop:chart-inv). The computed quantities use B + nu I,
whose nu I term is referred to the coordinate frame, so the regularised pencil
is NOT reparametrisation-invariant; the reported numbers are coordinate-dependent
approximations that recover the intrinsic quantities as nu -> 0 on non-degenerate
inputs. Stability across the swept nu range is what justifies treating them as
the intrinsic quantities (see the experimental setup).

"""

from typing import Optional, Tuple

import numpy as np
import scipy.linalg


def _nu_value(B: np.ndarray, nu_frac: float) -> float:
    """nu = nu_frac * tr(B) / q"""
    q = B.shape[0]
    return float(nu_frac * np.trace(B) / q)


def pullback_matrices(J_M: np.ndarray, J_P: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """A = J_M^T J_M, B = J_P^T J_P. Both q x q."""
    A = J_M.T @ J_M
    B = J_P.T @ J_P
    # Symmetrise to suppress floating-point asymmetry.
    A = 0.5 * (A + A.T)
    B = 0.5 * (B + B.T)
    return A, B


def lambda_star(A: np.ndarray, B: np.ndarray, nu_frac: float = 1e-3) -> Tuple[float, float]:
    """
    Top generalised eigenvalue of (A, B + nu I).
    Returns (lambda_max, nu).
    """
    q = A.shape[0]
    nu = _nu_value(B, nu_frac)
    B_reg = B + nu * np.eye(q)
    eigvals = scipy.linalg.eigvalsh(A, B_reg)
    return float(eigvals[-1]), nu


def trace_S(J_M: np.ndarray, B: np.ndarray, nu_frac: float = 1e-3) -> Tuple[float, float]:
    """tr S(x) = tr(J_M (B+nu I)^{-1} J_M^T) = tr((B+nu I)^{-1} A)"""
    q = B.shape[0]
    nu = _nu_value(B, nu_frac)
    B_reg = B + nu * np.eye(q)
    A = J_M.T @ J_M
    return float(np.trace(scipy.linalg.solve(B_reg, A, assume_a="pos"))), nu

def sigma_w(J_M: np.ndarray, B: np.ndarray, w: np.ndarray, nu_frac: float = 1e-3) -> Tuple[float, float]:
    """
    Sigma_w(x) = w^T J_M (B + nu I)^{-1} J_M^T w.
    Returns (sigma, nu).
    """
    q = B.shape[0]
    nu = _nu_value(B, nu_frac)
    B_reg = B + nu * np.eye(q)
    a = J_M.T @ w                                  # (q,)
    y = scipy.linalg.solve(B_reg, a, assume_a="pos")
    return float(a @ y), nu


def gamma_w(e_M_x: np.ndarray, w: np.ndarray, b: float) -> float:
    """Geometric margin |w^T e_M(x) + b|."""
    return float(abs(np.dot(w, e_M_x) + b))


def beta_min(B: np.ndarray, nu_frac: float = 1e-3) -> float:
    """
    beta = lambda_min(B + nu I), the smallest eigenvalue of the regularised
    proxy metric. This is the 1/beta constant in the finite-radius error
    r_M(eta) = O(eta^2 / beta) to normalise the linearisation residual per input.
    """
    q = B.shape[0]
    nu = _nu_value(B, nu_frac)
    B_reg = B + nu * np.eye(q)
    return float(scipy.linalg.eigvalsh(B_reg)[0])


def Z_w(
    e_M_x: np.ndarray,
    J_M: np.ndarray,
    B: np.ndarray,
    w: np.ndarray,
    b: float,
    nu_frac: float = 1e-3,
) -> Tuple[float, float]:
    """
    Attackability-adjusted margin Z_w(x) = gamma_w(x) / sqrt(Sigma_w(x)).
    Returns (Z, sigma).
    """
    sigma, _ = sigma_w(J_M, B, w, nu_frac)
    if sigma <= 0:
        return float("inf"), sigma
    return gamma_w(e_M_x, w, b) / np.sqrt(sigma), sigma


def linearised_worst_direction(
    J_M: np.ndarray,
    B: np.ndarray,
    w: np.ndarray,
    b: float,
    e_M_x: np.ndarray,
    eta: float,
    nu_frac: float = 1e-3,
) -> Tuple[Optional[np.ndarray], float]:
    """
    Closed-form u*_flip(eta)

    Returns (u_star, sigma); u_star is None if Sigma_w(x) = 0.
    """
    q = B.shape[0]
    nu = _nu_value(B, nu_frac)
    B_reg = B + nu * np.eye(q)

    s0 = float(np.dot(w, e_M_x) + b)
    sign_s0 = float(np.sign(s0)) if s0 != 0.0 else 1.0

    a = J_M.T @ w
    y = scipy.linalg.solve(B_reg, a, assume_a="pos")
    sigma = float(a @ y)
    if sigma <= 0:
        return None, sigma

    u_star = -sign_s0 * eta * y / np.sqrt(sigma)
    return u_star, sigma


def readout_aligned_direction(
    J_M: np.ndarray,
    B: np.ndarray,
    w: np.ndarray,
    b: float,
    e_M_x: np.ndarray,
    nu_frac: float = 1e-3,
) -> Tuple[Optional[np.ndarray], float]:
    """
    Unit (in the B-metric) readout-aligned direction r_w(x):

        r_w(x) = -sign(s0) (B + nu I)^{-1} J_M^T w / sqrt(Sigma_w(x)),
        ||r_w(x)||_B = 1.

    This is the direction along which a one-dimensional cover of [0, eta]
    recovers the continuous worst case; it is the right axis for the
    finite-search coverage diagnostic. Returns
    (r_w, sigma); r_w is None if Sigma_w(x) = 0.
    """
    u_star, sigma = linearised_worst_direction(J_M, B, w, b, e_M_x, 1.0, nu_frac)
    return u_star, sigma  # u*_flip at eta=1 is exactly r_w(x)
