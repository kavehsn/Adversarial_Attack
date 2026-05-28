"""
Local geometry of the attackability framework (Sections 2-3 of the paper).

All functions operate on numpy arrays; conversion from torch tensors is left
to the caller. The generalised eigenproblem (A, B + rho I) is solved via the
symmetric-definite path in scipy.linalg.

Quantities computed:

    A = J_M^T J_M, B = J_P^T J_P                            (pullback metrics)
    lambda*(x)     = top generalised eigenvalue of (A, B + rho I)
    Sigma_w(x)     = w^T J_M (B + rho I)^{-1} J_M^T w        (readout sensitivity)
    gamma_w(x)     = | w^T e_M(x) + b |                      (geometric margin)
    Z_w(x)         = gamma_w(x) / sqrt(Sigma_w(x))          (adjusted margin)

    u*_flip(eta;x) = -sign(s0) * eta * (B + rho I)^{-1} J_M^T w / sqrt(Sigma_w(x))

All theorems referenced are in the manuscript:
    Theorem 3.1   linearised flip condition
    Theorem 3.3   Sigma_w(x) <= lambda*(x)
    Theorem 4.2   margin-tail attackability bound
    Theorem 4.3   DKW concentration of the empirical curve
"""

from typing import Optional, Tuple

import numpy as np
import scipy.linalg


def _rho_value(B: np.ndarray, rho_frac: float) -> float:
    """rho = rho_frac * tr(B) / q  --  the regularisation scheme of Section 2.1."""
    q = B.shape[0]
    return float(rho_frac * np.trace(B) / q)


def pullback_matrices(J_M: np.ndarray, J_P: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """A = J_M^T J_M, B = J_P^T J_P. Both q x q."""
    A = J_M.T @ J_M
    B = J_P.T @ J_P
    # Symmetrise to suppress floating-point asymmetry.
    A = 0.5 * (A + A.T)
    B = 0.5 * (B + B.T)
    return A, B


def lambda_star(A: np.ndarray, B: np.ndarray, rho_frac: float = 1e-3) -> Tuple[float, float]:
    """
    Top generalised eigenvalue of (A, B + rho I).
    Returns (lambda_max, rho).
    """
    q = A.shape[0]
    rho = _rho_value(B, rho_frac)
    B_reg = B + rho * np.eye(q)
    eigvals = scipy.linalg.eigvalsh(A, B_reg)
    return float(eigvals[-1]), rho


def sigma_w(J_M: np.ndarray, B: np.ndarray, w: np.ndarray, rho_frac: float = 1e-3) -> Tuple[float, float]:
    """
    Sigma_w(x) = w^T J_M (B + rho I)^{-1} J_M^T w.
    Returns (sigma, rho).
    """
    q = B.shape[0]
    rho = _rho_value(B, rho_frac)
    B_reg = B + rho * np.eye(q)
    a = J_M.T @ w                                  # (q,)
    y = scipy.linalg.solve(B_reg, a, assume_a="pos")
    return float(a @ y), rho


def gamma_w(e_M_x: np.ndarray, w: np.ndarray, b: float) -> float:
    """Geometric margin |w^T e_M(x) + b|."""
    return float(abs(np.dot(w, e_M_x) + b))


def Z_w(
    e_M_x: np.ndarray,
    J_M: np.ndarray,
    B: np.ndarray,
    w: np.ndarray,
    b: float,
    rho_frac: float = 1e-3,
) -> Tuple[float, float]:
    """
    Attackability-adjusted margin Z_w(x) = gamma_w(x) / sqrt(Sigma_w(x)).
    Returns (Z, sigma).
    """
    sigma, _ = sigma_w(J_M, B, w, rho_frac)
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
    rho_frac: float = 1e-3,
) -> Tuple[Optional[np.ndarray], float]:
    """
    Closed-form u*_flip(eta) from Theorem 3.1.

    Returns (u_star, sigma); u_star is None if Sigma_w(x) = 0.
    """
    q = B.shape[0]
    rho = _rho_value(B, rho_frac)
    B_reg = B + rho * np.eye(q)

    s0 = float(np.dot(w, e_M_x) + b)
    sign_s0 = float(np.sign(s0)) if s0 != 0.0 else 1.0

    a = J_M.T @ w
    y = scipy.linalg.solve(B_reg, a, assume_a="pos")
    sigma = float(a @ y)
    if sigma <= 0:
        return None, sigma

    u_star = -sign_s0 * eta * y / np.sqrt(sigma)
    return u_star, sigma
