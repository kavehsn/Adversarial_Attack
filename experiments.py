"""
The experiments of Section 8.

Each function takes a list of per-input records and returns the aggregated
diagnostics needed for the corresponding figure. Record schema:

    {
        "e_M_x":   np.ndarray (d_M,)         -- target embedding of x
        "J_M":     np.ndarray (d_M, q)       -- target Jacobian in the chosen chart
        "J_P":     np.ndarray (d_P, q)       -- proxy Jacobian in the chosen chart
        "w":       np.ndarray (d_M,)         -- logit-gap readout direction
        "b":       float                     -- logit-gap readout bias
        # For Experiment 2 in the soft-token chart, the record additionally
        # stores enough state to apply u* through the nonlinear E_M:
        "perturb_fn": callable(u: np.ndarray) -> e_M_perturbed (np.ndarray (d_M,))
        # For Experiment 2' in the cloud chart, the record stores the
        # paraphrase displacements:
        "D_M":     np.ndarray (d_M, K)
        "D_P":     np.ndarray (d_P, K)
    }
"""

from typing import Callable, Iterable, List

import numpy as np

from geometry import (
    pullback_matrices,
    lambda_star,
    sigma_w,
    gamma_w,
    Z_w,
    linearised_worst_direction,
)


# ---------------------------------------------------------------------------
# Experiment 1 (and 1'): Sigma_w(x) <= lambda*(x)
# ---------------------------------------------------------------------------

def experiment_1_sigma_lambda(records: List[dict], rho_frac: float = 1e-3):
    """Scatter Sigma_w vs lambda* across inputs. Histogram of slack ratio."""
    lambdas, sigmas = [], []
    for r in records:
        A, B = pullback_matrices(r["J_M"], r["J_P"])
        l, _ = lambda_star(A, B, rho_frac)
        s, _ = sigma_w(r["J_M"], B, r["w"], rho_frac)
        lambdas.append(l)
        sigmas.append(s)
    lambdas = np.array(lambdas)
    sigmas = np.array(sigmas)
    with np.errstate(divide="ignore", invalid="ignore"):
        slack = np.where(sigmas > 0, lambdas / sigmas, np.nan)
    return {
        "lambda_star": lambdas,
        "sigma_w": sigmas,
        "slack_ratio": slack,
        "violations": int(np.sum(sigmas > lambdas + 1e-9)),
    }


# ---------------------------------------------------------------------------
# Experiment 2: linearisation error r_M(eta)  (soft-token chart only)
# ---------------------------------------------------------------------------

def experiment_2_linearisation_error(
    records: List[dict],
    eta_grid: np.ndarray,
    rho_frac: float = 1e-3,
):
    """
    For each record, evaluate the linearised worst direction u*_flip(eta)
    through the nonlinear E_M (via record["perturb_fn"]) and compare to the
    linearised prediction. Returns the mean residual as a function of eta.
    """
    residuals = np.full((len(records), len(eta_grid)), np.nan)

    for i, r in enumerate(records):
        _, B = pullback_matrices(r["J_M"], r["J_P"])
        s0 = float(np.dot(r["w"], r["e_M_x"]) + r["b"])
        sign_s0 = float(np.sign(s0)) if s0 != 0.0 else 1.0

        for j, eta in enumerate(eta_grid):
            u_star, sigma = linearised_worst_direction(
                r["J_M"], B, r["w"], r["b"], r["e_M_x"], float(eta), rho_frac,
            )
            if u_star is None:
                continue

            e_M_perturbed = r["perturb_fn"](u_star)
            delta_s_nl = float(np.dot(r["w"], e_M_perturbed - r["e_M_x"]))
            delta_s_lin = -sign_s0 * float(eta) * np.sqrt(sigma)
            residuals[i, j] = abs(delta_s_nl - delta_s_lin)

    return {
        "eta": eta_grid,
        "residuals": residuals,
        "mean_residual": np.nanmean(residuals, axis=0),
        "median_residual": np.nanmedian(residuals, axis=0),
    }


# ---------------------------------------------------------------------------
# Experiment 3 (and 3'): empirical attackability curve with DKW band
#                         and Theorem 4.2 quantile overlay.
# ---------------------------------------------------------------------------

def experiment_3_attackability_curve(
    records: List[dict],
    eta_grid: np.ndarray,
    delta: float = 0.05,
    rho_frac: float = 1e-3,
    beta_levels: Iterable[float] = (0.10, 0.25),
):
    """
    Empirical strict left-CDF of Z_w, with DKW band and Theorem 4.2 overlay.
    """
    Z_vals, lambda_vals, gamma_vals = [], [], []
    for r in records:
        A, B = pullback_matrices(r["J_M"], r["J_P"])
        Z, _ = Z_w(r["e_M_x"], r["J_M"], B, r["w"], r["b"], rho_frac)
        l, _ = lambda_star(A, B, rho_frac)
        g = gamma_w(r["e_M_x"], r["w"], r["b"])
        Z_vals.append(Z)
        lambda_vals.append(l)
        gamma_vals.append(g)

    Z_vals = np.array(Z_vals)
    lambda_vals = np.array(lambda_vals)
    gamma_vals = np.array(gamma_vals)
    n = len(records)

    # Strict empirical left-CDF (matches the strict inequality in Definition 4.1).
    A_hat = np.array([float((Z_vals < eta).mean()) for eta in eta_grid])

    # DKW half-width at level 1 - delta.
    dkw_half = float(np.sqrt(np.log(2.0 / delta) / (2.0 * n)))

    # Theorem 4.2 quantile bound, with the population quantile replaced by its
    # empirical version: pick Lambda so that P[lambda* > Lambda] ~ beta.
    thm4_2_bounds = {}
    for beta in beta_levels:
        Lambda_q = float(np.quantile(lambda_vals, 1.0 - beta))
        thm4_2_bounds[beta] = np.array([
            float((gamma_vals < eta * np.sqrt(Lambda_q)).mean()) + beta
            for eta in eta_grid
        ])

    return {
        "eta": eta_grid,
        "A_hat": A_hat,
        "dkw_half_width": dkw_half,
        "thm4_2_bounds": thm4_2_bounds,
        "Z_vals": Z_vals,
        "lambda_vals": lambda_vals,
        "gamma_vals": gamma_vals,
    }


# ---------------------------------------------------------------------------
# Experiment 2': finite-search attackability vs K (cloud chart)
# ---------------------------------------------------------------------------

def experiment_2prime_finite_search(
    records: List[dict],
    K_grid: Iterable[int],
    eta_grid: np.ndarray,
    seed: int = 0,
):
    """
    For each input and each K, evaluate the finite-search displacement
    D^fin_{K, eta}(x; w, b) = max{0} U { -sign(s0) w^T (e_M(x'_k) - e_M(x)) :
                                          x'_k in P_K(x) with d_P(x, x'_k) <= eta }
    and check whether D^fin >= gamma_w(x). The result averaged over inputs is
    the finite-search attackability A^fin_w(eta; P_K).

    Subsampling: for K < K_full, candidates are drawn without replacement.
    """
    K_grid = list(K_grid)
    rng = np.random.default_rng(seed)
    results = {K: np.zeros(len(eta_grid)) for K in K_grid}

    for r in records:
        D_M, D_P = r["D_M"], r["D_P"]            # (d_M, K_full), (d_P, K_full)
        w, b = r["w"], r["b"]
        e_M_x = r["e_M_x"]
        gamma = gamma_w(e_M_x, w, b)
        s0 = float(np.dot(w, e_M_x) + b)
        sign_s0 = float(np.sign(s0)) if s0 != 0.0 else 1.0

        proxy_dists = np.linalg.norm(D_P, axis=0)    # (K_full,)
        readout_disp = -sign_s0 * (w @ D_M)           # (K_full,)

        K_full = D_M.shape[1]

        for K in K_grid:
            if K >= K_full:
                idx = np.arange(K_full)
            else:
                idx = rng.choice(K_full, size=K, replace=False)
            pd_sub = proxy_dists[idx]
            rd_sub = readout_disp[idx]

            for j, eta in enumerate(eta_grid):
                valid = pd_sub <= eta
                if valid.any():
                    D_fin = max(0.0, float(rd_sub[valid].max()))
                else:
                    D_fin = 0.0
                if D_fin >= gamma:
                    results[K][j] += 1

    n = len(records)
    return {K: v / n for K, v in results.items()}
