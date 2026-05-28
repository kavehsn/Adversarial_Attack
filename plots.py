"""
Figure helpers for Section 8. matplotlib only; figures saved as PDF.

Produces:
    fig_sigma_lambda_<chart>.pdf      -- Experiment 1 / 1'
    fig_linearisation_error.pdf       -- Experiment 2
    fig_attackability_curve_<chart>.pdf -- Experiment 3 / 3'
    fig_finite_search.pdf             -- Experiment 2'
"""

import os

import matplotlib.pyplot as plt
import numpy as np


def _ensure_dir(path):
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)


def plot_sigma_vs_lambda(result, save_path):
    """Scatter of Sigma_w(x) vs lambda*(x), with the line Sigma = lambda."""
    _ensure_dir(save_path)
    lams = result["lambda_star"]
    sigs = result["sigma_w"]
    slack = result["slack_ratio"]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    lim = max(float(lams.max()), float(sigs.max())) * 1.05
    axes[0].scatter(lams, sigs, s=10, alpha=0.5)
    axes[0].plot([0, lim], [0, lim], "k--", lw=1, label=r"$\Sigma_w = \lambda^*$")
    axes[0].set_xlabel(r"$\lambda^*(x)$")
    axes[0].set_ylabel(r"$\Sigma_w(x)$")
    axes[0].set_xlim(0, lim)
    axes[0].set_ylim(0, lim)
    axes[0].legend(loc="upper left")
    axes[0].set_title(r"Theorem 3.3: $\Sigma_w(x) \leq \lambda^*(x)$")

    slack_clean = slack[np.isfinite(slack) & (slack > 0)]
    if slack_clean.size:
        axes[1].hist(np.log10(slack_clean), bins=40)
    axes[1].set_xlabel(r"$\log_{10}(\lambda^*(x) / \Sigma_w(x))$")
    axes[1].set_ylabel("count")
    axes[1].set_title("Slack-ratio distribution")

    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def plot_linearisation_error(result, save_path):
    """Mean residual |Delta s^nl - Delta s^lin| as a function of eta, log-log."""
    _ensure_dir(save_path)
    eta = result["eta"]
    mean_res = result["mean_residual"]
    median_res = result["median_residual"]

    fig, ax = plt.subplots(figsize=(5.5, 4))
    pos = mean_res > 0
    ax.loglog(eta[pos], mean_res[pos], "o-", lw=1.5, label="mean residual")
    pos_m = median_res > 0
    ax.loglog(eta[pos_m], median_res[pos_m], "s-", lw=1.5, label="median residual")

    # Slope-2 reference at the smallest eta with a positive mean.
    if pos.any():
        first = np.argmax(pos)
        ref = mean_res[first] * (eta / eta[first]) ** 2
        ax.loglog(eta, ref, "k--", lw=1, label=r"slope 2 ($O(\eta^2)$)")

    ax.set_xlabel(r"$\eta$")
    ax.set_ylabel(r"$|\Delta s^{\mathrm{nl}}(\eta;x) - \Delta s^{\mathrm{lin}}(\eta;x)|$")
    ax.legend()
    ax.set_title("Linearisation error (Remark 3.2)")
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def plot_attackability_curve(result, save_path):
    """Empirical curve + DKW band + Theorem 4.2 overlays."""
    _ensure_dir(save_path)
    eta = result["eta"]
    A_hat = result["A_hat"]
    h = result["dkw_half_width"]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(eta, A_hat, "b-", lw=2, label=r"$\hat A_{n,w}(\eta)$")
    ax.fill_between(
        eta,
        np.clip(A_hat - h, 0.0, 1.0),
        np.clip(A_hat + h, 0.0, 1.0),
        alpha=0.2,
        color="b",
        label=f"DKW band ($\\pm {h:.3f}$)",
    )

    for beta, bnd in sorted(result["thm4_2_bounds"].items()):
        ax.plot(eta, np.clip(bnd, 0.0, 1.0), "--", lw=1.2,
                label=fr"Thm 4.2, $\beta = {beta:.2f}$")

    ax.set_xlabel(r"$\eta$")
    ax.set_ylabel("attackability")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower right")
    ax.set_title("Empirical attackability curve (Theorems 4.2, 4.3)")
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def plot_finite_search(curves_by_K, eta_grid, save_path,
                       A_hat=None, dkw_half=None):
    """Finite-search attackability vs K, optionally overlaid with linearised."""
    _ensure_dir(save_path)
    fig, ax = plt.subplots(figsize=(6, 4))

    for K, vals in sorted(curves_by_K.items()):
        ax.plot(eta_grid, vals, "-", lw=1.5, label=f"K = {K}")

    if A_hat is not None:
        ax.plot(eta_grid, A_hat, "k-", lw=2, label=r"$\hat A_{n,w}(\eta)$ (linearised)")
        if dkw_half is not None:
            ax.fill_between(
                eta_grid,
                np.clip(A_hat - dkw_half, 0.0, 1.0),
                np.clip(A_hat + dkw_half, 0.0, 1.0),
                alpha=0.15, color="k",
            )

    ax.set_xlabel(r"$\eta$")
    ax.set_ylabel(r"$A^{\mathrm{fin}}_w(\eta; \mathcal{P}_K)$")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower right")
    ax.set_title("Finite-search attackability vs K (Proposition 7.3)")
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
