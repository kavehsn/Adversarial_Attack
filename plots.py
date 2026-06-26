"""
Matplotlib only; figures are saved as vector PDFs.  The functions deliberately
avoid set_title(), because the explanatory text belongs in the LaTeX captions.

Produces:
    fig_sigma_lambda_<chart>.pdf        
    fig_linearisation_error.pdf         
    fig_attackability_curve_<chart>.pdf 
    fig_finite_search.pdf               
    fig_predictive_validity.pdf         
"""

from __future__ import annotations

import os
from typing import Mapping, Optional

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import AutoMinorLocator, LogLocator, NullFormatter


# ---------------------------------------------------------------------------
# Global publication style
# ---------------------------------------------------------------------------


def _set_publication_style() -> None:
    """Use compact, journal-friendly defaults and embed editable vector fonts."""
    mpl.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "axes.labelsize": 10,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "legend.fontsize": 8.2,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "xtick.minor.width": 0.6,
            "ytick.minor.width": 0.6,
            "xtick.major.size": 3.5,
            "ytick.major.size": 3.5,
            "xtick.minor.size": 2.2,
            "ytick.minor.size": 2.2,
            "legend.frameon": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "lines.linewidth": 1.45,
            "lines.markersize": 4.0,
        }
    )


_set_publication_style()


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _ensure_dir(path: str) -> None:
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)


def _finite_array(x) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    return arr[np.isfinite(arr)]


def _format_linear_axis(ax: plt.Axes, *, minor: bool = True) -> None:
    ax.tick_params(axis="both", which="both", direction="out")
    if minor:
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.grid(True, which="major", linewidth=0.35, alpha=0.25)


def _format_log_axis(ax: plt.Axes) -> None:
    ax.tick_params(axis="both", which="both", direction="out")
    ax.xaxis.set_major_locator(LogLocator(base=10.0))
    ax.yaxis.set_major_locator(LogLocator(base=10.0))
    ax.xaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1))
    ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.grid(True, which="major", linewidth=0.35, alpha=0.25)
    ax.grid(True, which="minor", linewidth=0.25, alpha=0.10)


def _panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        0.015,
        0.985,
        label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        fontweight="bold",
    )


def _legend(ax: plt.Axes, **kwargs) -> None:
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handlelength=2.2, borderaxespad=0.3, labelspacing=0.35, **kwargs)


def _save(fig: plt.Figure, save_path: str) -> None:
    _ensure_dir(save_path)
    fig.savefig(save_path)
    plt.close(fig)


def _positive_limits(*arrays, pad_low: float = 0.75, pad_high: float = 1.35) -> tuple[float, float]:
    vals = []
    for arr in arrays:
        a = np.asarray(arr, dtype=float)
        vals.append(a[np.isfinite(a) & (a > 0)])
    vals = [v for v in vals if v.size]
    if not vals:
        return 1e-6, 1.0
    merged = np.concatenate(vals)
    lo = float(merged.min()) * pad_low
    hi = float(merged.max()) * pad_high
    if not np.isfinite(lo) or not np.isfinite(hi) or lo <= 0 or hi <= lo:
        return 1e-6, 1.0
    return lo, hi


# ---------------------------------------------------------------------------
#  Sigma_w(x) <= lambda*(x)
# ---------------------------------------------------------------------------


def plot_sigma_vs_lambda(result: Mapping, save_path: str) -> None:
    """Scatter of Sigma_w(x) versus lambda*(x), plus slack-ratio distribution."""
    lams = np.asarray(result["lambda_star"], dtype=float)
    sigs = np.asarray(result["sigma_w"], dtype=float)
    slack = np.asarray(result["slack_ratio"], dtype=float)

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.15), constrained_layout=True)

    pos = np.isfinite(lams) & np.isfinite(sigs) & (lams > 0) & (sigs > 0)
    ax = axes[0]
    if pos.any():
        lo, hi = _positive_limits(lams[pos], sigs[pos], pad_low=0.65, pad_high=1.55)
        ax.scatter(
            lams[pos],
            sigs[pos],
            s=12,
            alpha=0.55,
            linewidths=0.0,
            rasterized=True,
            label="inputs",
        )
        ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1.0, color="0.15",
                label=r"$\Sigma_w=\lambda^*$")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$\lambda^*(x)$")
    ax.set_ylabel(r"$\Sigma_w(x)$")
    _format_log_axis(ax)
    _panel_label(ax, "(a)")
    _legend(ax, loc="upper left")

    ax = axes[1]
    slack_clean = slack[np.isfinite(slack) & (slack > 0)]
    if slack_clean.size:
        z = np.log10(slack_clean)
        bins = min(32, max(10, int(np.sqrt(z.size))))
        ax.hist(z, bins=bins, histtype="stepfilled", alpha=0.65, edgecolor="0.2", linewidth=0.6)
        med = float(np.median(z))
        ax.axvline(med, linestyle="--", linewidth=1.0, color="0.15", label="median")
        _legend(ax, loc="upper right")
    ax.set_xlabel(r"$\log_{10}\{\lambda^*(x)/\Sigma_w(x)\}$")
    ax.set_ylabel("Frequency")
    _format_linear_axis(ax)
    _panel_label(ax, "(b)")

    _save(fig, save_path)


# ---------------------------------------------------------------------------
# local linearisation error
# ---------------------------------------------------------------------------


def plot_linearisation_error(result: Mapping, save_path: str) -> None:
    """
    Two-panel linearisation diagnostic.

    Panel (a): mean absolute nonlinear and linear readout displacements.
    Panel (b): residual normalised by eta^2 / beta_i.
    """
    eta = np.asarray(result["eta"], dtype=float)
    mean_nl = np.asarray(result["mean_abs_nl"], dtype=float)
    mean_lin = np.asarray(result["mean_abs_lin"], dtype=float)
    mean_nr = np.asarray(result["mean_norm_residual"], dtype=float)
    median_nr = np.asarray(result["median_norm_residual"], dtype=float)

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.15), constrained_layout=True)

    ax = axes[0]
    a_lin = np.isfinite(eta) & np.isfinite(mean_lin) & (eta > 0) & (mean_lin > 0)
    a_nl = np.isfinite(eta) & np.isfinite(mean_nl) & (eta > 0) & (mean_nl > 0)
    if a_lin.any():
        ax.loglog(eta[a_lin], mean_lin[a_lin], marker="s", markevery=max(1, a_lin.sum() // 8),
                  label=r"linear $|\Delta s|$")
    if a_nl.any():
        ax.loglog(eta[a_nl], mean_nl[a_nl], marker="o", markevery=max(1, a_nl.sum() // 8),
                  label=r"nonlinear $|\Delta s|$")
    if a_lin.any():
        first = np.flatnonzero(a_lin)[0]
        ref = mean_lin[first] * (eta / eta[first])
        ok = np.isfinite(ref) & (eta > 0) & (ref > 0)
        ax.loglog(eta[ok], ref[ok], linestyle=":", linewidth=1.0, color="0.15",
                  label="slope 1")
    ax.set_xlabel(r"proxy budget $\eta$")
    ax.set_ylabel("Mean readout displacement")
    _format_log_axis(ax)
    _panel_label(ax, "(a)")
    _legend(ax, loc="upper left")

    ax = axes[1]
    b_mean = np.isfinite(eta) & np.isfinite(mean_nr) & (eta > 0) & (mean_nr > 0)
    b_med = np.isfinite(eta) & np.isfinite(median_nr) & (eta > 0) & (median_nr > 0)
    if b_mean.any():
        ax.loglog(eta[b_mean], mean_nr[b_mean], marker="o", markevery=max(1, b_mean.sum() // 8),
                  label="mean")
    if b_med.any():
        ax.loglog(eta[b_med], median_nr[b_med], marker="s", markevery=max(1, b_med.sum() // 8),
                  label="median")
        idx = np.flatnonzero(b_med)
        m = max(1, len(idx) // 3)
        flat = float(np.nanmedian(median_nr[idx[:m]]))
        if np.isfinite(flat) and flat > 0:
            ax.axhline(flat, linestyle=":", linewidth=1.0, color="0.15", label="local reference")
    ax.set_xlabel(r"proxy budget $\eta$")
    ax.set_ylabel(r"Residual normalised by $\eta^2/\beta_i$")
    _format_log_axis(ax)
    _panel_label(ax, "(b)")
    _legend(ax, loc="best")

    _save(fig, save_path)


# ---------------------------------------------------------------------------
#  attackability curve
# ---------------------------------------------------------------------------


def plot_attackability_curve(result: Mapping, save_path: str) -> None:
    """Empirical attackability curve, DKW band, and empirical inclusion overlays."""
    eta = np.asarray(result["eta"], dtype=float)
    A_hat = np.asarray(result["A_hat"], dtype=float)
    h = float(result["dkw_half_width"])

    fig, ax = plt.subplots(figsize=(3.75, 3.05), constrained_layout=True)

    ax.plot(eta, A_hat, linewidth=1.9, label=r"$\widehat A_{n,w}(\eta)$")
    ax.fill_between(
        eta,
        np.clip(A_hat - h, 0.0, 1.0),
        np.clip(A_hat + h, 0.0, 1.0),
        alpha=0.18,
        linewidth=0.0,
        label=fr"DKW band, $\pm {h:.3f}$",
    )

    bnds_lambda = result.get("thm4_2_bounds_lambda", result.get("thm4_2_bounds", {}))
    bnds_sigma = result.get("thm4_2_bounds_sigma", {})

    line_styles = ["--", "-."]
    for j, (beta, bnd) in enumerate(sorted(bnds_sigma.items())):
        ax.plot(
            eta,
            np.clip(np.asarray(bnd, dtype=float), 0.0, 1.0),
            linestyle=line_styles[j % len(line_styles)],
            linewidth=1.25,
            label=fr"$\Sigma_w$ incl., $\beta={beta:.2f}$",
        )
    for j, (beta, bnd) in enumerate(sorted(bnds_lambda.items())):
        ax.plot(
            eta,
            np.clip(np.asarray(bnd, dtype=float), 0.0, 1.0),
            linestyle=":",
            linewidth=1.15,
            alpha=0.85,
            label=fr"$\lambda^*$ incl., $\beta={beta:.2f}$",
        )

    ax.set_xlabel(r"proxy budget $\eta$")
    ax.set_ylabel("Attackability")
    ax.set_xlim(float(np.nanmin(eta)), float(np.nanmax(eta)))
    ax.set_ylim(-0.02, 1.02)
    _format_linear_axis(ax)
    _legend(ax, loc="lower right")

    _save(fig, save_path)


# ---------------------------------------------------------------------------
#  finite-search approximation
# ---------------------------------------------------------------------------


def plot_finite_search(
    curves_by_K: Mapping[int, np.ndarray],
    eta_grid: np.ndarray,
    save_path: str,
    A_hat: Optional[np.ndarray] = None,
    dkw_half: Optional[float] = None,
    coverage_diag: Optional[Mapping] = None,
    oracle_K: Optional[int] = None,
) -> None:
    """Finite-search attackability curves, with optional coverage diagnostic."""
    eta_grid = np.asarray(eta_grid, dtype=float)
    two_panels = coverage_diag is not None
    fig, axes = plt.subplots(
        1,
        2 if two_panels else 1,
        figsize=(7.4 if two_panels else 3.9, 3.15),
        constrained_layout=True,
    )
    ax = axes[0] if two_panels else axes

    items = sorted(curves_by_K.items(), key=lambda kv: kv[0])
    cmap = plt.get_cmap("viridis")
    n_items = max(1, len(items) - 1)
    for j, (K, vals) in enumerate(items):
        vals = np.asarray(vals, dtype=float)
        is_oracle = oracle_K is not None and int(K) == int(oracle_K)
        if is_oracle:
            ax.plot(eta_grid, vals, color="0.05", linewidth=2.15,
                    label=fr"$K={K}$ oracle")
        else:
            ax.plot(eta_grid, vals, linewidth=1.25, color=cmap(j / n_items),
                    label=fr"$K={K}$")

    if A_hat is not None:
        A_hat = np.asarray(A_hat, dtype=float)
        ax.plot(eta_grid, A_hat, color="0.15", linestyle="--", linewidth=1.65,
                label=r"linearised $\widehat A_{n,w}$")
        if dkw_half is not None:
            ax.fill_between(
                eta_grid,
                np.clip(A_hat - float(dkw_half), 0.0, 1.0),
                np.clip(A_hat + float(dkw_half), 0.0, 1.0),
                color="0.3",
                alpha=0.12,
                linewidth=0.0,
            )

    ax.set_xlabel(r"proxy budget $\eta$")
    ax.set_ylabel(r"Finite-search attackability")
    ax.set_xlim(float(np.nanmin(eta_grid)), float(np.nanmax(eta_grid)))
    ax.set_ylim(-0.02, 1.02)
    _format_linear_axis(ax)
    _panel_label(ax, "(a)" if two_panels else "")

    if two_panels:
        axc = axes[1]
        c_eta = np.asarray(coverage_diag["eta"], dtype=float)
        med_gap = np.asarray(coverage_diag["median_norm_gap"], dtype=float)
        axc.plot(c_eta, med_gap, marker="o", linewidth=1.45, label="median gap")
        axc.axhline(1.0, linestyle=":", linewidth=1.0, color="0.15", label="uncovered")
        axc.set_xlabel(r"proxy budget $\eta$")
        axc.set_ylabel(r"Normalised coverage gap along $r_w$")
        axc.set_xlim(float(np.nanmin(c_eta)), float(np.nanmax(c_eta)))
        axc.set_ylim(-0.02, 1.05)
        _format_linear_axis(axc)
        _panel_label(axc, "(b)")
        _legend(axc, loc="lower left")

    # Panel (a)'s K-sweep legend is placed below the figure so it cannot overlap
    # the linearised curve and confidence band; savefig's tight bounding box
    # keeps it from being clipped.
    handles_a, labels_a = ax.get_legend_handles_labels()
    ncol_a = 4 if len(handles_a) >= 6 else max(1, len(handles_a))
    fig.legend(handles_a, labels_a, loc="upper center",
               bbox_to_anchor=(0.5, -0.02), ncol=ncol_a, frameon=False,
               fontsize=8.0, handlelength=2.0, columnspacing=1.3)

    _save(fig, save_path)


# ---------------------------------------------------------------------------
#  predictive validity
# ---------------------------------------------------------------------------


def plot_predictive_validity(result: Mapping, save_path: str) -> None:
    """Predictive validity of Z_w against realised paraphrase flips."""
    fig, axes = plt.subplots(1, 2, figsize=(7.3, 3.15), constrained_layout=True)

    Z = np.asarray(result["Z_vals"], dtype=float)
    rb = np.asarray(result["realised_flip_budget"], dtype=float)
    finite = np.isfinite(Z) & np.isfinite(rb) & (Z > 0) & (rb > 0)

    ax = axes[0]
    if finite.any():
        ax.scatter(Z[finite], rb[finite], s=15, alpha=0.58, linewidths=0.0, rasterized=True)
        lo, hi = _positive_limits(Z[finite], rb[finite], pad_low=0.65, pad_high=1.55)
        ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1.0, color="0.15", label="identity")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
    ax.set_xscale("log")
    ax.set_yscale("log")
    rho = float(result.get("spearman_Z_vs_budget", np.nan))
    n_used = int(result.get("n_used", int(finite.sum())))
    if np.isfinite(rho):
        ax.text(
            0.045,
            0.94,
            fr"Spearman $\rho={rho:.2f}$; $n={n_used}$",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8.4,
        )
    ax.set_xlabel(r"$Z_w(x)$")
    ax.set_ylabel("Smallest realised flip budget")
    _format_log_axis(ax)
    _panel_label(ax, "(a)")
    _legend(ax, loc="lower right")

    ax = axes[1]
    eta = np.asarray(result["eta"], dtype=float)
    auc = np.asarray(result["auc_vs_eta"], dtype=float)
    flip_rate = np.asarray(result["flip_rate_vs_eta"], dtype=float)
    me = max(1, len(eta) // 12)   # thin markers so they do not crowd the line
    ok = np.isfinite(auc)
    if ok.any():
        ax.plot(eta[ok], auc[ok], color="C0", linestyle="-", marker="o",
                markevery=me, markersize=3.8, linewidth=1.7, zorder=6,
                label=r"AUC of $-Z_w$")
    if "auc_gamma_vs_eta" in result:
        ax.plot(eta, np.asarray(result["auc_gamma_vs_eta"], float), color="C1",
                linestyle="--", marker="^", markevery=me, markersize=3.8,
                linewidth=1.3, zorder=4, label=r"AUC of $-\gamma_w$ (margin only)")
        ax.plot(eta, np.asarray(result["auc_sigma_vs_eta"], float), color="C2",
                linestyle=":", marker="D", markevery=me, markersize=3.4,
                linewidth=1.5, zorder=5, label=r"AUC of $\sqrt{\Sigma_w}$ (displ. only)")
    ax.plot(eta, flip_rate, color="C3", linestyle="-", marker="s", markevery=me,
            markersize=3.2, linewidth=1.0, alpha=0.75, zorder=2, label="flip rate")
    ax.axhline(0.5, linestyle=(0, (1, 1)), linewidth=1.0, color="0.45",
               zorder=1, label="no skill")
    auc_label = result.get("auc_at_label_budget", np.nan)
    label_budget = result.get("label_budget", np.nan)
    if np.isfinite(auc_label) and np.isfinite(label_budget):
        ax.text(
            0.97,
            0.60,
            fr"AUC$={float(auc_label):.2f}$ at $\eta={float(label_budget):.3g}$",
            transform=ax.transAxes,
            ha="right",
            va="center",
            fontsize=8.2,
        )
    ax.set_xlabel(r"proxy budget $\eta$")
    ax.set_ylabel("AUC / flip rate")
    ax.set_xlim(float(np.nanmin(eta)), float(np.nanmax(eta)))
    ax.set_ylim(-0.02, 1.02)
    _format_linear_axis(ax)
    _panel_label(ax, "(b)")

    # Panel (b)'s legend is placed below the figure so it never overlaps the
    # curves; savefig's tight bounding box keeps it from being clipped.
    handles_b, labels_b = ax.get_legend_handles_labels()
    fig.legend(handles_b, labels_b, loc="upper center",
               bbox_to_anchor=(0.5, -0.02), ncol=3, frameon=False, fontsize=8.2,
               handlelength=2.0, columnspacing=1.4)

    _save(fig, save_path)