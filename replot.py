"""
Restyle the Section-8 figures from cached records -- NO pipeline rerun.

The expensive run already saved everything the record-derivable figures need:
    <figdir>/soft_records.pkl
    <figdir>/cloud_records.pkl
    <figdir>/run_metadata.json
This script reloads those, recomputes the six diagnostics with the existing
(model-free, numpy/scipy) functions in experiments.py, and redraws them with
publication styling: framed semi-transparent legends placed off the data,
boxed text annotations, larger fonts, and editable vector fonts in the PDF.

It does NOT touch the pipeline, loads no model, and runs in seconds.

Covered (regenerated from cache):
    fig_sigma_lambda_soft.pdf, fig_sigma_lambda_cloud.pdf
    fig_attackability_curve.pdf, fig_attackability_curve_cloud.pdf
    fig_finite_search.pdf
    fig_predictive_validity.pdf

NOT covered: fig_linearisation_error.pdf. Its nonlinear curve needs the model
forward pass (record["perturb_fn"]), which is not cached, so it cannot be
recomputed here. If <figdir>/linearisation_results.npz exists it is restyled;
otherwise the figure is left as-is (see the note printed at the end).

Usage:
    python replot.py                      # reads ./figures, overwrites the PDFs
    python replot.py --figdir figures --outdir figures_pretty
"""

import argparse
import json
import os
import pickle
from typing import Mapping, Optional

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import AutoMinorLocator, LogLocator, NullFormatter

from experiments import (
    experiment_1_sigma_lambda,
    experiment_2prime_finite_search,
    experiment_3_attackability_curve,
    experiment_predictive_validity,
    finite_search_coverage_diagnostic,
    experiment_predictive_validity_crosschart,
)


# ---------------------------------------------------------------------------
# Publication style
# ---------------------------------------------------------------------------

# A compact, colour-blind-safe qualitative palette (Okabe-Ito subset).
C_PRIMARY = "#0072B2"   # blue
C_ACCENT = "#D55E00"    # vermillion
C_GREEN = "#009E73"
C_GREY = "#3a3a3a"
ANNOT_BBOX = dict(boxstyle="round,pad=0.32", facecolor="white",
                  edgecolor="0.75", alpha=0.92, linewidth=0.6)


def _set_style() -> None:
    mpl.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 400,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42,          # editable (TrueType) fonts in the PDF
        "ps.fonttype": 42,
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "axes.labelsize": 11,
        "axes.titlesize": 11,
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
        "legend.fontsize": 8.8,
        "axes.linewidth": 0.9,
        "xtick.major.width": 0.9,
        "ytick.major.width": 0.9,
        "xtick.minor.width": 0.6,
        "ytick.minor.width": 0.6,
        "xtick.major.size": 3.6,
        "ytick.major.size": 3.6,
        "xtick.minor.size": 2.2,
        "ytick.minor.size": 2.2,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "lines.linewidth": 1.7,
        "lines.markersize": 4.0,
        "axes.prop_cycle": mpl.cycler(color=[C_PRIMARY, C_ACCENT, C_GREEN,
                                             "#CC79A7", "#56B4E9", "#E69F00"]),
    })


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _ensure_dir(path: str) -> None:
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)


def _format_linear_axis(ax: plt.Axes) -> None:
    ax.tick_params(axis="both", which="both", direction="out")
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.grid(True, which="major", linewidth=0.4, alpha=0.28)


def _format_log_axis(ax: plt.Axes) -> None:
    ax.tick_params(axis="both", which="both", direction="out")
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_locator(LogLocator(base=10.0))
        axis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1))
        axis.set_minor_formatter(NullFormatter())
    ax.grid(True, which="major", linewidth=0.4, alpha=0.28)
    ax.grid(True, which="minor", linewidth=0.3, alpha=0.10)


def _panel_label(ax: plt.Axes, label: str) -> None:
    if label:
        ax.text(0.022, 0.978, label, transform=ax.transAxes, ha="left", va="top",
                fontsize=10.5, fontweight="bold")


def _legend(ax: plt.Axes, **kwargs) -> None:
    handles, _ = ax.get_legend_handles_labels()
    if not handles:
        return
    leg = ax.legend(handlelength=1.8, borderaxespad=0.4, labelspacing=0.32,
                    handletextpad=0.6, frameon=True, framealpha=0.92,
                    edgecolor="0.78", fancybox=False, **kwargs)
    leg.get_frame().set_linewidth(0.6)


def _save(fig: plt.Figure, path: str) -> None:
    _ensure_dir(path)
    fig.savefig(path)
    plt.close(fig)
    print(f"  wrote {path}")


def _positive_limits(*arrays, pad_low=0.65, pad_high=1.55):
    vals = []
    for arr in arrays:
        a = np.asarray(arr, dtype=float)
        vals.append(a[np.isfinite(a) & (a > 0)])
    vals = [v for v in vals if v.size]
    if not vals:
        return 1e-6, 1.0
    merged = np.concatenate(vals)
    lo, hi = float(merged.min()) * pad_low, float(merged.max()) * pad_high
    if not (np.isfinite(lo) and np.isfinite(hi)) or lo <= 0 or hi <= lo:
        return 1e-6, 1.0
    return lo, hi


# ---------------------------------------------------------------------------
# Figure: Sigma_w vs lambda*
# ---------------------------------------------------------------------------

def plot_sigma_vs_lambda(result: Mapping, save_path: str) -> None:
    lams = np.asarray(result["lambda_star"], dtype=float)
    sigs = np.asarray(result["sigma_w"], dtype=float)
    slack = np.asarray(result["slack_ratio"], dtype=float)

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2), constrained_layout=True)

    ax = axes[0]
    pos = np.isfinite(lams) & np.isfinite(sigs) & (lams > 0) & (sigs > 0)
    if pos.any():
        lo, hi = _positive_limits(lams[pos], sigs[pos])
        ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1.1, color=C_GREY,
                zorder=1, label=r"$\Sigma_w=\lambda^*$")
        ax.scatter(lams[pos], sigs[pos], s=14, alpha=0.55, linewidths=0.0,
                   color=C_PRIMARY, rasterized=True, zorder=2, label="inputs")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$\lambda^*(x)$")
    ax.set_ylabel(r"$\Sigma_w(x)$")
    _format_log_axis(ax)
    _panel_label(ax, "(a)")
    _legend(ax, loc="lower right")

    ax = axes[1]
    z = slack[np.isfinite(slack) & (slack > 0)]
    if z.size:
        z = np.log10(z)
        bins = min(30, max(12, int(np.sqrt(z.size))))
        ax.hist(z, bins=bins, color=C_PRIMARY, alpha=0.78, edgecolor="white",
                linewidth=0.5)
        med = float(np.median(z))
        ax.axvline(med, linestyle="--", linewidth=1.2, color=C_GREY,
                   label=fr"median $={med:.2f}$")
        _legend(ax, loc="upper right")
    ax.set_xlabel(r"$\log_{10}\{\lambda^*(x)/\Sigma_w(x)\}$")
    ax.set_ylabel("Frequency")
    _format_linear_axis(ax)
    _panel_label(ax, "(b)")

    _save(fig, save_path)


# ---------------------------------------------------------------------------
# Figure: attackability curve
# ---------------------------------------------------------------------------

def plot_attackability_curve(result: Mapping, save_path: str) -> None:
    eta = np.asarray(result["eta"], dtype=float)
    A_hat = np.asarray(result["A_hat"], dtype=float)
    h = float(result["dkw_half_width"])

    fig, ax = plt.subplots(figsize=(3.9, 3.15), constrained_layout=True)

    bnds_l = result.get("thm4_2_bounds_lambda", result.get("thm4_2_bounds", {}))
    bnds_s = result.get("thm4_2_bounds_sigma", {})
    sig_styles = ["--", "-."]
    for j, (beta, bnd) in enumerate(sorted(bnds_s.items())):
        ax.plot(eta, np.clip(np.asarray(bnd, float), 0, 1),
                linestyle=sig_styles[j % 2], linewidth=1.35, color=C_ACCENT,
                alpha=0.55 + 0.45 * (j == 0),
                label=fr"$\Sigma_w$ incl., $\beta={beta:.2f}$")
    for j, (beta, bnd) in enumerate(sorted(bnds_l.items())):
        ax.plot(eta, np.clip(np.asarray(bnd, float), 0, 1),
                linestyle=":", linewidth=1.25, color=C_GREEN,
                alpha=0.55 + 0.45 * (j == 0),
                label=fr"$\lambda^*$ incl., $\beta={beta:.2f}$")

    ax.fill_between(eta, np.clip(A_hat - h, 0, 1), np.clip(A_hat + h, 0, 1),
                    color=C_PRIMARY, alpha=0.16, linewidth=0.0,
                    label=fr"DKW band $\pm{h:.3f}$")
    ax.plot(eta, A_hat, linewidth=2.0, color=C_PRIMARY,
            label=r"$\widehat A_{n,w}(\eta)$")

    ax.set_xlabel(r"proxy budget $\eta$")
    ax.set_ylabel("Attackability")
    ax.set_xlim(float(np.nanmin(eta)), float(np.nanmax(eta)))
    ax.set_ylim(-0.02, 1.04)
    _format_linear_axis(ax)
    _legend(ax, loc="lower right")
    _save(fig, save_path)


# ---------------------------------------------------------------------------
# Figure: finite search (+ coverage diagnostic)
# ---------------------------------------------------------------------------

def plot_finite_search(curves_by_K, eta_grid, save_path, A_hat=None,
                       dkw_half=None, coverage_diag=None, oracle_K=None) -> None:
    eta_grid = np.asarray(eta_grid, dtype=float)
    two = coverage_diag is not None
    fig, axes = plt.subplots(1, 2 if two else 1,
                             figsize=(7.4 if two else 4.0, 3.2),
                             constrained_layout=True)
    ax = axes[0] if two else axes

    items = sorted(curves_by_K.items(), key=lambda kv: kv[0])
    non_oracle = [k for k, _ in items if oracle_K is None or int(k) != int(oracle_K)]
    cmap = plt.get_cmap("viridis")
    n = max(1, len(non_oracle) - 1)
    ci = 0
    for K, vals in items:
        vals = np.asarray(vals, dtype=float)
        if oracle_K is not None and int(K) == int(oracle_K):
            ax.plot(eta_grid, vals, color="0.05", linewidth=2.2,
                    label=fr"$K={K}$ (oracle)", zorder=5)
        else:
            ax.plot(eta_grid, vals, linewidth=1.4, color=cmap(0.08 + 0.84 * ci / n),
                    label=fr"$K={K}$")
            ci += 1

    if A_hat is not None:
        A_hat = np.asarray(A_hat, dtype=float)
        if dkw_half is not None:
            ax.fill_between(eta_grid, np.clip(A_hat - float(dkw_half), 0, 1),
                            np.clip(A_hat + float(dkw_half), 0, 1),
                            color="0.4", alpha=0.13, linewidth=0.0)
        ax.plot(eta_grid, A_hat, color=C_GREY, linestyle="--", linewidth=1.8,
                label=r"linearised $\widehat A_{n,w}$", zorder=6)

    ax.set_xlabel(r"proxy budget $\eta$")
    ax.set_ylabel("Finite-search attackability")
    ax.set_xlim(float(np.nanmin(eta_grid)), float(np.nanmax(eta_grid)))
    ax.set_ylim(-0.02, 1.04)
    _format_linear_axis(ax)
    _panel_label(ax, "(a)" if two else "")

    if two:
        axc = axes[1]
        ce = np.asarray(coverage_diag["eta"], dtype=float)
        gap = np.asarray(coverage_diag["median_norm_gap"], dtype=float)
        axc.axhline(1.0, linestyle=":", linewidth=1.1, color=C_GREY, label="uncovered")
        axc.plot(ce, gap, marker="o", markersize=3.4, linewidth=1.6,
                 color=C_PRIMARY, label="median gap")
        axc.set_xlabel(r"proxy budget $\eta$")
        axc.set_ylabel(r"Normalised coverage gap along $r_w$")
        axc.set_xlim(float(np.nanmin(ce)), float(np.nanmax(ce)))
        axc.set_ylim(-0.02, 1.06)
        _format_linear_axis(axc)
        _panel_label(axc, "(b)")
        _legend(axc, loc="lower left")

    # Panel (a)'s K-sweep legend below the figure so it cannot overlap the
    # linearised curve and band; savefig's tight bbox keeps it from clipping.
    handles_a, labels_a = ax.get_legend_handles_labels()
    ncol_a = 4 if len(handles_a) >= 6 else max(1, len(handles_a))
    fig.legend(handles_a, labels_a, loc="upper center",
               bbox_to_anchor=(0.5, -0.02), ncol=ncol_a, frameon=False,
               fontsize=8.0, handlelength=2.0, columnspacing=1.3)

    _save(fig, save_path)


# ---------------------------------------------------------------------------
# Figure: predictive validity
# ---------------------------------------------------------------------------

def plot_predictive_validity(result: Mapping, save_path: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.3, 3.2), constrained_layout=True)

    Z = np.asarray(result["Z_vals"], dtype=float)
    rb = np.asarray(result["realised_flip_budget"], dtype=float)
    fin = np.isfinite(Z) & np.isfinite(rb) & (Z > 0) & (rb > 0)

    ax = axes[0]
    if fin.any():
        lo, hi = _positive_limits(Z[fin], rb[fin])
        ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1.1, color=C_GREY,
                zorder=1, label="identity")
        ax.scatter(Z[fin], rb[fin], s=16, alpha=0.55, linewidths=0.0,
                   color=C_PRIMARY, rasterized=True, zorder=2)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
    ax.set_xscale("log")
    ax.set_yscale("log")
    rho = float(result.get("spearman_Z_vs_budget", np.nan))
    n_used = int(result.get("n_used", int(fin.sum())))
    if np.isfinite(rho):
        ax.text(0.055, 0.90, fr"Spearman $\rho={rho:.2f}$ ($n={n_used}$)",
                transform=ax.transAxes, ha="left", va="top", fontsize=9,
                bbox=ANNOT_BBOX)
    ax.set_xlabel(r"$Z_w(x)$")
    ax.set_ylabel("Smallest realised flip budget")
    _format_log_axis(ax)
    _panel_label(ax, "(a)")
    _legend(ax, loc="lower right")

    ax = axes[1]
    eta = np.asarray(result["eta"], dtype=float)
    auc = np.asarray(result["auc_vs_eta"], dtype=float)
    fr = np.asarray(result["flip_rate_vs_eta"], dtype=float)
    me = max(1, len(eta) // 12)   # thin markers so they do not crowd the line
    ax.axhline(0.5, linestyle=(0, (1, 1)), linewidth=1.0, color=C_GREY,
               zorder=1, label="no skill")
    ok = np.isfinite(auc)
    if ok.any():
        ax.plot(eta[ok], auc[ok], color=C_PRIMARY, linestyle="-", marker="o",
                markevery=me, markersize=3.8, linewidth=1.7, zorder=6,
                label=r"AUC of $-Z_w$")
    if "auc_gamma_vs_eta" in result:
        ax.plot(eta, np.asarray(result["auc_gamma_vs_eta"], float), color=C_ACCENT,
                linestyle="--", marker="^", markevery=me, markersize=3.8,
                linewidth=1.3, zorder=4, label=r"AUC of $-\gamma_w$ (margin only)")
        ax.plot(eta, np.asarray(result["auc_sigma_vs_eta"], float), color=C_GREEN,
                linestyle=":", marker="D", markevery=me, markersize=3.4,
                linewidth=1.5, zorder=5, label=r"AUC of $\sqrt{\Sigma_w}$ (displ. only)")
    ax.plot(eta, fr, color="#CC79A7", linestyle="-", marker="s", markevery=me,
            markersize=3.2, linewidth=1.0, alpha=0.85, zorder=2,
            label="realised flip rate")
    auc_lb = result.get("auc_at_label_budget", np.nan)
    lb = result.get("label_budget", np.nan)
    if np.isfinite(auc_lb) and np.isfinite(lb):
        ax.text(0.96, 0.60, fr"AUC $={float(auc_lb):.2f}$ at $\eta={float(lb):.3g}$",
                transform=ax.transAxes, ha="right", va="center", fontsize=8.6,
                bbox=ANNOT_BBOX)
    ax.set_xlabel(r"proxy budget $\eta$")
    ax.set_ylabel("AUC / flip rate")
    ax.set_xlim(float(np.nanmin(eta)), float(np.nanmax(eta)))
    ax.set_ylim(-0.02, 1.04)
    _format_linear_axis(ax)
    _panel_label(ax, "(b)")

    # Panel (b)'s legend below the figure so it never overlaps the curves;
    # savefig's tight bounding box keeps it from being clipped.
    handles_b, labels_b = ax.get_legend_handles_labels()
    fig.legend(handles_b, labels_b, loc="upper center",
               bbox_to_anchor=(0.5, -0.02), ncol=3, frameon=False, fontsize=8.4,
               handlelength=2.0, columnspacing=1.4)

    _save(fig, save_path)


# ---------------------------------------------------------------------------
# Optional: restyle linearisation from a cached npz (if present)
# ---------------------------------------------------------------------------

def plot_linearisation_from_npz(npz_path: str, save_path: str) -> None:
    d = np.load(npz_path)
    eta = np.asarray(d["eta"], dtype=float)
    mean_nl = np.asarray(d["mean_abs_nl"], dtype=float)
    mean_lin = np.asarray(d["mean_abs_lin"], dtype=float)
    mean_nr = np.asarray(d["mean_norm_residual"], dtype=float)
    med_nr = np.asarray(d["median_norm_residual"], dtype=float)

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2), constrained_layout=True)

    ax = axes[0]
    a = np.isfinite(eta) & (eta > 0)
    al = a & np.isfinite(mean_lin) & (mean_lin > 0)
    an = a & np.isfinite(mean_nl) & (mean_nl > 0)
    if al.any():
        ax.loglog(eta[al], mean_lin[al], marker="s", markersize=3.6,
                  color=C_PRIMARY, label=r"linear $|\Delta s|$")
        e0 = eta[al][0]
        y0 = mean_lin[al][0]
        ax.loglog(eta[al], y0 * (eta[al] / e0), linestyle=":", linewidth=1.1,
                  color=C_GREY, label="slope 1")
    if an.any():
        ax.loglog(eta[an], mean_nl[an], marker="o", markersize=3.6,
                  color=C_ACCENT, label=r"nonlinear $|\Delta s|$")
    ax.set_xlabel(r"proxy budget $\eta$")
    ax.set_ylabel("Mean readout displacement")
    _format_log_axis(ax)
    _panel_label(ax, "(a)")
    _legend(ax, loc="upper left")

    ax = axes[1]
    bm = a & np.isfinite(mean_nr) & (mean_nr > 0)
    bd = a & np.isfinite(med_nr) & (med_nr > 0)
    if bm.any():
        ax.loglog(eta[bm], mean_nr[bm], marker="o", markersize=3.4,
                  color=C_PRIMARY, label="mean")
    if bd.any():
        ax.loglog(eta[bd], med_nr[bd], marker="s", markersize=3.4,
                  color=C_ACCENT, label="median")
        idx = np.flatnonzero(bd)
        m = max(1, len(idx) // 3)
        flat = float(np.nanmedian(med_nr[idx[:m]]))
        if np.isfinite(flat) and flat > 0:
            ax.axhline(flat, linestyle=":", linewidth=1.1, color=C_GREY,
                       label="local reference")
    ax.set_xlabel(r"proxy budget $\eta$")
    ax.set_ylabel(r"Residual normalised by $\eta^2/\beta_i$")
    _format_log_axis(ax)
    _panel_label(ax, "(b)")
    _legend(ax, loc="lower left")

    _save(fig, save_path)


# ---------------------------------------------------------------------------
# Loading / parameter reconstruction
# ---------------------------------------------------------------------------

def _load_pickle(path: str):
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def _safe_K_grid(cloud_records):
    counts = [int(r["D_M"].shape[1]) for r in cloud_records]
    K_max = max(counts)
    base = [4, 8, 12, 18, 25]
    grid = sorted(set([k for k in base if k <= K_max] + [K_max]))
    return grid, K_max


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--figdir", default="figures",
                   help="directory holding *_records.pkl and run_metadata.json")
    p.add_argument("--outdir", default=None,
                   help="where to write PDFs (default: same as --figdir, overwrites)")
    args = p.parse_args()

    figdir = args.figdir
    outdir = args.outdir or figdir
    os.makedirs(outdir, exist_ok=True)
    _set_style()

    # Reconstruct experiment parameters from metadata (fall back to defaults).
    meta = {}
    meta_path = os.path.join(figdir, "run_metadata.json")
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
    nu_frac = float(meta.get("regularisation", {}).get("nu_frac", 1e-3))
    att = meta.get("eta_grids", {}).get("attackability", {})
    eta_max = float(att.get("max", 1.0))
    n_eta = int(att.get("n_eta", 40))
    eta_grid = np.linspace(0.01, eta_max, n_eta)
    seed = int(meta.get("dataset", {}).get("seed", 42))
    K_grid_meta = meta.get("cloud", {}).get("finite_search_K_grid")
    oracle_K_meta = meta.get("cloud", {}).get("oracle_K")
    print(f"figdir={figdir}  nu_frac={nu_frac}  eta in [0.01, {eta_max}] x{n_eta}")

    soft = _load_pickle(os.path.join(figdir, "soft_records.pkl"))
    cloud = _load_pickle(os.path.join(figdir, "cloud_records.pkl"))

    if soft:
        print(f"soft_records: {len(soft)}")
        r1 = experiment_1_sigma_lambda(soft, nu_frac=nu_frac)
        plot_sigma_vs_lambda(r1, os.path.join(outdir, "fig_sigma_lambda_soft.pdf"))
        r3 = experiment_3_attackability_curve(soft, eta_grid, nu_frac=nu_frac)
        plot_attackability_curve(r3, os.path.join(outdir, "fig_attackability_curve.pdf"))
    else:
        print("soft_records.pkl not found -- skipping soft-token figures")

    if cloud:
        print(f"cloud_records: {len(cloud)}")
        r1c = experiment_1_sigma_lambda(cloud, nu_frac=nu_frac)
        plot_sigma_vs_lambda(r1c, os.path.join(outdir, "fig_sigma_lambda_cloud.pdf"))
        r3c = experiment_3_attackability_curve(cloud, eta_grid, nu_frac=nu_frac)
        plot_attackability_curve(r3c, os.path.join(outdir, "fig_attackability_curve_cloud.pdf"))

        if K_grid_meta is not None and oracle_K_meta is not None:
            K_grid, oracle_K = [int(k) for k in K_grid_meta], int(oracle_K_meta)
        else:
            K_grid, oracle_K = _safe_K_grid(cloud)
        rfs = experiment_2prime_finite_search(cloud, K_grid, eta_grid, seed=seed + 2)
        cov = finite_search_coverage_diagnostic(cloud, eta_grid, nu_frac=nu_frac)
        plot_finite_search(rfs, eta_grid, os.path.join(outdir, "fig_finite_search.pdf"),
                           A_hat=r3c["A_hat"], dkw_half=r3c["dkw_half_width"],
                           coverage_diag=cov, oracle_K=oracle_K)

        pv = experiment_predictive_validity(cloud, eta_grid, nu_frac=nu_frac, mode="within_split")
        plot_predictive_validity(pv, os.path.join(outdir, "fig_predictive_validity.pdf"))
    else:
        print("cloud_records.pkl not found -- skipping cloud figures")

    # Linearisation: restyle only if its data was cached; cannot be recomputed here.
    lin_npz = os.path.join(figdir, "linearisation_results.npz")
    if os.path.exists(lin_npz):
        plot_linearisation_from_npz(lin_npz, os.path.join(outdir, "fig_linearisation_error.pdf"))
    else:
        print("\nNote: fig_linearisation_error.pdf was NOT regenerated.")
        print("  Its nonlinear curve needs the model forward pass, which is not")
        print("  cached. To restyle it too, add one line in main.py right after")
        print("  the linearisation block builds r2:")
        print("      np.savez(os.path.join(cfg.out, 'linearisation_results.npz'), **r2)")
        print("  then run a soft-token-only pass once (python main.py --run_soft_token),")
        print("  and re-run this script.")

    print("\nDone.")


if __name__ == "__main__":
    main()