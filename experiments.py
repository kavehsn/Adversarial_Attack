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
        "true_label": int                    -- true class in {0,1,2}
        # Soft-token chart, for the linearisation experiment:
        "perturb_fn": callable(u) -> e_M_perturbed (np.ndarray (d_M,))
        # Cloud chart, for finite search / predictive validity:
        "D_M":     np.ndarray (d_M, K)       -- target paraphrase displacements
        "D_P":     np.ndarray (d_P, K)       -- proxy paraphrase displacements
        "para_preds": np.ndarray (K,) int    -- realised FinBERT argmax per paraphrase
    }
"""

from typing import Iterable, List, Optional

import numpy as np
from scipy.stats import spearmanr

from geometry import (
    pullback_matrices,
    lambda_star,
    sigma_w,
    gamma_w,
    beta_min,
    Z_w,
    linearised_worst_direction,
    readout_aligned_direction,
)


# ---------------------------------------------------------------------------
# Small numerical helpers
# ---------------------------------------------------------------------------

def _auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """
    Rank-based ROC AUC (= normalised Mann-Whitney U), tie-aware.
    Returns nan if labels are all one class. ``scores`` higher => predict label 1.
    """
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)
    pos = labels == 1
    n_pos = int(pos.sum())
    n_neg = int((~pos).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1, dtype=float)
    # Average ranks within tie groups.
    s_sorted = scores[order]
    i = 0
    while i < len(s_sorted):
        j = i
        while j + 1 < len(s_sorted) and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = 0.5 * (ranks[order[i]] + ranks[order[j]])
        i = j + 1
    sum_ranks_pos = ranks[pos].sum()
    u_pos = sum_ranks_pos - n_pos * (n_pos + 1) / 2.0
    return float(u_pos / (n_pos * n_neg))


# ---------------------------------------------------------------------------
# Experiment 1A (and 1B): Sigma_w(x) <= lambda*(x)
# ---------------------------------------------------------------------------

def experiment_1_sigma_lambda(records: List[dict], nu_frac: float = 1e-3,
                              normalize_w: bool = False):
    """
    Scatter Sigma_w vs lambda* across inputs. Histogram of slack ratio.

    Sigma_w <= lambda*,is stated for a unit readout ||w|| = 1, whereas the logit-gap readout w_eff =
    W[t] - W[i*] is not unit-norm. Sigma_w scales as ||w||^2 while lambda* does
    not, so the raw scatter and the slack ratio depend on ||w||. Everything else
    in the pipeline (Z_w, the flip condition, the attackability curve, predictive
    validity) is scale-invariant in w and is unaffected. Set ``normalize_w=True``
    to reproduce the theorem's unit-w convention exactly; the default is False so
    existing numbers are preserved.
    """
    lambdas, sigmas = [], []
    for r in records:
        A, B = pullback_matrices(r["J_M"], r["J_P"])
        w = r["w"]
        if normalize_w:
            nrm = float(np.linalg.norm(w))
            if nrm > 0:
                w = w / nrm
        l, _ = lambda_star(A, B, nu_frac)
        s, _ = sigma_w(r["J_M"], B, w, nu_frac)
        lambdas.append(l)
        sigmas.append(s)
    lambdas = np.array(lambdas)
    sigmas = np.array(sigmas)
    with np.errstate(divide="ignore", invalid="ignore"):
        slack = np.where(sigmas > 0, lambdas / sigmas, np.nan)
    rng = np.random.default_rng(0)
    finite = np.isfinite(slack) & (slack > 0)
    logs = np.log10(slack[finite])
    boot = [float(np.median(rng.choice(logs, size=logs.size, replace=True)))
            for _ in range(1000)] if logs.size else []
    ci = (float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))) if boot else (np.nan, np.nan)
    return {
        "lambda_star": lambdas,
        "sigma_w": sigmas,
        "slack_ratio": slack,
        "violations": int(np.sum(sigmas > lambdas + 1e-9)),
        "log10_slack_median": float(np.median(logs)) if logs.size else np.nan, "log10_slack_median_ci": ci,
    }


# ---------------------------------------------------------------------------
# Experiment 2: linearisation error r_M(eta)  (soft-token chart only)
# ---------------------------------------------------------------------------

def experiment_2_linearisation_error(
    records: List[dict],
    eta_grid: np.ndarray,
    nu_frac: float = 1e-3,
):
    """
    For each record evaluate the linearised worst direction u*_flip(eta) through
    the nonlinear E_M (via record["perturb_fn"]) and record BOTH the realised and
    the linearised readout displacement.

    The flattening of the residual is the nonlinear
    displacement Delta s^nl saturating (the soft-token embedding moves within a
    fixed convex hull) while Delta s^lin keeps growing linearly; the break is
    therefore not a Taylor radius. We thus return:

      * delta_s_nl, delta_s_lin  -- signed, (n, n_eta), so they can be plotted
        separately (nl saturating vs lin linear);
      * residual                 -- |nl - lin|, (n, n_eta);
      * beta = lambda_min(B + nu I) per input;
      * norm_residual            -- residual / (eta^2 / beta_i), which removes the
        eta-scaling and the per-input 1/beta heterogeneity so that a flat curve
        (rather than slope 2) indicates the O(eta^2/beta).

    The experiment is a calibration of the budget range over which the
    first-order model is informative, not a pass/fail Taylor test.
    """
    n, m = len(records), len(eta_grid)
    delta_s_nl = np.full((n, m), np.nan)
    delta_s_lin = np.full((n, m), np.nan)
    betas = np.full(n, np.nan)

    for i, r in enumerate(records):
        _, B = pullback_matrices(r["J_M"], r["J_P"])
        betas[i] = beta_min(B, nu_frac)
        s0 = float(np.dot(r["w"], r["e_M_x"]) + r["b"])
        sign_s0 = float(np.sign(s0)) if s0 != 0.0 else 1.0

        for j, eta in enumerate(eta_grid):
            u_star, sigma = linearised_worst_direction(
                r["J_M"], B, r["w"], r["b"], r["e_M_x"], float(eta), nu_frac,
            )
            if u_star is None:
                continue
            e_M_perturbed = r["perturb_fn"](u_star)
            delta_s_nl[i, j] = float(np.dot(r["w"], e_M_perturbed - r["e_M_x"]))
            delta_s_lin[i, j] = -sign_s0 * float(eta) * np.sqrt(sigma)

    residual = np.abs(delta_s_nl - delta_s_lin)

    # Per-input normalisation by eta^2 / beta_i  (Remark 3.2 scale).
    eta_sq = (np.asarray(eta_grid, dtype=float) ** 2)[None, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        norm_residual = residual / (eta_sq / betas[:, None])

    return {
        "eta": np.asarray(eta_grid, dtype=float),
        "delta_s_nl": delta_s_nl,
        "delta_s_lin": delta_s_lin,
        "residual": residual,
        "beta": betas,
        "norm_residual": norm_residual,
        # Aggregates used by the plot.
        "mean_abs_nl": np.nanmean(np.abs(delta_s_nl), axis=0),
        "mean_abs_lin": np.nanmean(np.abs(delta_s_lin), axis=0),
        "mean_residual": np.nanmean(residual, axis=0),
        "median_residual": np.nanmedian(residual, axis=0),
        "mean_norm_residual": np.nanmean(norm_residual, axis=0),
        "median_norm_residual": np.nanmedian(norm_residual, axis=0),
    }


# ---------------------------------------------------------------------------
# Experiment 3A (and 3B): empirical attackability curve with DKW band
#                         and the empirical-inclusion overlay.
# ---------------------------------------------------------------------------

def experiment_3_attackability_curve(
    records: List[dict],
    eta_grid: np.ndarray,
    delta: float = 0.05,
    nu_frac: float = 1e-3,
    beta_levels: Iterable[float] = (0.10, 0.25),
    calibration_split: bool = False,
    split_frac: float = 0.5,
    split_seed: int = 0,
):
    """
    Empirical strict left-CDF of Z_w, with DKW band

    The overlay curves are the *deterministic empirical-
    inclusion* bound, not a population certificate. Writing P_n for the empirical
    measure and Lambda_hat_{1-beta} for the empirical (1-beta) quantile of
    lambda*(x_i),

        A_hat_n(eta) = P_n[Z_w < eta]
                     = P_n[gamma_w < eta sqrt(Sigma_w)]          
                     <= P_n[gamma_w < eta sqrt(lambda*)]      
                     <= P_n[gamma_w < eta sqrt(Lambda_hat)] + P_n[lambda* > Lambda_hat]
                     <= P_n[gamma_w < eta sqrt(Lambda_hat)] + beta,

    the last step holding because the empirical quantile guarantees
    P_n[lambda* > Lambda_hat_{1-beta}] <= beta. This is an exact in-sample
    inequality (no population claim). The analogous Sigma_w curve uses the
    quantile of Sigma_w directly. When ``calibration_split`` is True, the
    quantile is estimated on a held-out fraction and the bound evaluated on the
    complement, giving an out-of-sample version (the bound then holds up to the
    eval-set exceedance of the calibration quantile, which concentrates at beta).
    """
    Z_vals, lambda_vals, sigma_vals, gamma_vals = [], [], [], []
    for r in records:
        A, B = pullback_matrices(r["J_M"], r["J_P"])
        Z, sig = Z_w(r["e_M_x"], r["J_M"], B, r["w"], r["b"], nu_frac)
        l, _ = lambda_star(A, B, nu_frac)
        g = gamma_w(r["e_M_x"], r["w"], r["b"])
        Z_vals.append(Z)
        lambda_vals.append(l)
        sigma_vals.append(sig)
        gamma_vals.append(g)

    Z_vals = np.array(Z_vals)
    lambda_vals = np.array(lambda_vals)
    sigma_vals = np.array(sigma_vals)
    gamma_vals = np.array(gamma_vals)
    n = len(records)

    if calibration_split:
        rng = np.random.default_rng(split_seed)
        perm = rng.permutation(n)
        n_cal = max(1, int(round(split_frac * n)))
        cal_idx, eval_idx = perm[:n_cal], perm[n_cal:]
    else:
        cal_idx = eval_idx = np.arange(n)

    # Empirical strict left-CDF on the evaluation split.
    Z_eval = Z_vals[eval_idx]
    gamma_eval = gamma_vals[eval_idx]
    n_eval = len(eval_idx)
    A_hat = np.array([float((Z_eval < eta).mean()) for eta in eta_grid])
    dkw_half = float(np.sqrt(np.log(2.0 / delta) / (2.0 * n_eval)))

    thm4_2_bounds_lambda, thm4_2_bounds_sigma = {}, {}
    for beta in beta_levels:
        Lambda_q = float(np.quantile(lambda_vals[cal_idx], 1.0 - beta))
        Sigma_q = float(np.quantile(sigma_vals[cal_idx], 1.0 - beta))
        thm4_2_bounds_lambda[beta] = np.array([
            float((gamma_eval < eta * np.sqrt(Lambda_q)).mean()) + beta
            for eta in eta_grid
        ])
        thm4_2_bounds_sigma[beta] = np.array([
            float((gamma_eval < eta * np.sqrt(Sigma_q)).mean()) + beta
            for eta in eta_grid
        ])

    return {
        "eta": np.asarray(eta_grid, dtype=float),
        "A_hat": A_hat,
        "dkw_half_width": dkw_half,
        "thm4_2_bounds": thm4_2_bounds_lambda,        # backward-compat alias
        "thm4_2_bounds_lambda": thm4_2_bounds_lambda,
        "thm4_2_bounds_sigma": thm4_2_bounds_sigma,
        "calibration_split": bool(calibration_split),
        "n_eval": n_eval,
        "Z_vals": Z_vals,
        "lambda_vals": lambda_vals,
        "sigma_vals": sigma_vals,
        "gamma_vals": gamma_vals,
    }


# ---------------------------------------------------------------------------
# Experiment 4: finite-search attackability vs K (cloud chart)
# ---------------------------------------------------------------------------

def experiment_2prime_finite_search(
    records: List[dict],
    K_grid: Iterable[int],
    eta_grid: np.ndarray,
    seed: int = 0,
):
    """
    Finite-search attackability A^fin_w(eta; P_K), averaged over inputs.

    For each input and K, the finite-search displacement toward the boundary is
        D^fin = max{0} U { -sign(s0) w^T (e_M(x'_k) - e_M(x)) : d_P(x, x'_k) <= eta },
    and a flip is recorded when D^fin >= gamma_w(x). Including K = K_full in the
    grid yields the best-of-K_full oracle ceiling.
    """
    K_grid = list(K_grid)
    rng = np.random.default_rng(seed)
    results = {K: np.zeros(len(eta_grid)) for K in K_grid}

    for r in records:
        D_M, D_P = r["D_M"], r["D_P"]
        w, b, e_M_x = r["w"], r["b"], r["e_M_x"]
        gamma = gamma_w(e_M_x, w, b)
        s0 = float(np.dot(w, e_M_x) + b)
        sign_s0 = float(np.sign(s0)) if s0 != 0.0 else 1.0

        proxy_dists = np.linalg.norm(D_P, axis=0)
        readout_disp = -sign_s0 * (w @ D_M)
        K_full = D_M.shape[1]

        for K in K_grid:
            idx = (np.arange(K_full) if K >= K_full
                   else rng.choice(K_full, size=K, replace=False))
            pd_sub, rd_sub = proxy_dists[idx], readout_disp[idx]
            for j, eta in enumerate(eta_grid):
                valid = pd_sub <= eta
                D_fin = max(0.0, float(rd_sub[valid].max())) if valid.any() else 0.0
                if D_fin >= gamma:
                    results[K][j] += 1

    n = len(records)
    return {K: v / n for K, v in results.items()}


def finite_search_coverage_diagnostic(
    records: List[dict],
    eta_grid: np.ndarray,
    nu_frac: float = 1e-3,
):
    """
    In the cloud chart the k-th candidate has local coordinate u_k = e_k, so its
    signed position along r_w (in the B-metric) is alpha_k = <r_w, e_k>_B = (B r_w)_k,
    and its proxy budget is ||e_k||_B = sqrt(B_kk). For each input and budget eta we
    take the proxy-valid candidates whose positions fall in (0, eta], and measure the
    largest gap in {0} U {positions} U {eta}. A gap near eta means the candidate set
    barely populates the readout-relevant axis, so a finite-search shortfall is the
    expected consequence of sampling geometry rather than a verdict on generator
    quality. Returns the median (over inputs) of the gap, normalised by eta.
    """
    m = len(eta_grid)
    gaps = []
    for r in records:
        A, B = pullback_matrices(r["J_M"], r["J_P"])
        r_w, sigma = readout_aligned_direction(
            r["J_M"], B, r["w"], r["b"], r["e_M_x"], nu_frac,
        )
        if r_w is None:
            continue
        positions = B @ r_w                      # (K,) signed positions along r_w
        budgets = np.sqrt(np.clip(np.diag(B), 0.0, None))  # (K,) proxy budgets
        row = np.empty(m)
        for j, eta in enumerate(eta_grid):
            usable = (budgets <= eta) & (positions > 0) & (positions <= eta)
            pts = np.sort(positions[usable])
            grid_pts = np.concatenate(([0.0], pts, [float(eta)]))
            row[j] = float(np.max(np.diff(grid_pts))) / float(eta)
        gaps.append(row)
    gaps = np.array(gaps) if gaps else np.full((1, m), np.nan)
    return {
        "eta": np.asarray(eta_grid, dtype=float),
        "median_norm_gap": np.nanmedian(gaps, axis=0),
        "mean_norm_gap": np.nanmean(gaps, axis=0),
        "n_used": int(gaps.shape[0]),
    }


# ---------------------------------------------------------------------------
# Predictive validity of the attackability index
# ---------------------------------------------------------------------------

def experiment_predictive_validity(
    records: List[dict],
    eta_grid: np.ndarray,
    nu_frac: float = 1e-3,
    label_budget: Optional[float] = None,
    mode: str = "within_split",     # "within_split" (OOS) or "in_sample" (NOT for paper)
    n_splits: int = 25,
    split_frac: float = 0.5,
    n_boot: int = 1000,
    seed: int = 0,
):
    """
    Does Z_w(x) predict *realised* flips OUT OF SAMPLE?

    mode="within_split": for each input, the geometry half of its paraphrase
      cloud builds Sigma_w / Z_w; the disjoint evaluation half defines the flip
      label and realised flip budget. Metrics are averaged over n_splits random
      partitions. This breaks the feature<->label coupling of the in-sample test.

    mode="in_sample": the old behaviour (whole cloud builds Z_w AND the label).
      Kept only for an explicit leakage demonstration.

    Baselines (all oriented so larger => more attackable):
      score_Z     = -Z_w      (adjusted margin; the proposed index)
      score_gamma = -gamma_w  (bare classifier margin; label-free, OOS by design)
      score_sigma = +sqrt(Sigma_w)  (displacement only)
    """
    rng = np.random.default_rng(seed)

    # ---- per-input, per-split predictor/label assembly -------------------
    # We accumulate, for each split s, aligned per-input arrays.
    per_split = []  # list of dicts with Z, gamma, sigma, labels_lb, budget, proxy_lists, flip_lists
    n_eta = len(eta_grid)

    # First pass: cache raw cloud arrays per usable input.
    cache = []
    for r in records:
        if "para_flip" not in r or "D_P" not in r:
            continue
        D_M, D_P = np.asarray(r["D_M"]), np.asarray(r["D_P"])
        K = D_M.shape[1]
        if K < 2:
            continue
        flipped = np.asarray(r["para_flip"]).astype(bool)
        pd_all = np.linalg.norm(D_P, axis=0)
        gamma = gamma_w(r["e_M_x"], r["w"], r["b"])     # label-free
        cache.append(dict(r=r, D_M=D_M, D_P=D_P, K=K, flipped=flipped,
                          pd_all=pd_all, gamma=gamma))
    used = len(cache)

    n_pass = n_splits if mode == "within_split" else 1
    for s in range(n_pass):
        Z_vals, gamma_vals, sigma_vals = [], [], []
        proxy_lists, flip_lists, realised_budget = [], [], []
        for c in cache:
            r, D_M, D_P, K = c["r"], c["D_M"], c["D_P"], c["K"]
            if mode == "within_split":
                perm = rng.permutation(K)
                n_fit = max(1, int(round(split_frac * K)))
                fit, ev = perm[:n_fit], perm[n_fit:]
                if ev.size == 0:           # tiny cloud: fall back to leave-one-out
                    ev, fit = perm[:1], perm[1:]
            else:
                fit = ev = np.arange(K)    # in-sample
            D_M_fit, D_P_fit = D_M[:, fit], D_P[:, fit]
            B_fit = D_P_fit.T @ D_P_fit
            Z, sig = Z_w(r["e_M_x"], D_M_fit, B_fit, r["w"], r["b"], nu_frac)
            pd_ev, flip_ev = c["pd_all"][ev], c["flipped"][ev]
            Z_vals.append(Z); gamma_vals.append(c["gamma"]); sigma_vals.append(sig)
            proxy_lists.append(pd_ev); flip_lists.append(flip_ev)
            realised_budget.append(float(pd_ev[flip_ev].min()) if flip_ev.any() else np.inf)
        per_split.append(dict(
            Z=np.array(Z_vals), gamma=np.array(gamma_vals), sigma=np.array(sigma_vals),
            proxy_lists=proxy_lists, flip_lists=flip_lists,
            budget=np.array(realised_budget),
        ))

    # ---- label budget (median realised flip budget, pooled over splits) ---
    if label_budget is None:
        allb = np.concatenate([d["budget"][np.isfinite(d["budget"])] for d in per_split]) \
               if per_split else np.array([])
        label_budget = float(np.median(allb)) if allb.size else float("nan")

    # ---- metric helpers ---------------------------------------------------
    def _scores(d):
        sZ = np.where(np.isfinite(d["Z"]), -d["Z"], -np.inf)
        sg = -d["gamma"]
        ss = np.sqrt(np.clip(d["sigma"], 0.0, None))
        return sZ, sg, ss

    def _labels_at(d, budget):
        return np.array([
            int((pd <= budget).any() and fl[pd <= budget].any())
            for pd, fl in zip(d["proxy_lists"], d["flip_lists"])
        ])

    def _auc_at_lb(d):
        if not np.isfinite(label_budget):
            return (np.nan, np.nan, np.nan)
        lab = _labels_at(d, label_budget)
        sZ, sg, ss = _scores(d)
        return (_auc(sZ, lab), _auc(sg, lab), _auc(ss, lab))

    def _spearman(d):
        fin = np.isfinite(d["Z"]) & np.isfinite(d["budget"])
        if fin.sum() < 3:
            return (np.nan, np.nan, np.nan, np.nan)
        rZ, pZ = spearmanr(d["Z"][fin], d["budget"][fin])
        rg, _ = spearmanr(d["gamma"][fin], d["budget"][fin])
        rs, _ = spearmanr(np.sqrt(np.clip(d["sigma"][fin], 0.0, None)), d["budget"][fin])
        return (float(rZ), float(rg), float(rs), float(pZ))

    # Point estimates = mean across splits.
    aucs = np.array([_auc_at_lb(d) for d in per_split])           # (n_splits, 3)
    sps  = np.array([_spearman(d) for d in per_split])            # (n_splits, 4)
    auc_lb = np.nanmean(aucs, axis=0)                             # [Z, gamma, sigma]
    rho    = np.nanmean(sps[:, :3], axis=0)                       # [Z, gamma, sigma]
    p_Z    = float(np.nanmean(sps[:, 3]))

    # AUC vs eta (mean across splits), Z + baselines.
    def _auc_curve(d):
        sZ, sg, ss = _scores(d)
        aZ = np.empty(n_eta); ag = np.empty(n_eta); as_ = np.empty(n_eta); fr = np.empty(n_eta)
        for j, eta in enumerate(eta_grid):
            lab = _labels_at(d, eta)
            fr[j] = lab.mean()
            aZ[j] = _auc(sZ, lab); ag[j] = _auc(sg, lab); as_[j] = _auc(ss, lab)
        return aZ, ag, as_, fr
    curves = [ _auc_curve(d) for d in per_split ]
    auc_Z_eta     = np.nanmean([c[0] for c in curves], axis=0)
    auc_gamma_eta = np.nanmean([c[1] for c in curves], axis=0)
    auc_sigma_eta = np.nanmean([c[2] for c in curves], axis=0)
    flip_rate     = np.nanmean([c[3] for c in curves], axis=0)

    # ---- bootstrap CI over inputs (random split each round) ---------------
    boot_auc_Z, boot_rho_Z = [], []
    m = used
    for _ in range(n_boot):
        d = per_split[rng.integers(len(per_split))]
        bs = rng.integers(0, m, size=m)
        db = dict(Z=d["Z"][bs], gamma=d["gamma"][bs], sigma=d["sigma"][bs],
                  budget=d["budget"][bs],
                  proxy_lists=[d["proxy_lists"][i] for i in bs],
                  flip_lists=[d["flip_lists"][i] for i in bs])
        boot_auc_Z.append(_auc_at_lb(db)[0])
        boot_rho_Z.append(_spearman(db)[0])
    def _ci(v):
        v = np.asarray([x for x in v if np.isfinite(x)], float)
        return (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) if v.size else (np.nan, np.nan)

    # For panel (a) scatter: use the first split's Z and budget.
    d0 = per_split[0]
    return {
        "mode": mode,
        "eta": np.asarray(eta_grid, dtype=float),
        "n_used": used,
        "label_budget": float(label_budget),
        # scatter data (one split)
        "Z_vals": d0["Z"],
        "realised_flip_budget": d0["budget"],
        # Spearman (mean over splits) — Z, gamma, sigma
        "spearman_Z_vs_budget": float(rho[0]),
        "spearman_gamma_vs_budget": float(rho[1]),
        "spearman_sigma_vs_budget": float(rho[2]),
        "spearman_pvalue": p_Z,
        "spearman_Z_ci": _ci(boot_rho_Z),
        # AUC at label budget — Z, gamma, sigma
        "auc_at_label_budget": float(auc_lb[0]),
        "auc_gamma_at_label_budget": float(auc_lb[1]),
        "auc_sigma_at_label_budget": float(auc_lb[2]),
        "auc_at_label_budget_ci": _ci(boot_auc_Z),
        # AUC vs eta — Z (kept under old key), gamma, sigma
        "auc_vs_eta": auc_Z_eta,
        "auc_gamma_vs_eta": auc_gamma_eta,
        "auc_sigma_vs_eta": auc_sigma_eta,
        "flip_rate_vs_eta": flip_rate,
    }

######THIS IS WHERE YOU STOPPED. I WILL CONTINUE FROM HERE.

def experiment_predictive_validity_crosschart(
    soft_records: List[dict],
    cloud_records: List[dict],
    eta_grid: np.ndarray,
    nu_frac: float = 1e-3,
    label_budget: Optional[float] = None,
    n_boot: int = 1000,
    seed: int = 0,
):
    """
    A2: predict CLOUD realised flips using SOFT-TOKEN Z_w (which never sees a
    paraphrase) — a genuinely out-of-sample forecast. Z_w, gamma_w, sqrt(Sigma_w)
    are all taken from the soft chart so the only thing distinguishing Z from its
    baselines is the geometry term; labels come from the cloud paraphrases.
    Inputs are matched by record["text"].
    """
    rng = np.random.default_rng(seed)
    soft_by_text = {r["text"]: r for r in soft_records}

    Z_vals, gamma_vals, sigma_vals = [], [], []
    proxy_lists, flip_lists, realised_budget = [], [], []
    for cr in cloud_records:
        sr = soft_by_text.get(cr["text"])
        if sr is None or "para_flip" not in cr:
            continue
        _, B = pullback_matrices(sr["J_M"], sr["J_P"])      # soft-chart geometry
        Z, sig = Z_w(sr["e_M_x"], sr["J_M"], B, sr["w"], sr["b"], nu_frac)
        gamma = gamma_w(sr["e_M_x"], sr["w"], sr["b"])
        pd = np.linalg.norm(np.asarray(cr["D_P"]), axis=0)  # cloud labels
        flipped = np.asarray(cr["para_flip"]).astype(bool)
        Z_vals.append(Z); gamma_vals.append(gamma); sigma_vals.append(sig)
        proxy_lists.append(pd); flip_lists.append(flipped)
        realised_budget.append(float(pd[flipped].min()) if flipped.any() else np.inf)

    Z_vals = np.array(Z_vals); gamma_vals = np.array(gamma_vals)
    sigma_vals = np.array(sigma_vals); realised_budget = np.array(realised_budget)
    used = len(Z_vals)

    if label_budget is None:
        fb = realised_budget[np.isfinite(realised_budget)]
        label_budget = float(np.median(fb)) if fb.size else float("nan")

    sZ = np.where(np.isfinite(Z_vals), -Z_vals, -np.inf)
    sg = -gamma_vals
    ss = np.sqrt(np.clip(sigma_vals, 0.0, None))

    def _labels(budget):
        return np.array([int((pd <= budget).any() and fl[pd <= budget].any())
                         for pd, fl in zip(proxy_lists, flip_lists)])

    auc_Z_eta = np.empty(len(eta_grid)); auc_g_eta = np.empty(len(eta_grid))
    auc_s_eta = np.empty(len(eta_grid)); flip_rate = np.empty(len(eta_grid))
    for j, eta in enumerate(eta_grid):
        lab = _labels(eta); flip_rate[j] = lab.mean()
        auc_Z_eta[j] = _auc(sZ, lab); auc_g_eta[j] = _auc(sg, lab); auc_s_eta[j] = _auc(ss, lab)

    fin = np.isfinite(Z_vals) & np.isfinite(realised_budget)
    rZ, pZ = spearmanr(Z_vals[fin], realised_budget[fin]) if fin.sum() >= 3 else (np.nan, np.nan)
    rg = spearmanr(gamma_vals[fin], realised_budget[fin])[0] if fin.sum() >= 3 else np.nan
    rs = spearmanr(ss[fin], realised_budget[fin])[0] if fin.sum() >= 3 else np.nan
    lab_lb = _labels(label_budget) if np.isfinite(label_budget) else np.zeros(used, int)
    auc_lb = (_auc(sZ, lab_lb), _auc(sg, lab_lb), _auc(ss, lab_lb))

    boot_auc, boot_rho = [], []
    for _ in range(n_boot):
        bs = rng.integers(0, used, size=used)
        lab_b = np.array([int((proxy_lists[i] <= label_budget).any()
                              and flip_lists[i][proxy_lists[i] <= label_budget].any())
                          for i in bs]) if np.isfinite(label_budget) else np.zeros(used, int)
        boot_auc.append(_auc(sZ[bs], lab_b))
        fb = bs[np.isfinite(Z_vals[bs]) & np.isfinite(realised_budget[bs])]
        if fb.size >= 3:
            boot_rho.append(float(spearmanr(Z_vals[fb], realised_budget[fb])[0]))
    def _ci(v):
        v = np.asarray([x for x in v if np.isfinite(x)], float)
        return (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) if v.size else (np.nan, np.nan)

    return {
        "mode": "cross_chart_soft_to_cloud",
        "eta": np.asarray(eta_grid, float), "n_used": used,
        "label_budget": float(label_budget),
        "Z_vals": Z_vals, "realised_flip_budget": realised_budget,
        "spearman_Z_vs_budget": float(rZ), "spearman_gamma_vs_budget": float(rg),
        "spearman_sigma_vs_budget": float(rs), "spearman_pvalue": float(pZ),
        "spearman_Z_ci": _ci(boot_rho),
        "auc_at_label_budget": float(auc_lb[0]),
        "auc_gamma_at_label_budget": float(auc_lb[1]),
        "auc_sigma_at_label_budget": float(auc_lb[2]),
        "auc_at_label_budget_ci": _ci(boot_auc),
        "auc_vs_eta": auc_Z_eta, "auc_gamma_vs_eta": auc_g_eta,
        "auc_sigma_vs_eta": auc_s_eta, "flip_rate_vs_eta": flip_rate,
    }

def experiment_trace_sensitivity(records, nu_frac: float = 1e-3):
    """Empirical mean trace sensitivity T_bar_n = n^{-1} sum tr S(X_i) (Theorem 6.4)."""
    from geometry import trace_S
    tr = []
    for r in records:
        _, B = pullback_matrices(r["J_M"], r["J_P"])
        t, _ = trace_S(r["J_M"], B, nu_frac)
        tr.append(t)
    tr = np.asarray(tr, float)
    return {"trace_S": tr, "T_bar_n": float(tr.mean()),
            "trace_S_median": float(np.median(tr))}
