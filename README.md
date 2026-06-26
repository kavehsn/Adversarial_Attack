# Empirical exercise for Section 8

Implements the soft-token chart (8.2) and paraphrase-cloud chart (8.3) for the
two-embedding attackability framework. This revision incorporates the reviewer's
amendments to the empirical section.

## Files

| File | Purpose |
|---|---|
| `models.py` | Load FinBERT and all-MiniLM-L6-v2; soft-token forward passes; logit-gap readout. |
| `chart_soft_token.py` | Forward-mode JVP through the soft-token relaxation; **per-input** random projection. |
| `chart_cloud.py` | Cloud-chart Jacobians from paraphrase displacements. |
| `paraphrase.py` | Paraphrase generation interface. **Default is a PEGASUS placeholder — swap for Türetken & Leippold (2026).** |
| `paraphrase_gpt4o.py` | GPT-4o paraphraser with proxy-similarity filtering. |
| `geometry.py` | `lambda_star`, `Sigma_w`, `Z_w`, `beta_min`, the closed-form worst direction, and `readout_aligned_direction`. |
| `experiments.py` | Experiments 1, 2, 3, 2′, the finite-search **coverage diagnostic**, and the **predictive-validity** test. |
| `plots.py` | Figure helpers. |
| `main.py` | Orchestration / CLI. |
| `replot.py` | Re-run experiments and figures from cached records. |

## What changed in this revision

- **Ridge renamed `ρ → ν`** everywhere (`--nu_frac`, `nu_frac`), so the
  regularisation ridge is distinct from the Section-6 margin slack `ρ`.
- **`τ` default 10 → 20.** Over WordPiece (`|V| ≈ 30522`), `τ=10` puts only
  ~0.42 mass on the original token; `τ=20` gives > 0.9999, so the soft-token
  chart is centred at the intended near-one-hot input.
- **Per-input random projection `G`** is now the default (`--redraw_G`); the old
  single fixed `G` is available via `--no_redraw_G` for a sensitivity report.
- **Realised flips are logged** (`record["para_preds"]`), enabling the new
  predictive-validity test of `Z_w`.
- **Finite search** ships a full-`K` oracle curve and a 1-D coverage diagnostic
  along `r_w(x)`, and is framed as the asymmetry message rather than a test of
  the covering proposition.
- **Attackability overlays** are the deterministic empirical-inclusion bound
  (plug-in `(1−β)` quantile), optionally with a calibration split.
- **`run_metadata.json`** records model ids, split, seed, truncation, padding,
  `q`, `ν`, `τ`, `redraw_G`, paraphrase counts, proxy-valid counts, and
  degeneracy counts (`Σ_w=0`, near-zero `Σ_w`, degenerate `B`).

## Setup

```bash
pip install -r requirements.txt
```

Requires CUDA for tractable runtime. CPU-only works but is slow.

## Run

Soft-token chart only (Exps 1, 2, 3 — no paraphrase generation needed):

```bash
python main.py --n 1000 --q 64 --max_len 64 --run_soft_token
```

Full run including the cloud chart (Exps 1, 2, 3, 1′, 2′, 3′, predictive validity):

```bash
python main.py --n 1000 --q 64 --K 128 --run_soft_token --run_cloud
```

Sensitivity to the projection draw (fixed `G`, several seeds):

```bash
for s in 1 2 3 4 5; do python main.py --n 1000 --q 64 --seed $s --no_redraw_G; done
```

Outputs land in `./figures/`:

- `fig_sigma_lambda_soft.pdf` — Experiment 1.
- `fig_linearisation_error.pdf` — Experiment 2 (Δs^nl vs Δs^lin + normalised residual).
- `fig_attackability_curve.pdf` — Experiment 3.
- `fig_sigma_lambda_cloud.pdf` — Experiment 1′.
- `fig_finite_search.pdf` — Experiment 2′ (+ oracle + coverage diagnostic).
- `fig_attackability_curve_cloud.pdf` — Experiment 3′.
- `fig_predictive_validity.pdf` — predictive validity of `Z_w`.
- `soft_records.pkl`, `cloud_records.pkl` — per-input arrays for offline analysis.
- `run_metadata.json` — reproducibility metadata.

## What you almost certainly want to swap

**`paraphrase.py`**. The default `PegasusParaphraser` is a generic
high-temperature paraphraser, not the Türetken–Leippold attack. For the actual
experiment of Section 8.3, implement a class with a `paraphrase(text, K)` method
that calls into their attack code, or pre-generate paraphrases offline and use
`ExternalParaphraser` with a dict (see `generate_paraphrases.py`).

## Compute notes

The bottleneck is the forward-mode JVP loop in `chart_soft_token.py`: `q`
forward passes per model per input. With **per-input `G`**, each input also pays
a thin QR of an `(L·|V|) × q` matrix (a few seconds on GPU); for a one-off
sensitivity check use `--no_redraw_G`. The cloud chart is cheap by comparison.

The random projection `G_flat` is `(L_max · |V|) × q` floats; for `L_max = 64`,
`|V| ≈ 30522`, `q = 64`, this is ~500 MB, built (and discarded) once per input
under `--redraw_G`.

## Notes on correctness checks

- The vocab-alignment check in `models.check_vocab_alignment` runs at start-up;
  if it fails the soft-token chart is ill-defined for this model pair and
  `main.py` aborts.
- The `ν`-regularisation defaults to `ν = 10⁻³ · tr(B)/q`. Sweep via `--nu_frac`
  to verify stability (the paper recommends `[10⁻⁴, 10⁻²]`); `λ*` and `Σ_w` are
  chart-invariant only at `ν → 0`, so the swept values approximate the intrinsic
  quantities.
- `experiment_1_sigma_lambda` reports the number of inputs on which
  `Σ_w(x) > λ*(x)` (should be zero by Theorem 3.3, which assumes `‖w‖ = 1`; the
  logit-gap readout is not unit-norm, so pass `normalize_w=True` to reproduce the
  theorem's convention exactly — everything else is scale-invariant in `w`).
- The predictive-validity test needs `cloud_records` to carry `para_preds`;
  records produced by this revision of `main.py` do. Older caches must be
  regenerated.
