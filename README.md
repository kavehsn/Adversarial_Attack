# Empirical exercise for Section 8

Implements the soft-token chart (8.2) and paraphrase-cloud chart (8.3) for the
two-embedding attackability framework.

## Files

| File | Purpose |
|---|---|
| `models.py` | Load FinBERT and all-MiniLM-L6-v2; soft-token forward passes; logit-gap readout. |
| `chart_soft_token.py` | Forward-mode JVP through the soft-token relaxation, fixed Gaussian projection. |
| `chart_cloud.py` | Cloud-chart Jacobians from paraphrase displacements. |
| `paraphrase.py` | Paraphrase generation interface. **Default is a PEGASUS placeholder — swap for Türetken & Leippold (2026).** |
| `geometry.py` | `lambda_star`, `Sigma_w`, `Z_w`, the closed-form worst direction. |
| `experiments.py` | The four experiments (1, 2, 3, 2′). |
| `plots.py` | Figure helpers. |
| `main.py` | Orchestration / CLI. |

## Setup

```bash
pip install -r requirements.txt
```

Requires CUDA for tractable runtime. CPU-only works but is slow.

## Run

Soft-token chart only (Exps 1, 2, 3 — no paraphrase generation needed):

```bash
python main.py --n 500 --q 64 --max_len 64 --run_soft_token
```

Full run including the cloud chart (Exps 1, 2, 3, 1′, 2′, 3′):

```bash
python main.py --n 500 --q 64 --K 128 --run_soft_token --run_cloud
```

Outputs land in `./figures/`:

- `fig_sigma_lambda_soft.pdf` — Experiment 1.
- `fig_linearisation_error.pdf` — Experiment 2.
- `fig_attackability_curve.pdf` — Experiment 3.
- `fig_sigma_lambda_cloud.pdf` — Experiment 1′.
- `fig_finite_search.pdf` — Experiment 2′.
- `fig_attackability_curve_cloud.pdf` — Experiment 3′.
- `soft_records.pkl`, `cloud_records.pkl` — per-input arrays for offline analysis.

## What you almost certainly want to swap

**`paraphrase.py`**. The default `PegasusParaphraser` is a generic
high-temperature paraphraser, not the Türetken–Leippold attack. For the
actual experiment of Section 8.3, implement a class with a
`paraphrase(text, K)` method that calls into their attack code, or
pre-generate paraphrases offline and use `ExternalParaphraser` with a
dict.

## Compute notes

The bottleneck is the forward-mode JVP loop in `chart_soft_token.py`:
`q` forward passes per model per input. For `n = 500`, `q = 64`, two
models, that's `~64 000` BERT forward passes per chart, on the order of
10–30 minutes on a modern GPU. Lower `n` or `q` if memory or time is
tight. The cloud chart is cheap by comparison.

The random projection `G_flat` is `(L_max · |V|) × q` floats; for
`L_max = 64`, `|V| ≈ 30 522`, `q = 64`, this is ~500 MB. Held on GPU
once.

## Notes on correctness checks

- The vocab-alignment check in `models.check_vocab_alignment` is run at
  start-up; if it fails the soft-token chart is ill-defined for this
  model pair, and `main.py` aborts.
- The `ρ`-regularisation defaults to `ρ = 10⁻³ · tr(B)/q`. Sweep this
  via `--rho_frac` to verify stability (the paper recommends
  `[10⁻⁴, 10⁻²]`).
- `experiment_1_sigma_lambda` reports the number of inputs on which
  `Σ_w(x) > λ*(x)` (which should always be zero by Theorem 3.3). A
  non-zero count indicates a numerical issue, almost always from an
  ill-conditioned `B`.
