# Generalised Eigenvalue Geometry of Semantic Adversarial Attacks

[![arXiv](https://img.shields.io/badge/arXiv-2606.19212-b31b1b.svg)](https://arxiv.org/abs/2606.19212)

This repository accompanies the research paper:
> **Generalised Eigenvalue Geometry of Semantic Adversarial Attacks**
> *Martin Anthony & Kaveh Salehzadeh Nobari (2026)*
> <https://arxiv.org/abs/2606.19212>

It provides the code, local-geometry diagnostics, and empirical-verification
pipeline behind the two-embedding attackability framework of the paper. The code
implements the soft-token chart (Section 8.2) and the paraphrase-cloud chart
(Section 8.3), the generalised-eigenvalue geometry of the matrix pencil
`(A, B)`, the attackability index `λ*(x)`, the readout sensitivity `Σ_w(x)`, the
attackability-adjusted margin `Z_w(x)`, and the finite-search diagnostics, and
reproduces the figures and the proxy-embedding sweep of Section 8 on a deployed
FinBERT financial-sentiment classifier.

---

## What This Repository Provides

[#what-this-repository-provides](#what-this-repository-provides)

- Local pullback geometry of two embedders — the pencil `(A, B) = (Jₘᵀ Jₘ, J_Pᵀ J_P)`, the attackability index `λ*(x) = λ_max(A, B)`, the readout sensitivity `Σ_w(x)`, and the adjusted margin `Z_w(x) = γ_w(x) / √Σ_w(x)`
- Two charts for the discrete→continuous relaxation: a **soft-token chart** (forward-mode Jacobian–vector products) and a **paraphrase-cloud chart** (GPT-4o paraphrase displacements)
- The pointwise certificate `Σ_w(x) ≤ λ*(x)` (Theorem 3.3) and the linearised prediction-flip condition (Theorem 3.1)
- Population attackability curves with Dvoretzky–Kiefer–Wolfowitz bands and deterministic empirical-inclusion overlays (Section 4)
- Predictive-validity test of `Z_w` against realised paraphrase flips, with **bare-margin** and **displacement-only** baselines and a soft→cloud cross-chart variant (Section 8)
- A **proxy-embedding sweep** (FinBERT fixed; MiniLM / E5 / BGE proxies) reproducing the Section 8.5 table
- Finite-search attackability with a full-`K` oracle curve and a one-dimensional coverage diagnostic along `r_w(x)` (Section 7)
- Empirical trace sensitivity `T̄ₙ = n⁻¹ Σᵢ tr S(Xᵢ)` for the trace-controlled Rademacher bound (Theorem 6.4)
- Publication-ready figures, regenerable from cached records **without a model pass**

---

## Getting Started

[#getting-started](#getting-started)

### 1️⃣ Clone the repository

```
git clone https://github.com/kavehsn/Adversarial_Attack.git
cd Adversarial_Attack
```

### 2️⃣ Create & activate an environment

```
python -m venv .venv
# Windows:        .venv\Scripts\activate
# macOS / Linux:  source .venv/bin/activate
pip install -r requirements.txt
```

A CUDA GPU is recommended for the soft-token chart (it runs `q` forward passes
per input); CPU-only works but is slow. The cloud chart is cheap by comparison.

### 3️⃣ (Optional) Pre-generate the paraphrase cloud

The cloud chart (Section 8.3) uses GPT-4o paraphrases, generated once and cached
to `figures/paraphrases.json`:

```
export OPENAI_API_KEY=sk-...
python generate_paraphrases.py --n 200 --K 128 --seed 42
```

### 4️⃣ Run the pipeline

Soft-token chart only (Experiments 1, 2, 3 — no paraphrases needed):

```
python main.py --n 200 --q 32 --max_len 32 --run_soft_token --seed 42
```

Full run including the cloud chart (Experiments 1–3, 1′, 2′, 3′, predictive validity):

```
python main.py --n 200 --q 32 --max_len 32 --K 128 --run_soft_token --run_cloud --seed 42
```

Outputs (figures, per-input record pickles, and `run_metadata.json`) land in
`./figures/`.

---

## Files

| File | Purpose |
|---|---|
| `models.py` | Load the target (FinBERT) and a **configurable** proxy embedder (`--proxy_model`, `--target_model`); soft-token forward passes; logit-gap readout. |
| `geometry.py` | `lambda_star`, `sigma_w`, `Z_w`, `trace_S`, `beta_min`, the closed-form worst direction `u*_flip`, and the readout-aligned direction `r_w(x)`. |
| `chart_soft_token.py` | Forward-mode JVP through the soft-token relaxation; **per-input** random projection `G`. |
| `chart_cloud.py` | Cloud-chart Jacobians `J_M = D_M`, `J_P = D_P` from paraphrase displacements. |
| `paraphrase_gpt4o.py` | GPT-4o paraphraser with Sentence-BERT proxy-similarity filtering (the Section 8.3 cloud). |
| `paraphrase.py` | Generic paraphrase interface (PEGASUS placeholder / `ExternalParaphraser`). |
| `generate_paraphrases.py` | Pre-generate and cache the GPT-4o paraphrase sets consumed by `--run_cloud`. |
| `experiments.py` | Experiments 1–3 and 1′–3′, the finite-search coverage diagnostic, the trace-sensitivity measurement, and the predictive-validity test (with margin / displacement baselines and the soft→cloud cross-chart variant). |
| `plots.py` | Figure helpers used by `main.py`. |
| `replot.py` | Rebuild the Section-8 figures **from cached records — no model pass**. |
| `main.py` | Orchestration / CLI. |

---

## The Proxy-Embedding Sweep (Section 8.5)

[#the-proxy-embedding-sweep](#the-proxy-embedding-sweep)

The framework concerns the interaction of **two** embedding geometries through
the pencil `(A, B)`. To isolate the proxy's contribution, the target (FinBERT)
and its readout are held fixed while the proxy embedding is varied — the
paraphrases, the target Jacobian `J_M`, and the realised FinBERT flips are
unchanged, and only `B = J_Pᵀ J_P` (and hence `λ*`, `Σ_w`, and the proxy-metric
flip budget) changes. The proxy is selected with `--proxy_model`. Because the
soft-token chart requires a shared WordPiece vocabulary, the sweep runs on the
**cloud chart only**:

```
python main.py --n 200 --q 32 --max_len 32 --K 128 --run_cloud --seed 42 --proxy_model sentence-transformers/all-MiniLM-L6-v2
python main.py --n 200 --q 32 --max_len 32 --K 128 --run_cloud --seed 42 --proxy_model intfloat/e5-base-v2
python main.py --n 200 --q 32 --max_len 32 --K 128 --run_cloud --seed 42 --proxy_model BAAI/bge-base-en-v1.5
```

Each run prints the three quantities that populate the Section-8.5 table:

- the median slack `log₁₀(λ*/Σ_w)` with a bootstrap CI,
- the empirical trace sensitivity `T̄ₙ`, and
- the within-split AUC and Spearman correlation of `−Z_w`, the bare margin
  `γ_w`, and the displacement `√Σ_w` against the realised flip budget.

E5 and BGE share FinBERT's `bert-base-uncased` vocabulary, so they load as plain
`BertModel` encoders and pass the start-up vocab-alignment check; they are used
as mean-pooled encoders without instruction prefixes, consistent with how
all-MiniLM is used, which keeps the proxy-metric comparison clean.

> **Note on the generation gate vs. the proxy budget.** The cloud is built with
> a fixed Sentence-BERT cosine **floor** of `0.80` (i.e. proxy distance
> `≤ √(2 − 2·0.80) ≈ 0.63`), so the retained candidates lie within a fixed proxy
> neighbourhood of `x`. The proxy budget `η` of the theory is then swept *inside*
> this gate: no new candidates appear beyond `η ≈ 0.63`, so the large-`η` tail of
> the finite-search panel reflects the generation gate rather than the proxy ball.

---

## Regenerating Figures Without Rerunning the Pipeline

[#regenerating-figures](#regenerating-figures)

`replot.py` recomputes the six Section-8 diagnostics from the cached
`*_records.pkl` files and re-renders every figure. It loads no model and runs in
seconds:

```
python replot.py --figdir figures
```

To restyle only some figures and leave the others untouched, write to a separate
directory and copy back the ones you want:

```
python replot.py --figdir figures --outdir figures_restyled
# then e.g. copy figures_restyled/fig_predictive_validity.pdf and
# figures_restyled/fig_finite_search.pdf back into figures/
```

`fig_linearisation_error.pdf` is the one exception: its nonlinear curve needs a
model forward pass, which is not cached, so it is not regenerated from records.

---

## Outputs

[#outputs](#outputs)

Written to the `--figdir` / `--outdir` directory (default `./figures/`):

- `fig_sigma_lambda_soft.pdf` — Experiment 1 (`Σ_w ≤ λ*`, soft-token chart).
- `fig_linearisation_error.pdf` — Experiment 2 (`Δsⁿˡ` vs `Δsˡⁱⁿ` + normalised residual).
- `fig_attackability_curve.pdf` — Experiment 3 (soft-token).
- `fig_sigma_lambda_cloud.pdf` — Experiment 1′ (`Σ_w ≤ λ*`, cloud chart).
- `fig_finite_search.pdf` — Experiment 2′ (finite search + oracle + coverage diagnostic).
- `fig_attackability_curve_cloud.pdf` — Experiment 3′ (cloud).
- `fig_predictive_validity.pdf` — predictive validity of `Z_w` (with `γ_w` and `√Σ_w` baselines).
- `soft_records.pkl`, `cloud_records.pkl` — per-input arrays for offline analysis and `replot.py`.
- `run_metadata.json` — reproducibility metadata (model ids, split, seed, `q`, `ν`, `τ`, `redraw_G`, paraphrase / proxy-valid counts, degeneracy counts, and the cloud trace sensitivity `T̄ₙ`).

---

## Implementation Notes

[#implementation-notes](#implementation-notes)

- **Configurable target / proxy.** `--target_model` and `--proxy_model` select the
  two embedders; the Section-8.5 sweep varies `--proxy_model` over
  `all-MiniLM-L6-v2`, `e5-base-v2`, and `bge-base-en-v1.5` with FinBERT fixed.
- **Trace sensitivity.** `experiment_trace_sensitivity` reports
  `T̄ₙ = n⁻¹ Σᵢ tr S(Xᵢ)`, the dimension-free quantity entering the Rademacher
  bound (Theorem 6.4); it is printed and stored in `run_metadata.json`.
- **Slack reporting.** `experiment_1_sigma_lambda` prints the median
  `log₁₀(λ*/Σ_w)` with a bootstrap CI alongside the count of `Σ_w > λ*` violations.
- **Predictive-validity baselines.** The `Z_w` test also scores the bare margin
  `γ_w` and the displacement `√Σ_w`, in both the within-split and the soft→cloud
  cross-chart regimes.
- **Figures restyled.** Busy legends are placed below the panels and overlapping
  curves are separated by colour / linestyle / marker, so the predictive-validity
  and finite-search panels are legible; the change is purely in `plots.py` /
  `replot.py` and needs no rerun.
- **Ridge renamed `ρ → ν`** (`--nu_frac`, `nu_frac`), distinguishing the
  regularisation ridge from the Section-6 margin slack `ρ`.
- **`τ` default 10 → 20.** Over WordPiece (`|V| ≈ 30522`), `τ = 10` puts only
  ~0.42 mass on the original token; `τ = 20` gives `> 0.9999`, so the soft-token
  chart is centred at the intended near-one-hot input. Experiment 2 alone uses
  `τ = 10`, a smoother operating point at which the first-order regime is resolvable.
- **Per-input random projection `G`** is the default (`--redraw_G`); the old
  single fixed `G` is available via `--no_redraw_G` for a sensitivity report.
- **Realised flips logged** (`record["para_preds"]`), enabling the predictive-validity test.

---

## Compute Notes

[#compute-notes](#compute-notes)

The bottleneck is the forward-mode JVP loop in `chart_soft_token.py`: `q` forward
passes per model per input. With **per-input `G`**, each input also pays a thin QR
of an `(L·|V|) × q` matrix (a few seconds on GPU); for a one-off sensitivity check
use `--no_redraw_G`. The random projection `G_flat` is `(L_max · |V|) × q` floats;
for `L_max = 64`, `|V| ≈ 30522`, `q = 64` this is ~500 MB, built (and discarded)
once per input under `--redraw_G`. The cloud chart only re-embeds the cached
paraphrases, so the proxy sweep is cheap — swapping `--proxy_model` re-embeds the
existing candidates under the new proxy and recomputes the small `K × K` pencil.

---

## Correctness Checks

[#correctness-checks](#correctness-checks)

- The vocab-alignment check in `models.check_vocab_alignment` runs at start-up; if
  it fails the soft-token chart is ill-defined for the model pair and `main.py`
  aborts (use `--run_cloud` only for non-shared-vocab pairs).
- The `ν`-regularisation defaults to `ν = 10⁻³ · tr(B)/q`. Sweep via `--nu_frac`
  to verify stability (the paper recommends `[10⁻⁴, 10⁻²]`); `λ*` and `Σ_w` are
  chart-invariant only as `ν → 0`, so the swept values approximate the intrinsic
  quantities.
- `experiment_1_sigma_lambda` reports the number of inputs on which `Σ_w(x) > λ*(x)`
  (zero by Theorem 3.3, which assumes `‖w‖ = 1`; the logit-gap readout is not
  unit-norm, so pass `normalize_w=True` to reproduce the theorem's convention
  exactly — everything else is scale-invariant in `w`).
- The predictive-validity test needs `cloud_records` to carry `para_preds`;
  records produced by this revision of `main.py` do. Older caches must be regenerated.

---

## Citation

[#citation](#citation)

```bibtex
@article{anthony2026generalised,
  title   = {Generalised Eigenvalue Geometry of Semantic Adversarial Attacks},
  author  = {Anthony, Martin and Salehzadeh Nobari, Kaveh},
  journal = {arXiv preprint arXiv:2606.19212},
  year    = {2026},
  url     = {https://arxiv.org/abs/2606.19212}
}
```
