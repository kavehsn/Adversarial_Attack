"""
Entrypoint for the empirical exercise of Section 8.

Pipeline:
  1. Load FinBERT (target M) and all-MiniLM-L6-v2 (proxy P); verify shared vocab.
  2. Load the Financial PhraseBank all-agreement subset; subsample n inputs.
  3. Build a single fixed random projection G_flat for the soft-token chart.
  4. For each sentence:
       a) Tokenise to a fixed length L_max; compute base embeddings e_M(x), e_P(x).
       b) Construct the logit-gap effective readout (w, b).
       c) [soft-token] Compute J_M, J_P by forward-mode JVP through G_flat.
       d) [cloud]      Generate K paraphrases; compute D_M, D_P (the chart Jacobians).
  5. Run the three experiments per chart; save figures.

Typical use:

    # Soft-token chart only (no paraphrase generation):
    python main.py --n 500 --q 64 --run_soft_token

    # Full run including the cloud chart (requires PEGASUS or external paraphrases):
    python main.py --n 500 --q 64 --K 128 --run_soft_token --run_cloud

Compute notes:
  - Forward-mode JVP cost is q forward passes per model per input. For n=500,
    q=64, two models, that's ~64000 BERT forward passes -- order 10-30 minutes
    on a single modern GPU.
  - The PEGASUS paraphraser is a placeholder; swap in TuretkenLeippold for the
    actual experiment (paraphrase.py).
"""

import argparse
import os
import pickle
from dataclasses import dataclass

import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm

from chart_cloud import paraphrase_cloud_chart
from chart_soft_token import (
    apply_perturbation_soft_token,
    compute_jacobian_soft_token,
    make_random_projection,
)
from experiments import (
    experiment_1_sigma_lambda,
    experiment_2_linearisation_error,
    experiment_2prime_finite_search,
    experiment_3_attackability_curve,
)
from models import (
    check_vocab_alignment,
    get_logit_gap_readout,
    hard_forward_proxy,
    hard_forward_target,
    initial_soft_token_logits,
    load_proxy_model,
    load_target_model,
    soft_token_forward_proxy,
    soft_token_forward_target,
)
from paraphrase import PegasusParaphraser
from plots import (
    plot_attackability_curve,
    plot_finite_search,
    plot_linearisation_error,
    plot_sigma_vs_lambda,
)


@dataclass
class Config:
    n: int = 500
    q: int = 64
    K: int = 128
    tau: float = 10.0
    rho_frac: float = 1e-3
    eta_max: float = 1.0
    n_eta: int = 40
    max_len: int = 64
    seed: int = 42
    out: str = "figures"
    run_soft_token: bool = True
    run_cloud: bool = False
    save_records: bool = True


# ---------------------------------------------------------------------------
# Per-input processing
# ---------------------------------------------------------------------------

def process_input_soft_token(
    text, true_label,
    model_M, tok_M, model_P, G_flat, cfg, device,
):
    """Build a record for the soft-token chart for one input."""
    enc = tok_M(
        text,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=cfg.max_len,
    )
    input_ids = enc["input_ids"][0].to(device)
    attention_mask = enc["attention_mask"][0].to(device)
    V = tok_M.vocab_size

    logits_0 = initial_soft_token_logits(input_ids, V, tau=cfg.tau, device=device)

    # Base embeddings.
    e_M_x = soft_token_forward_target(model_M, logits_0, attention_mask)
    e_P_x = soft_token_forward_proxy(model_P, logits_0, attention_mask)

    # Logit-gap readout.
    w_t, b_t, _ = get_logit_gap_readout(model_M, e_M_x, true_label)

    # Jacobians via forward-mode JVP.
    J_M, _ = compute_jacobian_soft_token(
        lambda lg, am: soft_token_forward_target(model_M, lg, am),
        logits_0, G_flat, cfg.q, attention_mask,
    )
    J_P, _ = compute_jacobian_soft_token(
        lambda lg, am: soft_token_forward_proxy(model_P, lg, am),
        logits_0, G_flat, cfg.q, attention_mask,
    )

    # Bind a perturbation callback for Experiment 2 (closes over the input).
    def _perturb_fn(u_np):
        u = torch.as_tensor(u_np, device=device, dtype=logits_0.dtype)
        e_pert = apply_perturbation_soft_token(
            lambda lg, am: soft_token_forward_target(model_M, lg, am),
            logits_0, G_flat, u, attention_mask,
        )
        return e_pert.detach().cpu().numpy()

    return {
        "text": text,
        "true_label": true_label,
        "e_M_x": e_M_x.detach().cpu().numpy(),
        "e_P_x": e_P_x.detach().cpu().numpy(),
        "J_M": J_M.detach().cpu().numpy(),
        "J_P": J_P.detach().cpu().numpy(),
        "w": w_t.detach().cpu().numpy(),
        "b": float(b_t.detach().cpu().numpy()),
        "perturb_fn": _perturb_fn,
    }


def process_input_cloud(
    text, true_label, paraphrases,
    model_M, tok_M, model_P, tok_P, soft_record, cfg, device,
):
    """Build a record for the cloud chart for one input."""
    # Embed each paraphrase under both models.
    e_M_par, e_P_par = [], []
    for ptxt in paraphrases:
        enc_M = tok_M(
            ptxt, return_tensors="pt", truncation=True,
            padding="max_length", max_length=cfg.max_len,
        )
        enc_P = tok_P(
            ptxt, return_tensors="pt", truncation=True,
            padding="max_length", max_length=cfg.max_len,
        )
        ids_M = enc_M["input_ids"][0].to(device)
        am_M = enc_M["attention_mask"][0].to(device)
        ids_P = enc_P["input_ids"][0].to(device)
        am_P = enc_P["attention_mask"][0].to(device)

        e_M_par.append(hard_forward_target(model_M, ids_M, am_M).cpu().numpy())
        e_P_par.append(hard_forward_proxy(model_P, ids_P, am_P).cpu().numpy())

    e_M_par = np.stack(e_M_par)            # (K, d_M)
    e_P_par = np.stack(e_P_par)            # (K, d_P)

    J_M, J_P = paraphrase_cloud_chart(
        soft_record["e_M_x"], e_M_par,
        soft_record["e_P_x"], e_P_par,
    )

    return {
        "text": text,
        "true_label": true_label,
        "e_M_x": soft_record["e_M_x"],
        "e_P_x": soft_record["e_P_x"],
        "J_M": J_M,
        "J_P": J_P,
        "D_M": J_M,          # alias for clarity in Experiment 2'
        "D_P": J_P,
        "w": soft_record["w"],
        "b": soft_record["b"],
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(cfg: Config):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    os.makedirs(cfg.out, exist_ok=True)

    # --- Models ---
    print("Loading FinBERT (target) and all-MiniLM-L6-v2 (proxy)...")
    model_M, tok_M = load_target_model(device)
    model_P, tok_P = load_proxy_model(device)
    if not check_vocab_alignment(tok_M, tok_P):
        raise RuntimeError(
            "FinBERT and MiniLM tokenizers do not share an exact vocabulary mapping; "
            "the soft-token chart is not well-defined under this pairing."
        )
    print(f"Vocab alignment OK. |V| = {tok_M.vocab_size}")

    # --- Data ---
    print("Loading Financial PhraseBank (sentences_allagree)...")
    ds = load_dataset(
        "financial_phrasebank", "sentences_allagree", trust_remote_code=True,
    )["train"]
    rng = np.random.default_rng(cfg.seed)
    idx = rng.choice(len(ds), size=min(cfg.n, len(ds)), replace=False)
    sentences = [(ds[int(i)]["sentence"], int(ds[int(i)]["label"])) for i in idx]
    print(f"Selected n = {len(sentences)} sentences.")

    # --- Random projection (single, fixed) ---
    L_max = cfg.max_len
    V = tok_M.vocab_size
    print(f"Building fixed projection G_flat: shape = ({L_max * V}, {cfg.q})...")
    G_flat = make_random_projection(L_max, V, cfg.q, seed=cfg.seed + 1, device=device)

    # --- Paraphraser (optional) ---
    paraphraser = None
    if cfg.run_cloud:
        print("Loading PEGASUS paraphraser (placeholder)...")
        paraphraser = PegasusParaphraser(device=device)

    # --- Per-input processing ---
    soft_records, cloud_records = [], []
    for text, true_label in tqdm(sentences, desc="Inputs"):
        if cfg.run_soft_token:
            sr = process_input_soft_token(
                text, true_label, model_M, tok_M, model_P, G_flat, cfg, device,
            )
            soft_records.append(sr)
        else:
            sr = None

        if cfg.run_cloud:
            paraphrases = paraphraser.paraphrase(text, K=cfg.K)
            if sr is None:
                # Cheap base embeddings if we skipped the soft-token chart.
                enc = tok_M(
                    text, return_tensors="pt", truncation=True,
                    padding="max_length", max_length=cfg.max_len,
                )
                ids_M = enc["input_ids"][0].to(device)
                am_M = enc["attention_mask"][0].to(device)
                enc_P = tok_P(
                    text, return_tensors="pt", truncation=True,
                    padding="max_length", max_length=cfg.max_len,
                )
                ids_P = enc_P["input_ids"][0].to(device)
                am_P = enc_P["attention_mask"][0].to(device)
                e_M_x = hard_forward_target(model_M, ids_M, am_M).cpu().numpy()
                e_P_x = hard_forward_proxy(model_P, ids_P, am_P).cpu().numpy()
                with torch.no_grad():
                    e_M_t = torch.as_tensor(e_M_x, device=device)
                    w_t, b_t, _ = get_logit_gap_readout(model_M, e_M_t, true_label)
                sr_stub = {
                    "e_M_x": e_M_x, "e_P_x": e_P_x,
                    "w": w_t.cpu().numpy(),
                    "b": float(b_t.cpu().numpy()),
                }
            else:
                sr_stub = sr
            cr = process_input_cloud(
                text, true_label, paraphrases,
                model_M, tok_M, model_P, tok_P, sr_stub, cfg, device,
            )
            cloud_records.append(cr)

    # --- Persist records (without the perturbation closure) ---
    if cfg.save_records:
        def _strip(r):
            return {k: v for k, v in r.items() if k != "perturb_fn"}
        with open(os.path.join(cfg.out, "soft_records.pkl"), "wb") as f:
            pickle.dump([_strip(r) for r in soft_records], f)
        if cloud_records:
            with open(os.path.join(cfg.out, "cloud_records.pkl"), "wb") as f:
                pickle.dump(cloud_records, f)

    # --- Experiments ---
    eta_grid = np.linspace(0.01, cfg.eta_max, cfg.n_eta)

    if cfg.run_soft_token and soft_records:
        print("Soft-token: Experiment 1 (Sigma vs lambda*)")
        r1 = experiment_1_sigma_lambda(soft_records, rho_frac=cfg.rho_frac)
        print(f"  violations of Sigma <= lambda*: {r1['violations']} / {len(soft_records)}")
        plot_sigma_vs_lambda(r1, os.path.join(cfg.out, "fig_sigma_lambda_soft.pdf"))

        print("Soft-token: Experiment 2 (linearisation error)")
        r2 = experiment_2_linearisation_error(soft_records, eta_grid, rho_frac=cfg.rho_frac)
        plot_linearisation_error(r2, os.path.join(cfg.out, "fig_linearisation_error.pdf"))

        print("Soft-token: Experiment 3 (attackability curve)")
        r3 = experiment_3_attackability_curve(soft_records, eta_grid, rho_frac=cfg.rho_frac)
        plot_attackability_curve(r3, os.path.join(cfg.out, "fig_attackability_curve.pdf"))

    if cfg.run_cloud and cloud_records:
        print("Cloud: Experiment 1' (Sigma vs lambda*)")
        r1c = experiment_1_sigma_lambda(cloud_records, rho_frac=cfg.rho_frac)
        print(f"  violations of Sigma <= lambda*: {r1c['violations']} / {len(cloud_records)}")
        plot_sigma_vs_lambda(r1c, os.path.join(cfg.out, "fig_sigma_lambda_cloud.pdf"))

        print("Cloud: Experiment 2' (finite-search attackability)")
        K_grid = [8, 16, 32, 64, 128]
        rfs = experiment_2prime_finite_search(cloud_records, K_grid, eta_grid, seed=cfg.seed + 2)

        print("Cloud: Experiment 3' (attackability curve)")
        r3c = experiment_3_attackability_curve(cloud_records, eta_grid, rho_frac=cfg.rho_frac)
        plot_finite_search(
            rfs, eta_grid, os.path.join(cfg.out, "fig_finite_search.pdf"),
            A_hat=r3c["A_hat"], dkw_half=r3c["dkw_half_width"],
        )
        plot_attackability_curve(r3c, os.path.join(cfg.out, "fig_attackability_curve_cloud.pdf"))

    print(f"\nDone. Figures and records saved to: {cfg.out}/")


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=500)
    p.add_argument("--q", type=int, default=64)
    p.add_argument("--K", type=int, default=128)
    p.add_argument("--tau", type=float, default=10.0)
    p.add_argument("--rho_frac", type=float, default=1e-3)
    p.add_argument("--eta_max", type=float, default=1.0)
    p.add_argument("--n_eta", type=int, default=40)
    p.add_argument("--max_len", type=int, default=64)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=str, default="figures")
    p.add_argument("--run_soft_token", action="store_true")
    p.add_argument("--run_cloud", action="store_true")
    p.add_argument("--no_save_records", action="store_true")
    a = p.parse_args()
    return Config(
        n=a.n, q=a.q, K=a.K, tau=a.tau, rho_frac=a.rho_frac,
        eta_max=a.eta_max, n_eta=a.n_eta, max_len=a.max_len,
        seed=a.seed, out=a.out,
        run_soft_token=a.run_soft_token or (not a.run_cloud),
        run_cloud=a.run_cloud,
        save_records=not a.no_save_records,
    )


if __name__ == "__main__":
    cfg = _parse_args()
    main(cfg)
