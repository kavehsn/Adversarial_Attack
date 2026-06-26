"""
Pipeline:
  1. Load FinBERT (target M) and all-MiniLM-L6-v2 (proxy P); verify shared vocab.
  2. Load the Financial PhraseBank all-agreement subset; subsample n inputs.
  3. Keep only inputs whose hard FinBERT prediction agrees with the remapped
     Financial PhraseBank label. This makes realised paraphrase flips unambiguous:
     para_preds[k] != true_label is then also para_preds[k] != base_pred.
  4. For each retained sentence build the soft-token chart (per-input random
     projection G, see revision item 9b) and/or the paraphrase-cloud chart.
  5. Run the experiments per chart; save figures, records, and run metadata.

Typical use:

    # Soft-token chart only, per-input G:
    python main.py --n 1000 --q 64 --run_soft_token

    # Full run including the cloud chart, requiring cached paraphrases:
    python main.py --n 200 --q 32 --K 128 --run_soft_token --run_cloud

    # Sensitivity run only, not the main run:
    python main.py --n 200 --q 32 --K 128 --run_soft_token --run_cloud --no_redraw_G
"""

import argparse
import json
import os
import pickle
import zipfile
from dataclasses import dataclass, replace
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from huggingface_hub import hf_hub_download
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
    experiment_predictive_validity,
    finite_search_coverage_diagnostic,
    experiment_predictive_validity_crosschart,
    experiment_trace_sensitivity,
)
from geometry import pullback_matrices, sigma_w
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
from plots import (
    plot_attackability_curve,
    plot_finite_search,
    plot_linearisation_error,
    plot_predictive_validity,
    plot_sigma_vs_lambda,
)


@dataclass
class Config:
    n: int = 1000
    q: int = 64
    K: int = 128
    tau: float = 20.0
    tau_linearisation: float = 10.0   # Exp 2 only: lower tau so beta is non-degenerate
    nu_frac: float = 1e-3
    eta_max: float = 1.0
    n_eta: int = 40
    target_model: str = "ProsusAI/finbert"
    proxy_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    eta_min_linearisation: float = 1e-3
    eta_max_linearisation: float = 1e-1
    n_eta_linearisation: int = 40
    n_linearisation: int = 64   # inputs used for the linearisation diagnostic (memory + runtime cap)
    max_len: int = 64
    seed: int = 42
    out: str = "figures"
    run_soft_token: bool = True
    run_cloud: bool = False
    save_records: bool = True
    redraw_G: bool = True


def load_financial_phrasebank_train_split() -> List[dict]:
    """Load Financial PhraseBank all-agree split from the raw archive."""
    archive_path = hf_hub_download(
        repo_id="financial_phrasebank",
        repo_type="dataset",
        filename="data/FinancialPhraseBank-v1.0.zip",
    )
    label_to_id = {"negative": 0, "neutral": 1, "positive": 2}
    rows = []
    with zipfile.ZipFile(archive_path) as archive:
        with archive.open("FinancialPhraseBank-v1.0/Sentences_AllAgree.txt") as handle:
            for raw_line in handle:
                line = raw_line.decode("iso-8859-1").strip()
                if not line:
                    continue
                sentence, label = line.rsplit("@", 1)
                rows.append({"sentence": sentence, "label": label_to_id[label]})
    return rows


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _encode(tokenizer, text: str, cfg: Config, device: str):
    enc = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=cfg.max_len,
    )
    return enc["input_ids"][0].to(device), enc["attention_mask"][0].to(device)


@torch.no_grad()
def _finbert_pred(model_M, tok_M, text: str, cfg: Config, device: str) -> int:
    enc = tok_M(
        text,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=cfg.max_len,
    ).to(device)
    logits = model_M(**enc).logits[0]
    return int(logits.argmax().item())


def _normalise_readout(w_t: torch.Tensor, b_t: torch.Tensor, eps: float = 1e-12):
    """
    Normalise the binary logit-gap readout so that ||w||_2 = 1.

    This keeps the classifier and Z_w unchanged, but makes the geometric margin and
    the Sigma_w <= lambda* comparison match the pointwise theorem directly.
    """
    norm = w_t.norm().clamp(min=eps)
    return w_t / norm, b_t / norm


@torch.no_grad()
def _hard_base_record(text, true_label, model_M, tok_M, model_P, tok_P, cfg, device):
    """
    Compute the hard FinBERT and proxy embeddings of the original sentence.

    This is used by the cloud chart. It deliberately does not reuse the soft-token
    base embedding, because Section 8.3 defines D_M and D_P as hard paraphrase
    displacements from the actual input x.
    """
    ids_M, am_M = _encode(tok_M, text, cfg, device)
    ids_P, am_P = _encode(tok_P, text, cfg, device)

    e_M_x = hard_forward_target(model_M, ids_M, am_M)
    e_P_x = hard_forward_proxy(model_P, ids_P, am_P)
    base_logits = model_M.classifier(e_M_x.unsqueeze(0))[0]
    base_pred = int(base_logits.argmax().item())

    w_t, b_t, i_star = get_logit_gap_readout(model_M, e_M_x, true_label)
    w_t, b_t = _normalise_readout(w_t, b_t)

    return {
        "text": text,
        "true_label": int(true_label),
        "base_pred": int(base_pred),
        "competitor_label": int(i_star),
        "e_M_x": e_M_x.detach().cpu().numpy(),
        "e_P_x": e_P_x.detach().cpu().numpy(),
        "w": w_t.detach().cpu().numpy(),
        "b": float(b_t.detach().cpu().item()),
    }


def _remapped_label(row: dict, finbert_label_to_id: Dict[str, int]) -> int:
    """Map Financial PhraseBank ids into the model's actual FinBERT ids."""
    phrasebank_id_to_label = {0: "negative", 1: "neutral", 2: "positive"}
    label_name = phrasebank_id_to_label[int(row["label"])]
    if label_name not in finbert_label_to_id:
        raise RuntimeError(
            f"FinBERT label mapping does not contain {label_name!r}. "
            f"Available labels: {finbert_label_to_id}"
        )
    return int(finbert_label_to_id[label_name])


def _sample_and_filter_finbert_correct(
    ds: Sequence[dict],
    model_M,
    tok_M,
    finbert_label_to_id: Dict[str, int],
    cfg: Config,
    device: str,
) -> Tuple[List[Tuple[str, int]], dict]:
    """
    Reproduce the old sampling design, then keep only hard-FinBERT-correct inputs.

    We do not sample additional replacement sentences after filtering, because cached
    paraphrases are usually generated for exactly this seed/n subset by
    generate_paraphrases.py. This preserves alignment with figures/paraphrases.json.
    """
    rng = np.random.default_rng(cfg.seed)
    idx = rng.choice(len(ds), size=min(cfg.n, len(ds)), replace=False)

    selected = []
    for i in idx:
        row = ds[int(i)]
        selected.append((row["sentence"], _remapped_label(row, finbert_label_to_id)))

    filtered = []
    n_correct = 0
    for text, label in tqdm(selected, desc="FinBERT-correct filter"):
        pred = _finbert_pred(model_M, tok_M, text, cfg, device)
        if pred == label:
            n_correct += 1
            filtered.append((text, label))

    acc = n_correct / max(1, len(selected))
    filter_meta = {
        "n_selected_before_filter": int(len(selected)),
        "n_finbert_correct": int(n_correct),
        "n_used_after_filter": int(len(filtered)),
        "finbert_accuracy_on_selected": float(acc),
    }

    if acc < 0.5:
        raise RuntimeError(
            f"Label alignment looks wrong: FinBERT accuracy on the selected sample is "
            f"{acc:.1%}. Check finbert_label_to_id and the PhraseBank remapping."
        )
    if not filtered:
        raise RuntimeError("No FinBERT-correct inputs remain after filtering.")

    return filtered, filter_meta


# ---------------------------------------------------------------------------
# Per-input processing
# ---------------------------------------------------------------------------


def process_input_soft_token(
    text: str,
    true_label: int,
    model_M,
    tok_M,
    model_P,
    G_flat: torch.Tensor,
    cfg: Config,
    device: str,
):
    """Build a record for the soft-token chart for one input."""
    input_ids, attention_mask = _encode(tok_M, text, cfg, device)
    V = tok_M.vocab_size

    logits_0 = initial_soft_token_logits(input_ids, V, tau=cfg.tau, device=device)

    e_M_x = soft_token_forward_target(model_M, logits_0, attention_mask)
    e_P_x = soft_token_forward_proxy(model_P, logits_0, attention_mask)

    w_t, b_t, i_star = get_logit_gap_readout(model_M, e_M_x, true_label)
    w_t, b_t = _normalise_readout(w_t, b_t)

    J_M, _ = compute_jacobian_soft_token(
        lambda lg, am: soft_token_forward_target(model_M, lg, am),
        logits_0,
        G_flat,
        cfg.q,
        attention_mask,
    )
    J_P, _ = compute_jacobian_soft_token(
        lambda lg, am: soft_token_forward_proxy(model_P, lg, am),
        logits_0,
        G_flat,
        cfg.q,
        attention_mask,
    )

    def _perturb_fn(u_np):
        u = torch.as_tensor(u_np, device=device, dtype=logits_0.dtype)
        e_pert = apply_perturbation_soft_token(
            lambda lg, am: soft_token_forward_target(model_M, lg, am),
            logits_0,
            G_flat,
            u,
            attention_mask,
        )
        return e_pert.detach().cpu().numpy()

    return {
        "text": text,
        "true_label": int(true_label),
        "competitor_label": int(i_star),
        "e_M_x": e_M_x.detach().cpu().numpy(),
        "e_P_x": e_P_x.detach().cpu().numpy(),
        "J_M": J_M.detach().cpu().numpy(),
        "J_P": J_P.detach().cpu().numpy(),
        "w": w_t.detach().cpu().numpy(),
        "b": float(b_t.detach().cpu().item()),
        "perturb_fn": _perturb_fn,
    }


def process_input_cloud(
    text: str,
    true_label: int,
    paraphrases: Sequence[str],
    model_M,
    tok_M,
    model_P,
    tok_P,
    base_record: dict,
    cfg: Config,
    device: str,
):
    """Build a record for the cloud chart for one input, logging realised flips."""
    if len(paraphrases) == 0:
        raise ValueError("process_input_cloud received an empty paraphrase list.")

    e_M_par, e_P_par = [], []
    for ptxt in paraphrases:
        ids_M, am_M = _encode(tok_M, ptxt, cfg, device)
        ids_P, am_P = _encode(tok_P, ptxt, cfg, device)

        e_M_par.append(hard_forward_target(model_M, ids_M, am_M).detach().cpu().numpy())
        e_P_par.append(hard_forward_proxy(model_P, ids_P, am_P).detach().cpu().numpy())

    e_M_par = np.stack(e_M_par)  # (K_i, d_M)
    e_P_par = np.stack(e_P_par)  # (K_i, d_P)

    J_M, J_P = paraphrase_cloud_chart(
        base_record["e_M_x"],
        e_M_par,
        base_record["e_P_x"],
        e_P_par,
    )

    # log realised FinBERT predictions for paraphrases.
    # Since hard_forward_target returns FinBERT's pooler output, these logits equal
    # the deployed classifier head applied to the same representation.
    W = model_M.classifier.weight.detach().cpu().numpy()    # (3, d_M)
    b_cls = model_M.classifier.bias.detach().cpu().numpy()  # (3,)
    para_logits = e_M_par @ W.T + b_cls
    para_preds = para_logits.argmax(axis=1).astype(int)     # kept for reference

    # A flip must be across the SAME boundary the readout models (true vs i*),
    # not the full 3-class argmax, else the label scores a boundary gamma_w/Sigma_w
    # do not describe. Flip <=> readout score w·e_M(x') + b changes sign.
    w_np = base_record["w"]
    b_np = float(base_record["b"])
    s0 = float(w_np @ base_record["e_M_x"] + b_np)
    para_scores = e_M_par @ w_np + b_np                     # (K_i,)
    para_flip = (np.sign(para_scores) != np.sign(s0)).astype(int)

    return {
        "text": text,
        "true_label": int(true_label),
        "base_pred": int(base_record["base_pred"]),
        "competitor_label": int(base_record["competitor_label"]),
        "e_M_x": base_record["e_M_x"],
        "e_P_x": base_record["e_P_x"],
        "J_M": J_M,
        "J_P": J_P,
        "D_M": J_M,
        "D_P": J_P,
        "w": base_record["w"],
        "b": base_record["b"],
        "para_flip": para_flip,
        "para_preds": para_preds,
        "n_paraphrases": int(len(paraphrases)),
    }


# ---------------------------------------------------------------------------
# Degeneracy report (revision item 14)
# ---------------------------------------------------------------------------


def _degeneracy_report(records: Sequence[dict], nu_frac: float, near_zero_tol: float = 1e-8):
    """Count Sigma_w(x)=0, near-zero Sigma_w, and (near-)degenerate B across inputs."""
    n_sigma_zero = 0
    n_sigma_near = 0
    n_degen_B = 0
    sig_vals = []
    min_eigs = []
    cond_vals = []

    for r in records:
        _, B = pullback_matrices(r["J_M"], r["J_P"])
        s, _ = sigma_w(r["J_M"], B, r["w"], nu_frac)
        sig_vals.append(float(s))

        if s <= 0.0:
            n_sigma_zero += 1
        elif s < near_zero_tol:
            n_sigma_near += 1

        evs = np.linalg.eigvalsh(B)
        ev_min = float(evs[0])
        ev_max = float(evs[-1])
        min_eigs.append(ev_min)
        cond_vals.append(float(ev_max / max(ev_min, 1e-300)))
        if ev_min <= near_zero_tol * max(1.0, ev_max):
            n_degen_B += 1

    sig_vals = np.asarray(sig_vals, dtype=float)
    min_eigs = np.asarray(min_eigs, dtype=float)
    cond_vals = np.asarray(cond_vals, dtype=float)

    return {
        "n": int(len(records)),
        "n_sigma_zero": int(n_sigma_zero),
        "n_sigma_near_zero": int(n_sigma_near),
        "n_degenerate_B": int(n_degen_B),
        "sigma_w_min": float(sig_vals.min()) if sig_vals.size else None,
        "sigma_w_median": float(np.median(sig_vals)) if sig_vals.size else None,
        "B_min_eigenvalue_median": float(np.median(min_eigs)) if min_eigs.size else None,
        "B_condition_number_median": float(np.median(cond_vals)) if cond_vals.size else None,
    }


def _safe_K_grid(cloud_records: Sequence[dict]) -> Tuple[List[int], int]:
    """Choose finite-search K values that work even when K_i varies across inputs.

    The grid is the standard base ladder clipped to the largest available count;
    experiment_2prime_finite_search clamps each input to its own K_i, so a K above
    a given input's count just uses all of that input's candidates. The oracle is
    K_max, i.e. every input uses its FULL paraphrase set (best-of-available),
    rather than being bottlenecked by the worst-covered input.
    """
    counts = [int(r["D_M"].shape[1]) for r in cloud_records]
    K_max = max(counts)
    base = [4, 8, 12, 18, 25]
    grid = sorted(set([k for k in base if k <= K_max] + [K_max]))
    return grid, K_max


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(cfg: Config):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    os.makedirs(cfg.out, exist_ok=True)

    # --- Models ---
    print("Loading FinBERT (target) and all-MiniLM-L6-v2 (proxy)...")
    model_M, tok_M = load_target_model(device, cfg.target_model)
    model_P, tok_P = load_proxy_model(device, cfg.proxy_model)
    if cfg.run_soft_token and not check_vocab_alignment(tok_M, tok_P):
        raise RuntimeError("Soft-token chart requires a shared vocabulary; "
                           "this pairing only supports --run_cloud.")
    print(f"Vocab alignment OK. |V| = {tok_M.vocab_size}")
    print(f"FinBERT id2label: {model_M.config.id2label}")

    finbert_label_to_id = {str(v).lower(): int(k) for k, v in model_M.config.id2label.items()}

    # --- Data ---
    print("Loading Financial PhraseBank (sentences_allagree)...")
    ds = load_financial_phrasebank_train_split()
    sentences, filter_meta = _sample_and_filter_finbert_correct(
        ds, model_M, tok_M, finbert_label_to_id, cfg, device
    )
    print(
        "Selected {n_selected_before_filter} inputs; kept {n_used_after_filter} "
        "FinBERT-correct inputs (accuracy on selected sample = "
        "{finbert_accuracy_on_selected:.1%}).".format(**filter_meta)
    )

    # --- Random projection (per-input by default; revision item 9b) ---
    L_max = cfg.max_len
    V = tok_M.vocab_size
    G_fixed = None
    if cfg.run_soft_token and not cfg.redraw_G:
        print(
            f"Building a single FIXED projection G_flat: ({L_max * V}, {cfg.q}) "
            "[--no_redraw_G: use only for a sensitivity report]."
        )
        G_fixed = make_random_projection(L_max, V, cfg.q, seed=cfg.seed + 1, device=device)
    elif cfg.run_soft_token:
        print(
            f"Redrawing projection G_flat per input: ({L_max * V}, {cfg.q}) each "
            "[default main-specification behaviour]."
        )

    # --- Paraphraser (optional) ---
    paraphraser = None
    if cfg.run_cloud:
        from paraphrase import ExternalParaphraser

        paraphrase_path = os.path.join(cfg.out, "paraphrases.json")
        print(f"Loading cached GPT-4o paraphrases from {paraphrase_path}...")
        # paraphrases.json is sometimes written in Windows cp1252 (a smart quote
        # U+2019 -> byte 0x92), which breaks a strict utf-8 read. Try utf-8, then
        # cp1252, then latin-1 (which decodes any byte).
        _raw = open(paraphrase_path, "rb").read()
        for _enc in ("utf-8", "cp1252", "latin-1"):
            try:
                _text = _raw.decode(_enc)
                break
            except UnicodeDecodeError:
                continue
        table = json.loads(_text)
        paraphraser = ExternalParaphraser(table=table)
        print(f"  Loaded {len(table)} sentence -> paraphrase entries.")

    # --- Per-input processing ---
    soft_records = []
    cloud_records = []
    kept_paraphrase_counts = []
    missing_paraphrase_count = 0

    for i, (text, true_label) in enumerate(tqdm(sentences, desc="Inputs")):
        if cfg.run_soft_token:
            G_i = (
                G_fixed
                if not cfg.redraw_G
                else make_random_projection(
                    L_max, V, cfg.q, seed=cfg.seed + 1 + i, device=device
                )
            )
            sr = process_input_soft_token(
                text, true_label, model_M, tok_M, model_P, G_i, cfg, device
            )
            sr.pop("perturb_fn", None)  # Exps 1 & 3 never call it; dropping the
                                        # closure releases this input's ~125 MB G.
            soft_records.append(sr)

        if cfg.run_cloud:
            try:
                paraphrases = paraphraser.paraphrase(text, K=cfg.K)
            except KeyError:
                missing_paraphrase_count += 1
                print(f"  Skipping cloud chart; no cached paraphrases for: {text[:80]!r}")
                continue

            if len(paraphrases) == 0:
                print(f"  Skipping cloud chart; empty paraphrase list for: {text[:80]!r}")
                continue

            kept_paraphrase_counts.append(len(paraphrases))
            base_record = _hard_base_record(
                text, true_label, model_M, tok_M, model_P, tok_P, cfg, device
            )

            # The filter above should make this true. Keep the explicit guard so
            # realised flips remain para_preds != true_label without ambiguity.
            if base_record["base_pred"] != int(true_label):
                print(
                    "  Skipping cloud chart; hard base prediction no longer matches "
                    f"label for: {text[:80]!r}"
                )
                continue

            cr = process_input_cloud(
                text,
                true_label,
                paraphrases,
                model_M,
                tok_M,
                model_P,
                tok_P,
                base_record,
                cfg,
                device,
            )
            cloud_records.append(cr)

    if cfg.run_soft_token and not soft_records:
        raise RuntimeError("Soft-token chart was requested, but no soft records were built.")
    if cfg.run_cloud and not cloud_records:
        print("Warning: cloud chart was requested, but no cloud records were built.")

    # --- Persist records ---
    if cfg.save_records:
        def _strip(r):
            return {k: v for k, v in r.items() if k != "perturb_fn"}

        if soft_records:
            with open(os.path.join(cfg.out, "soft_records.pkl"), "wb") as f:
                pickle.dump([_strip(r) for r in soft_records], f)
        if cloud_records:
            with open(os.path.join(cfg.out, "cloud_records.pkl"), "wb") as f:
                pickle.dump(cloud_records, f)

    # --- Run metadata (revision item 14) ---
    meta = {
        "models": {
            "target": "ProsusAI/finbert",
            "proxy": "sentence-transformers/all-MiniLM-L6-v2",
            "shared_vocab_size": int(tok_M.vocab_size),
            "finbert_id2label": {int(k): v for k, v in model_M.config.id2label.items()},
        },
        "dataset": {
            "name": "financial_phrasebank/sentences_allagree",
            "split": "train",
            "n_requested": int(cfg.n),
            "n_used": int(len(sentences)),
            "seed": int(cfg.seed),
            "filter": filter_meta,
        },
        "tokenisation": {
            "truncation": True,
            "padding": "max_length",
            "max_len": int(cfg.max_len),
        },
        "chart": {
            "q": int(cfg.q),
            "tau": float(cfg.tau),
            "redraw_G": bool(cfg.redraw_G),
            "readout_normalised_to_unit_w": True,
        },
        "regularisation": {
            "nu_frac": float(cfg.nu_frac),
            "nu_scheme": "nu = nu_frac * tr(B) / q",
        },
        "eta_grids": {
            "attackability": {
                "min": 0.01,
                "max": float(cfg.eta_max),
                "n_eta": int(cfg.n_eta),
                "scale": "raw proxy distance units",
            },
            "linearisation": {
                "min": float(cfg.eta_min_linearisation),
                "max": float(cfg.eta_max_linearisation),
                "n_eta": int(cfg.n_eta_linearisation),
                "scale": "raw proxy distance units",
            },
        },
        "cloud": {
            "K_requested": int(cfg.K),
            "missing_cached_paraphrase_count": int(missing_paraphrase_count),
            "n_cloud_records": int(len(cloud_records)),
            "paraphrases_per_input_after_filter": {
                "mean": float(np.mean(kept_paraphrase_counts)) if kept_paraphrase_counts else None,
                "median": float(np.median(kept_paraphrase_counts)) if kept_paraphrase_counts else None,
                "min": int(np.min(kept_paraphrase_counts)) if kept_paraphrase_counts else None,
                "max": int(np.max(kept_paraphrase_counts)) if kept_paraphrase_counts else None,
            },
        },
    }

    if soft_records:
        meta["degeneracy_soft"] = _degeneracy_report(soft_records, cfg.nu_frac)
    if cloud_records:
        meta["degeneracy_cloud"] = _degeneracy_report(cloud_records, cfg.nu_frac)

        eta_probe = np.linspace(0.01, cfg.eta_max, 5)
        pv_counts = []
        for r in cloud_records:
            pd = np.linalg.norm(r["D_P"], axis=0)
            pv_counts.append([int((pd <= e).sum()) for e in eta_probe])
        pv_counts = np.asarray(pv_counts)
        meta["cloud"]["proxy_valid_mean_per_budget"] = {
            f"{e:.3f}": float(pv_counts[:, j].mean()) for j, e in enumerate(eta_probe)
        }

    metadata_path = os.path.join(cfg.out, "run_metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"Wrote run metadata -> {metadata_path}")

    # --- Experiments ---
    eta_grid = np.linspace(0.01, cfg.eta_max, cfg.n_eta)
    eta_grid_linearisation = np.logspace(
        np.log10(cfg.eta_min_linearisation),
        np.log10(cfg.eta_max_linearisation),
        cfg.n_eta_linearisation,
    )

    if cfg.run_soft_token and soft_records:
        print("Soft-token: Experiment 1 (Sigma vs lambda*)")
        r1 = experiment_1_sigma_lambda(soft_records, nu_frac=cfg.nu_frac)
        print(f"  violations of Sigma <= lambda*: {r1['violations']} / {len(soft_records)}")
        plot_sigma_vs_lambda(r1, os.path.join(cfg.out, "fig_sigma_lambda_soft.pdf"))

        ts = experiment_trace_sensitivity(soft_records, nu_frac=cfg.nu_frac)
        meta["trace_sensitivity_soft"] = {"T_bar_n": ts["T_bar_n"],
                                          "trace_S_median": ts["trace_S_median"]}
        print(f"  Empirical trace sensitivity  T_bar_n = {ts['T_bar_n']:.4g}")

        print(f"Soft-token: Experiment 2 (linearisation error) at tau={cfg.tau_linearisation}")
        # Exps 1 & 3 keep tau=20 (converged, faithful one-hot). Exp 2 needs a
        # non-degenerate beta = lambda_min(B), which tau=20 collapses to ~1e-20,
        # so its records are rebuilt at tau_linearisation. They are built and
        # CONSUMED ONE AT A TIME: each soft-token record pins a ~125 MB projection
        # G via its perturb_fn closure, so materialising all of them at once
        # exhausts memory. Streaming keeps a single G resident; n_linearisation
        # caps the inputs used (the aggregate slope-2 curve needs only a few dozen).
        cfg_lin = replace(cfg, tau=cfg.tau_linearisation)
        n_lin = min(len(sentences), cfg.n_linearisation)
        m_lin = len(eta_grid_linearisation)
        nl = np.full((n_lin, m_lin), np.nan)
        lin = np.full((n_lin, m_lin), np.nan)
        betas = np.full(n_lin, np.nan)
        for i, (text, true_label) in enumerate(
            tqdm(sentences[:n_lin], desc="Lin. records")
        ):
            G_i = (
                G_fixed
                if not cfg.redraw_G
                else make_random_projection(
                    L_max, V, cfg.q, seed=cfg.seed + 1 + i, device=device
                )
            )
            rec = process_input_soft_token(
                text, true_label, model_M, tok_M, model_P, G_i, cfg_lin, device
            )
            one = experiment_2_linearisation_error(
                [rec], eta_grid_linearisation, nu_frac=cfg.nu_frac
            )
            nl[i] = one["delta_s_nl"][0]
            lin[i] = one["delta_s_lin"][0]
            betas[i] = one["beta"][0]
            del rec, one, G_i  # release this input's projection before the next

        residual = np.abs(nl - lin)
        eta_sq = (eta_grid_linearisation.astype(float) ** 2)[None, :]
        with np.errstate(divide="ignore", invalid="ignore"):
            norm_residual = residual / (eta_sq / betas[:, None])
        r2 = {
            "eta": eta_grid_linearisation.astype(float),
            "delta_s_nl": nl,
            "delta_s_lin": lin,
            "residual": residual,
            "beta": betas,
            "norm_residual": norm_residual,
            "mean_abs_nl": np.nanmean(np.abs(nl), axis=0),
            "mean_abs_lin": np.nanmean(np.abs(lin), axis=0),
            "mean_residual": np.nanmean(residual, axis=0),
            "median_residual": np.nanmedian(residual, axis=0),
            "mean_norm_residual": np.nanmean(norm_residual, axis=0),
            "median_norm_residual": np.nanmedian(norm_residual, axis=0),
        }
        plot_linearisation_error(r2, os.path.join(cfg.out, "fig_linearisation_error.pdf"))

        print("Soft-token: Experiment 3 (attackability curve)")
        r3 = experiment_3_attackability_curve(soft_records, eta_grid, nu_frac=cfg.nu_frac)
        plot_attackability_curve(r3, os.path.join(cfg.out, "fig_attackability_curve.pdf"))

    if cfg.run_cloud and cloud_records:
        print("Cloud: Experiment 1' (Sigma vs lambda*)")
        r1c = experiment_1_sigma_lambda(cloud_records, nu_frac=cfg.nu_frac)
        print(f"  violations of Sigma <= lambda*: {r1c['violations']} / {len(cloud_records)}")
        print(f"  log10(lambda*/Sigma_w) median = {r1c['log10_slack_median']:.3f} "
              f"CI[{r1c['log10_slack_median_ci'][0]:.3f},{r1c['log10_slack_median_ci'][1]:.3f}]")
        plot_sigma_vs_lambda(r1c, os.path.join(cfg.out, "fig_sigma_lambda_cloud.pdf"))

        ts_c = experiment_trace_sensitivity(cloud_records, nu_frac=cfg.nu_frac)
        meta["trace_sensitivity_cloud"] = {"T_bar_n": ts_c["T_bar_n"],
                                           "trace_S_median": ts_c["trace_S_median"]}
        print(f"  Empirical trace sensitivity  T_bar_n = {ts_c['T_bar_n']:.4g}")

        print("Cloud: Experiment 2' (finite-search attackability + oracle + coverage)")
        K_grid, oracle_K = _safe_K_grid(cloud_records)
        rfs = experiment_2prime_finite_search(
            cloud_records, K_grid, eta_grid, seed=cfg.seed + 2
        )
        cov = finite_search_coverage_diagnostic(
            cloud_records, eta_grid, nu_frac=cfg.nu_frac
        )

        print("Cloud: Experiment 3' (attackability curve)")
        r3c = experiment_3_attackability_curve(cloud_records, eta_grid, nu_frac=cfg.nu_frac)
        plot_finite_search(
            rfs,
            eta_grid,
            os.path.join(cfg.out, "fig_finite_search.pdf"),
            A_hat=r3c["A_hat"],
            dkw_half=r3c["dkw_half_width"],
            coverage_diag=cov,
            oracle_K=oracle_K,
        )
        plot_attackability_curve(
            r3c, os.path.join(cfg.out, "fig_attackability_curve_cloud.pdf")
        )

        print("Cloud: Predictive validity (Z_w vs realised flips) — OUT OF SAMPLE")
        pv = experiment_predictive_validity(
            cloud_records, eta_grid, nu_frac=cfg.nu_frac, mode="within_split"
        )
        ciA = pv["auc_at_label_budget_ci"]; ciR = pv["spearman_Z_ci"]
        print(f"  [within-split] AUC(-Z_w) @ median budget = {pv['auc_at_label_budget']:.3f} "
              f"CI[{ciA[0]:.3f},{ciA[1]:.3f}]  vs gamma {pv['auc_gamma_at_label_budget']:.3f} "
              f"vs sqrtSigma {pv['auc_sigma_at_label_budget']:.3f}")
        print(f"  [within-split] Spearman(Z_w,budget) = {pv['spearman_Z_vs_budget']:.3f} "
              f"CI[{ciR[0]:.3f},{ciR[1]:.3f}]  vs gamma {pv['spearman_gamma_vs_budget']:.3f} "
              f"vs sqrtSigma {pv['spearman_sigma_vs_budget']:.3f}")
        plot_predictive_validity(pv, os.path.join(cfg.out, "fig_predictive_validity.pdf"))

        # A2 cross-chart (only if the soft chart was also run on the same inputs)
        if cfg.run_soft_token and soft_records:
            pv_x = experiment_predictive_validity_crosschart(
                soft_records, cloud_records, eta_grid, nu_frac=cfg.nu_frac
            )
            print(f"  [soft->cloud OOS] AUC(-Z_w) = {pv_x['auc_at_label_budget']:.3f} "
                  f"vs gamma {pv_x['auc_gamma_at_label_budget']:.3f}; "
                  f"Spearman = {pv_x['spearman_Z_vs_budget']:.3f}")
            plot_predictive_validity(pv_x, os.path.join(cfg.out, "fig_predictive_validity_crosschart.pdf"))
            meta["predictive_validity_crosschart"] = {
                k: pv_x[k] for k in ("auc_at_label_budget", "auc_gamma_at_label_budget",
                                     "auc_sigma_at_label_budget", "spearman_Z_vs_budget",
                                     "spearman_gamma_vs_budget", "n_used")
            }

        meta["predictive_validity"] = {
            "mode": pv["mode"],
            "auc_at_label_budget": pv["auc_at_label_budget"],
            "auc_at_label_budget_ci": pv["auc_at_label_budget_ci"],
            "auc_gamma_at_label_budget": pv["auc_gamma_at_label_budget"],
            "auc_sigma_at_label_budget": pv["auc_sigma_at_label_budget"],
            "spearman_Z_vs_budget": pv["spearman_Z_vs_budget"],
            "spearman_Z_ci": pv["spearman_Z_ci"],
            "spearman_gamma_vs_budget": pv["spearman_gamma_vs_budget"],
            "spearman_sigma_vs_budget": pv["spearman_sigma_vs_budget"],
            "label_budget": pv["label_budget"],
            "n_used": pv["n_used"],
        }
        meta["cloud"]["finite_search_K_grid"] = K_grid
        meta["cloud"]["oracle_K"] = int(oracle_K)

        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

    print(f"\nDone. Figures, records, and run_metadata.json saved to: {cfg.out}/")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> Config:
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=1000)
    p.add_argument("--q", type=int, default=64)
    p.add_argument("--K", type=int, default=128)
    p.add_argument("--tau", type=float, default=20.0)
    p.add_argument("--tau_linearisation", type=float, default=10.0)
    p.add_argument(
        "--nu_frac",
        type=float,
        default=1e-3,
        help="ridge fraction nu (was --rho_frac); nu = nu_frac * tr(B)/q",
    )
    p.add_argument("--target_model", type=str, default="ProsusAI/finbert")
    p.add_argument("--proxy_model", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    p.add_argument("--eta_max", type=float, default=1.0)
    p.add_argument("--n_eta", type=int, default=40)
    p.add_argument("--eta_min_linearisation", type=float, default=1e-3)
    p.add_argument("--eta_max_linearisation", type=float, default=1e-1)
    p.add_argument("--n_eta_linearisation", type=int, default=40)
    p.add_argument("--n_linearisation", type=int, default=64)
    p.add_argument("--max_len", type=int, default=64)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=str, default="figures")
    p.add_argument("--run_soft_token", action="store_true")
    p.add_argument("--run_cloud", action="store_true")
    p.add_argument("--no_save_records", action="store_true")
    p.add_argument(
        "--no_redraw_G",
        action="store_true",
        help="use one fixed G for all inputs; this is for sensitivity reports only",
    )

    a = p.parse_args()
    return Config(
        n=a.n,
        q=a.q,
        K=a.K,
        tau=a.tau,
        tau_linearisation=a.tau_linearisation,
        nu_frac=a.nu_frac,
        eta_max=a.eta_max,
        n_eta=a.n_eta,
        eta_min_linearisation=a.eta_min_linearisation,
        eta_max_linearisation=a.eta_max_linearisation,
        target_model=a.target_model,
        proxy_model=a.proxy_model,
        n_eta_linearisation=a.n_eta_linearisation,
        n_linearisation=a.n_linearisation,
        max_len=a.max_len,
        seed=a.seed,
        out=a.out,
        run_soft_token=a.run_soft_token or (not a.run_cloud),
        run_cloud=a.run_cloud,
        save_records=not a.no_save_records,
        redraw_G=not a.no_redraw_G,
    )


if __name__ == "__main__":
    main(_parse_args())