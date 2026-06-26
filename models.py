"""
Model loading and soft-token forward passes for the empirical exercise.

Two models:
- Target M: FinBERT (ProsusAI/finbert) -- 3-class financial sentiment classifier.
- Proxy  P: all-MiniLM-L6-v2 -- Sentence-BERT embedding model.

Both inherit BERT's WordPiece vocabulary (bert-base-uncased), so the soft-token
distribution sigma(logits) at each position is shared between the two models.
The embedding matrices E_tok are model-specific.

The soft-token forward functions take a logit tensor of shape (L, V) and return
the model's pooled representation, fully differentiable in the logits. This is
the construction underlying Section 8.2 of the paper.
"""

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, BertForSequenceClassification, BertModel


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_target_model(device="cuda", model_name="ProsusAI/finbert"):
    """Load FinBERT (frozen, eval mode)."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = BertForSequenceClassification.from_pretrained(
        model_name,
        attn_implementation="eager",
    )
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, tokenizer


def load_proxy_model(device="cuda",model_name="sentence-transformers/all-MiniLM-L6-v2"):
    """Load all-MiniLM-L6-v2 as a plain BertModel (frozen, eval mode)."""
    name = model_name
    tokenizer = AutoTokenizer.from_pretrained(name)
    model = BertModel.from_pretrained(
        name,
        attn_implementation="eager",
    )
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, tokenizer


def check_vocab_alignment(tokenizer_M, tokenizer_P):
    """Verify the two tokenizers map the same strings to the same integer IDs."""
    vocab_M = tokenizer_M.get_vocab()
    vocab_P = tokenizer_P.get_vocab()
    if len(vocab_M) != len(vocab_P):
        return False
    return all(vocab_P.get(tok) == idx for tok, idx in vocab_M.items())


# ---------------------------------------------------------------------------
# Soft-token forward passes
# ---------------------------------------------------------------------------

def soft_token_forward_target(model, logits, attention_mask):
    """
    Run FinBERT on a soft-token input.

    logits         : (L, V) per-position logit vector.
    attention_mask : (L,)   1 for real tokens, 0 for padding.

    Returns: pooled representation, shape (d_M,) = (768,).
    """
    probs = F.softmax(logits, dim=-1)                     # (L, V)
    E_tok = model.bert.embeddings.word_embeddings.weight  # (V, h)
    inputs_embeds = (probs @ E_tok).unsqueeze(0)          # (1, L, h)
    am = attention_mask.unsqueeze(0)                       # (1, L)

    outputs = model.bert(
        inputs_embeds=inputs_embeds,
        attention_mask=am,
        output_hidden_states=False,
        output_attentions=False,
    )
    return outputs.pooler_output.squeeze(0)               # (d_M,)


def soft_token_forward_proxy(model, logits, attention_mask):
    """
    Run all-MiniLM-L6-v2 on a soft-token input. Mean-pool + L2-normalise.

    Returns: proxy embedding, shape (d_P,) = (384,).
    """
    probs = F.softmax(logits, dim=-1)
    E_tok = model.embeddings.word_embeddings.weight
    inputs_embeds = (probs @ E_tok).unsqueeze(0)
    am = attention_mask.unsqueeze(0)

    outputs = model(inputs_embeds=inputs_embeds, attention_mask=am)
    last_hidden = outputs.last_hidden_state.squeeze(0)    # (L, h)

    mask = attention_mask.unsqueeze(-1).float()
    pooled = (last_hidden * mask).sum(dim=0) / mask.sum().clamp(min=1.0)
    pooled = pooled / pooled.norm(dim=-1, keepdim=True).clamp(min=1e-12)
    return pooled


# ---------------------------------------------------------------------------
# Hard (non-soft) forward passes -- used for evaluating paraphrases in 8.3
# ---------------------------------------------------------------------------

@torch.no_grad()
def hard_forward_target(model, input_ids, attention_mask):
    """Return FinBERT pooler_output for a hard input."""
    outputs = model.bert(
        input_ids=input_ids.unsqueeze(0),
        attention_mask=attention_mask.unsqueeze(0),
    )
    return outputs.pooler_output.squeeze(0)


@torch.no_grad()
def hard_forward_proxy(model, input_ids, attention_mask):
    """Mean-pool + L2-normalise output of all-MiniLM-L6-v2 for a hard input."""
    outputs = model(
        input_ids=input_ids.unsqueeze(0),
        attention_mask=attention_mask.unsqueeze(0),
    )
    last_hidden = outputs.last_hidden_state.squeeze(0)
    mask = attention_mask.unsqueeze(-1).float()
    pooled = (last_hidden * mask).sum(dim=0) / mask.sum().clamp(min=1.0)
    pooled = pooled / pooled.norm(dim=-1, keepdim=True).clamp(min=1e-12)
    return pooled


# ---------------------------------------------------------------------------
# Logit-gap readout
# ---------------------------------------------------------------------------

def get_logit_gap_readout(model, e_M_x, true_label):
    """
    Build the effective binary readout (w, b) for the multiclass logit-gap
    construction described in Section 8.1.

    e_M_x      : (d_M,) target embedding of x.
    true_label : int in {0, 1, 2}.

    Returns: w_eff (d_M,), b_eff (scalar), i_star (the closest competing class).
    """
    W = model.classifier.weight        # (3, d_M)
    b = model.classifier.bias          # (3,)

    logits = e_M_x @ W.T + b           # (3,)

    gaps = logits[true_label] - logits  # entry true_label is zero
    gaps_masked = gaps.clone()
    gaps_masked[true_label] = float("inf")
    i_star = int(torch.argmin(gaps_masked).item())

    w_eff = W[true_label] - W[i_star]
    b_eff = b[true_label] - b[i_star]
    return w_eff, b_eff, i_star


# ---------------------------------------------------------------------------
# Soft-token chart initialisation
# ---------------------------------------------------------------------------

def initial_soft_token_logits(input_ids, vocab_size, tau=20.0, device="cuda"):
    """
    Build the base logit tensor logits_0 of shape (L, V) such that
    softmax(logits_0)[i] is a sharp one-hot at input_ids[i].

    input_ids : (L,) -- token IDs of the input sentence (already padded if needed).
    tau       : sharpness of the one-hot approximation. The softmax mass on the
                target token is exp(tau) / (exp(tau) + |V| - 1), so it depends on
                the vocabulary size |V| (revision item 9). For BERT WordPiece
                (|V| ~ 30522): tau=10 gives only ~0.42 mass on the target token
                (far from one-hot, so the chart would be centred well away from
                the text it claims to approximate), tau=15 gives ~0.99, and
                tau=20 gives > 0.9999. Default is 20.0.
    """
    L = input_ids.shape[0]
    logits_0 = torch.zeros(L, vocab_size, device=device)
    logits_0[torch.arange(L, device=device), input_ids] = tau
    return logits_0
