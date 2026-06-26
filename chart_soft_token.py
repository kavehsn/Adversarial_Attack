"""
Jacobian computation for the soft-token chart

The local coordinate u in R^q parameterises perturbations of the position-wise
logit tensor through a fixed random projection:

    logits(u) = logits_0 + (G_flat @ u).reshape(L, V)

where G_flat is (L*V, q) with orthonormalised columns. This file computes
J = d(model output) / du at u=0 by forward-mode Jacobian-vector products,
one per column of G_flat.

Forward-mode is chosen over reverse-mode because q (typically 32-128) is
smaller than d_M + d_P = 1152; forward mode costs q forward passes whereas
reverse mode would cost d_M + d_P backward passes per input.
"""

import torch
import torch.func as func


def make_random_projection(L, V, q, seed=0, device="cuda"):
    """
    Build a random projection G_flat in R^{(L*V) x q} with orthonormal columns.

    The original pipeline drew this ONCE and reused the same
    q-dimensional subspace for every input, which is a confound (the measured
    geometry then depends on one fixed random slice rather than on directions
    adapted to each input). The fix is to redraw G per input -- pass a distinct
    ``seed`` for each input (e.g. seed = base_seed + input_index), which main.py
    now does by default via ``--redraw_G``. Holding G fixed is still available
    for a sensitivity-to-the-draw report (run several base seeds).

    Returns: G_flat of shape (L*V, q) on `device`.
    """
    gen = torch.Generator(device="cpu").manual_seed(seed)
    G = torch.randn(L * V, q, generator=gen)
    # QR gives orthonormal columns; we only need a thin QR.
    G, _ = torch.linalg.qr(G, mode="reduced")
    return G.to(device)


def compute_jacobian_soft_token(forward_fn, logits_0, G_flat, q, attention_mask):
    """
    Compute the Jacobian of u |-> forward_fn(logits_0 + (G_flat @ u).reshape(L, V),
                                              attention_mask) at u = 0.

    forward_fn     : callable(logits, attention_mask) -> tensor of shape (d,).
    logits_0       : (L, V) base logit tensor.
    G_flat         : (L*V, q) random projection.
    q              : local-coordinate dimension.
    attention_mask : (L,) tensor.

    Returns:
      J            : (d, q) Jacobian.
      out_0        : (d,) value of forward_fn at u=0 (free output).
    """
    L, V = logits_0.shape
    device = logits_0.device

    def f(u):
        delta = (G_flat @ u).reshape(L, V)
        return forward_fn(logits_0 + delta, attention_mask)

    u_0 = torch.zeros(q, device=device)

    # One forward pass to discover the output dimension.
    out_0 = f(u_0)
    d = out_0.shape[0]

    J = torch.empty(d, q, device=device, dtype=out_0.dtype)
    eye = torch.eye(q, device=device)
    for j in range(q):
        _, jvp_col = func.jvp(f, (u_0,), (eye[j],))
        J[:, j] = jvp_col
    return J, out_0


def apply_perturbation_soft_token(forward_fn, logits_0, G_flat, u, attention_mask):
    """
    Evaluate forward_fn(logits_0 + (G_flat @ u).reshape(L, V), attention_mask)
    without tracking gradients. Used by Experiment 2 to apply
    the linearised worst direction u* through the nonlinear E_M and measure
    the actual readout displacement.
    """
    L, V = logits_0.shape
    delta = (G_flat @ u).reshape(L, V)
    with torch.no_grad():
        return forward_fn(logits_0 + delta, attention_mask)
