"""
Paraphrase-cloud chart (Section 8.3).

For each input x and a set of K paraphrases x'_1, ..., x'_K, the chart
identifies the k-th local coordinate with the k-th paraphrase:

    J_M(x) = D_M(x) = [ e_M(x'_k) - e_M(x) ]_{k=1}^K  in R^{d_M x K},
    J_P(x) = D_P(x) = [ e_P(x'_k) - e_P(x) ]_{k=1}^K  in R^{d_P x K}.

The chart dimension is q = K. The pullback matrices A = D_M^T D_M and
B = D_P^T D_P are K x K and the generalised eigenproblem is small even
at K = 128.
"""

import numpy as np
import torch


def paraphrase_cloud_chart(e_M_x, e_M_paraphrases, e_P_x, e_P_paraphrases):
    """
    Build J_M = D_M and J_P = D_P from base and paraphrase embeddings.

    All inputs may be numpy arrays or torch tensors; they are returned as
    numpy arrays so the downstream geometry code (scipy.linalg) can act on
    them directly.

    e_M_x            : (d_M,)
    e_M_paraphrases  : (K, d_M)
    e_P_x            : (d_P,)
    e_P_paraphrases  : (K, d_P)

    Returns: J_M of shape (d_M, K), J_P of shape (d_P, K).
    """
    def _to_np(t):
        if isinstance(t, torch.Tensor):
            return t.detach().cpu().numpy()
        return np.asarray(t)

    e_M_x = _to_np(e_M_x)
    e_P_x = _to_np(e_P_x)
    e_M_paraphrases = _to_np(e_M_paraphrases)
    e_P_paraphrases = _to_np(e_P_paraphrases)

    J_M = (e_M_paraphrases - e_M_x).T   # (d_M, K)
    J_P = (e_P_paraphrases - e_P_x).T   # (d_P, K)
    return J_M, J_P
