import torch
from torch import Tensor


# ---------------------------------------------------------------------------
# CKA (Centered Kernel Alignment)
# ---------------------------------------------------------------------------

def compute_cka(X: Tensor, Y: Tensor) -> float:
    """
    Compute linear CKA between two representation matrices.

    Args:
        X: embeddings of shape (N, D1)
        Y: embeddings of shape (N, D2)

    Returns:
        CKA score in [0, 1]
    """
    K = torch.matmul(X, X.T)
    L = torch.matmul(Y, Y.T)

    n = K.size(0)
    H = torch.eye(n, device=X.device) - (1.0 / n) * torch.ones(n, n, device=X.device)

    K_centered = H @ K @ H
    L_centered = H @ L @ H

    hsic_kl = torch.sum(K_centered * L_centered)
    hsic_kk = torch.sum(K_centered * K_centered)
    hsic_ll = torch.sum(L_centered * L_centered)

    return (hsic_kl / torch.sqrt(hsic_kk * hsic_ll)).item()


def compute_crossmodal_cka(emb_img: Tensor, emb_sig: Tensor) -> float:
    """
    Cross-modal CKA between image and signal embeddings paired by class.

    Computes CKA(emb_img, emb_sig) without concatenating the two spaces.
    Both tensors must be row-paired: row i of emb_img and row i of emb_sig
    must belong to the same class (use build_paired_test to guarantee this).
    """
    return compute_cka(emb_img, emb_sig)


# ---------------------------------------------------------------------------
# Retrieval metrics
# ---------------------------------------------------------------------------

def recall_at_k(
    query_embs: Tensor,
    query_labels: Tensor,
    gallery_embs: Tensor,
    gallery_labels: Tensor,
    k: int,
) -> float:
    """
    Compute Recall@k for cross-modal retrieval.

    For each query embedding, retrieve the top-k gallery embeddings by cosine
    similarity and check whether at least one shares the same label.

    Args:
        query_embs:    (N, D) query embeddings (L2-normalised)
        query_labels:  (N,)   labels for each query
        gallery_embs:  (M, D) gallery embeddings (L2-normalised)
        gallery_labels:(M,)   labels for each gallery item
        k:             number of neighbours to retrieve

    Returns:
        Recall@k score in [0, 1]
    """
    sim_matrix = torch.matmul(query_embs, gallery_embs.T)   # (N, M)
    topk_indices = torch.topk(sim_matrix, k, dim=1, largest=True).indices  # (N, k)
    topk_labels = gallery_labels[topk_indices]               # (N, k)
    matches = (topk_labels == query_labels.unsqueeze(1))     # (N, k)
    return matches.any(dim=1).float().mean().item()


def evaluate_retrieval(model, digits_data, mnist1d_data, device, k: int):
    """
    Run Recall@k evaluation on the test split using a CLIP model.

    Returns:
        (recall_sig2img, recall_img2sig)
    """
    model.eval().to(device)
    with torch.no_grad():
        z_sig, z_img = model(
            torch.from_numpy(mnist1d_data["X_test"]).to(device),
            torch.from_numpy(digits_data["X_test"]).to(device),
        )
        recall_s2i = recall_at_k(
            z_sig, torch.from_numpy(mnist1d_data["y_test"]).to(device),
            z_img, torch.from_numpy(digits_data["y_test"]).to(device), k,
        )
        recall_i2s = recall_at_k(
            z_img, torch.from_numpy(digits_data["y_test"]).to(device),
            z_sig, torch.from_numpy(mnist1d_data["y_test"]).to(device), k,
        )
    model.train()
    return recall_s2i, recall_i2s


def evaluate_cka(model1, model2, digits_data, mnist1d_data, device,
                 emb1: Tensor = None, emb2: Tensor = None) -> float:
    """
    Compute CKA between two models (or directly between two embedding tensors).

    If emb1/emb2 are provided, the model arguments are ignored.
    """
    if emb1 is not None and emb2 is not None:
        n = min(emb1.shape[0], emb2.shape[0])
        return compute_cka(emb1[:n], emb2[:n])

    model1.eval().to(device)
    model2.eval().to(device)
    n = min(len(digits_data["X_test"]), len(mnist1d_data["X_test"]))

    with torch.no_grad():
        z_sig1, z_img1 = model1(
            torch.from_numpy(mnist1d_data["X_test"][:n]).to(device),
            torch.from_numpy(digits_data["X_test"][:n]).to(device),
        )
        z_sig2, z_img2 = model2(
            torch.from_numpy(mnist1d_data["X_test"][:n]).to(device),
            torch.from_numpy(digits_data["X_test"][:n]).to(device),
        )
        z1 = torch.cat([z_sig1, z_img1], dim=1)
        z2 = torch.cat([z_sig2, z_img2], dim=1)
        cka = compute_cka(z1, z2)

    model1.train()
    model2.train()
    return cka


# ---------------------------------------------------------------------------
# Modality gap
# ---------------------------------------------------------------------------

def compute_modality_gap(emb_img: Tensor, emb_sign: Tensor):
    """
    Characterise the modality gap via the residual (emb_img - emb_sign).

    Returns:
        residual_mean:   mean vector of the residual
        residual_cov:    covariance matrix of the residual
        eigenvalues:     sorted (descending) eigenvalues of the covariance
        mu_norm:         L2 norm of the residual mean
        cov_trace:       trace of the residual covariance
    """
    residual = emb_img - emb_sign
    residual_mean = residual.mean(dim=0)
    residual_cov = torch.cov(residual.T)
    eigenvalues = torch.flip(torch.linalg.eigvalsh(residual_cov), dims=[0])
    mu_norm = torch.norm(residual_mean)
    cov_trace = torch.trace(residual_cov).item()
    return residual_mean, residual_cov, eigenvalues, mu_norm, cov_trace
