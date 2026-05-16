import numpy as np
import torch


def coral_align(source_embs, target_embs, device=None):
    """
    Align source embeddings to target embeddings with CORAL method.

    Args:
        source_embs: source embeddings, shape (N, D)
        target_embs: target embeddings, shape (M, D)
    Returns:
        aligned_source_embs: source embeddings aligned to target space, shape (N, D)
    """

    #Check dimensions
    assert source_embs.shape[1] == target_embs.shape[1], \
    f"Embedding dimensions must match: {source_embs.shape[1]} vs {target_embs.shape[1]}"
    source_embs = source_embs.detach().cpu().numpy()
    target_embs = target_embs.detach().cpu().numpy()
    

    #We need 4 quantities
    mean_S = np.mean(source_embs, axis=0) # (D,)
    mean_T = np.mean(target_embs, axis=0) # (D,)
    cov_S = np.cov(source_embs, rowvar=False) # (D, D)
    cov_T = np.cov(target_embs, rowvar=False) # (D, D)

    #Compute the whitening transformation for source, squared root inverse of cov_S
    eigenvalues_s, eigenvectors_s = np.linalg.eigh(cov_S)

    #Clip eigenvalues to avoid numerical issues with small values
    eigenvalues_s = np.clip(eigenvalues_s, a_min=1e-8, a_max=None)

    #Compute the whitening transformation A^{1/2} = V @ diag(λ^{1/2}) @ Vᵀ
    cov_S_inv_sqrt = eigenvectors_s @ np.diag(1 / np.sqrt(eigenvalues_s)) @ eigenvectors_s.T # (D, D)

    #Compute the coloring transformation for target, squared root of cov_T
    eigenvalues_t, eigenvectors_t = np.linalg.eigh(cov_T)

    #Clip eigenvalues to avoid numerical issues with small values
    eigenvalues_t = np.clip(eigenvalues_t, a_min=1e-8, a_max=None)

    #Compute the coloring transformation B^{1/2} = V @ diag(λ^{1/2}) @ Vᵀ
    cov_T_sqrt = eigenvectors_t @ np.diag(np.sqrt(eigenvalues_t)) @ eigenvectors_t.T # (D, D)

    #Transform matrix
    W = cov_S_inv_sqrt @ cov_T_sqrt # (D, D)

    #Align source to target
    X_coral = (source_embs - mean_S) @ W + mean_T # (N, D)

    return torch.tensor(X_coral, dtype=torch.float32).to(device)