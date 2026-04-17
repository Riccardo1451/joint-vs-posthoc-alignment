import torch

def compute_cka(X, Y):

    K = torch.matmul(X, X.T)
    L = torch.matmul(Y, Y.T)

    H = torch.eye(K.size(0), device=X.device) - (1.0/K.size(0))*torch.ones(K.size(0), K.size(0), device=X.device)

    K_centered = torch.matmul(H, torch.matmul(K, H))
    L_centered = torch.matmul(H, torch.matmul(L, H))

    HSIC_K2L = torch.sum(K_centered * L_centered)
    HSIC_K2K = torch.sum(K_centered * K_centered)
    HSIC_L2L = torch.sum(L_centered * L_centered)

    CKA = HSIC_K2L / torch.sqrt(HSIC_K2K * HSIC_L2L)

    return CKA.item()