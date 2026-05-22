import torch
import torch.nn as nn

def info_nce_loss(z_img, z_sig, temperature=0.07):

    sim_matrix = torch.matmul(z_img, z_sig.T) / temperature # (B, B)

    N = z_img.size(0)

    pos_pair = torch.arange(N, device=z_img.device) # (B,)

    loss_img = nn.CrossEntropyLoss()(sim_matrix, pos_pair)
    loss_sig = nn.CrossEntropyLoss()(sim_matrix.T, pos_pair)

    loss = (loss_img + loss_sig) / 2

    return loss

def deep_coral_loss(embs_img, embs_sig):
    # Covariance matrices
    cov_img = (embs_img - embs_img.mean(dim = 0)).T @ (embs_img - embs_img.mean(dim = 0)) / (embs_img.size(0) - 1)
    cov_sig = (embs_sig - embs_sig.mean(dim = 0)).T @ (embs_sig - embs_sig.mean(dim = 0)) / (embs_sig.size(0) - 1)

    # Frobenius norm of the difference squared
    loss = torch.norm(cov_img - cov_sig, p='fro') ** 2

    # Scale the loss by the feature dimension
    loss = loss / (4 * embs_img.size(1) ** 2)

    return loss
