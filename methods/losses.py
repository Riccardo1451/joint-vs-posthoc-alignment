import torch
import torch.nn as nn

def info_nce_loss(z_img, z_sig, temperature=0.07):

    sim_matrix = torch.matmul(z_img, z_sig.T) / temperature # (B, B)

    N = z_img.size(0)

    pos_pair = torch.arange(N, device=z_img.device) # (B,)

    loss_img = nn.CrossEntropyLoss()(sim_matrix, pos_pair)
    loss_sig = nn.CrossEntropyLoss()(sim_matrix.T, pos_pair)

    loss = (loss_img + loss_sig) / 2

    return loss, loss_img.item(), loss_sig.item()