"""
Joint alignment training (CLIP-like).

Train a CLIP model on the MNIST-1D / Digits pair using InfoNCE contrastive loss.
Run from the project root:
    python scripts/train_clip.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import tqdm

from models.clip_model import CLIPModel
from data.dataloader import sample_batch
from data.dataset import load_all_datasets
from methods.losses import info_nce_loss, deep_coral_loss
from utils.metrics import evaluate_retrieval

os.makedirs("checkpoints", exist_ok=True)


def train_clip(
    seed: int,
    epochs: int,
    steps_per_epoch: int,
    batch_size: int,
    hidden_dim: int,
    projection_dim: int,
    temperature: float,
    mode: str = "mlp",
    force_reload: bool = False,
    y_train_digits: np.ndarray = None,
    y_train_mnist1d: np.ndarray = None,
    lambda_coral: float = 0.1,
    flip_rate: float = 0.0
):
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = CLIPModel(mode=mode, hidden_dim=hidden_dim, projection_dim=projection_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    digits_data, mnist1d_data = load_all_datasets(seed=seed, force_reload=force_reload)
    if y_train_digits is not None:
        digits_data["y_train"] = y_train_digits
    if y_train_mnist1d is not None:
        mnist1d_data["y_train"] = y_train_mnist1d 

    pbar = tqdm.tqdm(range(epochs), desc=f"Training - Seed {seed}")

    for epoch in pbar:
        model.train()
        epoch_loss = 0.0

        for step in range(steps_per_epoch):
            optimizer.zero_grad()

            batch_digits, batch_mnist1d, labels = sample_batch(
                digits_data, mnist1d_data=mnist1d_data, K=batch_size // 10
            )
            batch_digits = batch_digits.to(device)
            batch_mnist1d = batch_mnist1d.to(device)

            z_sig, z_img = model(batch_mnist1d, batch_digits)
            loss = info_nce_loss(z_img, z_sig, temperature=temperature) + lambda_coral * deep_coral_loss(z_img, z_sig)
            loss.backward()
            epoch_loss += loss.item()
            optimizer.step()

        avg_loss = epoch_loss / steps_per_epoch
        pbar.set_postfix(loss=f"{avg_loss:.4f}")

        if (epoch + 1) % 50 == 0:
            recall_s2i, recall_i2s = evaluate_retrieval(
                model, digits_data=digits_data, mnist1d_data=mnist1d_data, device=device, k=5
            )
            pbar.write(
                f"Epoch {epoch+1} - Recall@5  Sig→Img: {recall_s2i:.4f}, Img→Sig: {recall_i2s:.4f}"
            )

    recall_s2i, recall_i2s = evaluate_retrieval(
        model, digits_data=digits_data, mnist1d_data=mnist1d_data, device=device, k=5
    )

    ckpt_path = f"checkpoints/clip_{mode}_seed{seed}_hd{hidden_dim}_pd{projection_dim}_flipr{flip_rate}.pth"
    torch.save(model.state_dict(), ckpt_path)
    print(f"Model saved to {ckpt_path}")

    return recall_s2i, recall_i2s
