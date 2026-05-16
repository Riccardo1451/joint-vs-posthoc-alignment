"""
Train CLIP with InfoNCE + DeepCORAL loss (joint distribution alignment).

The DeepCORAL term penalises the difference in covariance between the two
modality embeddings, encouraging a more uniform modality gap.

Trains 3 seeds and saves:
    checkpoints/clip_deepcoral/cnn_seed{S}_hd64_pd32.pth

Run from the project root:
    python scripts/train_clip_deepcoral.py
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

os.makedirs("checkpoints/clip_deepcoral", exist_ok=True)

# ---------------------------------------------------------------------------
seeds          = [42, 123, 999]
epochs         = 200
steps_per_epoch = 50
batch_size     = 100
temperature    = 0.1
hidden_dim     = 64
projection_dim = 32
mode           = "cnn"
lambda_coral   = 0.1   # weight of the DeepCORAL term relative to InfoNCE
force_reload   = False
# ---------------------------------------------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
print(f"lambda_coral = {lambda_coral}")

results = []

for seed in seeds:
    print(f"\n=== Seed {seed} ===")
    torch.manual_seed(seed)
    np.random.seed(seed)

    digits_data, mnist1d_data = load_all_datasets(seed=seed, force_reload=force_reload)

    model = CLIPModel(mode=mode, hidden_dim=hidden_dim, projection_dim=projection_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    pbar = tqdm.tqdm(range(epochs), desc=f"InfoNCE+CORAL - Seed {seed}")
    for epoch in pbar:
        model.train()
        epoch_loss = 0.0
        for _ in range(steps_per_epoch):
            optimizer.zero_grad()
            batch_digits, batch_mnist1d, _ = sample_batch(digits_data, mnist1d_data=mnist1d_data, K=batch_size // 10)
            z_sig, z_img = model(batch_mnist1d.to(device), batch_digits.to(device))
            loss = info_nce_loss(z_img, z_sig, temperature=temperature) \
                 + lambda_coral * deep_coral_loss(z_img, z_sig)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        pbar.set_postfix(loss=f"{epoch_loss / steps_per_epoch:.4f}")

        if (epoch + 1) % 50 == 0:
            r_s2i, r_i2s = evaluate_retrieval(model, digits_data, mnist1d_data, device, k=5)
            pbar.write(f"Epoch {epoch+1}  Recall@5  Sig→Img: {r_s2i:.4f}  Img→Sig: {r_i2s:.4f}")

    r_s2i, r_i2s = evaluate_retrieval(model, digits_data, mnist1d_data, device, k=5)
    ckpt = f"checkpoints/clip_deepcoral/{mode}_seed{seed}_hd{hidden_dim}_pd{projection_dim}.pth"
    torch.save(model.state_dict(), ckpt)
    print(f"Saved → {ckpt}")
    results.append((seed, r_s2i, r_i2s))

# ---------------------------------------------------------------------------
print("\n=== Final Results ===")
s2i = [r[1] for r in results]
i2s = [r[2] for r in results]
print(f"Recall@5  Sig→Img: {np.mean(s2i):.4f} ± {np.std(s2i):.4f}")
print(f"Recall@5  Img→Sig: {np.mean(i2s):.4f} ± {np.std(i2s):.4f}")
