"""
Train CLIP with InfoNCE loss on corrupted labels (label flip robustness test).

For each combination of seed and flip rate, a fraction of training labels is
randomly reassigned to a different class before training.  The goal is to
check how retrieval (and later CKA) degrades as label noise increases.

Trains 3 seeds × 4 flip rates = 12 models and saves:
    checkpoints/clip_cnn_seed{S}_hd64_pd32_flipr{R}.pth

Run from the project root:
    python scripts/train_clip_labelflip.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import tqdm

from models.clip_model import CLIPModel
from data.dataloader import sample_batch, apply_label_flip
from data.dataset import load_all_datasets
from methods.losses import info_nce_loss
from utils.metrics import evaluate_retrieval

os.makedirs("checkpoints", exist_ok=True)

# ---------------------------------------------------------------------------
seeds          = [42, 123, 999]
flip_rates     = [0.05, 0.1, 0.2, 0.3]
epochs         = 200
steps_per_epoch = 50
batch_size     = 100
temperature    = 0.1
hidden_dim     = 64
projection_dim = 32
mode           = "cnn"
force_reload   = False
# ---------------------------------------------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

results = []

for seed in seeds:
    for flip_rate in flip_rates:
        print(f"\n=== Seed {seed}  flip_rate={flip_rate} ===")
        torch.manual_seed(seed)
        np.random.seed(seed)

        digits_data, mnist1d_data = load_all_datasets(seed=seed, force_reload=force_reload)

        digits_data["y_train"]  = apply_label_flip(digits_data["y_train"],  flip_rate=flip_rate, seed=seed)
        mnist1d_data["y_train"] = apply_label_flip(mnist1d_data["y_train"], flip_rate=flip_rate, seed=seed)

        model = CLIPModel(mode=mode, hidden_dim=hidden_dim, projection_dim=projection_dim).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

        pbar = tqdm.tqdm(range(epochs), desc=f"flip={flip_rate} seed={seed}")
        for epoch in pbar:
            model.train()
            epoch_loss = 0.0
            for _ in range(steps_per_epoch):
                optimizer.zero_grad()
                batch_digits, batch_mnist1d, _ = sample_batch(digits_data, mnist1d_data=mnist1d_data, K=batch_size // 10)
                z_sig, z_img = model(batch_mnist1d.to(device), batch_digits.to(device))
                loss = info_nce_loss(z_img, z_sig, temperature=temperature)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            pbar.set_postfix(loss=f"{epoch_loss / steps_per_epoch:.4f}")

            if (epoch + 1) % 50 == 0:
                r_s2i, r_i2s = evaluate_retrieval(model, digits_data, mnist1d_data, device, k=5)
                pbar.write(f"Epoch {epoch+1}  Recall@5  Sig→Img: {r_s2i:.4f}  Img→Sig: {r_i2s:.4f}")

        r_s2i, r_i2s = evaluate_retrieval(model, digits_data, mnist1d_data, device, k=5)
        ckpt = f"checkpoints/clip_{mode}_seed{seed}_hd{hidden_dim}_pd{projection_dim}_flipr{flip_rate}.pth"
        torch.save(model.state_dict(), ckpt)
        print(f"Saved → {ckpt}")
        results.append((seed, flip_rate, r_s2i, r_i2s))

# ---------------------------------------------------------------------------
print("\n=== Final Results (mean ± std over seeds) ===")
for fr in flip_rates:
    fr_res = [(r[2], r[3]) for r in results if r[1] == fr]
    s2i = [r[0] for r in fr_res]
    i2s = [r[1] for r in fr_res]
    print(f"flip_rate={fr:.2f}  Sig→Img: {np.mean(s2i):.4f} ± {np.std(s2i):.4f}  "
          f"Img→Sig: {np.mean(i2s):.4f} ± {np.std(i2s):.4f}")
