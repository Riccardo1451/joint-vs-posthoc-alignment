"""
Evaluate CLIP trained with InfoNCE + DeepCORAL loss.

Computes:
  - CKA between all seed pairs (representational convergence)
  - t-SNE of the embedding space for seed 42
  - Modality gap eigenspectrum vs Procrustes baseline

Requires:
  - checkpoints/clip_deepcoral/cnn_seed{42,123,999}_hd64_pd32.pth  (train_clip_deepcoral.py)
  - checkpoints/analysis/eigenvalues_procrustes.pth                (align_procrustes.py)

Run from the project root:
    python scripts/evaluate_clip_deepcoral.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from itertools import combinations

from data.dataset import load_all_datasets
from data.dataloader import build_paired_dataset
from models.clip_model import CLIPModel
from utils.metrics import evaluate_cka, compute_modality_gap
from utils.visualization import plot_tsne, plot_eigenspectrum

os.makedirs("figures", exist_ok=True)

# ---------------------------------------------------------------------------
seeds          = [42, 123, 999]
hidden_dim     = 64
projection_dim = 32
mode           = "cnn"
force_reload   = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

digits_data, mnist1d_data = load_all_datasets(seed=seeds[0], force_reload=force_reload)

model1 = CLIPModel(hidden_dim=hidden_dim, projection_dim=projection_dim, mode=mode).to(device)
model2 = CLIPModel(hidden_dim=hidden_dim, projection_dim=projection_dim, mode=mode).to(device)

def ckpt(seed):
    return f"checkpoints/clip_deepcoral/{mode}_seed{seed}_hd{hidden_dim}_pd{projection_dim}.pth"

# ---------------------------------------------------------------------------
# CKA across seed pairs
# ---------------------------------------------------------------------------
print("\n--- CKA across seeds ---")
cka_scores = []
for s1, s2 in combinations(seeds, 2):
    model1.load_state_dict(torch.load(ckpt(s1), weights_only=True))
    model2.load_state_dict(torch.load(ckpt(s2), weights_only=True))
    cka = evaluate_cka(model1, model2, digits_data=digits_data, mnist1d_data=mnist1d_data, device=device)
    cka_scores.append(cka)
    print(f"CKA (seed {s1} vs {s2}): {cka:.4f}")
print(f"Average CKA: {np.mean(cka_scores):.4f} ± {np.std(cka_scores):.4f}")

# ---------------------------------------------------------------------------
# t-SNE (seed 42)
# ---------------------------------------------------------------------------
print(f"\n--- t-SNE (seed {seeds[0]}) ---")
model1.load_state_dict(torch.load(ckpt(seeds[0]), weights_only=True))
model1.eval()
n = min(len(digits_data["X_test"]), len(mnist1d_data["X_test"]))
with torch.no_grad():
    z_sig, z_img = model1(
        torch.from_numpy(mnist1d_data["X_test"][:n]).to(device),
        torch.from_numpy(digits_data["X_test"][:n]).to(device),
    )
plot_tsne(z_img, z_sig,
          torch.from_numpy(digits_data["y_test"][:n]).to(device),
          torch.from_numpy(mnist1d_data["y_test"][:n]).to(device),
          title=f"t-SNE CLIP DeepCORAL (Seed {seeds[0]})",
          save_path=f"figures/tsne_clip_deepcoral_seed{seeds[0]}.png")

# ---------------------------------------------------------------------------
# Modality gap eigenspectrum vs Procrustes
# ---------------------------------------------------------------------------
print("\n--- Modality gap eigenspectrum ---")
eigenvalues_list = []
for seed in seeds:
    m = CLIPModel(hidden_dim=hidden_dim, projection_dim=projection_dim, mode=mode).to(device)
    m.load_state_dict(torch.load(ckpt(seed), weights_only=True))
    m.eval()
    align_set = build_paired_dataset(digits_data, mnist1d_data, seed=seed)
    with torch.no_grad():
        z_sig, z_img = m(
            torch.from_numpy(align_set["X_mnist1d"]).to(device),
            torch.from_numpy(align_set["X_digits"]).to(device),
        )
    _, _, eigenvalues, mu_norm, cov_trace = compute_modality_gap(z_img, z_sig)
    print(f"Seed {seed}  |mu|: {mu_norm:.4f}  tr(Σ): {cov_trace:.4f}")
    eigenvalues_list.append(eigenvalues)

avg_eigenvalues = torch.stack(eigenvalues_list).mean(dim=0)
plot_eigenspectrum(
    {
        "CLIP DeepCORAL": avg_eigenvalues,
        "Procrustes (paired)": torch.load("checkpoints/analysis/eigenvalues_procrustes.pth"),
    },
    title="Modality Gap Eigenspectrum - CLIP DeepCORAL vs Procrustes",
    save_path="figures/eigenspectrum_clip_deepcoral.png",
)
