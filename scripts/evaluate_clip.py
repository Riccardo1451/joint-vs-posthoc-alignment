"""
Evaluation of jointly-trained CLIP models.

Computes CKA across random seeds, plots t-SNE embeddings and the modality gap
eigenspectrum for the jointly-trained CLIP models.

Run from the project root:
    python scripts/evaluate_clip.py
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
seeds        = [42, 123, 999]
force_reload = False
hidden_dim   = 128
projection_dim = 32

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

digits_data, mnist1d_data = load_all_datasets(seed=seeds[0], force_reload=force_reload)

model1 = CLIPModel(hidden_dim=hidden_dim, projection_dim=projection_dim).to(device)
model2 = CLIPModel(hidden_dim=hidden_dim, projection_dim=projection_dim).to(device)

# ---------------------------------------------------------------------------
# CKA across seed pairs
# ---------------------------------------------------------------------------
cka_scores = []
for s1, s2 in combinations(seeds, 2):
    model1.load_state_dict(torch.load(
        f"checkpoints/clip_mlp_seed{s1}_hd{hidden_dim}_pd{projection_dim}.pth", weights_only=True))
    model2.load_state_dict(torch.load(
        f"checkpoints/clip_mlp_seed{s2}_hd{hidden_dim}_pd{projection_dim}.pth", weights_only=True))
    cka = evaluate_cka(model1, model2, digits_data=digits_data, mnist1d_data=mnist1d_data, device=device)
    cka_scores.append(cka)
    print(f"CKA (seed {s1} vs {s2}): {cka:.4f}")

print(f"Average CKA: {np.mean(cka_scores):.4f} ± {np.std(cka_scores):.4f}")

# ---------------------------------------------------------------------------
# t-SNE
# ---------------------------------------------------------------------------
model1.load_state_dict(torch.load(
    f"checkpoints/clip_mlp_seed{seeds[0]}_hd{hidden_dim}_pd{projection_dim}.pth", weights_only=True))
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
          title=f"t-SNE of CLIP Embeddings (Seed {seeds[0]})",
          save_path=f"figures/tsne_clip_seed{seeds[0]}.png")

# ---------------------------------------------------------------------------
# Modality gap eigenspectrum
# ---------------------------------------------------------------------------
clip_eigenvalues_list = []
for seed in seeds:
    m = CLIPModel(hidden_dim=hidden_dim, projection_dim=projection_dim).to(device)
    m.load_state_dict(torch.load(
        f"checkpoints/clip_mlp_seed{seed}_hd{hidden_dim}_pd{projection_dim}.pth", weights_only=True))
    m.eval()

    align_set = build_paired_dataset(digits_data, mnist1d_data, seed=seed)
    with torch.no_grad():
        z_sig, z_img = m(
            torch.from_numpy(align_set["X_mnist1d"]).to(device),
            torch.from_numpy(align_set["X_digits"]).to(device),
        )

    _, _, eigenvalues, mu_norm, cov_trace = compute_modality_gap(z_img, z_sig)
    print(f"Seed {seed}  |mu|: {mu_norm:.4f}  tr(Σ): {cov_trace:.4f}")
    clip_eigenvalues_list.append(eigenvalues)

avg_eigenvalues_clip = torch.stack(clip_eigenvalues_list).mean(dim=0)

plot_eigenspectrum(
    {
        "CLIP": avg_eigenvalues_clip,
        "Procrustes (paired)": torch.load("checkpoints/eigenvalues_procrustes.pth"),
    },
    title="Modality Gap Eigenspectrum - CLIP vs Procrustes",
    save_path="figures/eigenspectrum_clip.png",
)
