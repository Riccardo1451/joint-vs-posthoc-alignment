"""
Cross-architecture CKA comparison (MLP vs CNN encoders).

Measures how similar the representations are between a CLIP model trained with
MLP encoders and one trained with CNN encoders, seed-matched.

Related to the Platonic Representation Hypothesis: do different architectures
converge to similar representations when trained on the same task?

Run from the project root:
    python scripts/evaluate_architectures.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np

from data.dataset import load_all_datasets
from models.clip_model import CLIPModel
from utils.metrics import evaluate_cka

# ---------------------------------------------------------------------------
seeds          = [42, 123, 999]
force_reload   = False
hidden_dim_cnn = 64
hidden_dim_mlp = 128
projection_dim = 32

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

digits_data, mnist1d_data = load_all_datasets(seed=seeds[0], force_reload=force_reload)

model_cnn = CLIPModel(mode="cnn", hidden_dim=hidden_dim_cnn, projection_dim=projection_dim).to(device)
model_mlp = CLIPModel(mode="mlp", hidden_dim=hidden_dim_mlp, projection_dim=projection_dim).to(device)

# ---------------------------------------------------------------------------
# CKA: CNN vs MLP, same seed
# ---------------------------------------------------------------------------
cka_scores = []
for seed in seeds:
    model_cnn.load_state_dict(torch.load(
        f"checkpoints/clip_infonce/cnn_seed{seed}_hd{hidden_dim_cnn}_pd{projection_dim}.pth",
        weights_only=True))
    model_mlp.load_state_dict(torch.load(
        f"checkpoints/clip_infonce/mlp_seed{seed}_hd{hidden_dim_mlp}_pd{projection_dim}.pth",
        weights_only=True))

    cka = evaluate_cka(model_cnn, model_mlp,
                       digits_data=digits_data, mnist1d_data=mnist1d_data, device=device)
    cka_scores.append(cka)
    print(f"Seed {seed}  CKA (CNN hd{hidden_dim_cnn} vs MLP hd{hidden_dim_mlp}): {cka:.4f}")

print(f"\nAverage CKA: {np.mean(cka_scores):.4f} ± {np.std(cka_scores):.4f}")
