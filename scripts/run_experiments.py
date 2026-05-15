"""
Full experiment runner: trains CLIP models across seeds and reports Recall@5.

Run from the project root:
    python scripts/run_experiments.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from scripts.train_clip import train_clip
from utils.metrics import evaluate_cka
from models.clip_model import CLIPModel
from data.dataset import load_all_datasets

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ---------------------------------------------------------------------------
# Experiment config
# ---------------------------------------------------------------------------
seeds           = [42, 123, 999]
epochs          = 200
steps_per_epoch = 50
batch_size      = 100
temperature     = 0.1
hidden_dim      = 64
projection_dim  = 32
mode            = "cnn"
force_reload    = False

results = []

for seed in seeds:
    print(f"\nRunning experiment  seed={seed} ...")
    recall_s2i, recall_i2s = train_clip(
        seed=seed,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        batch_size=batch_size,
        mode=mode,
        hidden_dim=hidden_dim,
        projection_dim=projection_dim,
        temperature=temperature,
        force_reload=force_reload,
    )
    results.append((seed, recall_s2i, recall_i2s))

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print("\nFinal Results:")
recalls_s2i = [r[1] for r in results]
recalls_i2s = [r[2] for r in results]
print(f"Sig→Img Recall@5 : {np.mean(recalls_s2i):.4f} ± {np.std(recalls_s2i):.4f}")
print(f"Img→Sig Recall@5 : {np.mean(recalls_i2s):.4f} ± {np.std(recalls_i2s):.4f}")
