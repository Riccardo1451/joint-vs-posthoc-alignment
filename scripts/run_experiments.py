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
from data.dataloader import apply_label_flip

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
lambda_coral    = 0.0
flip_rate       = [0.05, 0.1, 0.2, 0.3]

results = []



for seed in seeds:
    for fr in flip_rate:
        print(f"\nRunning experiment  seed={seed}  flip_rate={fr} ...")
        digits_data, mnist1d_data = load_all_datasets(seed=seed, force_reload=force_reload)
        y_train_digits_flipped = apply_label_flip(digits_data["y_train"], flip_rate=fr, seed=seed)
        y_train_mnist1d_flipped = apply_label_flip(mnist1d_data["y_train"], flip_rate=fr, seed=seed)
    
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
            lambda_coral=lambda_coral,
            y_train_digits=y_train_digits_flipped,
            y_train_mnist1d=y_train_mnist1d_flipped,
            flip_rate=fr
        )
        results.append((seed, fr, recall_s2i, recall_i2s))

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print("\nFinal Results:")
for fr in flip_rate:
    fr_results = [(r[2], r[3]) for r in results if r[1] == fr]
    s2i = [r[0] for r in fr_results]
    i2s = [r[1] for r in fr_results]
    print(f"flip_rate={fr:.2f}  Sig→Img: {np.mean(s2i):.4f} ± {np.std(s2i):.4f}  Img→Sig: {np.mean(i2s):.4f} ± {np.std(i2s):.4f}")



# recalls_s2i = [r[2] for r in results]
# recalls_i2s = [r[3] for r in results]
# print(f"Sig→Img Recall@5 : {np.mean(recalls_s2i):.4f} ± {np.std(recalls_s2i):.4f}")
# print(f"Img→Sig Recall@5 : {np.mean(recalls_i2s):.4f} ± {np.std(recalls_i2s):.4f}")
