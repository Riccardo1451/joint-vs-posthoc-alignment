"""
Evaluate label flip robustness: Recall@5 and CKA at each flip rate.

Loads the CLIP models trained with corrupted labels and measures both
retrieval performance and representational convergence (CKA across seeds).
Showing both metrics together lets you verify whether CKA degrades in
lockstep with retrieval as label noise increases.

Set use_coral=True to evaluate the DeepCORAL variants (_flipr{R}_CORAL.pth),
or False for the InfoNCE-only variants (_flipr{R}.pth).

Requires:
  - checkpoints/clip_infonce/cnn_seed{S}_hd64_pd32.pth                      (baseline, flip_rate=0)
  - checkpoints/clip_labelflip/infonce/cnn_seed{S}_hd64_pd32_flipr{R}.pth  (lambda_coral=0)
  - checkpoints/clip_labelflip/deepcoral/cnn_seed{S}_hd64_pd32_flipr{R}.pth (lambda_coral>0)

Run from the project root:
    python scripts/evaluate_label_flip.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from itertools import combinations

from data.dataset import load_all_datasets
from models.clip_model import CLIPModel
from utils.metrics import evaluate_retrieval, evaluate_cka

# ---------------------------------------------------------------------------
seeds          = [42, 123, 999]
flip_rates     = [0.0, 0.05, 0.1, 0.2, 0.3]
hidden_dim     = 64
projection_dim = 32
mode           = "cnn"
use_coral      = True   # True → load _flipr{R}_CORAL.pth, False → load _flipr{R}.pth
force_reload   = False
# ---------------------------------------------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
print(f"Evaluating: {'InfoNCE + DeepCORAL' if use_coral else 'InfoNCE only'} variants")

digits_data, mnist1d_data = load_all_datasets(seed=seeds[0], force_reload=force_reload)

model      = CLIPModel(hidden_dim=hidden_dim, projection_dim=projection_dim, mode=mode).to(device)
model_pair = CLIPModel(hidden_dim=hidden_dim, projection_dim=projection_dim, mode=mode).to(device)

def ckpt(seed, flip_rate):
    if flip_rate == 0.0:
        return f"checkpoints/clip_infonce/{mode}_seed{seed}_hd{hidden_dim}_pd{projection_dim}.pth"
    subfolder = "deepcoral" if use_coral else "infonce"
    return f"checkpoints/clip_labelflip/{subfolder}/{mode}_seed{seed}_hd{hidden_dim}_pd{projection_dim}_flipr{flip_rate}.pth"

# ---------------------------------------------------------------------------
print(f"\n{'flip_rate':>10}  {'Recall S→I':>18}  {'Recall I→S':>18}  {'CKA':>15}")
print("-" * 68)

for flip_rate in flip_rates:
    recalls_s2i, recalls_i2s, cka_scores = [], [], []

    for seed in seeds:
        model.load_state_dict(torch.load(ckpt(seed, flip_rate), weights_only=True))
        r_s2i, r_i2s = evaluate_retrieval(model, digits_data, mnist1d_data, device, k=5)
        recalls_s2i.append(r_s2i)
        recalls_i2s.append(r_i2s)

    for s1, s2 in combinations(seeds, 2):
        model.load_state_dict(torch.load(ckpt(s1, flip_rate), weights_only=True))
        model_pair.load_state_dict(torch.load(ckpt(s2, flip_rate), weights_only=True))
        cka = evaluate_cka(model, model_pair, digits_data=digits_data, mnist1d_data=mnist1d_data, device=device)
        cka_scores.append(cka)

    print(
        f"{flip_rate:>10.2f}  "
        f"{np.mean(recalls_s2i):.4f} ± {np.std(recalls_s2i):.4f}  "
        f"{np.mean(recalls_i2s):.4f} ± {np.std(recalls_i2s):.4f}  "
        f"{np.mean(cka_scores):.4f} ± {np.std(cka_scores):.4f}"
    )
