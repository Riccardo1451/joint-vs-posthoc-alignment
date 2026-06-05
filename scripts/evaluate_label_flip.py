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
from data.dataloader import build_paired_test
from models.clip_model import CLIPModel
from utils.metrics import evaluate_retrieval, evaluate_cka, compute_crossmodal_cka

# ---------------------------------------------------------------------------
seeds          = [42, 123, 999]
flip_rates     = [0.0, 0.05, 0.1, 0.2, 0.3]
hidden_dim     = 64
projection_dim = 32
mode           = "cnn"
use_coral      = False   # True → load _flipr{R}_CORAL.pth, False → load _flipr{R}.pth
force_reload   = False
# ---------------------------------------------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
print(f"Evaluating: {'InfoNCE + DeepCORAL' if use_coral else 'InfoNCE only'} variants")

digits_data, mnist1d_data = load_all_datasets(seed=seeds[0], force_reload=force_reload)
paired_test = build_paired_test(digits_data, mnist1d_data, seed=42)

model      = CLIPModel(hidden_dim=hidden_dim, projection_dim=projection_dim, mode=mode).to(device)
model_pair = CLIPModel(hidden_dim=hidden_dim, projection_dim=projection_dim, mode=mode).to(device)

def ckpt(seed, flip_rate):
    if flip_rate == 0.0:
        return f"checkpoints/clip_infonce/{mode}_seed{seed}_hd{hidden_dim}_pd{projection_dim}.pth"
    subfolder = "deepcoral" if use_coral else "infonce"
    return f"checkpoints/clip_labelflip/{subfolder}/{mode}_seed{seed}_hd{hidden_dim}_pd{projection_dim}_flipr{flip_rate}.pth"

# ---------------------------------------------------------------------------
# Collect all results per flip_rate
# ---------------------------------------------------------------------------
results = {}
for flip_rate in flip_rates:
    recalls_s2i, recalls_i2s = [], []
    cka_interrun, cka_crossmodal = [], []

    for seed in seeds:
        model.load_state_dict(torch.load(ckpt(seed, flip_rate), weights_only=True))
        r_s2i, r_i2s = evaluate_retrieval(model, digits_data, mnist1d_data, device, k=5)
        recalls_s2i.append(r_s2i)
        recalls_i2s.append(r_i2s)

        model.eval()
        with torch.no_grad():
            z_sig, z_img = model(
                torch.from_numpy(paired_test["X_mnist1d"]).to(device),
                torch.from_numpy(paired_test["X_digits"]).to(device),
            )
        cka_crossmodal.append(compute_crossmodal_cka(z_img, z_sig))

    for s1, s2 in combinations(seeds, 2):
        model.load_state_dict(torch.load(ckpt(s1, flip_rate), weights_only=True))
        model_pair.load_state_dict(torch.load(ckpt(s2, flip_rate), weights_only=True))
        cka_interrun.append(
            evaluate_cka(model, model_pair, digits_data=digits_data, mnist1d_data=mnist1d_data, device=device)
        )

    results[flip_rate] = {
        "recalls_s2i": recalls_s2i, "recalls_i2s": recalls_i2s,
        "cka_interrun": cka_interrun, "cka_crossmodal": cka_crossmodal,
    }

# ---------------------------------------------------------------------------
# Print: Retrieval
# ---------------------------------------------------------------------------
print(f"\n{'flip_rate':>10}  {'Recall S→I':>22}  {'Recall I→S':>22}")
print("-" * 60)
for flip_rate in flip_rates:
    r = results[flip_rate]
    print(
        f"{flip_rate:>10.2f}  "
        f"{np.mean(r['recalls_s2i']):.4f} ± {np.std(r['recalls_s2i']):.4f}  "
        f"{np.mean(r['recalls_i2s']):.4f} ± {np.std(r['recalls_i2s']):.4f}"
    )

# ---------------------------------------------------------------------------
# Print: CKA Inter-Run
# ---------------------------------------------------------------------------
print(f"\n=== CKA Inter-Run (stabilità tra seed) ===")
print(f"{'flip_rate':>10}  {'CKA Inter-Run':>22}")
print("-" * 36)
for flip_rate in flip_rates:
    r = results[flip_rate]
    print(
        f"{flip_rate:>10.2f}  "
        f"{np.mean(r['cka_interrun']):.4f} ± {np.std(r['cka_interrun']):.4f}"
    )

# ---------------------------------------------------------------------------
# Print: CKA Cross-Modal
# ---------------------------------------------------------------------------
print(f"\n=== CKA Cross-Modal (geometria img vs sig) ===")
print(f"{'flip_rate':>10}  {'CKA Cross-Modal':>22}")
print("-" * 36)
for flip_rate in flip_rates:
    r = results[flip_rate]
    print(
        f"{flip_rate:>10.2f}  "
        f"{np.mean(r['cka_crossmodal']):.4f} ± {np.std(r['cka_crossmodal']):.4f}"
    )
