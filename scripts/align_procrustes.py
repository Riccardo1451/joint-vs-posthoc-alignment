"""
Post-hoc alignment with Procrustes analysis only.

Loads pre-trained unimodal models, computes the optimal rotation Q on a
paired alignment set, then evaluates retrieval (Recall@5) and CKA on the
test set and plots t-SNE before/after rotation.

For the Procrustes + CORAL variant see align_procrustes_coral.py.

Requires:
  - checkpoints/unimodal/{mnist1d,digits}_seed{S}.pth  (train_unimodal.py)

Run from the project root:
    python scripts/align_procrustes.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np

from models.unimodal import UnimodalModelMnist1DCNN, UnimodalModelDigitsCNN
from data.dataset import load_all_datasets
from data.dataloader import build_paired_dataset, build_paired_test
from methods.procrustes import procrustes_align
from utils.metrics import recall_at_k, compute_crossmodal_cka, compute_modality_gap
from utils.visualization import plot_tsne, plot_eigenspectrum

os.makedirs("figures/procrustes", exist_ok=True)

# ---------------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

seeds        = [42, 123, 999]
force_reload = False
results      = {}

model_mnist1d = UnimodalModelMnist1DCNN().to(device)
model_digits  = UnimodalModelDigitsCNN().to(device)

for seed in seeds:
    print(f"\n--- Seed {seed} ---")

    digits_dataset, mnist1d_dataset = load_all_datasets(force_reload=force_reload, seed=seed)
    align_set = build_paired_dataset(digits_dataset, mnist1d_dataset, seed=seed)

    model_mnist1d.load_state_dict(torch.load(f"checkpoints/unimodal/cnn_mnist1d_seed{seed}.pth", weights_only=True))
    model_digits.load_state_dict(torch.load(f"checkpoints/unimodal/cnn_digits_seed{seed}.pth", weights_only=True))
    model_mnist1d.eval()
    model_digits.eval()

    # --- Alignment set embeddings ---
    with torch.no_grad():
        embs_mnist1d_align = model_mnist1d.get_embedding(torch.from_numpy(align_set["X_mnist1d"]).to(device))
        embs_digits_align  = model_digits.get_embedding(torch.from_numpy(align_set["X_digits"]).to(device))

    Q = procrustes_align(embs_mnist1d_align, embs_digits_align)
    
    procrustes_error = torch.norm(embs_digits_align - embs_mnist1d_align @ Q.T, p='fro').item()
    print(f"Procrustes error: {procrustes_error:.4f}")

    # --- Test set embeddings ---
    with torch.no_grad():
        embs_mnist1d_test = model_mnist1d.get_embedding(torch.from_numpy(mnist1d_dataset["X_test"]).to(device))
        embs_digits_test  = model_digits.get_embedding(torch.from_numpy(digits_dataset["X_test"]).to(device))

    y_mnist1d_test = torch.from_numpy(mnist1d_dataset["y_test"]).to(device)
    y_digits_test  = torch.from_numpy(digits_dataset["y_test"]).to(device)

    mnist1d_aligned = embs_mnist1d_test @ Q.T

    # --- Retrieval ---
    recall_s2i = recall_at_k(mnist1d_aligned, y_mnist1d_test, embs_digits_test, y_digits_test, k=5)
    recall_i2s = recall_at_k(embs_digits_test, y_digits_test, mnist1d_aligned, y_mnist1d_test, k=5)
    print(f"Recall@5  Sig→Img: {recall_s2i:.4f}  Img→Sig: {recall_i2s:.4f}")

    # --- CKA Cross-Modal (paired test set) ---
    paired_test = build_paired_test(digits_dataset, mnist1d_dataset, seed=42)
    with torch.no_grad():
        embs_digits_paired  = model_digits.get_embedding(
            torch.from_numpy(paired_test["X_digits"]).to(device))
        embs_mnist1d_paired = model_mnist1d.get_embedding(
            torch.from_numpy(paired_test["X_mnist1d"]).to(device))
    embs_mnist1d_aligned_paired = embs_mnist1d_paired @ Q.T
    cka = compute_crossmodal_cka(embs_digits_paired, embs_mnist1d_aligned_paired)
    print(f"CKA Cross-Modal: {cka:.4f}")

    # --- Modality gap ---
    embs_mnist1d_align_rotated = embs_mnist1d_align @ Q.T
    _, _, eigenvalues, mu_norm, cov_trace = compute_modality_gap(embs_digits_align, embs_mnist1d_align_rotated)
    print(f"Modality gap  |mu|: {mu_norm:.4f}  tr(Σ): {cov_trace:.4f}")

    results[seed] = {
        "recall_s2i": recall_s2i, "recall_i2s": recall_i2s,
        "cka": cka, "eigenvalues": eigenvalues,
    }

    # --- t-SNE ---
    plot_tsne(embs_digits_test, embs_mnist1d_test, y_digits_test, y_mnist1d_test,
              title=f"Unimodal embeddings before Procrustes (Seed {seed})",
              save_path=f"figures/procrustes/tsne_before_seed{seed}.png")

    plot_tsne(embs_digits_test, mnist1d_aligned, y_digits_test, y_mnist1d_test,
              title=f"Unimodal embeddings after Procrustes (Seed {seed})",
              save_path=f"figures/procrustes/tsne_after_seed{seed}.png")

# ---------------------------------------------------------------------------
print("\n=== Final Results (mean ± std over seeds) ===")
s2i  = [results[s]["recall_s2i"] for s in seeds]
i2s  = [results[s]["recall_i2s"] for s in seeds]
ckas = [results[s]["cka"] for s in seeds]
print(f"Recall@5  Sig→Img : {np.mean(s2i):.4f} ± {np.std(s2i):.4f}")
print(f"Recall@5  Img→Sig : {np.mean(i2s):.4f} ± {np.std(i2s):.4f}")

print("\n=== CKA Cross-Modal (geometria img vs sig) ===")
print(f"CKA Cross-Modal   : {np.mean(ckas):.4f} ± {np.std(ckas):.4f}")

avg_eigenvalues = torch.stack([results[s]["eigenvalues"] for s in seeds]).mean(dim=0)
os.makedirs("checkpoints/analysis", exist_ok=True)
torch.save(avg_eigenvalues, "checkpoints/analysis/eigenvalues_procrustes.pth")

plot_eigenspectrum(
    {"Procrustes (paired)": avg_eigenvalues},
    title="Modality Gap Eigenspectrum - Procrustes",
    save_path="figures/procrustes/eigenspectrum.png",
)
