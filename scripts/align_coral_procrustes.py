"""
Post-hoc alignment: CORAL distribution matching followed by Procrustes rotation.

Pipeline:
  1. Load pre-trained unimodal CNN classifiers (train with train_unimodal.py --arch cnn first)
  2. Apply CORAL on the paired alignment set to match the covariance of the
     MNIST-1D embeddings to the Digits embeddings
  3. Fit Procrustes rotation Q on the CORAL-aligned MNIST-1D embeddings and the
     Digits embeddings
  4. At test time: apply the CORAL transform first, then the Q.T rotation
  5. Evaluate retrieval (Recall@5), cross-modal CKA on paired test set,
     modality gap, and Procrustes distance on class centroids

For the reversed order (Procrustes then CORAL) see align_procrustes_coral.py.

Requires:
  - checkpoints/unimodal/cnn_{mnist1d,digits}_seed{S}.pth  (train_unimodal.py --arch cnn)

Run from the project root:
    python scripts/align_coral_procrustes.py
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
from methods.coral import coral_align
from utils.metrics import recall_at_k, compute_crossmodal_cka, compute_modality_gap
from utils.visualization import plot_tsne, plot_eigenspectrum

os.makedirs("figures/coral_procrustes", exist_ok=True)

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

    # --- Step 1: CORAL on training embeddings ---
    embs_mnist1d_coral_align = coral_align(embs_mnist1d_align, embs_digits_align, device=device)

    # Compute and store CORAL transform parameters for reuse at test time
    src_np = embs_mnist1d_align.detach().cpu().numpy()
    tgt_np = embs_digits_align.detach().cpu().numpy()
    mean_S = src_np.mean(axis=0)
    mean_T = tgt_np.mean(axis=0)
    cov_S  = np.cov(src_np, rowvar=False)
    cov_T  = np.cov(tgt_np, rowvar=False)
    eigvals_s, eigvecs_s = np.linalg.eigh(cov_S)
    eigvals_t, eigvecs_t = np.linalg.eigh(cov_T)
    eigvals_s = np.clip(eigvals_s, 1e-8, None)
    eigvals_t = np.clip(eigvals_t, 1e-8, None)
    W_coral = (eigvecs_s @ np.diag(1.0 / np.sqrt(eigvals_s)) @ eigvecs_s.T
               @ eigvecs_t @ np.diag(np.sqrt(eigvals_t)) @ eigvecs_t.T)

    # --- Step 2: Procrustes on CORAL-aligned training embeddings ---
    Q = procrustes_align(embs_mnist1d_coral_align, embs_digits_align)
    procrustes_error = torch.norm(embs_digits_align - embs_mnist1d_coral_align @ Q.T, p='fro').item()
    print(f"Procrustes error (after CORAL): {procrustes_error:.4f}")

    # --- Test set embeddings ---
    with torch.no_grad():
        embs_mnist1d_test = model_mnist1d.get_embedding(torch.from_numpy(mnist1d_dataset["X_test"]).to(device))
        embs_digits_test  = model_digits.get_embedding(torch.from_numpy(digits_dataset["X_test"]).to(device))

    y_mnist1d_test = torch.from_numpy(mnist1d_dataset["y_test"]).to(device)
    y_digits_test  = torch.from_numpy(digits_dataset["y_test"]).to(device)

    # Apply CORAL transform (fitted on train) then Q.T rotation
    mnist1d_test_np = embs_mnist1d_test.detach().cpu().numpy()
    mnist1d_coral_test = torch.tensor(
        (mnist1d_test_np - mean_S) @ W_coral + mean_T, dtype=torch.float32).to(device)
    mnist1d_final_test = mnist1d_coral_test @ Q.T

    # --- Retrieval ---
    recall_s2i = recall_at_k(mnist1d_final_test, y_mnist1d_test, embs_digits_test, y_digits_test, k=5)
    recall_i2s = recall_at_k(embs_digits_test, y_digits_test, mnist1d_final_test, y_mnist1d_test, k=5)
    print(f"Recall@5  Sig→Img: {recall_s2i:.4f}  Img→Sig: {recall_i2s:.4f}")

    # --- CKA Cross-Modal (paired test set) ---
    paired_test = build_paired_test(digits_dataset, mnist1d_dataset, seed=42)
    with torch.no_grad():
        embs_digits_paired  = model_digits.get_embedding(
            torch.from_numpy(paired_test["X_digits"]).to(device))
        embs_mnist1d_paired = model_mnist1d.get_embedding(
            torch.from_numpy(paired_test["X_mnist1d"]).to(device))

    paired_np = embs_mnist1d_paired.detach().cpu().numpy()
    embs_mnist1d_paired_final = torch.tensor(
        (paired_np - mean_S) @ W_coral + mean_T, dtype=torch.float32).to(device) @ Q.T

    cka = compute_crossmodal_cka(embs_digits_paired, embs_mnist1d_paired_final)
    print(f"CKA Cross-Modal: {cka:.4f}")

    # --- Modality gap ---
    embs_mnist1d_align_final = embs_mnist1d_coral_align @ Q.T
    _, _, eigenvalues, mu_norm, cov_trace = compute_modality_gap(embs_digits_align, embs_mnist1d_align_final)
    print(f"Modality gap  |mu|: {mu_norm:.4f}  tr(Σ): {cov_trace:.4f}")

    # --- Procrustes distance on class centroids (paired test set) ---
    y_paired = torch.from_numpy(paired_test["y"]).to(device)
    centroids_digits  = torch.stack([embs_digits_paired[y_paired == c].mean(0) for c in range(10)])
    centroids_mnist1d = torch.stack([embs_mnist1d_paired_final[y_paired == c].mean(0) for c in range(10)])
    centroid_proc_dist = torch.norm(centroids_digits - centroids_mnist1d, p='fro').item()
    print(f"Procrustes centroid dist: {centroid_proc_dist:.4f}")

    results[seed] = {
        "recall_s2i": recall_s2i, "recall_i2s": recall_i2s,
        "cka": cka, "eigenvalues": eigenvalues,
        "centroid_proc_dist": centroid_proc_dist,
    }

    # --- t-SNE ---
    plot_tsne(embs_digits_test, mnist1d_final_test, y_digits_test, y_mnist1d_test,
              title=f"Unimodal embeddings after CORAL + Procrustes (Seed {seed})",
              save_path=f"figures/coral_procrustes/tsne_after_seed{seed}.png")

# ---------------------------------------------------------------------------
print("\n=== Final Results (mean ± std over seeds) ===")
s2i   = [results[s]["recall_s2i"] for s in seeds]
i2s   = [results[s]["recall_i2s"] for s in seeds]
ckas  = [results[s]["cka"] for s in seeds]
cdist = [results[s]["centroid_proc_dist"] for s in seeds]
print(f"Recall@5  Sig→Img : {np.mean(s2i):.4f} ± {np.std(s2i):.4f}")
print(f"Recall@5  Img→Sig : {np.mean(i2s):.4f} ± {np.std(i2s):.4f}")

print("\n=== CKA Cross-Modal (geometria img vs sig) ===")
print(f"CKA Cross-Modal   : {np.mean(ckas):.4f} ± {np.std(ckas):.4f}")

print("\n=== Procrustes Distance on Class Centroids ===")
print(f"Centroid dist     : {np.mean(cdist):.4f} ± {np.std(cdist):.4f}")

avg_eigenvalues = torch.stack([results[s]["eigenvalues"] for s in seeds]).mean(dim=0)
plot_eigenspectrum(
    {"CORAL + Procrustes (post-hoc)": avg_eigenvalues},
    title="Modality Gap Eigenspectrum - CORAL + Procrustes",
    save_path="figures/coral_procrustes/eigenspectrum.png",
)
