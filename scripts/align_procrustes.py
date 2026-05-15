"""
Post-hoc alignment evaluation using Procrustes analysis.

Loads pre-trained unimodal models, computes the Procrustes rotation Q on the
training set, evaluates cross-modal retrieval and CKA on the test set, and
plots t-SNE embeddings before/after alignment.

Run from the project root:
    python scripts/align_procrustes.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np

from models.unimodal import UnimodalModelMnist1D, UnimodalModelDigits
from data.dataset import load_all_datasets
from data.dataloader import build_paired_dataset
from methods.procrustes import procrustes_align
from utils.metrics import recall_at_k, evaluate_cka, compute_modality_gap
from utils.visualization import plot_tsne, plot_eigenspectrum
from methods.coral import coral_align

os.makedirs("figures", exist_ok=True)

# ---------------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

force_reload = False
seeds = [42, 123, 999]
results = {}

model_mnist1d = UnimodalModelMnist1D().to(device)
model_digits  = UnimodalModelDigits().to(device)

for seed in seeds:
    print(f"\n--- Seed {seed} ---")

    digits_dataset, mnist1d_dataset = load_all_datasets(force_reload=force_reload, seed=seed)
    align_set = build_paired_dataset(digits_dataset, mnist1d_dataset, seed=seed)

    model_mnist1d.load_state_dict(torch.load(f"checkpoints/mnist1d_unimodal_seed{seed}.pth", weights_only=True))
    model_digits.load_state_dict(torch.load(f"checkpoints/digits_unimodal_seed{seed}.pth", weights_only=True))
    model_mnist1d.eval()
    model_digits.eval()

    # --- Alignment set embeddings ---
    embs_mnist1d_align = model_mnist1d.get_embedding(torch.from_numpy(align_set["X_mnist1d"]).to(device))
    embs_digits_align  = model_digits.get_embedding(torch.from_numpy(align_set["X_digits"]).to(device))

    Q = procrustes_align(embs_mnist1d_align, embs_digits_align)

    # --- Test set embeddings ---
    embs_mnist1d_test = model_mnist1d.get_embedding(torch.from_numpy(mnist1d_dataset["X_test"]).to(device))
    embs_digits_test  = model_digits.get_embedding(torch.from_numpy(digits_dataset["X_test"]).to(device))
    y_mnist1d_test    = torch.from_numpy(mnist1d_dataset["y_test"]).to(device)
    y_digits_test     = torch.from_numpy(digits_dataset["y_test"]).to(device)

    mnist1d_aligned = embs_mnist1d_test @ Q.T

    # --- CORAL ---
    mnist1d_coral = coral_align(mnist1d_aligned, embs_digits_test, device=device)

    # --- Retrieval ---
    recall_s2i_procrustes = recall_at_k(mnist1d_aligned, y_mnist1d_test, embs_digits_test, y_digits_test, k=5)
    recall_i2s_procrustes = recall_at_k(embs_digits_test, y_digits_test, mnist1d_aligned, y_mnist1d_test, k=5)

    recall_s2i_coral = recall_at_k(mnist1d_coral, y_mnist1d_test, embs_digits_test, y_digits_test, k=5)
    recall_i2s_coral = recall_at_k(embs_digits_test, y_digits_test, mnist1d_coral, y_mnist1d_test, k=5)

    print(f"Recall@5 Procrustes Sig→Img: {recall_s2i_procrustes:.4f}  Img→Sig: {recall_i2s_procrustes:.4f}")
    print(f"Recall@5 CORAL Sig→Img: {recall_s2i_coral:.4f}  Img→Sig: {recall_i2s_coral:.4f}")

    # --- CKA ---
    n = min(len(embs_digits_test), len(mnist1d_aligned))
    cka_procrustes = evaluate_cka(None, None, None, None, device,
                       emb1=mnist1d_aligned[:n], emb2=embs_digits_test[:n])
    cka_coral = evaluate_cka(None, None, None, None, device,
                       emb1=mnist1d_coral[:n], emb2=embs_digits_test[:n])
    print(f"CKA Procrustes: {cka_procrustes:.4f}")
    print(f"CKA CORAL: {cka_coral:.4f}")

    results[seed] = {"recall_s2i": recall_s2i_procrustes, "recall_i2s": recall_i2s_procrustes, "cka_procrustes": cka_procrustes, 
                            "cka_coral": cka_coral, "recall_s2i_coral": recall_s2i_coral, "recall_i2s_coral": recall_i2s_coral}

    # --- t-SNE ---
    plot_tsne(embs_digits_test, embs_mnist1d_test, y_digits_test, y_mnist1d_test,
              title=f"Unimodal embeddings before Procrustes (Seed {seed})",
              save_path=f"figures/tsne_procrustes_before_seed{seed}.png")

    plot_tsne(embs_digits_test, mnist1d_aligned, y_digits_test, y_mnist1d_test,
              title=f"Unimodal embeddings after Procrustes (Seed {seed})",
              save_path=f"figures/tsne_procrustes_after_seed{seed}.png")

    plot_tsne(embs_digits_test, mnist1d_coral, y_digits_test, y_mnist1d_test,
              title=f"Unimodal embeddings after CORAL (Seed {seed})",
              save_path=f"figures/tsne_coral_after_seed{seed}.png")

    # --- Modality gap ---
    embs_mnist1d_align_rotated = embs_mnist1d_align @ Q.T
    _, _, eigenvalues, mu_norm, cov_trace = compute_modality_gap(embs_digits_align, embs_mnist1d_align_rotated)
    print(f"Modality gap  |mu|: {mu_norm:.4f}  tr(Σ): {cov_trace:.4f}")
    results[seed]["eigenvalues"] = eigenvalues

    mnist1d_coral_align = embs_mnist1d_align @ Q.T
    mnist1d_coral_align_corrected = coral_align(mnist1d_coral_align, embs_digits_align, device=device)
    _, _, eigenvalues_coral, mu_norm_coral, cov_trace_coral = compute_modality_gap(embs_digits_align, mnist1d_coral_align_corrected)
    print(f"Modality gap CORAL  |mu|: {mu_norm_coral:.4f}  tr(Σ): {cov_trace_coral:.4f}")

# ---------------------------------------------------------------------------
# Aggregate results
# ---------------------------------------------------------------------------
recalls_s2i_procrustes = [results[s]["recall_s2i"] for s in seeds]
recalls_i2s_procrustes = [results[s]["recall_i2s"] for s in seeds]
ckas_procrustes        = [results[s]["cka_procrustes"] for s in seeds]
recalls_s2i_coral      = [results[s]["recall_s2i_coral"] for s in seeds]
recalls_i2s_coral      = [results[s]["recall_i2s_coral"] for s in seeds]
ckas_coral             = [results[s]["cka_coral"]  for s in seeds]

print(f"\nFinal Results (mean ± std over seeds):")
print(f"  Recall Procrustes Sig→Img : {np.mean(recalls_s2i_procrustes):.4f} ± {np.std(recalls_s2i_procrustes):.4f}")
print(f"  Recall Procrustes Img→Sig : {np.mean(recalls_i2s_procrustes):.4f} ± {np.std(recalls_i2s_procrustes):.4f}")
print(f"  CKA Procrustes : {np.mean(ckas_procrustes):.4f} ± {np.std(ckas_procrustes):.4f}")
print(f"  Recall CORAL Sig→Img : {np.mean(recalls_s2i_coral):.4f} ± {np.std(recalls_s2i_coral):.4f}")
print(f"  Recall CORAL Img→Sig : {np.mean(recalls_i2s_coral):.4f} ± {np.std(recalls_i2s_coral):.4f}")
print(f"  CKA CORAL      : {np.mean(ckas_coral):.4f} ± {np.std(ckas_coral):.4f}")

avg_eigenvalues = torch.stack([results[s]["eigenvalues"] for s in seeds]).mean(dim=0)
torch.save(avg_eigenvalues, "checkpoints/eigenvalues_procrustes.pth")

plot_eigenspectrum(
    {"Procrustes (paired)": avg_eigenvalues},
    title="Modality Gap Eigenspectrum - Procrustes",
    save_path="figures/eigenspectrum_procrustes.png",
)
