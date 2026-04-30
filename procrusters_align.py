import torch
from models.unimodal import UnimodalModelMnist1D, UnimodalModelDigits
from data.dataset import load_all_datasets
from data.dataloader import build_paired_dataset
from methods.procruster import procruster_align
from utils.metrics import recall_at_k, evaluate_cka, compute_modality_gap
from utils.visualization import plot_tsne, plot_eigenspectrum
import numpy as np
import os
os.makedirs("figures", exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

force_reload = False
seeds = [42, 123, 999]
results = {}

Modelmnist1D = UnimodalModelMnist1D().to(device)
ModelDigits = UnimodalModelDigits().to(device)

for seed in seeds:
    print(f"\nRunning experiment with seed {seed}...")

    # Load datasets
    digits_dataset, mnist1d_dataset = load_all_datasets(force_reload=force_reload, seed=seed)

    # Build paired dataset from train set
    align_set = build_paired_dataset(digits_dataset, mnist1d_dataset, seed=seed)

    # Load unimodal models
    Modelmnist1D.load_state_dict(torch.load(f"checkpoints/mnist1d_unimodal_seed{seed}.pth", weights_only=True))
    ModelDigits.load_state_dict(torch.load(f"checkpoints/digits_unimodal_seed{seed}.pth", weights_only=True))
    Modelmnist1D.eval()
    ModelDigits.eval()

    # Extract embeddings from alignment set
    embs_mnist1d_align = Modelmnist1D.get_embedding(torch.from_numpy(align_set["X_mnist1d"]).to(device))
    embs_digits_align  = ModelDigits.get_embedding(torch.from_numpy(align_set["X_digits"]).to(device))

    # Compute Q matrix using paired embeddings 1-to-1
    Q = procruster_align(embs_mnist1d_align, embs_digits_align)

    # Extract embeddings from test set
    embs_mnist1d_test = Modelmnist1D.get_embedding(torch.from_numpy(mnist1d_dataset["X_test"]).to(device))
    embs_digits_test  = ModelDigits.get_embedding(torch.from_numpy(digits_dataset["X_test"]).to(device))
    y_mnist1d_test = torch.from_numpy(mnist1d_dataset["y_test"]).to(device)
    y_digits_test  = torch.from_numpy(digits_dataset["y_test"]).to(device)

    # Align MNIST-1D embeddings using Q matrix
    mnist1d_aligned = embs_mnist1d_test @ Q.T

    # Compute retrieval metrics on test set
    recall_sig2img = recall_at_k(mnist1d_aligned, y_mnist1d_test, embs_digits_test, y_digits_test, k=5)
    recall_img2sig = recall_at_k(embs_digits_test, y_digits_test, mnist1d_aligned, y_mnist1d_test, k=5)
    print(f"Recall@5 Sig→Img: {recall_sig2img:.4f}")
    print(f"Recall@5 Img→Sig: {recall_img2sig:.4f}")

    # Compute CKA on test set
    n = min(len(embs_digits_test), len(mnist1d_aligned))
    cka = evaluate_cka(None, None, None, None, device,
                       emb1=mnist1d_aligned[:n],
                       emb2=embs_digits_test[:n])
    print(f"CKA: {cka:.4f}")

    # Save results
    results[seed] = {
        "recall_sig2img": recall_sig2img,
        "recall_img2sig": recall_img2sig,
        "cka": cka
    }

    # t-SNE before alignment
    plot_tsne(embs_digits_test, embs_mnist1d_test,
              y_digits_test, y_mnist1d_test,
              title=f"Unimodal embeddings before Procrustes (Seed {seed})",
              save_path=f"figures/tsne_paired_before_seed{seed}.png")

    # t-SNE after alignment
    plot_tsne(embs_digits_test, mnist1d_aligned,
              y_digits_test, y_mnist1d_test,
              title=f"Unimodal embeddings after Procrustes (Seed {seed})",
              save_path=f"figures/tsne_paired_after_seed{seed}.png")
    
    # Compute modality gap
    # Align paired embeddings using Q matrix
    embs_mnist1d_align_rotated = embs_mnist1d_align @ Q.T
    residual_mean, residual_cov, eigenvalues, mu_norm, cov_trace = compute_modality_gap(embs_digits_align, embs_mnist1d_align_rotated)
    print(f"Modality Gap - |mu_e|: {mu_norm:.4f}")
    print(f"Modality Gap - tr(Sigma_e): {cov_trace:.4f}")

    # Save eigenvalues for plotting
    results[seed]["eigenvalues"] = eigenvalues

# Compute mean and std across seeds
recalls_sig2img = [results[seed]["recall_sig2img"] for seed in seeds]
recalls_img2sig = [results[seed]["recall_img2sig"] for seed in seeds]
ckas = [results[seed]["cka"] for seed in seeds]

print(f"\nFinal Results:")
print(f"Recall Sig→Img: {np.mean(recalls_sig2img):.4f} ± {np.std(recalls_sig2img):.4f}")
print(f"Recall Img→Sig: {np.mean(recalls_img2sig):.4f} ± {np.std(recalls_img2sig):.4f}")
print(f"CKA: {np.mean(ckas):.4f} ± {np.std(ckas):.4f}")

# Average eigenvalues across seeds
avg_eigenvalues = torch.stack([results[seed]["eigenvalues"] for seed in seeds]).mean(dim=0)
torch.save(avg_eigenvalues, "checkpoints/eigenvalues_procrustes.pth")

plot_eigenspectrum(
    {"Procrustes (paired)": avg_eigenvalues},
    title="Modality Gap Eigenspectrum - Procrustes",
    save_path="figures/eigenspectrum_procrustes.png"
)