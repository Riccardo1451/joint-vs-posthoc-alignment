from data.dataset import load_all_datasets
from data.dataloader import build_paired_dataset
from models.clip_model import CLIPModel
from utils.metrics import evaluate_cka, compute_modality_gap
from utils.visualization import plot_tsne, plot_eigenspectrum
from itertools import combinations
import torch
import numpy as np
import os
os.makedirs("figures", exist_ok=True)

seeds = [42, 123, 999]
force_reload = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load datasets with fixed seed to ensure same test set for all evaluations
digits_data, mnist1d_data = load_all_datasets(seed=seeds[0], force_reload=force_reload)

hiddend_dim = 64
projection_dim = 32

# Initialize models
model1 = CLIPModel(mode = "cnn", hiddend_dim=hiddend_dim, projection_dim=projection_dim).to(device)
model2 = CLIPModel(hiddend_dim=128, projection_dim=32).to(device)

#-------------------- CKA Evaluation ------------------
cka_scores = []

for s1 in seeds:
    model1.load_state_dict(torch.load(f"checkpoints/clip_cnn_seed{s1}_hd{hiddend_dim}_pd{projection_dim}.pth", weights_only=True))
    model2.load_state_dict(torch.load(f"checkpoints/clip_seed{s1}.pth", weights_only=True))
    cka = evaluate_cka(model1, model2, digits_data=digits_data, mnist1d_data=mnist1d_data, device=device)
    cka_scores.append(cka)
    print(f"CKA between models trained with seeds {s1}-hd{hiddend_dim}-pd{projection_dim} and {s1}-hd128-pd32: {cka:.4f}")

print(f"Average CKA across all pairs: {np.mean(cka_scores):.4f} ± {np.std(cka_scores):.4f}")

# #-------------------- t-SNE Visualization ------------------
# model = CLIPModel(hiddend_dim=hiddend_dim, projection_dim=projection_dim).to(device)
# model.load_state_dict(torch.load(f"checkpoints/clip_seed{seeds[0]}_hd{hiddend_dim}_pd{projection_dim}.pth", weights_only=True))
# model.eval()

# n = min(len(digits_data["X_test"]), len(mnist1d_data["X_test"]))

# with torch.no_grad():
#     z_sig, z_img = model(
#         torch.from_numpy(mnist1d_data["X_test"][:n]).to(device),
#         torch.from_numpy(digits_data["X_test"][:n]).to(device)
#     )

# plot_tsne(z_img, z_sig,
#           torch.from_numpy(digits_data["y_test"][:n]).to(device),
#           torch.from_numpy(mnist1d_data["y_test"][:n]).to(device),
#           title=f"t-SNE of CLIP Embeddings (Seed {seeds[0]})",
#           save_path=f"figures/tsne_clip_seed{seeds[0]}_hd{hiddend_dim}_pd{projection_dim}.png")

# #-------------------- Modality Gap Evaluation ------------------
# clip_eigenvalues_list = []

# for seed in seeds:
#     model = CLIPModel(hiddend_dim=hiddend_dim, projection_dim=projection_dim).to(device)
#     model.load_state_dict(torch.load(f"checkpoints/clip_seed{seed}_hd{hiddend_dim}_pd{projection_dim}.pth", weights_only=True))
#     model.eval()

#     # Build paired dataset for coupled residuals
#     align_set = build_paired_dataset(digits_data, mnist1d_data, seed=seed)

#     with torch.no_grad():
#         z_sig, z_img = model(
#             torch.from_numpy(align_set["X_mnist1d"]).to(device),
#             torch.from_numpy(align_set["X_digits"]).to(device)
#         )

#     residual_mean, residual_cov, eigenvalues, mu_norm, cov_trace = compute_modality_gap(z_img, z_sig)
#     print(f"Seed {seed} - |mu_e|: {mu_norm:.4f}, tr(Sigma_e): {cov_trace:.4f}")
#     clip_eigenvalues_list.append(eigenvalues)

# # Average eigenvalues across seeds
# avg_eigenvalues_clip = torch.stack(clip_eigenvalues_list).mean(dim=0)

# plot_eigenspectrum(
#     {"CLIP": avg_eigenvalues_clip,
#      "Procrustes (paired)": torch.load("checkpoints/eigenvalues_procrustes.pth")},
#     title="Modality Gap Eigenspectrum - CLIP",
#     save_path=f"figures/eigenspectrum_clip_hd{hiddend_dim}_pd{projection_dim}.png"
# )
