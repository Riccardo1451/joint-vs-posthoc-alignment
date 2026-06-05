"""
Evaluate label flip robustness for post-hoc Procrustes (and optionally CORAL) alignment.

Loads unimodal checkpoints trained at each flip rate and reports retrieval,
CKA, Procrustes error, and modality-gap statistics.  The test set is always
the clean (unflipped) split.

Set use_coral=True to apply CORAL on top of Procrustes.

Requires:
  - checkpoints/unimodal/{mlp,cnn}_{digits,mnist1d}_seed{S}.pth               (flip_rate=0.0)
  - checkpoints/unimodal/{mlp,cnn}_{digits,mnist1d}_seed{S}_flipr{R}.pth      (flip_rate>0)

Run from the project root:
    python scripts/evaluate_procrustes_labelflip.py
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

# ---------------------------------------------------------------------------
seeds        = [42, 123, 999]
flip_rates   = [0.0, 0.05, 0.1, 0.2, 0.3]
use_coral    = False
force_reload = False
mode         = "cnn"  # "mlp" or "cnn"
# ---------------------------------------------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
print(f"Method: Procrustes{' + CORAL' if use_coral else ''}")


def _ckpt_path(modality, seed, flip_rate):
    if flip_rate == 0.0:
        return f"checkpoints/unimodal/{mode}_{modality}_seed{seed}.pth"
    return f"checkpoints/unimodal/{mode}_{modality}_seed{seed}_flipr{flip_rate}.pth"


digits_dataset, mnist1d_dataset = load_all_datasets(force_reload=force_reload, seed=seeds[0])
paired_test = build_paired_test(digits_dataset, mnist1d_dataset, seed=42)

model_mnist1d = UnimodalModelMnist1DCNN().to(device)
model_digits  = UnimodalModelDigitsCNN().to(device)

# ---------------------------------------------------------------------------
print(f"\n{'flip_rate':>10}  {'Recall S→I':>22}  {'Recall I→S':>22}  "
      f"{'CKA Cross-Modal':>22}  {'Proc_error':>18}  {'|mu_e|':>14}  {'tr(Sigma_e)':>17}")
print("-" * 135)

for flip_rate in flip_rates:
    recalls_s2i, recalls_i2s, ckas = [], [], []
    proc_errors, mu_norms, cov_traces = [], [], []

    for seed in seeds:
        align_set = build_paired_dataset(digits_dataset, mnist1d_dataset, seed=seed)

        model_mnist1d.load_state_dict(
            torch.load(_ckpt_path("mnist1d", seed, flip_rate), weights_only=True))
        model_digits.load_state_dict(
            torch.load(_ckpt_path("digits", seed, flip_rate), weights_only=True))
        model_mnist1d.eval()
        model_digits.eval()

        # Alignment set embeddings (used for Procrustes fit and modality-gap stats)
        with torch.no_grad():
            embs_mnist1d_align = model_mnist1d.get_embedding(
                torch.from_numpy(align_set["X_mnist1d"]).to(device))
            embs_digits_align  = model_digits.get_embedding(
                torch.from_numpy(align_set["X_digits"]).to(device))

        Q = procrustes_align(embs_mnist1d_align, embs_digits_align)
        proc_error = torch.norm(
            embs_digits_align - embs_mnist1d_align @ Q.T, p='fro').item()

        # Clean test set embeddings
        with torch.no_grad():
            embs_mnist1d_test = model_mnist1d.get_embedding(
                torch.from_numpy(mnist1d_dataset["X_test"]).to(device))
            embs_digits_test  = model_digits.get_embedding(
                torch.from_numpy(digits_dataset["X_test"]).to(device))

        y_mnist1d_test = torch.from_numpy(mnist1d_dataset["y_test"]).to(device)
        y_digits_test  = torch.from_numpy(digits_dataset["y_test"]).to(device)

        mnist1d_aligned = embs_mnist1d_test @ Q.T
        if use_coral:
            mnist1d_aligned = coral_align(mnist1d_aligned, embs_digits_test, device=device)

        recall_s2i = recall_at_k(
            mnist1d_aligned, y_mnist1d_test, embs_digits_test, y_digits_test, k=5)
        recall_i2s = recall_at_k(
            embs_digits_test, y_digits_test, mnist1d_aligned, y_mnist1d_test, k=5)

        with torch.no_grad():
            embs_mnist1d_paired = model_mnist1d.get_embedding(
                torch.from_numpy(paired_test["X_mnist1d"]).to(device))
            embs_digits_paired  = model_digits.get_embedding(
                torch.from_numpy(paired_test["X_digits"]).to(device))
        mnist1d_paired_aligned = embs_mnist1d_paired @ Q.T
        if use_coral:
            # Derive CORAL transform from full test set statistics and apply to paired subset
            src_np = (embs_mnist1d_test @ Q.T).detach().cpu().numpy()
            tgt_np = embs_digits_test.detach().cpu().numpy()
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
            paired_proc_np = mnist1d_paired_aligned.detach().cpu().numpy()
            mnist1d_paired_aligned = torch.tensor(
                (paired_proc_np - mean_S) @ W_coral + mean_T, dtype=torch.float32).to(device)
        cka = compute_crossmodal_cka(embs_digits_paired, mnist1d_paired_aligned)

        # Modality gap on alignment set after rotation (and optional CORAL)
        embs_mnist1d_align_rot = embs_mnist1d_align @ Q.T
        if use_coral:
            embs_mnist1d_align_rot = coral_align(
                embs_mnist1d_align_rot, embs_digits_align, device=device)
        _, _, _, mu_norm, cov_trace = compute_modality_gap(
            embs_digits_align, embs_mnist1d_align_rot)

        recalls_s2i.append(recall_s2i)
        recalls_i2s.append(recall_i2s)
        ckas.append(cka)
        proc_errors.append(proc_error)
        mu_norms.append(mu_norm.item())
        cov_traces.append(cov_trace)

    print(
        f"{flip_rate:>10.2f}  "
        f"{np.mean(recalls_s2i):.4f} ± {np.std(recalls_s2i):.4f}  "
        f"{np.mean(recalls_i2s):.4f} ± {np.std(recalls_i2s):.4f}  "
        f"{np.mean(ckas):.4f} ± {np.std(ckas):.4f}  "
        f"{np.mean(proc_errors):.4f} ± {np.std(proc_errors):.4f}  "
        f"{np.mean(mu_norms):.4f} ± {np.std(mu_norms):.4f}  "
        f"{np.mean(cov_traces):.4f} ± {np.std(cov_traces):.4f}"
    )
