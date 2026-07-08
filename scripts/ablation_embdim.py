"""
Ablation study: effect of embedding dimension on Procrustes and CLIP InfoNCE alignment.

Tests projection_dim in [32, 16, 5] with hidden_dim=64 and 10 classes.
For each (projection_dim, seed) combination:
  - Trains unimodal CNN classifiers with the given embedding_dim
  - Fits Procrustes alignment on the filtered embeddings
  - Trains CLIP InfoNCE with the given projection_dim
  - Evaluates Recall@5 (both directions), cross-modal CKA (paired test, seed=42),
    and modality gap (mu_norm, cov_trace)

Checkpoints saved under checkpoints/ablation_embdim/dim{D}/.

Run from the project root:
    python scripts/ablation_embdim.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import tqdm
from torch.utils.data import DataLoader, TensorDataset

from data.dataset import load_all_datasets
from data.dataloader import build_paired_dataset, build_paired_test, sample_batch
from models.unimodal import UnimodalModelDigitsCNN, UnimodalModelMnist1DCNN
from models.clip_model import CLIPModel
from methods.losses import info_nce_loss
from methods.procrustes import procrustes_align
from utils.metrics import recall_at_k, compute_crossmodal_cka, compute_modality_gap

# ---------------------------------------------------------------------------
proj_dim_list        = [32, 16, 5]
seeds                = [42, 123, 999]
hidden_dim           = 64
uni_epochs           = 300
uni_batch_size       = 64
uni_lr               = 1e-4
clip_epochs          = 200
clip_steps_per_epoch = 50
clip_K               = 10
clip_temperature     = 0.1
force_reload         = False
# ---------------------------------------------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

digits_raw, mnist1d_raw = load_all_datasets(seed=seeds[0], force_reload=force_reload)

paired_test = build_paired_test(digits_raw, mnist1d_raw, seed=42)

all_results = {}

for proj_dim in proj_dim_list:
    ckpt_dir = f"checkpoints/ablation_embdim/dim{proj_dim}"
    os.makedirs(ckpt_dir, exist_ok=True)

    for seed in seeds:
        print(f"\n{'='*70}")
        print(f"  proj_dim={proj_dim}  seed={seed}")
        print(f"{'='*70}")
        torch.manual_seed(seed)
        np.random.seed(seed)

        # ----------------------------------------------------------------
        # 1. Unimodal CNN training
        # ----------------------------------------------------------------
        model_digits  = UnimodalModelDigitsCNN(embedding_dim=proj_dim, num_classes=10).to(device)
        model_mnist1d = UnimodalModelMnist1DCNN(embedding_dim=proj_dim, num_classes=10).to(device)
        criterion = nn.CrossEntropyLoss()

        for tag, model, data in [
            ("digits",  model_digits,  digits_raw),
            ("mnist1d", model_mnist1d, mnist1d_raw),
        ]:
            opt    = optim.Adam(model.parameters(), lr=uni_lr)
            X_tr   = torch.from_numpy(data["X_train"]).to(device)
            y_tr   = torch.from_numpy(data["y_train"]).to(device)
            X_te   = torch.from_numpy(data["X_test"]).to(device)
            y_te   = torch.from_numpy(data["y_test"]).to(device)
            loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=uni_batch_size, shuffle=True, drop_last=True)

            pbar = tqdm.tqdm(range(uni_epochs), desc=f"Unimodal {tag} dim={proj_dim} s={seed}")
            for epoch in pbar:
                model.train()
                ep_loss = 0.0
                for X, y in loader:
                    opt.zero_grad()
                    loss = criterion(model(X), y)
                    loss.backward()
                    opt.step()
                    ep_loss += loss.item()
                pbar.set_postfix(loss=f"{ep_loss / len(loader):.4f}")
                if (epoch + 1) % 100 == 0:
                    model.eval()
                    with torch.no_grad():
                        acc = (model(X_te).argmax(1) == y_te).float().mean().item()
                    pbar.write(f"Epoch {epoch+1}  Test Acc: {acc:.4f}")

            torch.save(model.state_dict(), f"{ckpt_dir}/cnn_{tag}_seed{seed}_dim{proj_dim}.pth")

        model_digits.eval()
        model_mnist1d.eval()

        # ----------------------------------------------------------------
        # 2. Procrustes alignment
        # ----------------------------------------------------------------
        align_set = build_paired_dataset(digits_raw, mnist1d_raw, seed=seed)

        with torch.no_grad():
            embs_d_align = model_digits.get_embedding(torch.from_numpy(align_set["X_digits"]).to(device))
            embs_m_align = model_mnist1d.get_embedding(torch.from_numpy(align_set["X_mnist1d"]).to(device))

        Q = procrustes_align(embs_m_align, embs_d_align)

        y_d_test = torch.from_numpy(digits_raw["y_test"]).to(device)
        y_m_test = torch.from_numpy(mnist1d_raw["y_test"]).to(device)

        with torch.no_grad():
            embs_d_test = model_digits.get_embedding(torch.from_numpy(digits_raw["X_test"]).to(device))
            embs_m_test = model_mnist1d.get_embedding(torch.from_numpy(mnist1d_raw["X_test"]).to(device))

        m_aligned  = embs_m_test @ Q.T
        proc_r_s2i = recall_at_k(m_aligned,   y_m_test, embs_d_test, y_d_test, k=5)
        proc_r_i2s = recall_at_k(embs_d_test, y_d_test, m_aligned,   y_m_test, k=5)

        with torch.no_grad():
            embs_d_paired = model_digits.get_embedding(
                torch.from_numpy(paired_test["X_digits"]).to(device))
            embs_m_paired = model_mnist1d.get_embedding(
                torch.from_numpy(paired_test["X_mnist1d"]).to(device))

        m_paired_aligned = embs_m_paired @ Q.T
        proc_cka = compute_crossmodal_cka(embs_d_paired, m_paired_aligned)
        _, _, _, proc_mu_norm, proc_cov_trace = compute_modality_gap(embs_d_paired, m_paired_aligned)

        print(f"[Procrustes]   R S→I: {proc_r_s2i:.4f}  R I→S: {proc_r_i2s:.4f}  "
              f"CKA: {proc_cka:.4f}  |mu|: {proc_mu_norm:.4f}  tr(Σ): {proc_cov_trace:.4f}")

        # ----------------------------------------------------------------
        # 3. CLIP InfoNCE training
        # ----------------------------------------------------------------
        clip_model = CLIPModel(mode="cnn", hidden_dim=hidden_dim, projection_dim=proj_dim).to(device)
        clip_opt   = optim.Adam(clip_model.parameters(), lr=1e-4)

        pbar = tqdm.tqdm(range(clip_epochs), desc=f"CLIP InfoNCE dim={proj_dim} s={seed}")
        for epoch in pbar:
            clip_model.train()
            ep_loss = 0.0
            for _ in range(clip_steps_per_epoch):
                clip_opt.zero_grad()
                bd, bm, _ = sample_batch(digits_raw, mnist1d_raw, K=clip_K)
                z_sig, z_img = clip_model(bm.to(device), bd.to(device))
                loss = info_nce_loss(z_img, z_sig, temperature=clip_temperature)
                loss.backward()
                clip_opt.step()
                ep_loss += loss.item()
            pbar.set_postfix(loss=f"{ep_loss / clip_steps_per_epoch:.4f}")

        torch.save(
            clip_model.state_dict(),
            f"{ckpt_dir}/clip_infonce_cnn_seed{seed}_hd{hidden_dim}_pd{proj_dim}.pth",
        )

        # ----------------------------------------------------------------
        # 4. CLIP evaluation
        # ----------------------------------------------------------------
        clip_model.eval()
        with torch.no_grad():
            z_sig_test, z_img_test = clip_model(
                torch.from_numpy(mnist1d_raw["X_test"]).to(device),
                torch.from_numpy(digits_raw["X_test"]).to(device),
            )
        clip_r_s2i = recall_at_k(z_sig_test, y_m_test, z_img_test, y_d_test, k=5)
        clip_r_i2s = recall_at_k(z_img_test, y_d_test, z_sig_test, y_m_test, k=5)

        with torch.no_grad():
            z_sig_p, z_img_p = clip_model(
                torch.from_numpy(paired_test["X_mnist1d"]).to(device),
                torch.from_numpy(paired_test["X_digits"]).to(device),
            )
        clip_cka = compute_crossmodal_cka(z_img_p, z_sig_p)
        _, _, _, clip_mu_norm, clip_cov_trace = compute_modality_gap(z_img_p, z_sig_p)

        print(f"[CLIP InfoNCE] R S→I: {clip_r_s2i:.4f}  R I→S: {clip_r_i2s:.4f}  "
              f"CKA: {clip_cka:.4f}  |mu|: {clip_mu_norm:.4f}  tr(Σ): {clip_cov_trace:.4f}")

        all_results[(proj_dim, seed)] = {
            "proc_r_s2i": proc_r_s2i, "proc_r_i2s": proc_r_i2s, "proc_cka": proc_cka,
            "proc_mu": proc_mu_norm.item(), "proc_cov": proc_cov_trace,
            "clip_r_s2i": clip_r_s2i, "clip_r_i2s": clip_r_i2s, "clip_cka": clip_cka,
            "clip_mu": clip_mu_norm.item(), "clip_cov": clip_cov_trace,
        }

# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------
W = 130
print(f"\n{'='*W}")
print(f"{'':>14}  {'Procrustes':^57}  {'CLIP InfoNCE':^57}")
print(f"{'dim / seed':>14}  "
      f"{'R S→I':>9}  {'R I→S':>9}  {'CKA':>9}  {'|mu|':>9}  {'tr(Σ)':>9}  "
      f"{'R S→I':>9}  {'R I→S':>9}  {'CKA':>9}  {'|mu|':>9}  {'tr(Σ)':>9}")
print(f"{'-'*W}")


def _mv(key, proj_dim):
    vals = [all_results[(proj_dim, s)][key] for s in seeds]
    return f"{np.mean(vals):.4f}±{np.std(vals):.4f}"


for proj_dim in proj_dim_list:
    for seed in seeds:
        r   = all_results[(proj_dim, seed)]
        tag = f"d={proj_dim} s={seed}"
        print(f"{tag:>14}  "
              f"{r['proc_r_s2i']:>9.4f}  {r['proc_r_i2s']:>9.4f}  {r['proc_cka']:>9.4f}  "
              f"{r['proc_mu']:>9.4f}  {r['proc_cov']:>9.4f}  "
              f"{r['clip_r_s2i']:>9.4f}  {r['clip_r_i2s']:>9.4f}  {r['clip_cka']:>9.4f}  "
              f"{r['clip_mu']:>9.4f}  {r['clip_cov']:>9.4f}")
    print(f"{'mean±std':>14}  "
          f"{_mv('proc_r_s2i', proj_dim):>14}  {_mv('proc_r_i2s', proj_dim):>14}  "
          f"{_mv('proc_cka',   proj_dim):>14}  {_mv('proc_mu',    proj_dim):>14}  "
          f"{_mv('proc_cov',   proj_dim):>14}  "
          f"{_mv('clip_r_s2i', proj_dim):>14}  {_mv('clip_r_i2s', proj_dim):>14}  "
          f"{_mv('clip_cka',   proj_dim):>14}  {_mv('clip_mu',    proj_dim):>14}  "
          f"{_mv('clip_cov',   proj_dim):>14}")
    print(f"{'-'*W}")
