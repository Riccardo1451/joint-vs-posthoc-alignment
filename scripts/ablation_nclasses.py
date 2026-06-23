"""
Ablation study: effect of number of classes on Procrustes and CLIP InfoNCE alignment.

Tests n_classes in [10, 8, 6], keeping always classes 0..(n_classes-1).
For each (n_classes, seed) combination:
  - Trains unimodal CNN classifiers from scratch on the filtered dataset
  - Fits Procrustes alignment on the filtered embeddings
  - Trains CLIP InfoNCE from scratch on the filtered dataset
  - Evaluates Recall@5 (both directions) and cross-modal CKA (paired test, seed=42)

Checkpoints saved under checkpoints/ablation_nclasses/{n_classes}/.

Run from the project root:
    python scripts/ablation_nclasses.py
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
from models.unimodal import UnimodalModelDigitsCNN, UnimodalModelMnist1DCNN
from models.clip_model import CLIPModel
from methods.losses import info_nce_loss
from methods.procrustes import procrustes_align
from utils.metrics import recall_at_k, compute_crossmodal_cka

# ---------------------------------------------------------------------------
n_classes_list       = [10, 8, 6]
seeds                = [42, 123, 999]
uni_epochs           = 300
uni_batch_size       = 64
uni_lr               = 1e-4
clip_epochs          = 200
clip_steps_per_epoch = 50
clip_K               = 10
clip_temperature     = 0.1
hidden_dim           = 64
projection_dim       = 32
force_reload         = False
# ---------------------------------------------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def filter_dataset(data, n_classes):
    mask_tr = data["y_train"] < n_classes
    mask_te = data["y_test"]  < n_classes
    return {
        "X_train": data["X_train"][mask_tr],
        "y_train": data["y_train"][mask_tr],
        "X_test":  data["X_test"][mask_te],
        "y_test":  data["y_test"][mask_te],
    }


def build_paired_dataset_nc(digits_data, mnist1d_data, seed, n_classes):
    paired_d, paired_m, paired_y = [], [], []
    np.random.seed(seed)
    for cls in range(n_classes):
        idx_d = np.where(digits_data["y_train"] == cls)[0]
        idx_m = np.where(mnist1d_data["y_train"] == cls)[0]
        np.random.shuffle(idx_d)
        np.random.shuffle(idx_m)
        n = min(len(idx_d), len(idx_m))
        paired_d.append(digits_data["X_train"][idx_d[:n]])
        paired_m.append(mnist1d_data["X_train"][idx_m[:n]])
        paired_y.append(np.full(n, cls))
    return {
        "X_digits":  np.concatenate(paired_d),
        "X_mnist1d": np.concatenate(paired_m),
        "y":         np.concatenate(paired_y),
    }


def build_paired_test_nc(digits_data, mnist1d_data, seed, n_classes):
    paired_d, paired_m, paired_y = [], [], []
    rng = np.random.default_rng(seed)
    for cls in range(n_classes):
        idx_d = np.where(digits_data["y_test"] == cls)[0]
        idx_m = np.where(mnist1d_data["y_test"] == cls)[0]
        rng.shuffle(idx_d)
        rng.shuffle(idx_m)
        n = min(len(idx_d), len(idx_m))
        paired_d.append(digits_data["X_test"][idx_d[:n]])
        paired_m.append(mnist1d_data["X_test"][idx_m[:n]])
        paired_y.append(np.full(n, cls))
    return {
        "X_digits":  np.concatenate(paired_d),
        "X_mnist1d": np.concatenate(paired_m),
        "y":         np.concatenate(paired_y),
    }


def sample_batch_nc(digits_data, mnist1d_data, K, n_classes):
    batch_d, batch_m = [], []
    for cls in range(n_classes):
        idx_d = np.random.choice(np.where(digits_data["y_train"] == cls)[0], K, replace=False)
        idx_m = np.random.choice(np.where(mnist1d_data["y_train"] == cls)[0], K, replace=False)
        batch_d.append(digits_data["X_train"][idx_d])
        batch_m.append(mnist1d_data["X_train"][idx_m])
    return torch.from_numpy(np.concatenate(batch_d)), torch.from_numpy(np.concatenate(batch_m))


# ---------------------------------------------------------------------------
digits_raw, mnist1d_raw = load_all_datasets(seed=seeds[0], force_reload=force_reload)

all_results = {}

for n_classes in n_classes_list:
    ckpt_dir = f"checkpoints/ablation_nclasses/{n_classes}"
    os.makedirs(ckpt_dir, exist_ok=True)

    digits_data  = filter_dataset(digits_raw,  n_classes)
    mnist1d_data = filter_dataset(mnist1d_raw, n_classes)

    for seed in seeds:
        print(f"\n{'='*70}")
        print(f"  n_classes={n_classes}  seed={seed}")
        print(f"{'='*70}")
        torch.manual_seed(seed)
        np.random.seed(seed)

        # ----------------------------------------------------------------
        # 1. Unimodal CNN training
        # ----------------------------------------------------------------
        model_digits  = UnimodalModelDigitsCNN(num_classes=n_classes).to(device)
        model_mnist1d = UnimodalModelMnist1DCNN(num_classes=n_classes).to(device)
        criterion = nn.CrossEntropyLoss()

        for tag, model, data in [
            ("digits",  model_digits,  digits_data),
            ("mnist1d", model_mnist1d, mnist1d_data),
        ]:
            opt   = optim.Adam(model.parameters(), lr=uni_lr)
            X_tr  = torch.from_numpy(data["X_train"]).to(device)
            y_tr  = torch.from_numpy(data["y_train"]).to(device)
            X_te  = torch.from_numpy(data["X_test"]).to(device)
            y_te  = torch.from_numpy(data["y_test"]).to(device)
            loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=uni_batch_size, shuffle=True)

            pbar = tqdm.tqdm(range(uni_epochs), desc=f"Unimodal {tag} nc={n_classes} s={seed}")
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

            torch.save(model.state_dict(), f"{ckpt_dir}/cnn_{tag}_seed{seed}.pth")

        model_digits.eval()
        model_mnist1d.eval()

        # ----------------------------------------------------------------
        # 2. Procrustes alignment
        # ----------------------------------------------------------------
        align_set = build_paired_dataset_nc(digits_data, mnist1d_data, seed=seed, n_classes=n_classes)

        with torch.no_grad():
            embs_d_align = model_digits.get_embedding(torch.from_numpy(align_set["X_digits"]).to(device))
            embs_m_align = model_mnist1d.get_embedding(torch.from_numpy(align_set["X_mnist1d"]).to(device))

        Q = procrustes_align(embs_m_align, embs_d_align)

        with torch.no_grad():
            embs_d_test = model_digits.get_embedding(torch.from_numpy(digits_data["X_test"]).to(device))
            embs_m_test = model_mnist1d.get_embedding(torch.from_numpy(mnist1d_data["X_test"]).to(device))

        y_d_test = torch.from_numpy(digits_data["y_test"]).to(device)
        y_m_test = torch.from_numpy(mnist1d_data["y_test"]).to(device)

        m_aligned   = embs_m_test @ Q.T
        proc_r_s2i  = recall_at_k(m_aligned,   y_m_test, embs_d_test, y_d_test, k=5)
        proc_r_i2s  = recall_at_k(embs_d_test, y_d_test, m_aligned,   y_m_test, k=5)

        paired_test = build_paired_test_nc(digits_data, mnist1d_data, seed=42, n_classes=n_classes)
        with torch.no_grad():
            embs_d_paired = model_digits.get_embedding(torch.from_numpy(paired_test["X_digits"]).to(device))
            embs_m_paired = model_mnist1d.get_embedding(torch.from_numpy(paired_test["X_mnist1d"]).to(device))
        proc_cka = compute_crossmodal_cka(embs_d_paired, embs_m_paired @ Q.T)

        print(f"[Procrustes]   Recall S→I: {proc_r_s2i:.4f}  I→S: {proc_r_i2s:.4f}  CKA: {proc_cka:.4f}")

        # ----------------------------------------------------------------
        # 3. CLIP InfoNCE training
        # ----------------------------------------------------------------
        clip_model = CLIPModel(mode="cnn", hidden_dim=hidden_dim, projection_dim=projection_dim).to(device)
        clip_opt   = optim.Adam(clip_model.parameters(), lr=1e-4)

        pbar = tqdm.tqdm(range(clip_epochs), desc=f"CLIP InfoNCE nc={n_classes} s={seed}")
        for epoch in pbar:
            clip_model.train()
            ep_loss = 0.0
            for _ in range(clip_steps_per_epoch):
                clip_opt.zero_grad()
                bd, bm = sample_batch_nc(digits_data, mnist1d_data, K=clip_K, n_classes=n_classes)
                z_sig, z_img = clip_model(bm.to(device), bd.to(device))
                loss = info_nce_loss(z_img, z_sig, temperature=clip_temperature)
                loss.backward()
                clip_opt.step()
                ep_loss += loss.item()
            pbar.set_postfix(loss=f"{ep_loss / clip_steps_per_epoch:.4f}")

        torch.save(
            clip_model.state_dict(),
            f"{ckpt_dir}/clip_infonce_cnn_seed{seed}_hd{hidden_dim}_pd{projection_dim}.pth",
        )

        # ----------------------------------------------------------------
        # 4. CLIP evaluation
        # ----------------------------------------------------------------
        clip_model.eval()
        with torch.no_grad():
            z_sig_test, z_img_test = clip_model(
                torch.from_numpy(mnist1d_data["X_test"]).to(device),
                torch.from_numpy(digits_data["X_test"]).to(device),
            )
        clip_r_s2i = recall_at_k(z_sig_test, y_m_test, z_img_test, y_d_test, k=5)
        clip_r_i2s = recall_at_k(z_img_test, y_d_test, z_sig_test, y_m_test, k=5)

        with torch.no_grad():
            z_sig_p, z_img_p = clip_model(
                torch.from_numpy(paired_test["X_mnist1d"]).to(device),
                torch.from_numpy(paired_test["X_digits"]).to(device),
            )
        clip_cka = compute_crossmodal_cka(z_img_p, z_sig_p)

        print(f"[CLIP InfoNCE] Recall S→I: {clip_r_s2i:.4f}  I→S: {clip_r_i2s:.4f}  CKA: {clip_cka:.4f}")

        all_results[(n_classes, seed)] = {
            "proc_r_s2i": proc_r_s2i, "proc_r_i2s": proc_r_i2s, "proc_cka": proc_cka,
            "clip_r_s2i": clip_r_s2i, "clip_r_i2s": clip_r_i2s, "clip_cka": clip_cka,
        }

# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------
W = 100
print(f"\n{'='*W}")
print(f"{'':>14}  {'Procrustes':^38}  {'CLIP InfoNCE':^38}")
print(f"{'n_cls / seed':>14}  {'Recall S→I':>11}  {'Recall I→S':>11}  {'CKA':>11}  "
      f"{'Recall S→I':>11}  {'Recall I→S':>11}  {'CKA':>11}")
print(f"{'-'*W}")

def _mv(key, n_classes):
    vals = [all_results[(n_classes, s)][key] for s in seeds]
    return f"{np.mean(vals):.4f}±{np.std(vals):.4f}"

for n_classes in n_classes_list:
    for seed in seeds:
        r   = all_results[(n_classes, seed)]
        tag = f"nc={n_classes} s={seed}"
        print(f"{tag:>14}  "
              f"{r['proc_r_s2i']:>11.4f}  {r['proc_r_i2s']:>11.4f}  {r['proc_cka']:>11.4f}  "
              f"{r['clip_r_s2i']:>11.4f}  {r['clip_r_i2s']:>11.4f}  {r['clip_cka']:>11.4f}")
    print(f"{'mean±std':>14}  "
          f"{_mv('proc_r_s2i', n_classes):>11}  {_mv('proc_r_i2s', n_classes):>11}  {_mv('proc_cka', n_classes):>11}  "
          f"{_mv('clip_r_s2i', n_classes):>11}  {_mv('clip_r_i2s', n_classes):>11}  {_mv('clip_cka', n_classes):>11}")
    print(f"{'-'*W}")
