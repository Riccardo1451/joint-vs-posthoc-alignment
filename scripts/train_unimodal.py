"""
Unimodal classifier training (pre-step for post-hoc Procrustes alignment).

Trains independent classifiers on MNIST-1D and Digits across multiple seeds.
Run from the project root:
    python scripts/train_unimodal.py --modality digits
    python scripts/train_unimodal.py --modality mnist1d
    python scripts/train_unimodal.py --modality both   (default)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import tqdm
from torch.utils.data import DataLoader, TensorDataset

from data.dataset import load_digits_dataset, load_mnist1d_dataset
from models.unimodal import UnimodalModelDigits, UnimodalModelMnist1D

os.makedirs("checkpoints", exist_ok=True)

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------
NUM_EPOCHS   = 300
BATCH_SIZE   = 64
LEARNING_RATE = 1e-4
SEEDS        = [42, 123, 999]
FORCE_RELOAD = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def train_digits():
    for seed in SEEDS:
        torch.manual_seed(seed)
        np.random.seed(seed)

        data = load_digits_dataset(force_reload=FORCE_RELOAD)
        X_train = torch.from_numpy(data["X_train"]).to(device)
        y_train = torch.from_numpy(data["y_train"]).to(device)
        X_test  = torch.from_numpy(data["X_test"]).to(device)
        y_test  = torch.from_numpy(data["y_test"]).to(device)

        dataloader = DataLoader(TensorDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
        model = UnimodalModelDigits().to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

        pbar = tqdm.tqdm(range(NUM_EPOCHS), desc=f"Digits  - Seed {seed}")
        for epoch in pbar:
            model.train()
            batch_loss = 0.0
            for X, y in dataloader:
                optimizer.zero_grad()
                loss = criterion(model(X), y)
                loss.backward()
                optimizer.step()
                batch_loss += loss.item()
            pbar.set_postfix(loss=f"{batch_loss / len(dataloader):.4f}")

            if (epoch + 1) % 20 == 0:
                model.eval()
                with torch.no_grad():
                    _, predicted = torch.max(model(X_test), 1)
                    acc = (predicted == y_test).float().mean().item()
                pbar.write(f"Epoch {epoch+1}/{NUM_EPOCHS}  Test Acc: {acc:.4f}")

        ckpt = f"checkpoints/digits_unimodal_seed{seed}.pth"
        torch.save(model.state_dict(), ckpt)
        print(f"Saved → {ckpt}")


def train_mnist1d():
    for seed in SEEDS:
        torch.manual_seed(seed)
        np.random.seed(seed)

        data = load_mnist1d_dataset(force_reload=FORCE_RELOAD, seed=seed)
        X_train = torch.from_numpy(data["X_train"]).to(device)
        y_train = torch.from_numpy(data["y_train"]).to(device)
        X_test  = torch.from_numpy(data["X_test"]).to(device)
        y_test  = torch.from_numpy(data["y_test"]).to(device)

        dataloader = DataLoader(TensorDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
        model = UnimodalModelMnist1D().to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

        pbar = tqdm.tqdm(range(NUM_EPOCHS), desc=f"MNIST-1D - Seed {seed}")
        for epoch in pbar:
            model.train()
            batch_loss = 0.0
            for X, y in dataloader:
                optimizer.zero_grad()
                loss = criterion(model(X), y)
                loss.backward()
                optimizer.step()
                batch_loss += loss.item()
            pbar.set_postfix(loss=f"{batch_loss / len(dataloader):.4f}")

            if (epoch + 1) % 20 == 0:
                model.eval()
                with torch.no_grad():
                    _, predicted = torch.max(model(X_test), 1)
                    acc = (predicted == y_test).float().mean().item()
                pbar.write(f"Epoch {epoch+1}/{NUM_EPOCHS}  Test Acc: {acc:.4f}")

        ckpt = f"checkpoints/mnist1d_unimodal_seed{seed}.pth"
        torch.save(model.state_dict(), ckpt)
        print(f"Saved → {ckpt}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--modality",
        choices=["digits", "mnist1d", "both"],
        default="both",
        help="Which modality to train (default: both)",
    )
    args = parser.parse_args()

    if args.modality in ("digits", "both"):
        train_digits()
    if args.modality in ("mnist1d", "both"):
        train_mnist1d()
