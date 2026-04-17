from train_clip import train_clip
import numpy as np
from utils.metrics import evaluate_cka
from models.clip_model import CLIPModel
import torch
from data.dataset import load_all_datasets

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print (f"Using device: {device}")
seeds = [42, 123, 999]
epochs = 200
steps_per_epoch = 50
batch_size = 100
temperature = 0.1
force_reload = False

results = []

for seed in seeds:
    print(f"Running experiment with seed {seed}...")
    recall_sig2img, recall_img2sig = train_clip(seed=seed, epochs=epochs, steps_per_epoch=steps_per_epoch, batch_size=batch_size, temperature=temperature, force_reload=force_reload)
    results.append((seed, recall_sig2img, recall_img2sig))


print("\nFinal Results:")
#-------------------- Recall@5 Evaluation ------------------
recalls_imgs = [r[1] for r in results]
recalls_sigs = [r[2] for r in results]

print(f"Img2Sig Recall@5: Mean = {np.mean(recalls_imgs):.4f}, Std = {np.std(recalls_imgs):.4f}")
print(f"Sig2Img Recall@5: Mean = {np.mean(recalls_sigs):.4f}, Std = {np.std(recalls_sigs):.4f}")
    