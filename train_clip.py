import torch
import numpy as np
import os
os.makedirs("checkpoints", exist_ok=True)
import tqdm

from models.clip_model import CLIPModel
from data.dataloader import sample_batch
from data.dataset import load_all_datasets
from methods.losses import info_nce_loss
from utils.metrics import evaluate_retrival


def train_clip(seed, epochs, steps_per_epoch, batch_size, temperature, force_reload=False):

    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")


    model = CLIPModel(projection_dim=32).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    digits_data, mnist1d_data = load_all_datasets(seed=seed, force_reload=force_reload)
    pbar = tqdm.tqdm(range(epochs), desc=f"Training - Seed {seed}", )

    for epoch in pbar:

        model.train()
        epoch_loss = 0.0
        

        for step in range(steps_per_epoch):

            optimizer.zero_grad()

            batch_digits, batch_mnist1d, labels = sample_batch(digits_data, mnist1d_data=mnist1d_data, K=batch_size//10)
            batch_digits, batch_mnist1d, labels = batch_digits.to(device), batch_mnist1d.to(device), labels.to(device)


            z_sig, z_img = model(batch_mnist1d, batch_digits)

            loss, loss_img, loss_sig = info_nce_loss(z_img, z_sig, temperature=temperature)

            
            loss.backward()
            epoch_loss += loss.item()        
            optimizer.step()

        avg_loss = epoch_loss / steps_per_epoch
        pbar.set_postfix(loss = f"{avg_loss:.4f}")
        if (epoch + 1) % 50 == 0:
            recall_sig2img, recall_img2sig = evaluate_retrival(model, digits_data=digits_data, mnist1d_data=mnist1d_data, device=device, k=5)
            pbar.write(f"Epoch {epoch+1} - Recall@5 Sig2Img: {recall_sig2img:.4f}, Img2Sig: {recall_img2sig:.4f}")
        
            

    recall_sig2img, recall_img2sig = evaluate_retrival(model, digits_data=digits_data, mnist1d_data=mnist1d_data, device=device, k=5)
    torch.save(model.state_dict(), f"checkpoints/clip_seed{seed}.pth")
    print(f"Training completed and model saved in checkpoints/clip_seed{seed}.pth.")

    return recall_sig2img, recall_img2sig