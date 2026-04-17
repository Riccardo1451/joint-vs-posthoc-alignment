import torch
import numpy as np
import os
os.makedirs("checkpoints", exist_ok=True)

from models.clip_model import CLIPModel
from data.dataloader import sample_batch
from data.dataset import load_all_datasets
from methods.losses import info_nce_loss
from utils.metrics import evaluate_retrival

seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

epochs = 200
steps_per_epoch = 50 
batch_size = 100
temperature = 0.1 

model = CLIPModel(projection_dim=32).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

digits_data, mnist1d_data = load_all_datasets(seed=seed, force_reload=True)

for epoch in range(epochs):

    model.train()
    epoch_loss = 0.0
    

    for step in range(steps_per_epoch):

        optimizer.zero_grad()

        batch_digits, batch_mnist1d, labels = sample_batch(digits_data, mnist1d_data=mnist1d_data, K=batch_size//10)
        batch_digits, batch_mnist1d, labels = batch_digits.to(device), batch_mnist1d.to(device), labels.to(device)


        z_sig, z_img = model(batch_mnist1d, batch_digits)

        loss, loss_img, loss_sig = info_nce_loss(z_img, z_sig, temperature=temperature)

        
        loss.backward()
        epoch_loss = loss.item() / steps_per_epoch
    
        optimizer.step()

    if (epoch + 1) % 20 == 0:
        recall_sig2img, recall_img2sig = evaluate_retrival(model, digits_data=digits_data, mnist1d_data=mnist1d_data, device=device, k=5)
        print(f"Epoch {epoch+1}/{epochs} - Recall@5 Sig2Img: {recall_sig2img:.4f}, Img2Sig: {recall_img2sig:.4f}")

    print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f} (Img: {loss_img:.4f}, Sig: {loss_sig:.4f})")

torch.save(model.state_dict(), f"checkpoints/clip_seed{seed}.pth")