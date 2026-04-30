from data.dataset import load_all_datasets
from models.clip_model import CLIPModel
from utils.metrics import evaluate_cka
from utils.visualization import plot_tsne
import torch

seeds = [42, 123, 999]
force_reload = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print (f"Using device: {device}")
digits_data, mnist1d_data = load_all_datasets(seed=seeds[0], force_reload=force_reload) #Same seed to ensure same test set for CKA evaluation

#-------------------- CKA Evaluation ------------------

#--------------------cka seed 42 vs 123------------------
model1 = CLIPModel(projection_dim=32).to(device)
model2 = CLIPModel(projection_dim=32).to(device)
model1.load_state_dict(torch.load(f"checkpoints/clip_seed{seeds[0]}.pth"))
model2.load_state_dict(torch.load(f"checkpoints/clip_seed{seeds[1]}.pth"))
cka1 = evaluate_cka(model1, model2, digits_data=digits_data, mnist1d_data=mnist1d_data, device=device)
print(f"CKA between models trained with seeds {seeds[0]} and {seeds[1]}: {cka1:.4f}")

#--------------------cka seed 999 vs 123------------------
model1 = CLIPModel(projection_dim=32).to(device)
model2 = CLIPModel(projection_dim=32).to(device)
model1.load_state_dict(torch.load(f"checkpoints/clip_seed{seeds[1]}.pth"))
model2.load_state_dict(torch.load(f"checkpoints/clip_seed{seeds[2]}.pth"))
cka2 = evaluate_cka(model1, model2, digits_data=digits_data, mnist1d_data=mnist1d_data, device=device)
print(f"CKA between models trained with seeds {seeds[1]} and {seeds[2]}: {cka2:.4f}")

#--------------------cka seed 999 vs 42------------------
model1 = CLIPModel(projection_dim=32).to(device)
model2 = CLIPModel(projection_dim=32).to(device)
model1.load_state_dict(torch.load(f"checkpoints/clip_seed{seeds[0]}.pth"))
model2.load_state_dict(torch.load(f"checkpoints/clip_seed{seeds[2]}.pth"))
cka3 = evaluate_cka(model1, model2, digits_data=digits_data, mnist1d_data=mnist1d_data, device=device)
print(f"CKA between models trained with seeds {seeds[0]} and {seeds[2]}: {cka3:.4f}")

avg_cka = (cka1 + cka2 + cka3) / 3
print(f"Average CKA across all pairs: {avg_cka:.4f}")


#-------------------- t-SNE Visualization ------------------
model = CLIPModel(projection_dim=32).to(device)
model.load_state_dict(torch.load(f"checkpoints/clip_seed{seeds[0]}.pth", weights_only=True))
model.eval()

n = min(len(digits_data["X_test"]), len(mnist1d_data["X_test"]))

with torch.no_grad():
    z_sig, z_img = model(
        torch.from_numpy(mnist1d_data["X_test"][:n]).to(device),
        torch.from_numpy(digits_data["X_test"][:n]).to(device)
    )

plot_tsne(z_img, z_sig, 
          torch.from_numpy(digits_data["y_test"][:n]).to(device),
          torch.from_numpy(mnist1d_data["y_test"][:n]).to(device),
          title=f"t-SNE of CLIP Embeddings (Seed {seeds[0]})",
          save_path=f"figures/tsne_clip_seed{seeds[0]}.png")