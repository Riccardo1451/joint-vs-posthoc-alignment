import torch
from models.unimodal import UnimodalModelMnist1D, UnimodalModelDigits
from data.dataset import load_all_datasets
from methods.procruster import procruster_align
from utils.metrics import recall_at_k, evaluate_cka

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print (f"Using device: {device}")

force_reload = False

digits_dataset, mnist1d_dataset = load_all_datasets(force_reload=force_reload)

Modelmnist1D = UnimodalModelMnist1D().to(device)
ModelDigits = UnimodalModelDigits().to(device)

Modelmnist1D.load_state_dict(torch.load("checkpoints/mnist1d_unimodal.pth", weights_only=True))
ModelDigits.load_state_dict(torch.load("checkpoints/digits_unimodal.pth", weights_only=True))

Modelmnist1D.eval()
ModelDigits.eval()

#Embeddings from train set
embs_mnist1d_train = Modelmnist1D.get_embedding(torch.from_numpy(mnist1d_dataset["X_train"]).to(device))
embs_digits_train = ModelDigits.get_embedding(torch.from_numpy(digits_dataset["X_train"]).to(device))

print(embs_mnist1d_train[:3])  # dovrebbero essere valori coerenti, non tutti uguali
print(embs_digits_train[:3])

Q = procruster_align(embs_mnist1d_train, embs_digits_train)

#Embeddings from test set
embs_mnist1d_test = Modelmnist1D.get_embedding(torch.from_numpy(mnist1d_dataset["X_test"]).to(device))
embs_digits_test = ModelDigits.get_embedding(torch.from_numpy(digits_dataset["X_test"]).to(device))

#Use Q matrix to align embs
mnist1D_aligned = embs_mnist1d_test @ Q.T

#Compute retrival
recall_sig2img = recall_at_k(mnist1D_aligned, torch.from_numpy(mnist1d_dataset["y_test"]).to(device), embs_digits_test, torch.from_numpy(digits_dataset["y_test"]).to(device), k = 5)
recall_img2sig = recall_at_k(embs_digits_test, torch.from_numpy(digits_dataset["y_test"]).to(device), mnist1D_aligned, torch.from_numpy(mnist1d_dataset["y_test"]).to(device), k = 5)

#Compute CKA
cka = evaluate_cka(None, None, None, None, device, emb1 = mnist1D_aligned, emb2 = embs_digits_test)

print (f"Recall mnist1D to digits: {recall_sig2img:.4f}")
print (f"Recall digits to mnist1D: {recall_img2sig:.4f}")
print (f"CKA: {cka:.4f}")

print("mnist1D_aligned shape:", mnist1D_aligned.shape)
print("embs_digits_test shape:", embs_digits_test.shape)
print("y_test mnist1d:", mnist1d_dataset["y_test"][:10])
print("y_test digits:", digits_dataset["y_test"][:10])
sims = mnist1D_aligned[0] @ embs_digits_test.T
print("Similarities:", sims)
print("Max sim:", sims.max().item())
print("Min sim:", sims.min().item())
