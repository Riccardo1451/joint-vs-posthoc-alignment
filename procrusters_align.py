import torch
from models.unimodal import UnimodalModelMnist1D, UnimodalModelDigits
from data.dataset import load_all_datasets
from methods.procruster import procruster_align, procruster_align_centroid
from utils.metrics import recall_at_k, evaluate_cka
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print (f"Using device: {device}")

force_reload = False

Modelmnist1D = UnimodalModelMnist1D().to(device)
ModelDigits = UnimodalModelDigits().to(device)
seeds = [42, 123, 999]

results = {}

for seed in seeds:
    digits_dataset, mnist1d_dataset = load_all_datasets(force_reload=force_reload, seed=seed)

    Modelmnist1D.load_state_dict(torch.load(f"checkpoints/mnist1d_unimodal_seed{seed}.pth", weights_only=True))
    ModelDigits.load_state_dict(torch.load(f"checkpoints/digits_unimodal_seed{seed}.pth", weights_only=True))

    Modelmnist1D.eval()
    ModelDigits.eval()

    #Embeddings from train set
    embs_mnist1d_train = Modelmnist1D.get_embedding(torch.from_numpy(mnist1d_dataset["X_train"]).to(device))
    embs_digits_train = ModelDigits.get_embedding(torch.from_numpy(digits_dataset["X_train"]).to(device))

    #Q = procruster_align(embs_mnist1d_train, embs_digits_train)
    Q = procruster_align_centroid(embs_mnist1d_train, torch.from_numpy(mnist1d_dataset["y_train"]).to(device), embs_digits_train, torch.from_numpy(digits_dataset["y_train"]).to(device))

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

    results[seed] = {
        "recall_sig2img": recall_sig2img,
        "recall_img2sig": recall_img2sig,
        "cka": cka
    }

    print (f"Recall mnist1D to digits seed {seed}: {recall_sig2img:.4f}")
    print (f"Recall digits to mnist1D seed {seed}: {recall_img2sig:.4f}")
    print (f"CKA seed {seed}: {cka:.4f}")

recalls_sig2img = [results[seed]["recall_sig2img"] for seed in seeds]
recalls_img2sig = [results[seed]["recall_img2sig"] for seed in seeds]
ckas = [results[seed]["cka"] for seed in seeds]

print(f"Recall Sig→Img: {np.mean(recalls_sig2img):.4f} ± {np.std(recalls_sig2img):.4f}")
print(f"Recall Img→Sig: {np.mean(recalls_img2sig):.4f} ± {np.std(recalls_img2sig):.4f}")
print(f"CKA: {np.mean(ckas):.4f} ± {np.std(ckas):.4f}")