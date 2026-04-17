import torch
from methods.cka import compute_cka

def recall_at_k(query_embs, query_labels, gallery_embs, gallery_labels, k):

    sim_matrix = torch.matmul(query_embs, gallery_embs.T)  # (num_queries, num_gallery)
    topk = torch.topk(sim_matrix, k, dim = 1, largest = True)
    topk_indices = topk.indices  # (num_queries, k)

    topk_labels = gallery_labels[topk_indices]  # (num_queries, k)
    matches = (topk_labels == query_labels.unsqueeze(1))  # (num_queries, k)
    recall = matches.any(dim=1).float().mean().item()  # Average over queries

    return recall

def evaluate_retrival(model, digits_data, mnist1d_data, device, k):
    model.eval().to(device)
    with torch.no_grad():

        z_sig, z_img = model(torch.from_numpy(mnist1d_data["X_test"]).to(device), torch.from_numpy(digits_data["X_test"]).to(device))

        recall_sig2img = recall_at_k(z_sig, torch.from_numpy(mnist1d_data["y_test"]).to(device), z_img, torch.from_numpy(digits_data["y_test"]).to(device), k)
        recall_img2sig = recall_at_k(z_img, torch.from_numpy(digits_data["y_test"]).to(device), z_sig, torch.from_numpy(mnist1d_data["y_test"]).to(device), k)
    model.train()

    return recall_sig2img, recall_img2sig

def evaluate_cka(model1, model2, digits_data, mnist1d_data, device):
    model1.eval().to(device)
    model2.eval().to(device)
    n = min(len(digits_data["X_test"]), len(mnist1d_data["X_test"]))

    with torch.no_grad():

        z_sig1, z_img1 = model1(torch.from_numpy(mnist1d_data["X_test"][:n]).to(device), torch.from_numpy(digits_data["X_test"][:n]).to(device))
        z_sig2, z_img2 = model2(torch.from_numpy(mnist1d_data["X_test"][:n]).to(device), torch.from_numpy(digits_data["X_test"][:n]).to(device))

        z1 = torch.cat([z_sig1, z_img1], dim=1)
        z2 = torch.cat([z_sig2, z_img2], dim=1)

        cka = compute_cka(z1, z2)
        

    model1.train()
    model2.train()

    return cka