import torch
from data.dataset import load_mnist1d_dataset
from torch.utils.data import DataLoader, TensorDataset
from models.unimodal import UnimodalModelMnist1D
import torch.nn as nn
import torch.optim as optim
import os
os.makedirs("checkpoints", exist_ok=True)
import tqdm

num_epochs = 300
batch_size = 64
learning_rate = 1e-4
force_reload = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

data = load_mnist1d_dataset(force_reload=force_reload)
X_train = torch.from_numpy(data["X_train"]).to(device)
y_train = torch.from_numpy(data["y_train"]).to(device)
X_test = torch.from_numpy(data["X_test"]).to(device)
y_test = torch.from_numpy(data["y_test"]).to(device)

dataset = TensorDataset(X_train, y_train)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

model = UnimodalModelMnist1D().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

pbar = tqdm.tqdm(range(num_epochs), desc="Training MNIST1D Model")

for epoch in pbar:
    model.train()
    batch_loss = 0.0
    
    for X, y in dataloader:

        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

        batch_loss += loss.item()
    
    pbar.set_postfix({"loss": batch_loss / len(dataloader)})

    if (epoch + 1) % 20 == 0:
        model.eval()
        with torch.no_grad():
            test_outputs = model(X_test)
            _, predicted = torch.max(test_outputs, 1)
            accuracy = (predicted == y_test).float().mean().item()
        pbar.write(f"Epoch {epoch+1}/{num_epochs}, Test Accuracy: {accuracy:.4f}")

# Save the trained model
torch.save(model.state_dict(), "checkpoints/mnist1d_unimodal.pth")
print("Model saved to checkpoints/mnist1d_unimodal.pth")
