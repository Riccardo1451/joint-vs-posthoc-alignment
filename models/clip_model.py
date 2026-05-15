import torch
import torch.nn as nn
from models.encoders import MNIST1DEncoder, DigitsEncoder, MNIST1DEncoderCNN, DigitsEncoderCNN

class CLIPModel(nn.Module):
    def __init__(self, mode = "mlp", hidden_dim = 128, projection_dim = 32):
        super(CLIPModel, self).__init__()

        if mode == "cnn":
            self.mnist_encoder = MNIST1DEncoderCNN(input_dim=40, hidden_dim=hidden_dim, output_dim=projection_dim)
            self.digits_encoder = DigitsEncoderCNN(input_dim=64, hidden_dim=hidden_dim, output_dim=projection_dim)
        if mode == "mlp":
            self.mnist_encoder = MNIST1DEncoder(input_dim=40, hidden_dim=hidden_dim, output_dim=projection_dim)
            self.digits_encoder = DigitsEncoder(input_dim=64, hidden_dim=hidden_dim, output_dim=projection_dim)

        self.mnist_projection = nn.Linear(projection_dim, projection_dim) #Projection head is in the clip model definition, not in the encoders
        self.digits_projection = nn.Linear(projection_dim, projection_dim)

    def forward(self, x_mnist, x_digits):
        
        mnist_emb = self.mnist_encoder(x_mnist)          # (B, projection_dim)
        digits_emb = self.digits_encoder(x_digits)       # (B, projection_dim)

        mnist_proj = self.mnist_projection(mnist_emb)   # (B, projection_dim)
        digits_proj = self.digits_projection(digits_emb) # (B, projection_dim)

        return nn.functional.normalize(mnist_proj, p=2, dim=1), nn.functional.normalize(digits_proj, p=2, dim=1)

