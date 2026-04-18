import torch
import torch.nn as nn
from models.encoders import MNIST1DEncoder, DigitsEncoder


class UnimodalModelMnist1D(nn.Module):

    def __init__(self, embedding_dim = 32, num_classes = 10):
        super(UnimodalModelMnist1D, self).__init__()

        self.mnist_encoder = MNIST1DEncoder(input_dim=40, hidden_dim=128, output_dim=embedding_dim)
        self.class_head = nn.Linear(embedding_dim, num_classes)


        self.net = nn.Sequential(
            self.mnist_encoder, #Feature are already L2 normalized in the encoder, no need for a normalization layer
            self.class_head
        )
    
    def get_embedding(self, x_mnist):
        with torch.no_grad():
            embedding = self.mnist_encoder(x_mnist)
        return embedding

    def forward (self, x_mnist):
        x = self.net(x_mnist)
        return x
    
    
    
class UnimodalModelDigits(nn.Module):

    def __init__(self, embedding_dim = 32, num_classes = 10):
        super(UnimodalModelDigits, self).__init__()

        self.digits_encoder = DigitsEncoder(input_dim=64, hidden_dim=128, output_dim=embedding_dim)
        self.class_head = nn.Linear(embedding_dim, num_classes)


        self.net = nn.Sequential(
            self.digits_encoder,
            self.class_head
        )

    def get_embedding(self, x_digits):
        with torch.no_grad():
            embedding = self.digits_encoder(x_digits)
        return embedding
        

    def forward (self, x_digits):
        x = self.net(x_digits)
        return x