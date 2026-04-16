import torch
import torch.nn as nn

class DigitsEncoder(nn.Module):
    def __init__(self, input_dim: int = 64, hidden_dim: int = 128, output_dim: int = 32):
        super(DigitsEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embedding = self.encoder(x)
        return nn.functional.normalize(embedding, p=2, dim=1)                   #L2 to normalize the output
    
class MNIST1DEncoder(nn.Module):
    def __init__(self, input_dim: int = 40, hidden_dim: int = 128, output_dim: int = 32):
        super(MNIST1DEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embedding = self.encoder(x)
        return nn.functional.normalize(embedding, p=2, dim=1)                   #L2 to normalize the output
    
if __name__ == "__main__":
    # Quick sanity check
    batch_size = 16
 
    digits_enc  = DigitsEncoder(input_dim=64, hidden_dim=128, output_dim=32)
    mnist1d_enc = MNIST1DEncoder(input_dim=40, hidden_dim=128, output_dim=32)
 
    x_digits  = torch.randn(batch_size, 64)
    x_mnist1d = torch.randn(batch_size, 40)
 
    out_digits  = digits_enc(x_digits)
    out_mnist1d = mnist1d_enc(x_mnist1d)
 
    print(f"Digits  encoder output shape: {out_digits.shape}")   # (16, 32)
    print(f"MNIST1D encoder output shape: {out_mnist1d.shape}")  # (16, 32)
 
    # Check L2 normalization
    norms_digits  = out_digits.norm(dim=1)
    norms_mnist1d = out_mnist1d.norm(dim=1)
    print(f"Digits  embedding norms (should be ~1): {norms_digits[:4]}")
    print(f"MNIST1D embedding norms (should be ~1): {norms_mnist1d[:4]}")