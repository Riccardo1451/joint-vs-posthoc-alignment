import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import torch


def plot_tsne(emb_digits, emb_mnist, y_digits, y_mnist, title, save_path=None):
    """
    Plot t-SNE visualization of embeddings from two modalities.
    
    Args:
        emb_digits:  torch.Tensor or np.ndarray of shape (N_digits, D)
        emb_mnist:   torch.Tensor or np.ndarray of shape (N_mnist, D)
        y_digits:    labels for digits, shape (N_digits,)
        y_mnist:     labels for mnist1d, shape (N_mnist,)
        title:       plot title
        save_path:   if provided, saves the figure to this path
    """

    # Convert to numpy if needed
    if isinstance(emb_digits, torch.Tensor):
        emb_digits = emb_digits.detach().cpu().numpy()
    if isinstance(emb_mnist, torch.Tensor):
        emb_mnist = emb_mnist.detach().cpu().numpy()
    if isinstance(y_digits, torch.Tensor):
        y_digits = y_digits.detach().cpu().numpy()
    if isinstance(y_mnist, torch.Tensor):
        y_mnist = y_mnist.detach().cpu().numpy()

    # Align sizes
    n = min(len(emb_digits), len(emb_mnist))
    emb_digits = emb_digits[:n]
    emb_mnist  = emb_mnist[:n]
    y_digits   = y_digits[:n]
    y_mnist    = y_mnist[:n]

    # Concatenate for joint t-SNE
    emb_all = np.concatenate([emb_digits, emb_mnist], axis=0)  # (2N, D)
    modality = np.array([0] * n + [1] * n)                     # 0=Digits, 1=MNIST-1D

    # Fit t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
    emb_2d = tsne.fit_transform(emb_all)                        # (2N, 2)

    # Split back
    emb_2d_digits = emb_2d[:n]
    emb_2d_mnist  = emb_2d[n:]

    # Plot
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, 10))

    for cls in range(10):
        mask_d = (y_digits == cls)
        mask_m = (y_mnist  == cls)

        ax.scatter(emb_2d_digits[mask_d, 0], emb_2d_digits[mask_d, 1],
                   color=colors[cls], marker='o', s=30, alpha=0.7,
                   label=f'Class {cls} (Digits)' if cls == 0 else "")

        ax.scatter(emb_2d_mnist[mask_m, 0], emb_2d_mnist[mask_m, 1],
                   color=colors[cls], marker='x', s=30, alpha=0.7,
                   label=f'Class {cls} (MNIST-1D)' if cls == 0 else "")

    # Custom legend — one entry per class + modality markers
    from matplotlib.lines import Line2D
    class_handles = [Line2D([0], [0], marker='o', color='w',
                             markerfacecolor=colors[i], markersize=8, label=f'Class {i}')
                     for i in range(10)]
    modality_handles = [
        Line2D([0], [0], marker='o', color='gray', markersize=8, label='Digits (○)'),
        Line2D([0], [0], marker='x', color='gray', markersize=8, label='MNIST-1D (×)'),
    ]
    ax.legend(handles=class_handles + modality_handles,
              bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)

    ax.set_title(title, fontsize=13)
    ax.set_xlabel('t-SNE dim 1')
    ax.set_ylabel('t-SNE dim 2')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")

    plt.show()