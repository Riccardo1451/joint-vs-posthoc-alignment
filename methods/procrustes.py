import torch


def procrustes_align(X_train, Y_train):
    """
    Compute the Procrustes rotation matrix Q that best aligns X_train to Y_train.

    Solves: min ||Y - X @ Q.T||_F  subject to Q^T Q = I
    via SVD of M = Y^T X.

    Args:
        X_train: source embeddings, shape (N, D)
        Y_train: target embeddings, shape (N, D)

    Returns:
        Q: orthogonal rotation matrix, shape (D, D)
    """
    n = min(X_train.shape[0], Y_train.shape[0])
    M = Y_train[:n].T @ X_train[:n]
    U, S, Vt = torch.linalg.svd(M)
    Q = U @ Vt
    return Q


def procrustes_align_centroid(X_train, y_train_X, Y_train, y_train_Y):
    """
    Compute the Procrustes rotation matrix Q using class centroids instead of
    individual paired samples.

    Args:
        X_train:   source embeddings, shape (N, D)
        y_train_X: labels for X_train, shape (N,)
        Y_train:   target embeddings, shape (M, D)
        y_train_Y: labels for Y_train, shape (M,)

    Returns:
        Q: orthogonal rotation matrix, shape (D, D)
    """
    unique_classes = torch.unique(y_train_X)
    centroids_X = torch.stack([X_train[y_train_X == cls].mean(dim=0) for cls in unique_classes])
    centroids_Y = torch.stack([Y_train[y_train_Y == cls].mean(dim=0) for cls in unique_classes])

    M = centroids_Y.T @ centroids_X
    U, S, Vt = torch.linalg.svd(M)
    Q = U @ Vt
    return Q
