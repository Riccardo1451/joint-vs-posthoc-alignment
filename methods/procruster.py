import torch

def procruster_align(X_train, Y_train): #In input both of the parameters are embedding
    #Compute Q matrix to rotate the embedding

    n = min(X_train.shape[0], Y_train.shape[0])

    M = Y_train[:n].T @ X_train[:n]

    U, S, Vt = torch.linalg.svd(M)

    Q = U @ Vt
    
    return Q