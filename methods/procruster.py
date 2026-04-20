import torch

def procruster_align(X_train, Y_train): #In input both of the parameters are embedding
    #Compute Q matrix to rotate the embedding

    n = min(X_train.shape[0], Y_train.shape[0])

    M = Y_train[:n].T @ X_train[:n]

    U, S, Vt = torch.linalg.svd(M)

    Q = U @ Vt
    
    return Q

def procruster_align_centroid(X_train, y_train_X, Y_train, y_train_Y):
    #Compute Q matrix to rotate the embedding and also align the centroids of the classes

    #Compute centroids
    unique_classes = torch.unique(y_train_X)
    centroids_X = torch.stack([X_train[y_train_X == cls].mean(dim=0) for cls in unique_classes])
    centroids_Y = torch.stack([Y_train[y_train_Y == cls].mean(dim=0) for cls in unique_classes])

    

    M = centroids_Y.T @ centroids_X #shape (num_classes, num_classes)

    U, S, Vt = torch.linalg.svd(M)

    Q = U @ Vt
    
    return Q