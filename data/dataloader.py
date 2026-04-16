import numpy as np
import torch

def sample_batch(digits_data, mnist1d_data, K):

    complete_batch_digits = []
    complete_batch_mnist1d = []

    for classes in range(10): #Classes goes from 0 to 9

        #Sample K samples of the current class

        idx_digits = np.random.choice(np.where(digits_data["y"]== classes)[0], K, replace=False)
        idx_mnist1d = np.random.choice(np.where(mnist1d_data["y_train"]== classes)[0], K, replace=False)

        batch_digits = digits_data["X"][idx_digits] # (K, 8, 8)
        batch_mnist1d = mnist1d_data["X_train"][idx_mnist1d] # (K, 40)

        complete_batch_digits.append(batch_digits)
        complete_batch_mnist1d.append(batch_mnist1d)
    
    #Concatenate all classe to get final batch
    complete_batch_digits = np.concatenate(complete_batch_digits, axis=0) # (10*K, 8, 8)
    complete_batch_mnist1d = np.concatenate(complete_batch_mnist1d, axis=0) # (10*K, 40)

    label = np.repeat(np.arange(10), K) # (10*K,)

    return torch.from_numpy(complete_batch_digits), torch.from_numpy(complete_batch_mnist1d), torch.from_numpy(label)


        
