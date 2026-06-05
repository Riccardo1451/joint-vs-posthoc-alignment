import numpy as np
import torch
from sklearn.model_selection import train_test_split 

def sample_batch(digits_data, mnist1d_data, K):

    complete_batch_digits = []
    complete_batch_mnist1d = []

    for classes in range(10): #Classes goes from 0 to 9

        #Sample K samples of the current class

        idx_digits = np.random.choice(np.where(digits_data["y_train"]== classes)[0], K, replace=False)
        idx_mnist1d = np.random.choice(np.where(mnist1d_data["y_train"]== classes)[0], K, replace=False)

        batch_digits = digits_data["X_train"][idx_digits] # (K, 8, 8)
        batch_mnist1d = mnist1d_data["X_train"][idx_mnist1d] # (K, 40)

        complete_batch_digits.append(batch_digits)
        complete_batch_mnist1d.append(batch_mnist1d)
    
    #Concatenate all classe to get final batch
    complete_batch_digits = np.concatenate(complete_batch_digits, axis=0) # (10*K, 8, 8)
    complete_batch_mnist1d = np.concatenate(complete_batch_mnist1d, axis=0) # (10*K, 40)

    label = np.repeat(np.arange(10), K) # (10*K,)

    return torch.from_numpy(complete_batch_digits), torch.from_numpy(complete_batch_mnist1d), torch.from_numpy(label)

def build_paired_dataset(digits_data, mnist1d_data, seed=42):
    
    paired_digits = []
    paired_mnist1d = []
    paired_labels = []

    np.random.seed(seed)

    for cls in range(10):
        idx_digits  = np.where(digits_data["y_train"] == cls)[0]
        idx_mnist1d = np.where(mnist1d_data["y_train"] == cls)[0]

        np.random.shuffle(idx_digits)
        np.random.shuffle(idx_mnist1d)

        n_per_class = min(len(idx_digits), len(idx_mnist1d))

        paired_digits.append(digits_data["X_train"][idx_digits[:n_per_class]])
        paired_mnist1d.append(mnist1d_data["X_train"][idx_mnist1d[:n_per_class]])
        paired_labels.append(np.full(n_per_class, cls))

    paired_digits = np.concatenate(paired_digits, axis=0)
    paired_mnist1d = np.concatenate(paired_mnist1d, axis=0)
    paired_labels = np.concatenate(paired_labels, axis=0)

    align_set = {
        "X_digits": paired_digits,
        "X_mnist1d": paired_mnist1d,
        "y": paired_labels
    }

    return align_set
        
def build_paired_test(digits_data, mnist1d_data, seed=42):
    """
    Build a class-paired test set: for every class 0-9, shuffles the test
    samples of each modality with a fixed seed and takes min(n_digits, n_mnist1d)
    samples, so that row i of X_digits and row i of X_mnist1d share the same label.
    """
    paired_digits = []
    paired_mnist1d = []
    paired_labels = []

    rng = np.random.default_rng(seed)

    for cls in range(10):
        idx_digits  = np.where(digits_data["y_test"] == cls)[0]
        idx_mnist1d = np.where(mnist1d_data["y_test"] == cls)[0]

        rng.shuffle(idx_digits)
        rng.shuffle(idx_mnist1d)

        n_per_class = min(len(idx_digits), len(idx_mnist1d))

        paired_digits.append(digits_data["X_test"][idx_digits[:n_per_class]])
        paired_mnist1d.append(mnist1d_data["X_test"][idx_mnist1d[:n_per_class]])
        paired_labels.append(np.full(n_per_class, cls))

    return {
        "X_digits":  np.concatenate(paired_digits,  axis=0),
        "X_mnist1d": np.concatenate(paired_mnist1d, axis=0),
        "y":         np.concatenate(paired_labels,  axis=0),
    }


def apply_label_flip(y_train, flip_rate=0.1, seed=42):
    n_flip = int(len(y_train) * flip_rate)
    # Choose n_flip random indices to flip
    np.random.seed(seed)
    flip_indices = np.random.choice(len(y_train), n_flip, replace=False)

    classes = np.unique(y_train)

    flipped_labels = y_train.copy()

    # Flip the labels with a random different class
    for i in flip_indices:
        new_label = np.random.choice([c for c in classes if c!= y_train[i]])
        flipped_labels[i] = new_label
        
    return flipped_labels