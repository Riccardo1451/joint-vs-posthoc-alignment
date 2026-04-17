import os
import pickle
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from mnist1d.data import make_dataset, get_dataset_args

CACHE_DIR = os.path.join(os.path.dirname(__file__), "cached")
DIGITS_CACHE = os.path.join(CACHE_DIR, "digits.pkl")
MNIST1D_CACHE = os.path.join(CACHE_DIR, "mnist1d.pkl")

#Make sure the cache directory exists or create it
def _ensure_cache_dir():
    if not os.path.exists(CACHE_DIR):
        os.makedirs(CACHE_DIR)


#Load digits dataset with caching
def load_digits_dataset(force_reload=False) -> dict:
    _ensure_cache_dir()


    if os.path.exists(DIGITS_CACHE) and not force_reload:
        print("Loading digits dataset from cache...")
        with open(DIGITS_CACHE, "rb") as f:
            return pickle.load(f)
        
    print("Loading digits dataset from source...")
    raw = load_digits()

    X = raw.data.astype(np.float32)/16.0            #Normalize to [0,1]
    X_images = X.reshape(-1, 8, 8)                  #(N, 1, 8, 8) for CNN
    y = raw.target.astype(np.int64)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    data = {"X": X, "X_images": X_images, "y": y, "X_train": X_train, "X_test": X_test, "y_train": y_train, "y_test": y_test}


    with open(DIGITS_CACHE, "wb") as f:
        pickle.dump(data, f)
    print(f"Digits saved to cache: {DIGITS_CACHE}")

    return data

#Load MNIST1D dataset with caching
def load_mnist1d_dataset(force_reload=False, num_samples: int = 5000, seed: int = 42) -> dict:
    _ensure_cache_dir()

    if os.path.exists(MNIST1D_CACHE) and not force_reload:
        print("Loading MNIST1D dataset from cache...")
        with open(MNIST1D_CACHE, "rb") as f:
            return pickle.load(f)

    print("Generating MNIST1D dataset...")
    args = get_dataset_args()
    args.num_samples = num_samples
    args.seed = seed
    raw = make_dataset(args)
    
    data = {
        "X_train": raw["x"].astype(np.float32),
        "y_train": raw["y"].astype(np.int64),
        "X_test": raw["x_test"].astype(np.float32),
        "y_test": raw["y_test"].astype(np.int64),
    }

    with open(MNIST1D_CACHE, "wb") as f:
        pickle.dump(data, f)
    print(f"MNIST1D saved to cache: {MNIST1D_CACHE}")

    return data

def load_all_datasets(force_reload=False, mnist1d_samples: int = 5000, seed: int = 42) -> tuple[dict, dict]:

    digits_data = load_digits_dataset(force_reload=force_reload)
    mnist1d_data = load_mnist1d_dataset(force_reload=force_reload, num_samples=mnist1d_samples, seed=seed)

    return digits_data, mnist1d_data

if __name__ == "__main__":
    # Quick sanity check
    digits, mnist1d = load_all_datasets(force_reload=True)
 
    print("\n--- Digits ---")
    print(f"  X shape:        {digits['X'].shape}")
    print(f"  X_images shape: {digits['X_images'].shape}")
    print(f"  y shape:        {digits['y'].shape}")
    print(f"  Classes:        {np.unique(digits['y'])}")
 
    print("\n--- MNIST-1D ---")
    print(f"  X_train shape:  {mnist1d['X_train'].shape}")
    print(f"  y_train shape:  {mnist1d['y_train'].shape}")
    print(f"  X_test shape:   {mnist1d['X_test'].shape}")
    print(f"  y_test shape:   {mnist1d['y_test'].shape}")
    print(f"  Classes:        {np.unique(mnist1d['y_train'])}")