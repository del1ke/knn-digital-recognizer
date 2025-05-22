import numpy as np

def normalize_pixels(X: np.ndarray) -> np.ndarray:
    """Rescale already 0–1 pixels to zero mean, unit variance."""
    mean = X.mean(axis=0)
    std  = X.std(axis=0) + 1e-8
    return (X - mean) / std