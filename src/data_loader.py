import numpy as np
import pandas as pd

def load_csv(path: str):
    """
    Expect CSV with first column=label (0–9), next 784 cols=pixel values 0–255.
    Returns:
      X: np.ndarray, shape (n_samples, 784), normalized to [0,1]
      y: np.ndarray, shape (n_samples,)
    """
    df = pd.read_csv(path)
    y = df.iloc[:, 0].to_numpy(dtype=int)
    X = df.iloc[:, 1:].to_numpy(dtype=float) / 255.0
    return X, y