import numpy as np
from typing import Optional

class KNNClassifier:
    def __init__(self, k: int = 3):
        self.k = k
        self.X_train: Optional[np.ndarray] = None
        self.y_train: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.X_train = X
        self.y_train = y

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.X_train is None or self.y_train is None:
            raise ValueError("Model has not been trained (call .fit first).")
        dists = np.linalg.norm(self.X_train[None, :, :] - X[:, None, :], axis=2)
        neighbors = np.argsort(dists, axis=1)[:, : self.k]
        preds = []
        for nbr_idxs in neighbors:
            votes = self.y_train[nbr_idxs]
            pred  = np.bincount(votes).argmax()
            preds.append(pred)
        return np.array(preds)