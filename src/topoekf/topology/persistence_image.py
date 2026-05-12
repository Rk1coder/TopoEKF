import numpy as np


class PersistenceImageVectorizer:
    def __init__(self, grid_size: int = 20, sigma: float = 1.0, weight_power: float = 1.2):
        self.grid_size = grid_size
        self.sigma = sigma
        self.weight_power = weight_power

    def vectorize(self, dgm1: np.ndarray) -> np.ndarray:
        if len(dgm1) == 0:
            return np.zeros(self.grid_size * self.grid_size)

        lifetimes = dgm1[:, 1] - dgm1[:, 0]
        births = dgm1[:, 0]
        birth_edges = np.linspace(0.0, max(float(np.max(births)), 1.0), self.grid_size + 1)
        life_edges = np.linspace(0.0, max(float(np.max(lifetimes)), 1.0), self.grid_size + 1)
        weights = np.maximum(lifetimes, 0.0) ** self.weight_power
        image, _, _ = np.histogram2d(births, lifetimes, bins=[birth_edges, life_edges], weights=weights)
        return image.astype(float).flatten()
