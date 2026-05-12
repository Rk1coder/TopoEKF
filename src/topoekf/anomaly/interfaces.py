from abc import ABC, abstractmethod

import numpy as np

from .anomaly_result import AnomalyResult


class IAnomalyDetector(ABC):
    @abstractmethod
    def fit(self, features: np.ndarray) -> None:
        """Fit detector from feature rows."""

    @abstractmethod
    def predict(self, features: np.ndarray) -> AnomalyResult:
        """Predict anomaly labels and scores."""
