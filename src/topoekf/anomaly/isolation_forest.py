import numpy as np
from sklearn.ensemble import IsolationForest

from .anomaly_result import AnomalyResult
from .interfaces import IAnomalyDetector


class IsolationForestDetector(IAnomalyDetector):
    def __init__(
        self,
        n_estimators: int = 100,
        contamination: float = 0.15,
        max_samples: int | str = 256,
        random_state: int = 42,
        n_jobs: int | None = 1,
        anomaly_score_threshold: float = -0.1,
    ):
        self.anomaly_score_threshold = anomaly_score_threshold
        self.is_fitted = False
        self.model = IsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,
            max_samples=max_samples,
            random_state=random_state,
            n_jobs=n_jobs,
        )

    def fit(self, features: np.ndarray) -> None:
        self.model.fit(features)
        self.is_fitted = True

    def predict(self, features: np.ndarray) -> AnomalyResult:
        scores = self.model.decision_function(features)
        labels = np.where(scores < self.anomaly_score_threshold, -1, 1)
        return AnomalyResult(labels=labels.astype(int), scores=scores)
