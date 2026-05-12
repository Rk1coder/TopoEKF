import numpy as np

from .base_adapter import AdaptationContext, INoiseAdapter


class Tier1ConfidenceAdapter(INoiseAdapter):
    def __init__(self, alpha: float = 0.80):
        self.alpha = alpha
        self._beta_smooth = 1.0

    def adapt(
        self, Q: np.ndarray, R: np.ndarray, context: AdaptationContext
    ) -> tuple[np.ndarray, np.ndarray]:
        raw_beta = 2.0 - context.confidence
        self._beta_smooth = self.alpha * self._beta_smooth + (1 - self.alpha) * raw_beta
        return Q, R * self._beta_smooth
