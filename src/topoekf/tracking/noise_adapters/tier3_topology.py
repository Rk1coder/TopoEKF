import numpy as np

from .base_adapter import AdaptationContext, INoiseAdapter


class Tier3TopologyAdapter(INoiseAdapter):
    def __init__(self, lam: float = 0.4, theta_beta: float = 3.0, eps_max: float = 10.0):
        self.lam = lam
        self.theta_beta = theta_beta
        self.eps_max = eps_max
        self._gamma_k = 1.0
        self._delta_k = 1.0

    def update_topology(self, beta1: int, p_max: float) -> None:
        self._gamma_k = 1.0 + self.lam * (beta1 / self.theta_beta)
        self._delta_k = max(0.5, 1.0 - p_max / self.eps_max)

    def adapt(
        self, Q: np.ndarray, R: np.ndarray, context: AdaptationContext
    ) -> tuple[np.ndarray, np.ndarray]:
        return Q * self._gamma_k, R * self._delta_k
