import numpy as np

from .base_adapter import AdaptationContext, INoiseAdapter


class Tier2OcclusionAdapter(INoiseAdapter):
    def __init__(self, lambda_v: float = 0.05, scene_complexity: float = 1.0):
        self.lambda_v = lambda_v
        self.gamma_scene = scene_complexity

    def adapt(
        self, Q: np.ndarray, R: np.ndarray, context: AdaptationContext
    ) -> tuple[np.ndarray, np.ndarray]:
        velocity_norm = np.linalg.norm(context.velocity)
        alpha_k = 1.0 + 0.2 * min(context.miss_count, 5)
        velocity_term = 1.0 + self.lambda_v * velocity_norm
        return Q * alpha_k * self.gamma_scene * velocity_term, R
