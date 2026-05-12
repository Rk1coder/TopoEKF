from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TopologyResult:
    dgm0: np.ndarray
    dgm1: np.ndarray
    beta0: int
    beta1: int
    p_max: float


class PersistenceComputer:
    def __init__(self, max_dim: int = 1, thresh: float = 50.0):
        self.max_dim = max_dim
        self.thresh = thresh

    def compute(self, positions: np.ndarray) -> TopologyResult:
        from ripser import ripser

        result = ripser(positions, maxdim=self.max_dim, thresh=self.thresh)
        dgms = result["dgms"]
        dgm0 = dgms[0]
        dgm1 = dgms[1] if len(dgms) > 1 else np.empty((0, 2))
        finite_mask = np.isfinite(dgm1[:, 1]) if len(dgm1) > 0 else np.array([], dtype=bool)
        dgm1_finite = dgm1[finite_mask] if len(dgm1) > 0 else np.empty((0, 2))
        beta1 = len(dgm1_finite)
        p_max = float(np.max(dgm1_finite[:, 1] - dgm1_finite[:, 0])) if beta1 else 0.0
        return TopologyResult(
            dgm0=dgm0,
            dgm1=dgm1_finite,
            beta0=len(dgm0),
            beta1=beta1,
            p_max=p_max,
        )
