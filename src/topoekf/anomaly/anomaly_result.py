from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class AnomalyResult:
    labels: np.ndarray
    scores: np.ndarray
