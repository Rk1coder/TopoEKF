from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DetectionResult:
    bbox: np.ndarray
    centroid: np.ndarray
    confidence: float
    class_id: int
