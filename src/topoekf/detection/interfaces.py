from abc import ABC, abstractmethod

import numpy as np

from .detection_result import DetectionResult


class IDetector(ABC):
    @abstractmethod
    def detect(self, frame: np.ndarray) -> list[DetectionResult]:
        """Return detections for a single frame."""
