from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np


@dataclass
class AdaptationContext:
    confidence: float = 1.0
    miss_count: int = 0
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(2))
    object_count: int = 0


class INoiseAdapter(ABC):
    @abstractmethod
    def adapt(
        self, Q: np.ndarray, R: np.ndarray, context: AdaptationContext
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return adapted Q and R matrices."""
