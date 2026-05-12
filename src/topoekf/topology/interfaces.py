from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from .persistence import TopologyResult


class ITopologyComputer(ABC):
    @abstractmethod
    def compute(self, positions: np.ndarray) -> TopologyResult:
        """Compute persistent homology for trajectory positions."""


class IVectorizer(ABC):
    @abstractmethod
    def vectorize(self, diagram: Any) -> np.ndarray:
        """Convert a persistence diagram to a numeric vector."""
