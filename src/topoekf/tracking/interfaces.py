from abc import ABC, abstractmethod

import numpy as np

from .state import TrackState
from .noise_adapters.base_adapter import AdaptationContext


class IFilter(ABC):
    @abstractmethod
    def predict(self, state: TrackState, Q: np.ndarray) -> TrackState:
        """Predict the next state."""

    @abstractmethod
    def update(self, state: TrackState, z: np.ndarray, R: np.ndarray) -> TrackState:
        """Update a state with a measurement."""


class INoiseAdapter(ABC):
    @abstractmethod
    def adapt(
        self, Q: np.ndarray, R: np.ndarray, context: AdaptationContext
    ) -> tuple[np.ndarray, np.ndarray]:
        """Adapt process and measurement covariances."""


class IAssociator(ABC):
    @abstractmethod
    def associate(self, cost_matrix: np.ndarray) -> tuple[list[tuple[int, int]], list[int], list[int]]:
        """Return matches, unmatched tracks, and unmatched detections."""
