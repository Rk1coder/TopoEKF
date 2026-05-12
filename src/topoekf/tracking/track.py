from collections import deque

import numpy as np

from .ekf_core import EKFCore, EKFState
from .noise_adapters.base_adapter import AdaptationContext
from .noise_adapters.tier1_confidence import Tier1ConfidenceAdapter
from .noise_adapters.tier2_occlusion import Tier2OcclusionAdapter
from .noise_adapters.tier3_topology import Tier3TopologyAdapter


class Track:
    def __init__(
        self,
        track_id: int,
        initial_detection,
        ekf: EKFCore,
        P0: np.ndarray,
        buffer_size: int = 50,
    ):
        self.track_id = track_id
        self.ekf = ekf
        initial_position = initial_detection.centroid
        self.state = EKFState(x=np.array([initial_position[0], initial_position[1], 0.0, 0.0]), P=P0)
        self.miss_count = 0
        self.age = 1
        self.position_buffer = deque([initial_position.astype(float)], maxlen=buffer_size)
        self.trail = deque([initial_position.astype(float)], maxlen=buffer_size)
        self.tier1_adapter = Tier1ConfidenceAdapter()
        self.tier2_adapter = Tier2OcclusionAdapter()
        self.tier3_adapter = Tier3TopologyAdapter()
        self.last_Q: np.ndarray | None = None
        self.last_R: np.ndarray | None = None
        self.bbox = initial_detection.bbox.astype(float)
        self.confidence = float(initial_detection.confidence)
        self.class_id = int(initial_detection.class_id)
        self.anomaly_label = "unknown"
        self.anomaly_score: float | None = None

    def adapt_covariances(
        self, Q_base: np.ndarray, R_base: np.ndarray, confidence: float
    ) -> tuple[np.ndarray, np.ndarray]:
        context = self.adaptation_context(confidence=confidence)
        Q, R = self.tier1_adapter.adapt(Q_base.copy(), R_base.copy(), context)
        Q, R = self.tier2_adapter.adapt(Q, R, context)
        Q, R = self.tier3_adapter.adapt(Q, R, context)
        self.last_Q = Q.copy()
        self.last_R = R.copy()
        return Q, R

    def predict(self, Q: np.ndarray) -> EKFState:
        self.state = self.ekf.predict(self.state, Q)
        return self.state

    def update(self, detection, R: np.ndarray) -> EKFState:
        self.state = self.ekf.update(self.state, detection.centroid, R)
        self.miss_count = 0
        self.age += 1
        self.position_buffer.append(self.state.x[:2].copy())
        self.trail.append(self.state.x[:2].copy())
        self.bbox = detection.bbox.astype(float)
        self.confidence = float(detection.confidence)
        self.class_id = int(detection.class_id)
        return self.state

    def mark_missed(self) -> None:
        self.miss_count += 1
        self.age += 1
        self.trail.append(self.state.x[:2].copy())

    def adaptation_context(self, confidence: float = 1.0) -> AdaptationContext:
        return AdaptationContext(
            confidence=confidence,
            miss_count=self.miss_count,
            velocity=self.state.x[2:4].copy(),
        )
