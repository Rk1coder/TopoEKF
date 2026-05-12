import numpy as np

from topoekf.anomaly.interfaces import IAnomalyDetector
from topoekf.topology.feature_builder import FeatureBuilder
from topoekf.topology.persistence import PersistenceComputer


class AnomalyPipeline:
    def __init__(
        self,
        topo_computer: PersistenceComputer,
        feature_builder: FeatureBuilder,
        anomaly_detector: IAnomalyDetector,
    ):
        self.topo_computer = topo_computer
        self.feature_builder = feature_builder
        self.anomaly_detector = anomaly_detector

    def extract_features(self, trajectories: list[np.ndarray]) -> np.ndarray:
        return np.vstack(
            [self.feature_builder.build(self.topo_computer.compute(trajectory)) for trajectory in trajectories]
        )
