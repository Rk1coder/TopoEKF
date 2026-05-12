import numpy as np

from topoekf.anomaly.interfaces import IAnomalyDetector
from topoekf.detection.interfaces import IDetector
from topoekf.topology.feature_builder import FeatureBuilder
from topoekf.topology.persistence import PersistenceComputer
from topoekf.tracking.track_manager import TrackManager


class TopoEKFPipeline:
    def __init__(
        self,
        detector: IDetector,
        track_manager: TrackManager,
        topo_computer: PersistenceComputer,
        feature_builder: FeatureBuilder,
        anomaly_detector: IAnomalyDetector | None = None,
        tda_frequency: int = 5,
        min_tda_length: int = 30,
    ):
        self.detector = detector
        self.track_manager = track_manager
        self.topo_computer = topo_computer
        self.feature_builder = feature_builder
        self.anomaly_detector = anomaly_detector
        self.tda_frequency = tda_frequency
        self.min_tda_length = min_tda_length
        self._frame_count = 0

    def process_frame(self, frame: np.ndarray) -> dict:
        detections = self.detector.detect(frame)
        self.track_manager.update(detections)
        if self._frame_count % self.tda_frequency == 0:
            self._run_tda_feedback()
        self._frame_count += 1
        result = self.track_manager.get_active_tracks()
        result["detections"] = detections
        return result

    def _run_tda_feedback(self) -> None:
        for track in self.track_manager.active_tracks:
            if len(track.position_buffer) < self.min_tda_length:
                continue
            positions = np.array(track.position_buffer)[-50:]
            topo = self.topo_computer.compute(positions)
            features = self.feature_builder.build(topo)
            track.tier3_adapter.update_topology(topo.beta1, topo.p_max)
            self._update_anomaly(track, features)

    def _update_anomaly(self, track, features: np.ndarray) -> None:
        if self.anomaly_detector is None:
            return
        if not getattr(self.anomaly_detector, "is_fitted", False):
            return
        result = self.anomaly_detector.predict(features.reshape(1, -1))
        label = int(result.labels[0])
        score = float(result.scores[0])
        track.anomaly_label = "anomaly" if label == -1 else "normal"
        track.anomaly_score = score
