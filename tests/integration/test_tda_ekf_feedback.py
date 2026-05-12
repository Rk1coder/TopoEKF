import numpy as np

from topoekf.detection.detection_result import DetectionResult
from topoekf.pipeline.topoekf_pipeline import TopoEKFPipeline
from topoekf.topology.persistence import TopologyResult
from topoekf.tracking.track_manager import TrackManager
from topoekf.tracking.noise_adapters.tier3_topology import Tier3TopologyAdapter


def test_tda_feedback_changes_tier3_covariance_scaling():
    adapter = Tier3TopologyAdapter(lam=0.4, theta_beta=4.0, eps_max=10.0)
    adapter.update_topology(beta1=2, p_max=5.0)

    Q, R = adapter.adapt(np.eye(4), np.eye(2), context=None)

    np.testing.assert_allclose(Q, np.eye(4) * 1.2)
    np.testing.assert_allclose(R, np.eye(2) * 0.5)


class MovingDetector:
    def __init__(self):
        self.index = 0

    def detect(self, frame):
        x = float(self.index)
        self.index += 1
        return [
            DetectionResult(
                bbox=np.array([x, 0.0, x + 10.0, 10.0]),
                centroid=np.array([x + 5.0, 5.0]),
                confidence=0.9,
                class_id=2,
            )
        ]


class FakeTopoComputer:
    def __init__(self):
        self.calls = 0

    def compute(self, positions):
        self.calls += 1
        return TopologyResult(
            dgm0=np.array([[0.0, 1.0]]),
            dgm1=np.array([[0.1, 4.1]]),
            beta0=1,
            beta1=2,
            p_max=4.0,
        )


class FakeFeatureBuilder:
    def build(self, topo):
        return np.ones(420)


class FakeAnomalyDetector:
    is_fitted = True

    def predict(self, features):
        from topoekf.anomaly.anomaly_result import AnomalyResult

        return AnomalyResult(labels=np.array([-1]), scores=np.array([-0.2]))


def test_pipeline_runs_tda_feedback_and_sets_anomaly_label():
    topo = FakeTopoComputer()
    pipeline = TopoEKFPipeline(
        detector=MovingDetector(),
        track_manager=TrackManager(dt=1.0, init_confidence_threshold=0.1),
        topo_computer=topo,
        feature_builder=FakeFeatureBuilder(),
        anomaly_detector=FakeAnomalyDetector(),
        tda_frequency=1,
        min_tda_length=3,
    )

    result = None
    for _ in range(4):
        result = pipeline.process_frame(np.zeros((20, 20, 3), dtype=np.uint8))

    assert topo.calls >= 1
    assert result["tracks"][0]["anomaly_label"] == "anomaly"
    assert result["tracks"][0]["anomaly_score"] == -0.2
