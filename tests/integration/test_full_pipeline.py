import numpy as np

from topoekf.detection.detection_result import DetectionResult
from topoekf.pipeline.topoekf_pipeline import TopoEKFPipeline
from topoekf.topology.feature_builder import FeatureBuilder
from topoekf.topology.persistence import PersistenceComputer
from topoekf.tracking.track_manager import TrackManager


class FakeDetector:
    def detect(self, frame):
        return [
            DetectionResult(
                bbox=np.array([0.0, 0.0, 10.0, 10.0]),
                centroid=np.array([5.0, 5.0]),
                confidence=0.9,
                class_id=0,
            )
        ]


def test_pipeline_process_frame_returns_active_tracks():
    pipeline = TopoEKFPipeline(
        detector=FakeDetector(),
        track_manager=TrackManager(),
        topo_computer=PersistenceComputer(),
        feature_builder=FeatureBuilder(),
    )

    result = pipeline.process_frame(np.zeros((20, 20, 3), dtype=np.uint8))

    assert len(result["tracks"]) == 1
    assert len(result["detections"]) == 1
    np.testing.assert_allclose(result["detections"][0].bbox, [0.0, 0.0, 10.0, 10.0])
