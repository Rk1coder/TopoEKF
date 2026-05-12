import numpy as np

from topoekf.detection.detection_result import DetectionResult
from topoekf.tracking.track_manager import TrackManager


def make_detection(x: float, y: float, confidence: float = 0.9) -> DetectionResult:
    return DetectionResult(
        bbox=np.array([x - 5.0, y - 5.0, x + 5.0, y + 5.0]),
        centroid=np.array([x, y]),
        confidence=confidence,
        class_id=1,
    )


def test_update_keeps_stable_track_id_for_nearby_detection():
    manager = TrackManager(dt=1.0, init_confidence_threshold=0.1)
    manager.update([make_detection(10.0, 10.0)])
    first = manager.get_active_tracks()["tracks"]

    manager.update([make_detection(11.0, 10.5)])
    second = manager.get_active_tracks()["tracks"]

    assert len(second) == 1
    assert second[0]["track_id"] == first[0]["track_id"]
    assert second[0]["confidence"] == 0.9
    np.testing.assert_allclose(second[0]["bbox"], [6.0, 5.5, 16.0, 15.5])
    assert len(second[0]["trail"]) >= 2
    assert second[0]["anomaly_score"] is None


def test_update_creates_new_track_for_far_unmatched_detection():
    manager = TrackManager(dt=1.0, init_confidence_threshold=0.1)
    manager.update([make_detection(10.0, 10.0)])

    manager.update([make_detection(200.0, 200.0)])
    tracks = manager.get_active_tracks()["tracks"]

    assert [track["track_id"] for track in tracks] == [1, 2]
    assert tracks[0]["miss_count"] == 1


def test_update_removes_track_after_max_misses():
    manager = TrackManager(dt=1.0, max_miss_count=1, init_confidence_threshold=0.1)
    manager.update([make_detection(10.0, 10.0)])

    manager.update([])
    manager.update([])

    assert manager.get_active_tracks()["tracks"] == []


def test_matched_update_uses_three_tier_adaptive_covariances():
    manager = TrackManager(dt=1.0, init_confidence_threshold=0.1)
    manager.update([make_detection(10.0, 10.0, confidence=0.9)])
    track = manager.active_tracks[0]
    track.tier3_adapter.update_topology(beta1=3, p_max=2.0)

    manager.update([make_detection(11.0, 10.5, confidence=0.5)])

    assert track.last_Q is not None
    assert track.last_R is not None
    assert np.trace(track.last_Q) > np.trace(manager.Q_base)
    assert not np.allclose(track.last_R, manager.R_base)
    assert track.anomaly_label == "unknown"
