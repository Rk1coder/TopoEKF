import numpy as np

from topoekf.detection.detection_result import DetectionResult
from topoekf.visualization.track_visualizer import TrackVisualizer


def test_draw_detections_draws_bbox_pixels():
    frame = np.zeros((30, 30, 3), dtype=np.uint8)
    detections = [
        DetectionResult(
            bbox=np.array([5.0, 6.0, 20.0, 22.0]),
            centroid=np.array([12.5, 14.0]),
            confidence=0.85,
            class_id=2,
        )
    ]

    output = TrackVisualizer().draw_detections(frame, detections)

    assert output[6, 5].sum() > 0
    assert output[22, 20].sum() > 0
    assert tuple(output[6, 5]) == TrackVisualizer.DETECTION_COLOR


def test_draw_frame_draws_detections_and_tracks():
    frame = np.zeros((30, 30, 3), dtype=np.uint8)
    detections = [
        DetectionResult(
            bbox=np.array([2.0, 3.0, 12.0, 13.0]),
            centroid=np.array([7.0, 8.0]),
            confidence=0.75,
            class_id=1,
        )
    ]
    tracks = [{"track_id": 4, "state": np.array([20.0, 21.0, 0.0, 0.0])}]

    output = TrackVisualizer().draw_frame(frame, detections=detections, tracks=tracks)

    assert output[3, 2].sum() > 0
    assert output[21, 20].sum() > 0


def test_draw_tracks_draws_track_bbox_from_metadata():
    frame = np.zeros((40, 40, 3), dtype=np.uint8)
    tracks = [
        {
            "track_id": 7,
            "state": np.array([20.0, 21.0, 0.0, 0.0]),
            "bbox": np.array([4.0, 5.0, 18.0, 19.0]),
            "confidence": 0.91,
            "class_id": 3,
            "anomaly_label": "unknown",
        }
    ]

    output = TrackVisualizer().draw_tracks(frame, tracks)

    assert output[5, 4].sum() > 0
    assert output[21, 20].sum() > 0
    assert tuple(output[5, 4]) == TrackVisualizer.NORMAL_COLOR


def test_draw_tracks_draws_trajectory_trail():
    frame = np.zeros((40, 40, 3), dtype=np.uint8)
    tracks = [
        {
            "track_id": 7,
            "state": np.array([20.0, 21.0, 0.0, 0.0]),
            "trail": np.array([[5.0, 5.0], [10.0, 10.0], [15.0, 15.0]]),
            "anomaly_label": "normal",
        }
    ]

    output = TrackVisualizer().draw_tracks(frame, tracks)

    assert output[10, 10].sum() > 0


def test_draw_frame_draws_info_panel_text():
    frame = np.zeros((60, 160, 3), dtype=np.uint8)

    output = TrackVisualizer().draw_frame(
        frame,
        detections=[],
        tracks=[],
        frame_index=12,
        fps=24.5,
        anomaly_count=2,
    )

    assert output[:35, :150].sum() > 0


def test_draw_tracks_uses_yellow_for_anomaly_and_red_for_crash():
    frame = np.zeros((80, 80, 3), dtype=np.uint8)
    tracks = [
        {
            "track_id": 1,
            "state": np.array([20.0, 20.0, 0.0, 0.0]),
            "bbox": np.array([5.0, 5.0, 30.0, 30.0]),
            "confidence": 0.9,
            "class_id": 2,
            "anomaly_label": "crash",
        },
        {
            "track_id": 2,
            "state": np.array([60.0, 60.0, 0.0, 0.0]),
            "bbox": np.array([45.0, 45.0, 70.0, 70.0]),
            "confidence": 0.8,
            "class_id": 2,
            "anomaly_label": "anomaly",
        },
    ]

    output = TrackVisualizer().draw_tracks(frame, tracks)

    assert output[5, 5, 2] > output[5, 5, 1]
    assert output[45, 45, 1] > 0 and output[45, 45, 2] > 0
    assert output[45, 45, 0] == 0


def test_normal_tracks_use_one_status_color_instead_of_id_palette():
    visualizer = TrackVisualizer()

    assert visualizer.status_color({"track_id": 4, "anomaly_label": "normal"}, (1, 2, 3)) == visualizer.NORMAL_COLOR
    assert visualizer.status_color({"track_id": 5, "anomaly_label": "unknown"}, (9, 8, 7)) == visualizer.NORMAL_COLOR
