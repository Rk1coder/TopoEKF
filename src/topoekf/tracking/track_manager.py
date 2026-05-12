import numpy as np

from topoekf.detection.detection_result import DetectionResult

from .association.hungarian import HungarianAssociator
from .association.mahalanobis import mahalanobis_distance
from .ekf_core import EKFCore
from .track import Track


class TrackManager:
    def __init__(
        self,
        dt: float = 0.033,
        max_miss_count: int = 10,
        init_confidence_threshold: float = 0.7,
        association_threshold: float = 9.21,
    ):
        self.ekf = EKFCore(dt=dt)
        self.max_miss_count = max_miss_count
        self.init_confidence_threshold = init_confidence_threshold
        self.associator = HungarianAssociator(max_cost=association_threshold)
        self.active_tracks: list[Track] = []
        self._next_track_id = 1
        self.Q_base = np.diag([1.0, 1.0, 0.01, 0.01])
        self.R_base = np.diag([5.0, 5.0])
        self.P0 = np.diag([10.0, 10.0, 100.0, 100.0])

    def update(self, detections: list[DetectionResult]) -> None:
        for track in self.active_tracks:
            Q, _ = track.adapt_covariances(self.Q_base, self.R_base, track.confidence)
            track.predict(Q)

        matches, unmatched_tracks, unmatched_detections = self._associate(detections)

        for track_idx, detection_idx in matches:
            track = self.active_tracks[track_idx]
            _, R = track.adapt_covariances(
                self.Q_base, self.R_base, detections[detection_idx].confidence
            )
            track.update(detections[detection_idx], R)

        for track_idx in unmatched_tracks:
            self.active_tracks[track_idx].mark_missed()

        for detection_idx in unmatched_detections:
            detection = detections[detection_idx]
            if detection.confidence >= self.init_confidence_threshold:
                self._start_track(detection)

        self.active_tracks = [
            track for track in self.active_tracks if track.miss_count <= self.max_miss_count
        ]

    def _associate(
        self, detections: list[DetectionResult]
    ) -> tuple[list[tuple[int, int]], list[int], list[int]]:
        if not self.active_tracks or not detections:
            return [], list(range(len(self.active_tracks))), list(range(len(detections)))

        cost_matrix = np.zeros((len(self.active_tracks), len(detections)), dtype=float)
        for track_idx, track in enumerate(self.active_tracks):
            z_pred = self.ekf.H @ track.state.x
            S = self.ekf.H @ track.state.P @ self.ekf.H.T + self.R_base
            for detection_idx, detection in enumerate(detections):
                cost_matrix[track_idx, detection_idx] = mahalanobis_distance(
                    detection.centroid, z_pred, S
                )
        return self.associator.associate(cost_matrix)

    def _start_track(self, detection: DetectionResult) -> None:
        self.active_tracks.append(Track(self._next_track_id, detection, self.ekf, self.P0.copy()))
        self._next_track_id += 1

    def get_active_tracks(self) -> dict:
        return {
            "tracks": [
                {
                    "track_id": track.track_id,
                    "state": track.state.x.copy(),
                    "miss_count": track.miss_count,
                    "bbox": track.bbox.copy(),
                    "confidence": track.confidence,
                    "class_id": track.class_id,
                    "anomaly_label": track.anomaly_label,
                    "anomaly_score": track.anomaly_score,
                    "trail": np.array(track.trail, dtype=float),
                }
                for track in self.active_tracks
                if track.miss_count <= self.max_miss_count
            ]
        }
