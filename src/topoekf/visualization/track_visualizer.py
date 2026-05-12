import cv2
import numpy as np

from topoekf.detection.detection_result import DetectionResult


class TrackVisualizer:
    NORMAL_COLOR = (0, 255, 0)
    ANOMALY_COLOR = (0, 255, 255)
    CRASH_COLOR = (0, 0, 255)
    DETECTION_COLOR = (180, 180, 180)

    def draw_frame(
        self,
        frame: np.ndarray,
        detections: list[DetectionResult],
        tracks: list[dict],
        frame_index: int | None = None,
        fps: float | None = None,
        anomaly_count: int = 0,
    ) -> np.ndarray:
        output = self.draw_detections(frame, detections)
        output = self.draw_tracks(output, tracks)
        if frame_index is not None:
            self.draw_info_panel(output, frame_index, len(detections), len(tracks), anomaly_count, fps)
        return output

    def draw_detections(self, frame: np.ndarray, detections: list[DetectionResult]) -> np.ndarray:
        output = frame.copy()
        for detection in detections:
            x1, y1, x2, y2 = detection.bbox.astype(int)
            cv2.rectangle(output, (x1, y1), (x2, y2), self.DETECTION_COLOR, 1)
            label = f"cls {detection.class_id} {detection.confidence:.2f}"
            cv2.putText(
                output,
                label,
                (x1, max(y1 - 6, 12)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                self.DETECTION_COLOR,
                1,
                cv2.LINE_AA,
            )
        return output

    def draw_tracks(self, frame: np.ndarray, tracks: list[dict]) -> np.ndarray:
        output = frame.copy()
        for track in tracks:
            x, y = track["state"][:2]
            color = self.status_color(track, self.NORMAL_COLOR)
            thickness = 3 if track.get("anomaly_label") in {"anomaly", "crash", "collision", "warning"} else 2
            self._draw_trail(output, track, color)
            if "bbox" in track:
                x1, y1, x2, y2 = track["bbox"].astype(int)
                cv2.rectangle(output, (x1, y1), (x2, y2), color, thickness)
                label = self._track_label(track)
                cv2.putText(
                    output,
                    label,
                    (x1, max(y1 - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    color,
                    2,
                    cv2.LINE_AA,
                )
            cv2.circle(output, (int(x), int(y)), 4, color, -1)
        return output

    def draw_info_panel(
        self,
        output: np.ndarray,
        frame_index: int,
        detection_count: int,
        track_count: int,
        anomaly_count: int,
        fps: float | None = None,
    ) -> None:
        panel_width = min(output.shape[1], 360)
        panel_height = 58
        overlay = output.copy()
        cv2.rectangle(overlay, (0, 0), (panel_width, panel_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.45, output, 0.55, 0, output)
        fps_text = f" | FPS:{fps:.1f}" if fps is not None else ""
        lines = [
            f"Frame:{frame_index}{fps_text}",
            f"Detections:{detection_count} | Tracks:{track_count} | Anomalies:{anomaly_count}",
        ]
        for idx, line in enumerate(lines):
            cv2.putText(
                output,
                line,
                (10, 22 + idx * 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.58,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

    def track_color(self, track_id: int) -> tuple[int, int, int]:
        hue = (track_id * 37) % 180
        hsv = np.uint8([[[hue, 220, 255]]])
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
        return tuple(int(value) for value in bgr)

    def status_color(self, track: dict, base_color: tuple[int, int, int]) -> tuple[int, int, int]:
        label = track.get("anomaly_label")
        if label in {"crash", "collision"}:
            return self.CRASH_COLOR
        if label in {"anomaly", "warning", "deceleration", "erratic"}:
            return self.ANOMALY_COLOR
        return self.NORMAL_COLOR

    def _track_label(self, track: dict) -> str:
        label = f"ID:{track['track_id']}"
        if "class_id" in track and "confidence" in track:
            label += f" cls:{track['class_id']} conf:{track['confidence']:.2f}"
        anomaly_label = track.get("anomaly_label")
        if anomaly_label and anomaly_label != "unknown":
            label += f" [{anomaly_label.upper()}]"
        return label

    def _draw_trail(self, output: np.ndarray, track: dict, color: tuple[int, int, int]) -> None:
        trail = track.get("trail")
        if trail is None or len(trail) < 2:
            return
        points = np.asarray(trail, dtype=int)
        if len(points) > 75:
            points = points[-75:]
        for start, end in zip(points[:-1], points[1:]):
            cv2.line(output, tuple(start), tuple(end), color, 2, cv2.LINE_AA)
