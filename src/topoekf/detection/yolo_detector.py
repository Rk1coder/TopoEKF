import numpy as np

from .detection_result import DetectionResult
from .interfaces import IDetector


class YOLODetector(IDetector):
    def __init__(
        self,
        model_name: str = "yolo12n.pt",
        confidence_threshold: float = 0.10,
        image_size: int = 960,
        classes: list[int] | None = None,
    ):
        from ultralytics import YOLO

        self.model = YOLO(model_name)
        self.confidence_threshold = confidence_threshold
        self.image_size = image_size
        self.classes = classes

    def detect(self, frame: np.ndarray) -> list[DetectionResult]:
        results = self.model(
            frame,
            verbose=False,
            conf=self.confidence_threshold,
            imgsz=self.image_size,
            classes=self.classes,
        )
        detections: list[DetectionResult] = []
        for result in results:
            boxes = getattr(result, "boxes", None)
            if boxes is None:
                continue
            for box in boxes:
                confidence = float(box.conf[0])
                if confidence < self.confidence_threshold:
                    continue
                bbox = box.xyxy[0].detach().cpu().numpy().astype(float)
                centroid = np.array([(bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0])
                detections.append(
                    DetectionResult(
                        bbox=bbox,
                        centroid=centroid,
                        confidence=confidence,
                        class_id=int(box.cls[0]),
                    )
                )
        return detections
