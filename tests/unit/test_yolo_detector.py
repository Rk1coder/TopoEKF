import sys
import types

import numpy as np


class FakeTensor:
    def __init__(self, value):
        self.value = value

    def __getitem__(self, index):
        return self.value[index]

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return np.array(self.value)


class FakeBox:
    def __init__(self):
        self.conf = [0.42]
        self.cls = [2]
        self.xyxy = [FakeTensor([1.0, 2.0, 11.0, 22.0])]


class FakeResult:
    boxes = [FakeBox()]


class FakeYOLO:
    last_instance = None

    def __init__(self, model_name):
        self.model_name = model_name
        self.calls = []
        FakeYOLO.last_instance = self

    def __call__(self, frame, verbose=False, conf=None, imgsz=None, classes=None):
        self.calls.append({"verbose": verbose, "conf": conf, "imgsz": imgsz, "classes": classes})
        return [FakeResult()]


def test_yolo_detector_passes_uav_detection_options(monkeypatch):
    fake_ultralytics = types.SimpleNamespace(YOLO=FakeYOLO)
    monkeypatch.setitem(sys.modules, "ultralytics", fake_ultralytics)

    from topoekf.detection.yolo_detector import YOLODetector

    detector = YOLODetector(
        model_name="fake.pt",
        confidence_threshold=0.1,
        image_size=960,
        classes=[2, 3, 5, 7],
    )
    detections = detector.detect(np.zeros((10, 10, 3), dtype=np.uint8))

    assert FakeYOLO.last_instance.calls == [
        {"verbose": False, "conf": 0.1, "imgsz": 960, "classes": [2, 3, 5, 7]}
    ]
    assert len(detections) == 1
    assert detections[0].class_id == 2
