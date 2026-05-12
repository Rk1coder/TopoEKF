from pathlib import Path

import pytest

from scripts.run_tracking import parse_args


def test_parse_args_accepts_model_override():
    args = parse_args(
        [
            "video.mp4",
            "--model",
            "custom.pt",
            "--confidence",
            "0.4",
            "--imgsz",
            "1280",
            "--classes",
            "2,5,7",
            "--debug",
            "--output",
            "data/results/out.mp4",
        ]
    )

    assert args.video_path == Path("video.mp4")
    assert args.model == "custom.pt"
    assert args.confidence == 0.4
    assert args.imgsz == 1280
    assert args.classes == [2, 5, 7]
    assert args.debug is True
    assert args.output == Path("data/results/out.mp4")


def test_parse_args_uses_uav_vehicle_detection_defaults():
    args = parse_args(["video.mp4"])

    assert args.model == "yolo12n.pt"
    assert args.confidence == 0.10
    assert args.imgsz == 960
    assert args.classes == [2, 3, 5, 7]
    assert args.debug is False


def test_parse_args_rejects_placeholder_video_path():
    with pytest.raises(SystemExit):
        parse_args([r"path\to\video.mp4"])
