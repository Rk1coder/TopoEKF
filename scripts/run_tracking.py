import argparse
from pathlib import Path

import cv2

from topoekf.detection.yolo_detector import YOLODetector
from topoekf.pipeline.topoekf_pipeline import TopoEKFPipeline
from topoekf.topology.feature_builder import FeatureBuilder
from topoekf.topology.persistence import PersistenceComputer
from topoekf.tracking.track_manager import TrackManager
from topoekf.visualization.track_visualizer import TrackVisualizer


DEFAULT_MODEL = "yolo12n.pt"
DEFAULT_VEHICLE_CLASSES = [2, 3, 5, 7]


def parse_class_ids(value: str) -> list[int]:
    if not value.strip():
        return []
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run TopoEKF tracking on a video.")
    parser.add_argument("video_path", type=Path)
    parser.add_argument("--model", default=DEFAULT_MODEL, help="YOLO weights path or Ultralytics model name.")
    parser.add_argument("--confidence", type=float, default=0.10, help="Detection confidence threshold.")
    parser.add_argument("--imgsz", type=int, default=960, help="YOLO inference image size.")
    parser.add_argument(
        "--classes",
        type=parse_class_ids,
        default=DEFAULT_VEHICLE_CLASSES,
        help="Comma-separated YOLO class ids. Default vehicles: 2,3,5,7.",
    )
    parser.add_argument("--debug", action="store_true", help="Print per-frame detection and track counts.")
    parser.add_argument("--output", type=Path, help="Optional path for an annotated output video.")
    args = parser.parse_args(argv)
    if str(args.video_path).lower() == r"path\to\video.mp4":
        parser.error(r"Replace path\to\video.mp4 with a real video file path.")
    return args


def main() -> None:
    args = parse_args()
    if not args.video_path.exists():
        raise FileNotFoundError(f"Video file not found: {args.video_path}")

    pipeline = TopoEKFPipeline(
        detector=YOLODetector(
            model_name=args.model,
            confidence_threshold=args.confidence,
            image_size=args.imgsz,
            classes=args.classes,
        ),
        track_manager=TrackManager(init_confidence_threshold=args.confidence),
        topo_computer=PersistenceComputer(),
        feature_builder=FeatureBuilder(),
    )
    capture = cv2.VideoCapture(str(args.video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video file: {args.video_path}")

    writer = None
    visualizer = TrackVisualizer()
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(args.output), fourcc, fps, (width, height))
        if not writer.isOpened():
            capture.release()
            raise RuntimeError(f"Could not open output video writer: {args.output}")

    frame_count = 0
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        result = pipeline.process_frame(frame)
        frame_count += 1
        anomaly_count = sum(
            1
            for track in result["tracks"]
            if track.get("anomaly_label") in {"anomaly", "crash", "warning"}
        )
        if args.debug:
            print(
                f"frame={frame_count} detections={len(result['detections'])} "
                f"tracks={len(result['tracks'])} anomalies={anomaly_count}"
            )
        if writer is not None:
            annotated = visualizer.draw_frame(
                frame,
                detections=result["detections"],
                tracks=result["tracks"],
                frame_index=frame_count,
                fps=fps,
                anomaly_count=anomaly_count,
            )
            writer.write(annotated)
    capture.release()
    if writer is not None:
        writer.release()
        print(f"Wrote annotated video to {args.output}")
    print(f"Processed {frame_count} frames")


if __name__ == "__main__":
    main()
