import argparse

import numpy as np

from topoekf.anomaly.isolation_forest import IsolationForestDetector
from topoekf.pipeline.anomaly_pipeline import AnomalyPipeline
from topoekf.topology.feature_builder import FeatureBuilder
from topoekf.topology.persistence import PersistenceComputer


def main() -> None:
    parser = argparse.ArgumentParser(description="Run anomaly analysis on saved trajectories.")
    parser.add_argument("trajectory_npy")
    args = parser.parse_args()

    trajectories = list(np.load(args.trajectory_npy, allow_pickle=True))
    pipeline = AnomalyPipeline(PersistenceComputer(), FeatureBuilder(), IsolationForestDetector())
    features = pipeline.extract_features(trajectories)
    pipeline.anomaly_detector.fit(features)
    print(pipeline.anomaly_detector.predict(features))


if __name__ == "__main__":
    main()
