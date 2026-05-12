import numpy as np

from topoekf.anomaly.isolation_forest import IsolationForestDetector


def test_isolation_forest_detector_returns_one_result_per_feature_row():
    detector = IsolationForestDetector(random_state=42, contamination=0.2, n_estimators=10)
    train_features = np.vstack([np.zeros((8, 4)), np.ones((2, 4)) * 10])

    detector.fit(train_features)
    result = detector.predict(np.array([[0.0, 0.0, 0.0, 0.0], [12.0, 12.0, 12.0, 12.0]]))

    assert result.labels.shape == (2,)
    assert result.scores.shape == (2,)
