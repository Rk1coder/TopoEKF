import numpy as np


def mahalanobis_distance(z: np.ndarray, z_pred: np.ndarray, S: np.ndarray) -> float:
    innovation = z - z_pred
    solved = np.linalg.solve(S, innovation)
    return float(np.sqrt(innovation.T @ solved))


def gating_threshold(
    trace_s: float,
    trace_high_threshold: float = 50.0,
    trace_low_threshold: float = 10.0,
) -> float:
    if trace_s > trace_high_threshold:
        return 5.99
    if trace_s < trace_low_threshold:
        return 13.82
    return 9.21
