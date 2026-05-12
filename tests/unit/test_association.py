import numpy as np

from topoekf.tracking.association.hungarian import HungarianAssociator
from topoekf.tracking.association.mahalanobis import mahalanobis_distance, gating_threshold


def test_mahalanobis_distance_uses_innovation_covariance():
    z = np.array([3.0, 4.0])
    z_pred = np.array([0.0, 0.0])
    S = np.eye(2)

    assert mahalanobis_distance(z, z_pred, S) == 5.0


def test_gating_threshold_tightens_for_high_uncertainty_and_relaxes_for_low():
    assert gating_threshold(trace_s=60.0) == 5.99
    assert gating_threshold(trace_s=5.0) == 13.82
    assert gating_threshold(trace_s=25.0) == 9.21


def test_hungarian_associator_returns_matches_and_unmatched_indices():
    associator = HungarianAssociator(max_cost=5.0)
    cost_matrix = np.array([[1.0, 9.0], [8.0, 2.0], [7.0, 6.0]])

    matches, unmatched_tracks, unmatched_detections = associator.associate(cost_matrix)

    assert matches == [(0, 0), (1, 1)]
    assert unmatched_tracks == [2]
    assert unmatched_detections == []
