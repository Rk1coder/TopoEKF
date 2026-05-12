import numpy as np

from topoekf.tracking.noise_adapters.base_adapter import AdaptationContext
from topoekf.tracking.noise_adapters.tier1_confidence import Tier1ConfidenceAdapter
from topoekf.tracking.noise_adapters.tier2_occlusion import Tier2OcclusionAdapter
from topoekf.tracking.noise_adapters.tier3_topology import Tier3TopologyAdapter


def test_tier1_scales_measurement_noise_from_confidence_ema():
    adapter = Tier1ConfidenceAdapter(alpha=0.5)
    ctx = AdaptationContext(confidence=0.5)

    Q, R = adapter.adapt(np.eye(4), np.eye(2), ctx)

    np.testing.assert_allclose(Q, np.eye(4))
    np.testing.assert_allclose(R, np.eye(2) * 1.25)


def test_tier2_scales_process_noise_from_occlusion_velocity_and_scene():
    adapter = Tier2OcclusionAdapter(lambda_v=0.05, scene_complexity=1.3)
    ctx = AdaptationContext(miss_count=3, velocity=np.array([3.0, 4.0]))

    Q, R = adapter.adapt(np.eye(4), np.eye(2), ctx)

    np.testing.assert_allclose(Q, np.eye(4) * 2.6)
    np.testing.assert_allclose(R, np.eye(2))


def test_tier3_uses_latest_topology_feedback_for_both_covariances():
    adapter = Tier3TopologyAdapter(lam=0.4, theta_beta=2.0, eps_max=10.0)
    adapter.update_topology(beta1=3, p_max=2.5)

    Q, R = adapter.adapt(np.eye(4), np.eye(2), AdaptationContext())

    np.testing.assert_allclose(Q, np.eye(4) * 1.6)
    np.testing.assert_allclose(R, np.eye(2) * 0.75)
