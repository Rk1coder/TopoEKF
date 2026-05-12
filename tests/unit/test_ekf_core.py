import numpy as np

from topoekf.tracking.ekf_core import EKFCore, EKFState


def test_predict_advances_position_with_constant_velocity():
    ekf = EKFCore(dt=1.0)
    state = EKFState(x=np.array([0.0, 0.0, 10.0, 5.0]), P=np.eye(4))

    predicted = ekf.predict(state, Q=np.zeros((4, 4)))

    np.testing.assert_allclose(predicted.x[:2], [10.0, 5.0])


def test_predict_increases_uncertainty_when_process_noise_is_added():
    ekf = EKFCore(dt=0.033)
    state = EKFState(x=np.array([100.0, 200.0, 5.0, -3.0]), P=np.eye(4) * 10)

    predicted = ekf.predict(state, Q=np.eye(4) * 0.01)

    assert np.trace(predicted.P) > np.trace(state.P)


def test_update_moves_state_toward_measurement_and_reduces_uncertainty():
    ekf = EKFCore(dt=0.033)
    state = EKFState(x=np.array([100.0, 200.0, 0.0, 0.0]), P=np.eye(4) * 100)

    updated = ekf.update(state, z=np.array([105.0, 198.0]), R=np.eye(2) * 5)

    assert updated.x[0] > state.x[0]
    assert updated.x[1] < state.x[1]
    assert np.trace(updated.P) < np.trace(state.P)
