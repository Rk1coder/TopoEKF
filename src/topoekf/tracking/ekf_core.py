from dataclasses import dataclass

import numpy as np


@dataclass
class EKFState:
    x: np.ndarray
    P: np.ndarray


class EKFCore:
    def __init__(self, dt: float):
        self.dt = dt
        self.F = np.array(
            [[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]],
            dtype=float,
        )
        self.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=float)

    def predict(self, state: EKFState, Q: np.ndarray) -> EKFState:
        x_pred = self.F @ state.x
        P_pred = self.F @ state.P @ self.F.T + Q
        return EKFState(x=x_pred, P=P_pred)

    def update(self, state: EKFState, z: np.ndarray, R: np.ndarray) -> EKFState:
        y = z - self.H @ state.x
        S = self.H @ state.P @ self.H.T + R
        gain_t = np.linalg.solve(S, self.H @ state.P).T
        x_upd = state.x + gain_t @ y
        P_upd = (np.eye(4) - gain_t @ self.H) @ state.P
        return EKFState(x=x_upd, P=P_upd)
