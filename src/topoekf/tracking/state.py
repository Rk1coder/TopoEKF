from dataclasses import dataclass

import numpy as np


@dataclass
class TrackState:
    x: np.ndarray
    P: np.ndarray
    track_id: int | None = None
