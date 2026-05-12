import numpy as np
from scipy.stats import entropy as scipy_entropy


class StatisticalFeatureExtractor:
    def extract(self, beta0: int, beta1: int, dgm1: np.ndarray) -> np.ndarray:
        features: list[float] = [float(beta0), float(beta1)]

        if len(dgm1) == 0:
            features.extend([0.0] * 18)
        else:
            lifetimes = dgm1[:, 1] - dgm1[:, 0]
            births = dgm1[:, 0]
            deaths = dgm1[:, 1]
            features.extend(
                [
                    float(np.mean(lifetimes)),
                    float(np.std(lifetimes)),
                    float(np.max(lifetimes)),
                    float(np.sum(lifetimes)),
                    float(np.mean(births)),
                    float(np.median(births)),
                    float(np.percentile(births, 25)),
                    float(np.percentile(births, 75)),
                    float(np.mean(deaths)),
                    float(np.median(deaths)),
                    float(np.percentile(deaths, 25)),
                    float(np.percentile(deaths, 75)),
                    float(scipy_entropy(lifetimes + 1e-10)),
                    float(np.sum(lifetimes > np.mean(lifetimes))),
                    float(np.min(lifetimes)),
                    float(np.var(lifetimes)),
                    float(np.argmax(lifetimes)),
                    float(len(lifetimes)),
                ]
            )

        return np.array(features[:20], dtype=float)
