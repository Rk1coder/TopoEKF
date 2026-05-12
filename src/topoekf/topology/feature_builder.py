import numpy as np

from .persistence import TopologyResult
from .persistence_image import PersistenceImageVectorizer
from .statistical_features import StatisticalFeatureExtractor


class FeatureBuilder:
    def __init__(
        self,
        stat_extractor: StatisticalFeatureExtractor | None = None,
        pi_vectorizer: PersistenceImageVectorizer | None = None,
    ):
        self.stat_extractor = stat_extractor or StatisticalFeatureExtractor()
        self.pi_vectorizer = pi_vectorizer or PersistenceImageVectorizer()

    def build(self, topo: TopologyResult) -> np.ndarray:
        stat_features = self.stat_extractor.extract(topo.beta0, topo.beta1, topo.dgm1)
        pi_features = self.pi_vectorizer.vectorize(topo.dgm1)
        return np.concatenate([stat_features, pi_features])
