import numpy as np

from topoekf.topology.feature_builder import FeatureBuilder
from topoekf.topology.persistence import TopologyResult
from topoekf.topology.persistence_image import PersistenceImageVectorizer
from topoekf.topology.statistical_features import StatisticalFeatureExtractor


def test_statistical_features_are_always_twenty_dimensional():
    extractor = StatisticalFeatureExtractor()
    dgm1 = np.array([[0.1, 0.6], [0.2, 0.9]])

    features = extractor.extract(beta0=3, beta1=2, dgm1=dgm1)

    assert features.shape == (20,)
    np.testing.assert_allclose(features[:2], [3.0, 2.0])


def test_persistence_image_vectorizer_returns_flat_grid_for_empty_diagram():
    vectorizer = PersistenceImageVectorizer(grid_size=20)

    image = vectorizer.vectorize(np.empty((0, 2)))

    assert image.shape == (400,)
    assert np.all(image == 0.0)


def test_feature_builder_returns_420_dimensions():
    topo = TopologyResult(
        dgm0=np.array([[0.0, 1.0]]),
        dgm1=np.array([[0.1, 0.6], [0.2, 0.9]]),
        beta0=1,
        beta1=2,
        p_max=0.7,
    )

    features = FeatureBuilder().build(topo)

    assert features.shape == (420,)
