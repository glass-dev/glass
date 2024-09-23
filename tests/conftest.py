import pytest


@pytest.fixture
def rng():
    import numpy as np

    return np.random.default_rng(seed=42)
