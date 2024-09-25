import pytest


@pytest.fixture(scope="session")
def rng():
    import numpy as np

    return np.random.default_rng(seed=42)
