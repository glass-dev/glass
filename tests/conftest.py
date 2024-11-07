import numpy as np
import pytest


@pytest.fixture(scope="session")
def rng() -> np.random.Generator:
    import numpy as np

    return np.random.default_rng(seed=42)
