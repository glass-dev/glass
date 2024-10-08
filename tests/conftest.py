import numpy as np
import pytest  # type: ignore[import-not-found]


@pytest.fixture(scope="session")  # type: ignore[misc]
def rng() -> np.random.Generator:
    import numpy as np

    return np.random.default_rng(seed=42)
