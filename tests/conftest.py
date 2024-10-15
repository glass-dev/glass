import pytest


@pytest.fixture(scope="session")
def rng():  # type: ignore[no-untyped-def]
    import numpy as np

    return np.random.default_rng(seed=42)
