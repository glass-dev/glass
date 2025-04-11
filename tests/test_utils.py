import numpy as np
import pytest

import glass._utils

jax = pytest.importorskip("jax")


def test_get_namespace() -> None:
    arrays = [np.array([1, 2]), np.array([2, 3])]
    namespace = glass._utils.get_namespace(*arrays)
    assert namespace == np

    arrays = [np.array([1, 2]), jax.numpy.array([2, 3])]
    with pytest.raises(ValueError, match="input arrays should"):
        glass._utils.get_namespace(*arrays)
