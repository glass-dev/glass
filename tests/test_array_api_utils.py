import jax.numpy as jnp
import numpy as np
import pytest

import glass._array_api_utils


def test_get_namespace() -> None:
    arrays = [np.array([1, 2]), np.array([2, 3])]
    namespace = glass._array_api_utils.get_namespace(*arrays)
    assert namespace == np

    arrays = [np.array([1, 2]), jnp.array([2, 3])]
    with pytest.raises(ValueError, match="input arrays should"):
        glass._array_api_utils.get_namespace(*arrays)
