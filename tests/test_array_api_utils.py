import numpy as np
import pytest

import glass._array_api_utils
import jax.numpy as jnp


def test_get_namespace() -> None:
    arrays = [np.array([1, 2]), np.array([2, 3])]
    namespace = glass._array_api_utils.get_namespace(*arrays)
    assert namespace == np

    arrays = [np.array([1, 2]), jnp.array([2, 3])]
    with pytest.raises(ValueError, match="input arrays should"):
        glass._array_api_utils.get_namespace(*arrays)


def test_rng_dispatcher() -> None:
    rng = glass._array_api_utils.rng_dispatcher(np.array([1, 2]))
    assert isinstance(rng, np.random.Generator)

    rng = glass._array_api_utils.rng_dispatcher(jnp.array([1, 2]))
    assert isinstance(rng, glass._array_api_utils.JAXGenerator)
