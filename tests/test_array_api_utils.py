import jax.numpy as jnp
import numpy as np
import pytest

import array_api_strict
from array_api_strict._array_object import Array

import glass._array_api_utils


def test_get_namespace() -> None:
    arrays = [np.array([1, 2]), np.array([2, 3])]
    namespace = glass._array_api_utils.get_namespace(*arrays)
    assert namespace == np

    arrays = [np.array([1, 2]), jnp.array([2, 3])]
    with pytest.raises(ValueError, match="input arrays should"):
        glass._array_api_utils.get_namespace(*arrays)


def test_rng_dispatcher() -> None:
    rng = glass._array_api_utils.rng_dispatcher(np.asarray([1, 2]))
    assert isinstance(rng, np.random.Generator)

    rng = glass._array_api_utils.rng_dispatcher(jnp.asarray([1, 2]))
    assert isinstance(rng, glass.jax.Generator)

    rng = glass._array_api_utils.rng_dispatcher(array_api_strict.asarray([1, 2]))
    assert isinstance(rng, glass._array_api_utils.Generator)


def test_init() -> None:
    rng = glass._array_api_utils.Generator(42)
    assert isinstance(rng, glass._array_api_utils.Generator)


def test_random() -> None:
    rng = glass._array_api_utils.Generator(42)
    rvs = rng.random(size=10_000)
    assert rvs.shape == (10_000,)
    assert array_api_strict.min(rvs) >= 0.0
    assert array_api_strict.max(rvs) < 1.0
    assert isinstance(rvs, Array)


def test_normal() -> None:
    rng = glass._array_api_utils.Generator(42)
    rvs = rng.normal(1, 2, size=10_000)
    assert rvs.shape == (10_000,)
    assert isinstance(rvs, Array)


def test_standard_normal() -> None:
    rng = glass._array_api_utils.Generator(42)
    rvs = rng.standard_normal(size=10_000)
    assert rvs.shape == (10_000,)
    assert isinstance(rvs, Array)


def test_poisson() -> None:
    rng = glass._array_api_utils.Generator(42)
    rvs = rng.poisson(lam=1, size=10_000)
    assert rvs.shape == (10_000,)
    assert isinstance(rvs, Array)


def test_uniform() -> None:
    rng = glass._array_api_utils.Generator(42)
    rvs = rng.uniform(size=10_000)
    assert rvs.shape == (10_000,)
    assert array_api_strict.min(rvs) >= 0.0
    assert array_api_strict.max(rvs) < 1.0
    assert isinstance(rvs, Array)
