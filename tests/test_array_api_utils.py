import importlib

import numpy as np

import glass._array_api_utils

# check if available for testing
HAVE_ARRAY_API_STRICT = importlib.util.find_spec("array_api_strict") is not None
HAVE_JAX = importlib.util.find_spec("jax") is not None


def test_rng_dispatcher_numpy() -> None:
    rng = glass._array_api_utils.rng_dispatcher(np.asarray([1, 2]))
    assert isinstance(rng, np.random.Generator)


@pytest.mark.skipif(not HAVE_ARRAY_API_STRICT, reason="test requires array_api_strict")
def test_rng_dispatcher_array_api_strict() -> None:
    import array_api_strict

    rng = glass._array_api_utils.rng_dispatcher(array_api_strict.asarray([1, 2]))
    assert isinstance(rng, glass._array_api_utils.Generator)


@pytest.mark.skipif(not HAVE_JAX, reason="test requires jax")
def test_rng_dispatcher_jax() -> None:
    import jax.numpy as jnp

    rng = glass._array_api_utils.rng_dispatcher(jnp.asarray([1, 2]))
    assert isinstance(rng, glass.jax.Generator)


@pytest.mark.skipif(not HAVE_ARRAY_API_STRICT, reason="test requires array_api_strict")
def test_init() -> None:
    rng = glass._array_api_utils.Generator(42)
    assert isinstance(rng, glass._array_api_utils.Generator)


@pytest.mark.skipif(not HAVE_ARRAY_API_STRICT, reason="test requires array_api_strict")
def test_random() -> None:
    import array_api_strict
    from array_api_strict._array_object import Array

    rng = glass._array_api_utils.Generator(42)
    rvs = rng.random(size=10_000)
    assert rvs.shape == (10_000,)
    assert array_api_strict.min(rvs) >= 0.0
    assert array_api_strict.max(rvs) < 1.0
    assert isinstance(rvs, Array)


@pytest.mark.skipif(not HAVE_ARRAY_API_STRICT, reason="test requires array_api_strict")
def test_normal() -> None:
    from array_api_strict._array_object import Array

    rng = glass._array_api_utils.Generator(42)
    rvs = rng.normal(1, 2, size=10_000)
    assert rvs.shape == (10_000,)
    assert isinstance(rvs, Array)


@pytest.mark.skipif(not HAVE_ARRAY_API_STRICT, reason="test requires array_api_strict")
def test_standard_normal() -> None:
    from array_api_strict._array_object import Array

    rng = glass._array_api_utils.Generator(42)
    rvs = rng.standard_normal(size=10_000)
    assert rvs.shape == (10_000,)
    assert isinstance(rvs, Array)


@pytest.mark.skipif(not HAVE_ARRAY_API_STRICT, reason="test requires array_api_strict")
def test_poisson() -> None:
    from array_api_strict._array_object import Array

    rng = glass._array_api_utils.Generator(42)
    rvs = rng.poisson(lam=1, size=10_000)
    assert rvs.shape == (10_000,)
    assert isinstance(rvs, Array)


@pytest.mark.skipif(not HAVE_ARRAY_API_STRICT, reason="test requires array_api_strict")
def test_uniform() -> None:
    import array_api_strict
    from array_api_strict._array_object import Array

    rng = glass._array_api_utils.Generator(42)
    rvs = rng.uniform(size=10_000)
    assert rvs.shape == (10_000,)
    assert array_api_strict.min(rvs) >= 0.0
    assert array_api_strict.max(rvs) < 1.0
    assert isinstance(rvs, Array)
