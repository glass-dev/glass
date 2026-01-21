import contextlib
import importlib.util

import numpy as np
import pytest

from glass import _rng

with contextlib.suppress(ImportError):
    # only import if jax is available
    import glass.jax

# check if available for testing
HAVE_ARRAY_API_STRICT = importlib.util.find_spec("array_api_strict") is not None
HAVE_JAX = importlib.util.find_spec("jax") is not None


def test_rng_dispatcher_numpy() -> None:
    rng = _rng.rng_dispatcher(xp=np)
    assert isinstance(rng, np.random.Generator)


@pytest.mark.skipif(not HAVE_JAX, reason="test requires jax")
def test_rng_dispatcher_jax() -> None:
    import jax.numpy as jnp

    rng = _rng.rng_dispatcher(xp=jnp)
    assert isinstance(rng, glass.jax.Generator)


@pytest.mark.skipif(not HAVE_ARRAY_API_STRICT, reason="test requires array_api_strict")
def test_rng_dispatcher_array_api_strict() -> None:
    import array_api_strict

    rng = _rng.rng_dispatcher(xp=array_api_strict)
    assert isinstance(rng, _rng.Generator)


@pytest.mark.skipif(not HAVE_ARRAY_API_STRICT, reason="test requires array_api_strict")
def test_init() -> None:
    rng = _rng.Generator()
    assert isinstance(rng, _rng.Generator)


@pytest.mark.skipif(not HAVE_ARRAY_API_STRICT, reason="test requires array_api_strict")
def test_random() -> None:
    import array_api_strict

    rng = _rng.rng_dispatcher(xp=array_api_strict)
    rvs = rng.random(size=10_000)
    assert rvs.shape == (10_000,)
    assert array_api_strict.min(rvs) >= 0.0
    assert array_api_strict.max(rvs) < 1.0
    assert isinstance(rvs, array_api_strict._array_object.Array)


@pytest.mark.skipif(not HAVE_ARRAY_API_STRICT, reason="test requires array_api_strict")
def test_normal() -> None:
    import array_api_strict

    rng = _rng.rng_dispatcher(xp=array_api_strict)
    rvs = rng.normal(1, 2, size=10_000)
    assert rvs.shape == (10_000,)
    assert isinstance(rvs, array_api_strict._array_object.Array)


@pytest.mark.skipif(not HAVE_ARRAY_API_STRICT, reason="test requires array_api_strict")
def test_standard_normal() -> None:
    import array_api_strict

    rng = _rng.rng_dispatcher(xp=array_api_strict)
    rvs = rng.standard_normal(size=10_000)
    assert rvs.shape == (10_000,)
    assert isinstance(rvs, array_api_strict._array_object.Array)


@pytest.mark.skipif(not HAVE_ARRAY_API_STRICT, reason="test requires array_api_strict")
def test_poisson() -> None:
    import array_api_strict

    rng = _rng.rng_dispatcher(xp=array_api_strict)
    rvs = rng.poisson(lam=1, size=10_000)
    assert rvs.shape == (10_000,)
    assert isinstance(rvs, array_api_strict._array_object.Array)


@pytest.mark.skipif(not HAVE_ARRAY_API_STRICT, reason="test requires array_api_strict")
def test_uniform() -> None:
    import array_api_strict

    rng = _rng.rng_dispatcher(xp=array_api_strict)
    rvs = rng.uniform(size=10_000)
    assert rvs.shape == (10_000,)
    assert array_api_strict.min(rvs) >= 0.0
    assert array_api_strict.max(rvs) < 1.0
    assert isinstance(rvs, array_api_strict._array_object.Array)
