import contextlib
import importlib.util

import numpy as np
import pytest

import glass.rng

with contextlib.suppress(ImportError):
    # only import if jax is available
    import glass.jax

# check if available for testing
HAVE_ARRAY_API_STRICT = importlib.util.find_spec("array_api_strict") is not None
HAVE_JAX = importlib.util.find_spec("jax") is not None


def test_default_rng_numpy() -> None:
    rng = glass.rng.default_rng(xp=np)
    assert isinstance(rng, np.random.Generator)


@pytest.mark.skipif(not HAVE_JAX, reason="test requires jax")
def test_default_rng_jax() -> None:
    import jax.numpy as jnp

    rng = glass.rng.default_rng(xp=jnp)
    assert isinstance(rng, glass.jax.Generator)


@pytest.mark.skipif(not HAVE_ARRAY_API_STRICT, reason="test requires array_api_strict")
def test_default_rng_array_api_strict() -> None:
    import array_api_strict

    rng = glass.rng.default_rng(xp=array_api_strict)
    assert isinstance(rng, glass.rng.Generator)


@pytest.mark.skipif(not HAVE_ARRAY_API_STRICT, reason="test requires array_api_strict")
def test_init() -> None:
    import array_api_strict

    rng = glass.rng.Generator(xp=array_api_strict)
    assert isinstance(rng, glass.rng.Generator)


@pytest.mark.skipif(not HAVE_ARRAY_API_STRICT, reason="test requires array_api_strict")
def test_init_mix_of_backends_np_array_api_strict() -> None:
    import array_api_strict as xp

    rng = glass.rng.Generator(rng=np.random.default_rng(), xp=xp)
    assert rng.random(1).__array_namespace__().__name__ == "array_api_strict"
    assert rng.poisson(1).__array_namespace__().__name__ == "array_api_strict"
    assert rng.standard_normal(1).__array_namespace__().__name__ == "array_api_strict"
    assert rng.uniform().__array_namespace__().__name__ == "array_api_strict"
    assert (
        rng.multinomial(1, xp.ones(2)).__array_namespace__().__name__
        == "array_api_strict"
    )


@pytest.mark.skipif(not HAVE_JAX, reason="test requires jax")
def test_init_mix_of_backends_jax_np() -> None:
    rng = glass.rng.Generator(rng=glass.jax.Generator(42), xp=np)
    assert rng.random(1).__array_namespace__().__name__ == "numpy"
    assert rng.poisson(1).__array_namespace__().__name__ == "numpy"
    assert rng.standard_normal(1).__array_namespace__().__name__ == "numpy"
    assert rng.uniform().__array_namespace__().__name__ == "numpy"
    assert rng.multinomial(1, np.ones(2)).__array_namespace__().__name__ == "numpy"


@pytest.mark.skipif(not HAVE_ARRAY_API_STRICT, reason="test requires array_api_strict")
def test_random() -> None:
    import array_api_strict

    rng = glass.rng.default_rng(xp=array_api_strict)
    rvs = rng.random(size=10_000)
    assert rvs.shape == (10_000,)
    assert array_api_strict.min(rvs) >= 0.0
    assert array_api_strict.max(rvs) < 1.0
    assert isinstance(rvs, array_api_strict._array_object.Array)


@pytest.mark.skipif(not HAVE_ARRAY_API_STRICT, reason="test requires array_api_strict")
def test_normal() -> None:
    import array_api_strict

    rng = glass.rng.default_rng(xp=array_api_strict)
    rvs = rng.normal(1, 2, size=10_000)
    assert rvs.shape == (10_000,)
    assert isinstance(rvs, array_api_strict._array_object.Array)


@pytest.mark.skipif(not HAVE_ARRAY_API_STRICT, reason="test requires array_api_strict")
def test_standard_normal() -> None:
    import array_api_strict

    rng = glass.rng.default_rng(xp=array_api_strict)
    rvs = rng.standard_normal(size=10_000)
    assert rvs.shape == (10_000,)
    assert isinstance(rvs, array_api_strict._array_object.Array)


@pytest.mark.skipif(not HAVE_ARRAY_API_STRICT, reason="test requires array_api_strict")
def test_poisson() -> None:
    import array_api_strict

    rng = glass.rng.default_rng(xp=array_api_strict)
    rvs = rng.poisson(lam=1, size=10_000)
    assert rvs.shape == (10_000,)
    assert isinstance(rvs, array_api_strict._array_object.Array)


@pytest.mark.skipif(not HAVE_ARRAY_API_STRICT, reason="test requires array_api_strict")
def test_uniform() -> None:
    import array_api_strict

    rng = glass.rng.default_rng(xp=array_api_strict)
    rvs = rng.uniform(size=10_000)
    assert rvs.shape == (10_000,)
    assert array_api_strict.min(rvs) >= 0.0
    assert array_api_strict.max(rvs) < 1.0
    assert isinstance(rvs, array_api_strict._array_object.Array)
