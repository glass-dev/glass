import pytest

jax = pytest.importorskip("jax")

import jax.numpy as jnp  # noqa: E402
from jax.typing import ArrayLike  # noqa: E402

from glass.jax import Generator  # noqa: E402


def test_init() -> None:
    rng = Generator(42)
    assert isinstance(rng, Generator)
    assert isinstance(rng.key, jax.Array)
    assert jax.dtypes.issubdtype(rng.key.dtype, jax.dtypes.prng_key)
    assert jnp.all(rng.key == jax.random.key(42))


def test_from_key() -> None:
    key = jax.random.key(42)
    rng = Generator.from_key(key)
    assert rng.key is key

    with pytest.raises(ValueError, match="not a random key"):
        Generator.from_key(object())

    with pytest.raises(ValueError, match="not a random key"):
        Generator.from_key(jnp.zeros(()))


def test_key() -> None:
    rng = Generator(42)
    rngkey, outkey = jax.random.split(rng.key, 2)
    key = rng.split()
    assert jnp.all(rng.key == rngkey)
    assert jnp.all(key == outkey)


def test_spawn() -> None:
    rng = Generator(42)
    key, *subkeys = jax.random.split(rng.key, 4)
    subrngs = rng.spawn(3)
    assert rng.key == key
    assert isinstance(subrngs, list)
    assert len(subrngs) == 3
    for subrng, subkey in zip(subrngs, subkeys, strict=False):
        assert isinstance(subrng, Generator)
        assert subrng.key == subkey


def test_random() -> None:
    rng = Generator(42)
    key = rng.key
    rvs = rng.random(size=10_000)
    assert rng.key != key
    assert rvs.shape == (10_000,)
    assert jnp.min(rvs) >= 0.0
    assert jnp.max(rvs) < 1.0
    assert isinstance(rvs, ArrayLike)


def test_normal() -> None:
    rng = Generator(42)
    key = rng.key
    rvs = rng.normal(1, 2, size=10_000)
    assert rng.key != key
    assert rvs.shape == (10_000,)
    assert isinstance(rvs, ArrayLike)


def test_standard_normal() -> None:
    rng = Generator(42)
    key = rng.key
    rvs = rng.standard_normal(size=10_000)
    assert rng.key != key
    assert rvs.shape == (10_000,)
    assert isinstance(rvs, ArrayLike)


def test_poisson() -> None:
    rng = Generator(42)
    key = rng.key
    rvs = rng.poisson(lam=1, size=10_000)
    assert rng.key != key
    assert rvs.shape == (10_000,)
    assert isinstance(rvs, ArrayLike)


def test_uniform() -> None:
    rng = Generator(42)
    key = rng.key
    rvs = rng.uniform(size=10_000)
    assert rng.key != key
    assert rvs.shape == (10_000,)
    assert jnp.min(rvs) >= 0.0
    assert jnp.max(rvs) < 1.0
    assert isinstance(rvs, ArrayLike)
