import pytest

import jax
import jax.numpy as jnp
from glass._array_api_utils import JAXGenerator
from jax import dtypes, random


def test_init() -> None:
    rng = JAXGenerator(42)
    assert isinstance(rng, JAXGenerator)
    assert isinstance(rng.key, jax.Array)
    assert dtypes.issubdtype(rng.key.dtype, dtypes.prng_key)
    assert jnp.all(rng.key == random.key(42))


def test_from_key() -> None:
    key = random.key(42)
    rng = JAXGenerator.from_key(key)
    assert rng.key is key

    with pytest.raises(ValueError, match="not a random key"):
        JAXGenerator.from_key(object())

    with pytest.raises(ValueError, match="not a random key"):
        JAXGenerator.from_key(jnp.zeros(()))


def test_key() -> None:
    rng = JAXGenerator(42)
    rngkey, outkey = random.split(rng.key, 2)
    key = rng.split()
    assert jnp.all(rng.key == rngkey)
    assert jnp.all(key == outkey)


def test_spawn() -> None:
    rng = JAXGenerator(42)
    key, *subkeys = random.split(rng.key, 4)
    subrngs = rng.spawn(3)
    assert rng.key == key
    assert isinstance(subrngs, list)
    assert len(subrngs) == 3
    for subrng, subkey in zip(subrngs, subkeys, strict=False):
        assert isinstance(subrng, JAXGenerator)
        assert subrng.key == subkey


def test_random():
    rng = JAXGenerator(42)
    key = rng.key
    rvs = rng.random(size=10000)
    assert rng.key != key
    assert rvs.shape == (10000,)
    assert rvs.min() >= 0.0
    assert rvs.max() < 1.0


def test_normal():
    rng = JAXGenerator(42)
    key = rng.key
    rvs = rng.normal(1, 2, size=10000)
    assert rng.key != key
    assert rvs.shape == (10000,)


def test_standard_normal():
    rng = JAXGenerator(42)
    key = rng.key
    rvs = rng.standard_normal(size=10000)
    assert rng.key != key
    assert rvs.shape == (10000,)


def test_poisson():
    rng = JAXGenerator(42)
    key = rng.key
    rvs = rng.poisson(lam=1, size=10000)
    assert rng.key != key
    assert rvs.shape == (10000,)


def test_uniform():
    rng = JAXGenerator(42)
    key = rng.key
    rvs = rng.uniform(size=10000)
    assert rng.key != key
    assert rvs.shape == (10000,)
    assert rvs.min() >= 0.0
    assert rvs.max() < 1.0
