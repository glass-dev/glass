import pytest

jax = pytest.importorskip("jax", reason="tests require jax")

import jax.numpy as jnp  # noqa: E402
from jax.typing import ArrayLike  # noqa: E402

from glass import _rng  # noqa: E402
from glass.jax import Generator  # noqa: E402


def test_init() -> None:
    rng = _rng.rng_dispatcher(xp=jnp)
    assert isinstance(rng, Generator)
    assert isinstance(rng.key, jax.Array)
    assert jax.dtypes.issubdtype(rng.key.dtype, jax.dtypes.prng_key)
    assert jnp.all(rng.key == jax.random.key(_rng.SEED))


def test_from_key() -> None:
    key = jax.random.key(_rng.SEED)
    rng = Generator.from_key(key)
    assert rng.key is key

    with pytest.raises(ValueError, match="not a random key"):
        Generator.from_key(object())

    with pytest.raises(ValueError, match="not a random key"):
        Generator.from_key(jnp.zeros(()))


def test_key() -> None:
    rng = _rng.rng_dispatcher(xp=jnp)
    rngkey, outkey = jax.random.split(rng.key, 2)  # ty: ignore[unresolved-attribute]
    key = rng.split()  # ty: ignore[unresolved-attribute]
    assert jnp.all(rng.key == rngkey)  # ty: ignore[unresolved-attribute]
    assert jnp.all(key == outkey)


def test_spawn() -> None:
    rng = _rng.rng_dispatcher(xp=jnp)
    key, *subkeys = jax.random.split(rng.key, 4)  # ty: ignore[unresolved-attribute]
    subrngs = rng.spawn(3)  # ty: ignore[unresolved-attribute]
    assert rng.key == key  # ty: ignore[unresolved-attribute]
    assert isinstance(subrngs, list)
    assert len(subrngs) == 3
    for subrng, subkey in zip(subrngs, subkeys, strict=False):
        assert isinstance(subrng, Generator)
        assert subrng.key == subkey


@pytest.mark.parametrize(
    ("size_input", "shape_output"),
    [
        (10_000, (10_000,)),
    ],
)
def test_random(
    shape_output: tuple[int, ...],
    size_input: int,
) -> None:
    """Test passing glass.jax.Generator.random."""
    rng = _rng.rng_dispatcher(xp=jnp)
    key = rng.key  # ty: ignore[unresolved-attribute]
    rvs = rng.random(size=size_input)
    assert rng.key != key  # ty: ignore[unresolved-attribute]
    assert rvs.shape == shape_output
    assert jnp.min(rvs) >= 0.0
    assert jnp.max(rvs) < 1.0
    assert isinstance(rvs, ArrayLike)


@pytest.mark.parametrize(
    ("loc", "scale", "size_input", "shape_output"),
    [
        (1, 2, 10_000, (10_000,)),
    ],
)
def test_normal(
    loc: float,
    scale: float,
    shape_output: tuple[int, ...],
    size_input: int,
) -> None:
    """Test passing glass.jax.Generator.normal."""
    rng = _rng.rng_dispatcher(xp=jnp)
    key = rng.key  # ty: ignore[unresolved-attribute]
    rvs = rng.normal(loc, scale, size=size_input)
    assert rng.key != key  # ty: ignore[unresolved-attribute]
    assert rvs.shape == shape_output
    assert isinstance(rvs, ArrayLike)


@pytest.mark.parametrize(
    ("size_input", "shape_output"),
    [
        (10_000, (10_000,)),
    ],
)
def test_standard_normal(
    shape_output: tuple[int, ...],
    size_input: int,
) -> None:
    """Test passing glass.jax.Generator.standard_normal."""
    rng = _rng.rng_dispatcher(xp=jnp)
    key = rng.key  # ty: ignore[unresolved-attribute]
    rvs = rng.standard_normal(size=size_input)
    assert rng.key != key  # ty: ignore[unresolved-attribute]
    assert rvs.shape == shape_output
    assert isinstance(rvs, ArrayLike)


@pytest.mark.parametrize(
    ("lam", "size_input", "shape_output"),
    [
        (1, 10_000, (10_000,)),
    ],
)
def test_poisson(
    lam: int,
    shape_output: tuple[int, ...],
    size_input: int,
) -> None:
    """Test passing glass.jax.Generator.poisson."""
    rng = _rng.rng_dispatcher(xp=jnp)
    key = rng.key  # ty: ignore[unresolved-attribute]
    rvs = rng.poisson(lam=lam, size=size_input)
    assert rng.key != key  # ty: ignore[unresolved-attribute]
    assert rvs.shape == shape_output
    assert isinstance(rvs, ArrayLike)


@pytest.mark.parametrize(
    ("low", "high", "size_input", "shape_output"),
    [
        (0.0, 1.0, 10_000, (10_000,)),
    ],
)
def test_uniform(
    low: float,
    high: float,
    shape_output: tuple[int, ...],
    size_input: int,
) -> None:
    """Test passing glass.jax.Generator.uniform."""
    rng = _rng.rng_dispatcher(xp=jnp)
    key = rng.key  # ty: ignore[unresolved-attribute]
    rvs = rng.uniform(size=size_input, low=low, high=high)
    assert rng.key != key  # ty: ignore[unresolved-attribute]
    assert rvs.shape == shape_output
    assert jnp.min(rvs) >= low
    assert jnp.max(rvs) < high
    assert isinstance(rvs, ArrayLike)
