from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

jax = pytest.importorskip("jax", reason="tests require jax")

import jax.numpy as jnp  # noqa: E402
from jax.typing import ArrayLike  # noqa: E402

from glass import _rng  # noqa: E402
from glass.jax import Generator  # noqa: E402

if TYPE_CHECKING:
    from glass._types import FloatArray, IntArray


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
        ((10_000, 2), (10_000, 2)),
        (None, ()),
    ],
    ids=[
        "explicit_int_size",
        "explicit_tuple_size",
        "no_size_scalar_output",
    ],
)
def test_random(
    size_input: int | tuple[int, ...] | None,
    shape_output: tuple[int, ...],
) -> None:
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
        (1, 2 * jnp.ones(10_000), 10_000, (10_000,)),
        (jnp.ones(10_000), 2, 10_000, (10_000,)),
        (jnp.ones(10_000), 2 * jnp.ones(10_000), 10_000, (10_000,)),
        (jnp.ones(10_000), 2 * jnp.ones(10_000), None, (10_000,)),
        (1, 2, (10_000, 2), (10_000, 2)),
        (1, 2, None, ()),
    ],
    ids=[
        "scalar_inputs_explicit_size",
        "array_scale_explicit_size",
        "array_loc_explicit_size",
        "array_inputs_explicit_size",
        "array_inputs_shape_inferred",
        "scalar_inputs_2D_explicit_size",
        "scalar_inputs_scalar_output",
    ],
)
def test_normal(
    loc: float | FloatArray,
    scale: float | FloatArray,
    size_input: int | tuple[int, ...] | None,
    shape_output: tuple[int, ...],
) -> None:
    rng = _rng.rng_dispatcher(xp=jnp)
    key = rng.key  # ty: ignore[unresolved-attribute]
    rvs = rng.normal(loc=loc, scale=scale, size=size_input)
    assert rng.key != key  # ty: ignore[unresolved-attribute]
    assert rvs.shape == shape_output
    assert isinstance(rvs, ArrayLike)


def test_normal_shape_mismatch_explicit() -> None:
    """Explicit size incompatible with input broadcast shape."""
    rng = _rng.rng_dispatcher(xp=jnp)
    with pytest.raises(
        ValueError,
        match="is incompatible with input shapes that broadcast to",
    ):
        rng.normal(loc=jnp.ones(5), scale=2, size=3)


def test_normal_shape_mismatch_broadcast() -> None:
    """Input shapes that cannot be broadcast together."""
    rng = _rng.rng_dispatcher(xp=jnp)
    with pytest.raises(
        ValueError,
        match="Incompatible shapes for broadcasting",
    ):
        rng.normal(loc=jnp.ones(5), scale=jnp.ones(3), size=None)


@pytest.mark.parametrize(
    ("size_input", "shape_output"),
    [
        (10_000, (10_000,)),
        ((10_000, 2), (10_000, 2)),
        (None, ()),
    ],
    ids=[
        "explicit_int_size",
        "explicit_tuple_size",
        "no_size_scalar_output",
    ],
)
def test_standard_normal(
    size_input: int | tuple[int, ...] | None,
    shape_output: tuple[int, ...],
) -> None:
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
        (jnp.ones(10_000), 10_000, (10_000,)),
        (jnp.ones(10_000), None, (10_000,)),
        (1, (10_000, 2), (10_000, 2)),
        (1, None, ()),
    ],
    ids=[
        "scalar_lam_explicit_size",
        "array_lam_explicit_size",
        "array_lam_shape_inferred",
        "scalar_lam_2D_explicit_size",
        "scalar_lam_scalar_output",
    ],
)
def test_poisson(
    lam: float | FloatArray,
    size_input: int | tuple[int, ...] | None,
    shape_output: tuple[int, ...],
) -> None:
    rng = _rng.rng_dispatcher(xp=jnp)
    key = rng.key  # ty: ignore[unresolved-attribute]
    rvs = rng.poisson(lam=lam, size=size_input)
    assert rng.key != key  # ty: ignore[unresolved-attribute]
    assert rvs.shape == shape_output
    assert isinstance(rvs, ArrayLike)


def test_poisson_shape_mismatch_explicit() -> None:
    """Explicit size incompatible with input broadcast shape."""
    rng = _rng.rng_dispatcher(xp=jnp)
    with pytest.raises(
        ValueError,
        match="is incompatible with input shapes that broadcast to",
    ):
        rng.poisson(lam=jnp.ones(5), size=3)


@pytest.mark.parametrize(
    ("low", "high", "size_input", "shape_output"),
    [
        (0.0, 1.0, 10_000, (10_000,)),
        (jnp.zeros(10_000), 1.0, 10_000, (10_000,)),
        (0, jnp.ones(10_000), 10_000, (10_000,)),
        (jnp.zeros(10_000), jnp.ones(10_000), 10_000, (10_000,)),
        (jnp.zeros(10_000), jnp.ones(10_000), None, (10_000,)),
        (0.0, 1.0, (10_000, 2), (10_000, 2)),
        (0.0, 1.0, None, ()),
        (-5.0, 5.0, 10_000, (10_000,)),
    ],
    ids=[
        "scalar_inputs_explicit_size",
        "array_low_explicit_size",
        "array_high_explicit_size",
        "array_inputs_explicit_size",
        "array_inputs_shape_inferred",
        "scalar_inputs_2D_explicit_size",
        "scalar_inputs_scalar_output",
        "negative_low_explicit_size",
    ],
)
def test_uniform(
    low: float | FloatArray,
    high: float | FloatArray,
    size_input: int | tuple[int, ...] | None,
    shape_output: tuple[int, ...],
) -> None:
    rng = _rng.rng_dispatcher(xp=jnp)
    key = rng.key  # ty: ignore[unresolved-attribute]
    rvs = rng.uniform(low=low, high=high, size=size_input)
    assert rng.key != key  # ty: ignore[unresolved-attribute]
    assert rvs.shape == shape_output
    assert jnp.all(rvs >= low)
    assert jnp.all(rvs < high)
    assert isinstance(rvs, ArrayLike)


def test_uniform_shape_mismatch_explicit() -> None:
    """Explicit size incompatible with input broadcast shape."""
    rng = _rng.rng_dispatcher(xp=jnp)
    with pytest.raises(
        ValueError,
        match="is incompatible with input shapes that broadcast to",
    ):
        rng.uniform(low=jnp.zeros(5), high=1.0, size=3)


def test_uniform_shape_mismatch_broadcast() -> None:
    """Input shapes that cannot be broadcast together."""
    rng = _rng.rng_dispatcher(xp=jnp)
    with pytest.raises(
        ValueError,
        match="Incompatible shapes for broadcasting",
    ):
        rng.uniform(low=jnp.zeros(5), high=jnp.ones(3), size=None)


@pytest.mark.parametrize(
    ("n", "pvals", "size_input", "shape_output"),
    [
        (10_000, jnp.array([0.1, 0.2, 0.3, 0.4]), None, (4,)),
        (10_000, jnp.ones((3, 4)) / 4, None, (3, 4)),
        (jnp.array([10, 20, 30]), jnp.ones((3, 4)) / 4, None, (3, 4)),
        (10_000, jnp.ones((3, 4)) / 4, (3,), (3, 4)),
        (jnp.array([10, 20, 30]), jnp.ones((3, 4)) / 4, (3,), (3, 4)),
    ],
    ids=[
        "scalar_n_1D_pvals_no_size",
        "scalar_n_2D_pvals_no_size",
        "batched_n_2D_pvals_no_size",
        "scalar_n_2D_pvals_explicit_size",
        "batched_n_2D_pvals_explicit_size",
    ],
)
def test_multinomial(
    n: int | IntArray,
    pvals: FloatArray,
    size_input: int | tuple[int, ...] | None,
    shape_output: tuple[int, ...],
) -> None:
    rng = _rng.rng_dispatcher(xp=jnp)
    key = rng.key  # ty: ignore[unresolved-attribute]
    rvs = rng.multinomial(n, pvals, size=size_input)
    assert rng.key != key  # ty: ignore[unresolved-attribute]
    assert jnp.all(jnp.sum(rvs, axis=-1) == n)
    assert jax.dtypes.issubdtype(rvs.dtype, jnp.integer)
    assert rvs.shape == shape_output
    assert isinstance(rvs, ArrayLike)


def test_multinomial_shape_mismatch_explicit() -> None:
    """Explicit size incompatible with input broadcast shape."""
    rng = _rng.rng_dispatcher(xp=jnp)
    with pytest.raises(
        ValueError,
        match="is incompatible with input shapes that broadcast to",
    ):
        rng.multinomial(jnp.array([10, 20, 30]), jnp.ones((3, 4)) / 4, size=5)


def test_multinomial_shape_mismatch_broadcast() -> None:
    """Input shapes that cannot be broadcast together."""
    rng = _rng.rng_dispatcher(xp=jnp)
    with pytest.raises(
        ValueError,
        match="Incompatible shapes for broadcasting",
    ):
        rng.multinomial(jnp.array([10, 20, 30]), jnp.ones((5, 4)) / 4)
