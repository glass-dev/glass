"""Wrapper for JAX RNG with a NumPy-like interface."""

from __future__ import annotations

import math
import threading
from typing import TYPE_CHECKING

import jax.dtypes
import jax.numpy as jnp
import jax.random
import jax.scipy
import jax.typing

if TYPE_CHECKING:
    from typing import Self

    from jaxtyping import PRNGKeyArray

    from glass._types import AnyArray, DTypeLike, FloatArray, IntArray


def _shape(
    shape: int | tuple[int, ...] | None,
    *bcast: AnyArray,
) -> tuple[int, ...]:
    """
    Computes the shape of an array given an explicit shape and a broadcast shape.

    A check will be performed to ensure that the explicit shape is compatible
    with the broadcast shape.

    Parameters
    ----------
    shape
        Non-negative integers representing the result shape.
    bcast
        Arrays whose shapes will be broadcast together to determine the broadcast.

    Returns
    -------
        The resulting shape as a tuple of non-negative integers.

    Raises
    ------
    ValueError
        If the explicit shape is incompatible with the broadcast shape.

    """
    bcast_shape = jnp.broadcast_shapes(*map(jnp.shape, bcast)) if bcast else ()

    if shape is None:
        return bcast_shape

    explicit_shape = (shape,) if isinstance(shape, int) else shape

    if bcast_shape and explicit_shape != bcast_shape:
        raise ValueError(
            f"shape {explicit_shape} is incompatible with input shapes "
            f"that broadcast to {bcast_shape}",
        )

    return explicit_shape


def trapezoid(
    y: AnyArray,
    x: AnyArray | None = None,
    dx: float | AnyArray = 1.0,
    axis: int = -1,
) -> FloatArray:
    """Wrapper for jax.scipy.integrate.trapezoid."""
    return jax.scipy.integrate.trapezoid(y, x=x, dx=dx, axis=axis)


class Generator:
    """JAX random number generation as a NumPy generator."""

    __slots__ = ("key", "lock")
    key: PRNGKeyArray
    lock: threading.Lock

    @classmethod
    def from_key(cls, key: PRNGKeyArray) -> Self:
        """Wrap a JAX random key."""
        if not isinstance(key, jax.typing.ArrayLike) or not jax.dtypes.issubdtype(
            key.dtype,
            jax.dtypes.prng_key,
        ):
            msg = "not a random key"
            raise ValueError(msg)
        rng = object.__new__(cls)
        rng.key = key
        rng.lock = threading.Lock()
        return rng

    def __init__(self, seed: int | AnyArray, *, impl: str | None = None) -> None:
        """Create a wrapper instance with a new key."""
        self.key = jax.random.key(seed, impl=impl)
        self.lock = threading.Lock()

    @property
    def __key(self) -> AnyArray:
        """Return next key for sampling while updating internal state."""
        with self.lock:
            self.key, key = jax.random.split(self.key)
        return key

    def split(self, size: int | tuple[int, ...] | None = None) -> AnyArray:
        """Split random key."""
        shape = _shape(size)
        with self.lock:
            keys = jax.random.split(self.key, 1 + math.prod(shape))
            self.key = keys[0]
        return keys[1:].reshape(shape)

    def spawn(self, n_children: int) -> list[Self]:
        """Create new independent child generators."""
        with self.lock:
            self.key, *keys = jax.random.split(self.key, num=n_children + 1)
        return list(map(self.from_key, keys))

    def random(
        self,
        size: int | tuple[int, ...] | None = None,
        dtype: DTypeLike = float,
    ) -> FloatArray:
        """Return random floats in the half-open interval [0.0, 1.0)."""
        return jax.random.uniform(
            self.__key,
            _shape(size),
            dtype,
        )

    def normal(
        self,
        loc: float | FloatArray = 0.0,
        scale: float | FloatArray = 1.0,
        size: int | tuple[int, ...] | None = None,
        dtype: DTypeLike = float,
    ) -> FloatArray:
        """Draw samples from a Normal distribution (mean=loc, stdev=scale)."""
        return loc + scale * jax.random.normal(
            self.__key,
            _shape(size, loc, scale),
            dtype,
        )

    def poisson(
        self,
        lam: float | FloatArray,
        size: int | tuple[int, ...] | None = None,
        dtype: DTypeLike = int,
    ) -> IntArray:
        """Draw samples from a Poisson distribution."""
        return jax.random.poisson(
            self.__key,
            lam,
            _shape(size, lam),
            dtype,
        )

    def standard_normal(
        self,
        size: int | tuple[int, ...] | None = None,
        dtype: DTypeLike = float,
    ) -> FloatArray:
        """Draw samples from a standard Normal distribution (mean=0, stdev=1)."""
        return jax.random.normal(
            self.__key,
            _shape(size),
            dtype,
        )

    def uniform(
        self,
        low: float | FloatArray = 0.0,
        high: float | FloatArray = 1.0,
        size: int | tuple[int, ...] | None = None,
        dtype: DTypeLike = float,
    ) -> FloatArray:
        """Draw samples from a Uniform distribution."""
        return jax.random.uniform(
            self.__key,
            _shape(size, low, high),
            dtype,
            low,
            high,
        )
        return jax.random.uniform(self.__key, _size(size), dtype, low, high)

    def multinomial(
        self,
        n: int | IntArray,
        pvals: FloatArray,
        size: int | tuple[int, ...] | None = None,
    ) -> IntArray:
        """Draw samples from a multinomial distribution."""
        shape = _size(size) if size is not None else None
        return jax.random.multinomial(self.__key, n, pvals, shape=shape)
