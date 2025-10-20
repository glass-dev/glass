"""Wrapper for JAX RNG with a NumPy-like interface."""

from __future__ import annotations

import math
from threading import Lock
from typing import TYPE_CHECKING, TypeAlias

import jax.dtypes
import jax.numpy as jnp
import jax.random
import jax.scipy
from jax.typing import ArrayLike
from typing_extensions import Self

if TYPE_CHECKING:
    from jaxtyping import Array, Integer, PRNGKeyArray, Shaped

    RealArray: TypeAlias = Array
    Size: TypeAlias = int | tuple[int, ...] | None


def _size(size: Size, *bcast: Array) -> tuple[int, ...]:
    """
    Return a size, which can be a single int or None, as a shape, which
    is a tuple of int.
    """
    if size is None:
        if bcast:
            return jnp.broadcast_shapes(*map(jnp.shape, bcast))  # type: ignore[no-any-return]
        return ()
    if isinstance(size, int):
        return (size,)
    return size


def trapezoid(y: Array, x: Array = None, dx: Array = 1.0, axis: int = -1) -> Array:
    """Wrapper for jax.scipy.integrate.trapezoid."""
    return jax.scipy.integrate.trapezoid(y, x=x, dx=dx, axis=axis)


class Generator:
    """JAX random number generation as a NumPy generator."""

    __slots__ = ("key", "lock")
    key: PRNGKeyArray
    lock: Lock

    @classmethod
    def from_key(cls, key: PRNGKeyArray) -> Self:
        """Wrap a JAX random key."""
        if not isinstance(key, ArrayLike) or not jax.dtypes.issubdtype(
            key.dtype,
            jax.dtypes.prng_key,
        ):
            msg = "not a random key"
            raise ValueError(msg)
        rng = object.__new__(cls)
        rng.key = key
        rng.lock = Lock()
        return rng

    def __init__(self, seed: int | Array, *, impl: str | None = None) -> None:
        """Create a wrapper instance with a new key."""
        self.key = jax.random.key(seed, impl=impl)
        self.lock = Lock()

    @property
    def __key(self) -> Array:
        """Return next key for sampling while updating internal state."""
        with self.lock:
            self.key, key = jax.random.split(self.key)
        return key

    def split(self, size: Size = None) -> Array:
        """Split random key."""
        shape = _size(size)
        with self.lock:
            keys = jax.random.split(self.key, 1 + math.prod(shape))
            self.key = keys[0]
        return keys[1:].reshape(shape)

    def spawn(self, n_children: int) -> list[Self]:
        """Create new independent child generators."""
        with self.lock:
            self.key, *keys = jax.random.split(self.key, num=n_children + 1)
        return list(map(self.from_key, keys))

    def random(self, size: Size = None, dtype: Shaped[Array, ...] = float) -> Array:
        """Return random floats in the half-open interval [0.0, 1.0)."""
        return jax.random.uniform(self.__key, _size(size), dtype)

    def normal(
        self,
        loc: float = 0.0,
        scale: float = 1.0,
        size: Size = None,
        dtype: Shaped[Array, ...] = float,
    ) -> Array:
        """Draw samples from a Normal distribution (mean=loc, stdev=scale)."""
        return loc + scale * jax.random.normal(self.__key, _size(size), dtype)

    def poisson(
        self,
        lam: float,
        size: Size = None,
        dtype: Integer[Array, ...] = int,
    ) -> Array:
        """Draw samples from a Poisson distribution."""
        return jax.random.poisson(self.__key, lam, size, dtype)

    def standard_normal(
        self,
        size: Size = None,
        dtype: Shaped[Array, ...] = float,
    ) -> Array:
        """Draw samples from a standard Normal distribution (mean=0, stdev=1)."""
        return jax.random.normal(self.__key, _size(size), dtype)

    def uniform(
        self,
        low: int = 0,
        high: int = 1,
        size: Size = None,
        dtype: Shaped[Array, ...] = float,
    ) -> Array:
        """Draw samples from a Uniform distribution."""
        return jax.random.uniform(self.__key, _size(size), dtype, low, high)
