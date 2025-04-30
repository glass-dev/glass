"""Dispatcher functionality to unify JAX and NumPy RNG behavior."""

import math
from threading import Lock
from typing import Any, Self, TypeAlias

import numpy as np
from jax.dtypes import issubdtype, prng_key
from jax.numpy import broadcast_shapes, shape
from jax.random import (
    key,
    normal,
    poisson,
    split,
    uniform,
)
from jax.typing import DTypeLike
from jaxtyping import Array
from numpy.typing import NDArray

RealArray: TypeAlias = Array
Size: TypeAlias = int | tuple[int, ...] | None


def _s(size: Size, *bcast: Array) -> tuple[int, ...]:
    """
    Return a size, which can be a single int or None, as a shape, which
    is a tuple of int.
    """
    if size is None:
        if bcast:
            return broadcast_shapes(*map(shape, bcast))  # type: ignore[no-any-return]
        return ()
    if isinstance(size, int):
        return (size,)
    return size


class JAXGenerator:
    """JAX random number generation as a NumPy generator."""

    __slots__ = ("key", "lock")
    key: Array
    lock: Lock

    @classmethod
    def from_key(cls, key: Array) -> Self:
        """Wrap a JAX random key."""
        if not isinstance(key, Array) or not issubdtype(key.dtype, prng_key):
            msg = "not a random key"
            raise ValueError(msg)
        rng = object.__new__(cls)
        rng.key = key
        rng.lock = Lock()
        return rng

    def __init__(self, seed: int | Array, *, impl: str | None = None) -> None:
        """Create a wrapper instance with a new key."""
        self.key = key(seed, impl=impl)
        self.lock = Lock()

    @property
    def __key(self) -> Array:
        """Return next key for sampling while updating internal state."""
        with self.lock:
            self.key, key = split(self.key)
        return key

    def split(self, size: Size = None) -> Array:
        """Split random key."""
        shape = _s(size)
        with self.lock:
            keys = split(self.key, 1 + math.prod(shape))
            self.key = keys[0]
        return keys[1:].reshape(shape)

    def spawn(self, n_children: int) -> list[Self]:
        """Create new independent child generators."""
        with self.lock:
            self.key, *keys = split(self.key, num=n_children + 1)
        return list(map(self.from_key, keys))

    def random(self, size: Size = None, dtype: DTypeLike = float) -> Array:
        """Return random floats in the half-open interval [0.0, 1.0)."""
        return uniform(self.__key, _s(size), dtype)

    def normal(
        self, loc: float, scale: float, size: Size = None, dtype: DTypeLike = float
    ) -> Array:
        """Draw samples from a Normal distribution (mean=loc, stdev=scale)."""
        return loc + scale * normal(self.__key, _s(size), dtype)

    def poisson(self, lam: float, size: Size = None, dtype: DTypeLike = float) -> Array:
        """Draw samples from a Poisson distribution."""
        return poisson(self.__key, lam, size, dtype)

    def standard_normal(self, size: Size = None, dtype: DTypeLike = float) -> Array:
        """Draw samples from a standard Normal distribution (mean=0, stdev=1)."""
        return normal(self.__key, _s(size), dtype)

    def uniform(
        self, low: int = 0, high: int = 1, size: Size = None, dtype: DTypeLike = float
    ) -> Array:
        """Draw samples from a Uniform distribution."""
        return uniform(self.__key, _s(size), dtype, low, high)


def rng(array: NDArray[Any] | Array) -> JAXGenerator | np.random.Generator:
    """Dispatch RNG on the basis of the provided array or backend."""
    if array.__array_namespace__().__name__ == "jax.numpy":
        return JAXGenerator(seed=42)
    return np.random.default_rng()
