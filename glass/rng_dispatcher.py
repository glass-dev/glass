"""Dispatcher functionality to unify JAX and NumPy RNG behavior."""

import math
import types
from threading import Lock
from typing import Any, Literal, Self, TypeAlias

import numpy as np
from jax.dtypes import issubdtype, prng_key
from jax.numpy import array, broadcast_shapes, shape
from jax.random import (
    beta,
    binomial,
    chisquare,
    dirichlet,
    exponential,
    f,
    gamma,
    key,
    multivariate_normal,
    normal,
    poisson,
    split,
    uniform,
)
from jax.typing import ArrayLike, DTypeLike
from jaxtyping import Array
from numpy.typing import NDArray

RealArray: TypeAlias = ArrayLike
Size: TypeAlias = int | tuple[int, ...] | None


def _s(size: Size, *bcast: ArrayLike) -> tuple[int, ...]:
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

    def __init__(self, seed: int | ArrayLike, *, impl: str | None = None) -> None:
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

    def beta(self, a: RealArray, b: RealArray, size: Size = None) -> Array:
        """Draw samples from a Beta distribution."""
        return beta(self.__key, a, b, _s(size))

    def binomial(self, n: RealArray, p: RealArray, size: Size = None) -> Array:
        """Draw samples from a binomial distribution."""
        return binomial(self.__key, n, p, _s(size))

    def chisquare(self, df: RealArray, size: Size = None) -> Array:
        """Draw samples from a chi-square distribution."""
        return chisquare(self.__key, df, _s(size))

    def dirichlet(self, alpha: RealArray, size: Size = None) -> Array:
        """Draw samples from the Dirichlet distribution."""
        return dirichlet(self.__key, alpha, _s(size))

    def exponential(self, scale: RealArray = 1.0, size: Size = None) -> Array:
        """Draw samples from an exponential distribution."""
        return array(scale) * exponential(self.__key, _s(size, scale))

    def f(self, dfnum: RealArray, dfden: RealArray, size: Size = None) -> Array:
        """Draw samples from an F distribution."""
        return f(self.__key, dfnum, dfden, _s(size))

    def gamma(
        self,
        a: RealArray,
        scale: RealArray = 1.0,
        size: Size = None,
    ) -> Array:
        """Draw samples from a Gamma distribution."""
        return array(scale) * gamma(self.__key, a, _s(size, a, scale))

    def multivariate_normal(
        self,
        mean: RealArray,
        cov: RealArray,
        size: Size = None,
        *,
        method: Literal["svd", "eigh", "cholesky"] = "svd",
    ) -> Array:
        """Draw random samples from a multivariate normal distribution."""
        return multivariate_normal(
            self.__key,
            mean,
            cov,
            shape=_s(size),
            method=method,
        )

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


def rng(
    *,
    array: NDArray[Any] | Array | None = None,
    backend: types.ModuleType | None = None,
) -> JAXGenerator | np.random.Generator:
    """Dispatch RNG on the basis of provided array or backend."""
    if (array is not None and array.__array_namespace__().__name__ == "jax.numpy") or (
        backend is not None and backend.__name__ == "jax.numpy"
    ):
        return JAXGenerator(seed=42)
    return np.random.default_rng(seed=42)
