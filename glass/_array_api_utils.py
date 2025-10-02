from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypeAlias

import array_api_strict
import numpy as np
import numpy.random

import glass.jax

if TYPE_CHECKING:
    from types import ModuleType

    from array_api_strict._array_object import Array as AArray
    from jaxtyping import Array as JAXArray
    from numpy.typing import DTypeLike, NDArray

    Size: TypeAlias = int | tuple[int, ...] | None
    GLASSFloatArray: TypeAlias = JAXArray | NDArray[np.float64]


def get_namespace(*arrays: NDArray[Any] | JAXArray) -> ModuleType:
    """
    Return the array library (array namespace) of input arrays
    if they belong to the same library or raise a :class:`ValueError`
    if they do not.
    """
    namespace = arrays[0].__array_namespace__()
    if any(
        array.__array_namespace__() != namespace
        for array in arrays
        if array is not None
    ):
        msg = "input arrays should belong to the same array library"
        raise ValueError(msg)

    return namespace


def rng_dispatcher(array: NDArray[Any] | JAXArray) -> UnifiedGenerator:
    """Dispatch RNG on the basis of the provided array."""
    backend = array.__array_namespace__().__name__
    if backend == "jax.numpy":
        return glass.jax.Generator(seed=42)
    if backend == "numpy":
        return np.random.default_rng()
    if backend == "array_api_strict":
        return Generator(seed=42)
    msg = "the array backend in not supported"
    raise NotImplementedError(msg)


class Generator:
    """NumPy random number generator returning array_api_strict Array."""

    __slots__ = ("rng",)

    def __init__(
        self, seed: int | bool | NDArray[np.int_ | np.bool] | None = None
    ) -> None:
        self.rng = numpy.random.default_rng(seed=seed)  # type: ignore[arg-type]

    def random(
        self,
        size: Size = None,
        dtype: DTypeLike | None = np.float64,
        out: NDArray[Any] | None = None,
    ) -> AArray:
        """Return random floats in the half-open interval [0.0, 1.0)."""
        return array_api_strict.asarray(self.rng.random(size, dtype, out))  # type: ignore[arg-type]

    def normal(
        self,
        loc: float | NDArray[np.floating] = 0.0,
        scale: float | NDArray[np.floating] = 1.0,
        size: Size = None,
    ) -> AArray:
        """Draw samples from a Normal distribution (mean=loc, stdev=scale)."""
        return array_api_strict.asarray(self.rng.normal(loc, scale, size))

    def poisson(self, lam: float | NDArray[np.floating], size: Size = None) -> AArray:
        """Draw samples from a Poisson distribution."""
        return array_api_strict.asarray(self.rng.poisson(lam, size))

    def standard_normal(
        self,
        size: Size = None,
        dtype: DTypeLike | None = np.float64,
        out: NDArray[Any] | None = None,
    ) -> AArray:
        """Draw samples from a standard Normal distribution (mean=0, stdev=1)."""
        return array_api_strict.asarray(self.rng.standard_normal(size, dtype, out))  # type: ignore[arg-type]

    def uniform(
        self,
        low: float | NDArray[np.floating] = 0.0,
        high: float | NDArray[np.floating] = 1.0,
        size: Size = None,
    ) -> AArray:
        """Draw samples from a Uniform distribution."""
        return array_api_strict.asarray(self.rng.uniform(low, high, size))


UnifiedGenerator: TypeAlias = np.random.Generator | glass.jax.Generator | Generator
