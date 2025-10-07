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

    GlassAnyArray: TypeAlias = NDArray[Any] | JAXArray
    GlassFloatArray: TypeAlias = NDArray[np.float64] | JAXArray


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


class GlassXPAdditions:
    """Additional functions missing from both array-api-strict and array-api-extra."""

    xp: ModuleType

    def __init__(self, xp: ModuleType) -> None:
        self.xp = xp

    def trapezoid(
        self, y: GlassAnyArray, x: GlassAnyArray = None, dx: float = 1.0, axis: int = -1
    ) -> GlassAnyArray:
        """Integrate along the given axis using the composite trapezoidal rule."""
        backend = self.xp.__name__
        if backend == "jax.numpy":
            return glass.jax.trapezoid(y, x=x, dx=dx, axis=axis)
        if backend == "numpy":
            return np.trapezoid(y, x=x, dx=dx, axis=axis)
        if backend == "array_api_strict":
            # Using design principle of scipy (i.e. copy, use np, copy back)
            y_np = np.asarray(y, copy=True)
            x_np = np.asarray(x, copy=True)
            result_np = np.trapezoid(y_np, x_np, dx=dx, axis=axis)
            return self.xp.asarray(result_np, copy=True)

        msg = "the array backend in not supported"
        raise NotImplementedError(msg)

    def union1d(self, ar1: GlassAnyArray, ar2: GlassAnyArray) -> GlassAnyArray:
        """Compute the set union of two 1D arrays."""
        backend = self.xp.__name__
        if backend == "jax.numpy":
            return glass.jax.union1d(ar1, ar2)
        if backend == "numpy":
            return np.union1d(ar1, ar2)
        if backend == "array_api_strict":
            # Using design principle of scipy (i.e. copy, use np, copy back)
            ar1_np = np.asarray(ar1, copy=True)
            ar2_np = np.asarray(ar2, copy=True)
            result_np = np.union1d(ar1_np, ar2_np)
            return self.xp.asarray(result_np, copy=True)

        msg = "the array backend in not supported"
        raise NotImplementedError(msg)

    def interp(  # noqa: PLR0913
        self,
        x: GlassAnyArray,
        x_points: GlassAnyArray,
        y_points: GlassAnyArray,
        left: float | None = None,
        right: float | None = None,
        period: float | None = None,
    ) -> GlassAnyArray:
        """
        One-dimensional linear interpolation for monotonically increasing
        sample points.
        """
        backend = self.xp.__name__
        if backend == "jax.numpy":
            return glass.jax.interp(
                x, x_points, y_points, left=left, right=right, period=period
            )
        if backend == "numpy":
            return np.interp(
                x, x_points, y_points, left=left, right=right, period=period
            )
        if backend == "array_api_strict":
            # Using design principle of scipy (i.e. copy, use np, copy back)
            x_np = np.asarray(x, copy=True)
            x_points_np = np.asarray(x_points, copy=True)
            y_points_np = np.asarray(y_points, copy=True)
            result_np = np.interp(
                x_np, x_points_np, y_points_np, left=left, right=right, period=period
            )
            return self.xp.asarray(result_np, copy=True)

        msg = "the array backend in not supported"
        raise NotImplementedError(msg)

    def gradient(self, f: GlassAnyArray) -> GlassAnyArray:
        """Return the gradient of an N-dimensional array."""
        backend = self.xp.__name__
        if backend == "jax.numpy":
            return glass.jax.gradient(f)
        if backend == "numpy":
            return np.gradient(f)
        if backend == "array_api_strict":
            # Using design principle of scipy (i.e. copy, use np, copy back)
            f_np = np.asarray(f, copy=True)
            result_np = np.gradient(f_np)
            return self.xp.asarray(result_np, copy=True)

        msg = "the array backend in not supported"
        raise NotImplementedError(msg)

    def linalg_lstsq(
        self, a: GlassAnyArray, b: GlassAnyArray, rcond: float | None = None
    ) -> tuple[GlassAnyArray, GlassAnyArray, GlassAnyArray, GlassAnyArray]:
        """Return the gradient of an N-dimensional array."""
        backend = self.xp.__name__
        if backend == "jax.numpy":
            return glass.jax.linalg_lstsq(a, b, rcond=rcond)
        if backend == "numpy":
            return np.linalg.lstsq(a, b, rcond=rcond)
        if backend == "array_api_strict":
            # Using design principle of scipy (i.e. copy, use np, copy back)
            a_np = np.asarray(a, copy=True)
            b_np = np.asarray(b, copy=True)
            result_np = np.linalg.lstsq(a_np, b_np, rcond=rcond)
            return tuple(self.xp.asarray(res, copy=True) for res in result_np)

        msg = "the array backend in not supported"
        raise NotImplementedError(msg)
