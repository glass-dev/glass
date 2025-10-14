from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypeAlias

import array_api_strict
import numpy as np
import numpy.random

import glass.jax

if TYPE_CHECKING:
    from types import FunctionType, ModuleType

    from array_api_strict._array_object import Array as AArray
    from jaxtyping import Array as JAXArray
    from numpy.typing import DTypeLike, NDArray

    Size: TypeAlias = int | tuple[int, ...] | None
    GLASSAnyArray: TypeAlias = JAXArray | NDArray[Any]
    GLASSFloatArray: TypeAlias = JAXArray | NDArray[np.float64]
    GLASSComplexArray: TypeAlias = JAXArray | NDArray[np.complex128]

    GlassAnyArray: TypeAlias = NDArray[Any] | JAXArray
    GlassFloatArray: TypeAlias = NDArray[np.float64] | JAXArray


def get_namespace(*arrays: GlassAnyArray) -> ModuleType:
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
    """
    Additional functions missing from both array-api-strict and array-api-extra.

    This is intended as a temporary solution. See https://github.com/glass-dev/glass/issues/645
    for details.
    """

    xp: ModuleType
    backend: str

    def __init__(self, xp: ModuleType) -> None:
        self.xp = xp
        self.backend = xp.__name__

    def trapezoid(
        self, y: GlassAnyArray, x: GlassAnyArray = None, dx: float = 1.0, axis: int = -1
    ) -> GlassAnyArray:
        """
        Integrate along the given axis using the composite trapezoidal rule.

        See https://github.com/glass-dev/glass/issues/646
        """
        self.backend = self.xp.__name__
        if self.backend == "jax.numpy":
            return glass.jax.trapezoid(y, x=x, dx=dx, axis=axis)
        if self.backend == "numpy":
            return np.trapezoid(y, x=x, dx=dx, axis=axis)
        if self.backend == "array_api_strict":
            # Using design principle of scipy (i.e. copy, use np, copy back)
            y_np = np.asarray(y, copy=True)
            x_np = np.asarray(x, copy=True)
            result_np = np.trapezoid(y_np, x_np, dx=dx, axis=axis)
            return self.xp.asarray(result_np, copy=True)

        msg = "the array backend in not supported"
        raise NotImplementedError(msg)

    def union1d(self, ar1: GlassAnyArray, ar2: GlassAnyArray) -> GlassAnyArray:
        """
        Compute the set union of two 1D arrays.

        See https://github.com/glass-dev/glass/issues/647
        """
        if self.backend == "jax.numpy":
            return glass.jax.union1d(ar1, ar2)
        if self.backend == "numpy":
            return np.union1d(ar1, ar2)
        if self.backend == "array_api_strict":
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

        See https://github.com/glass-dev/glass/issues/650
        """
        if self.backend == "jax.numpy":
            return glass.jax.interp(
                x, x_points, y_points, left=left, right=right, period=period
            )
        if self.backend == "numpy":
            return np.interp(
                x, x_points, y_points, left=left, right=right, period=period
            )
        if self.backend == "array_api_strict":
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
        """
        Return the gradient of an N-dimensional array.

        See https://github.com/glass-dev/glass/issues/648
        """
        if self.backend == "jax.numpy":
            return glass.jax.gradient(f)
        if self.backend == "numpy":
            return np.gradient(f)
        if self.backend == "array_api_strict":
            # Using design principle of scipy (i.e. copy, use np, copy back)
            f_np = np.asarray(f, copy=True)
            result_np = np.gradient(f_np)
            return self.xp.asarray(result_np, copy=True)

        msg = "the array backend in not supported"
        raise NotImplementedError(msg)

    def linalg_lstsq(
        self, a: GlassAnyArray, b: GlassAnyArray, rcond: float | None = None
    ) -> tuple[GlassAnyArray, GlassAnyArray, GlassAnyArray, GlassAnyArray]:
        """
        Return the gradient of an N-dimensional array.

        See https://github.com/glass-dev/glass/issues/649
        """
        if self.backend == "jax.numpy":
            return glass.jax.linalg_lstsq(a, b, rcond=rcond)
        if self.backend == "numpy":
            return np.linalg.lstsq(a, b, rcond=rcond)
        if self.backend == "array_api_strict":
            # Using design principle of scipy (i.e. copy, use np, copy back)
            a_np = np.asarray(a, copy=True)
            b_np = np.asarray(b, copy=True)
            result_np = np.linalg.lstsq(a_np, b_np, rcond=rcond)
            return tuple(self.xp.asarray(res, copy=True) for res in result_np)

        msg = "the array backend in not supported"
        raise NotImplementedError(msg)

    def einsum(self, subscripts: str, *operands: GlassAnyArray) -> GlassAnyArray:
        """
        Evaluates the Einstein summation convention on the operands.

        See https://github.com/glass-dev/glass/issues/657
        """
        if self.backend == "jax.numpy":
            return glass.jax.einsum(subscripts, *operands)
        if self.backend == "numpy":
            return np.einsum(subscripts, *operands)
        if self.backend == "array_api_strict":
            # Using design principle of scipy (i.e. copy, use np, copy back)
            operands_np = (np.asarray(op, copy=True) for op in operands)
            result_np = np.einsum(subscripts, *operands_np)
            return self.xp.asarray(result_np, copy=True)

        msg = "the array backend in not supported"
        raise NotImplementedError(msg)

    def apply_along_axis(
        self,
        func1d: FunctionType,
        axis: int,
        arr: GlassAnyArray,
        *args: object,
        **kwargs: object,
    ) -> GlassAnyArray:
        """
        Apply a function to 1-D slices along the given axis.

        See https://github.com/glass-dev/glass/issues/651
        """
        if self.backend == "jax.numpy":
            return glass.jax.apply_along_axis(func1d, axis, arr, *args, **kwargs)
        if self.backend == "numpy":
            return np.apply_along_axis(func1d, axis, arr, *args, **kwargs)
        if self.backend == "array_api_strict":
            # Using design principle of scipy (i.e. copy, use np, copy back)
            arr_np = np.asarray(arr, copy=True)
            result_np = np.apply_along_axis(func1d, axis, arr_np, *args, **kwargs)
            return self.xp.asarray(result_np, copy=True)

        msg = "the array backend in not supported"
        raise NotImplementedError(msg)
