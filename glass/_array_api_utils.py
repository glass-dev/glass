"""
Array API Utilities for glass.
============================

This module provides utility functions and classes for working with multiple array
backends in the glass project, including NumPy, JAX, and array-api-strict. It includes
functions for importing backends, determining array namespaces, dispatching random
number generators, and providing missing functionality for array-api-strict through the
XPAdditions class.

Classes and functions in this module help ensure consistent behavior and compatibility
across different array libraries, and provide wrappers for common operations such as
integration, interpolation, and linear algebra.

"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypeAlias

if TYPE_CHECKING:
    from collections.abc import Callable
    from types import ModuleType

    import numpy as np
    from jaxtyping import Array as JAXArray
    from numpy.typing import DTypeLike, NDArray

    from array_api_strict._array_object import Array as AArray

    import glass.jax

    Size: TypeAlias = int | tuple[int, ...] | None

    AnyArray: TypeAlias = NDArray[Any] | JAXArray | AArray
    ComplexArray: TypeAlias = NDArray[np.complex128] | JAXArray | AArray
    DoubleArray: TypeAlias = NDArray[np.double] | JAXArray | AArray
    FloatArray: TypeAlias = NDArray[np.float64] | JAXArray | AArray


class CompatibleBackendNotFoundError(Exception):
    """
    Exception raised when an array library backend that
    implements a requested function, is not found.
    """

    def __init__(self, missing_backend: str, users_backend: str) -> None:
        self.message = (
            f"{missing_backend} is required here as some functions required by GLASS "
            f"are not supported by {users_backend}"
        )
        super().__init__(self.message)


def import_numpy(backend: str) -> ModuleType:
    """
    Import the NumPy module, raising a helpful error if NumPy is not installed.

    Parameters
    ----------
    backend
        The name of the backend requested by the user.

    Returns
    -------
        The NumPy module.

    Raises
    ------
    ModuleNotFoundError
        If NumPy is not found in the user's environment.

    Notes
    -----
    This is useful for explaining to the user why NumPy is required when their chosen
    backend does not implement a needed function.
    """
    try:
        import numpy  # noqa: ICN001, PLC0415

    except ModuleNotFoundError as err:
        raise CompatibleBackendNotFoundError("numpy", backend) from err
    else:
        return numpy


def get_namespace(*arrays: AnyArray) -> ModuleType:
    """
    Return the array library (namespace) of input arrays if they all belong to the same
    library.

    Parameters
    ----------
    *arrays
        Arrays whose namespace is to be determined.

    Returns
    -------
        The array namespace module.

    Raises
    ------
    ValueError
        If input arrays do not all belong to the same array library.
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


def rng_dispatcher(
    array: AnyArray,
) -> np.random.Generator | glass.jax.Generator | Generator:
    """
    Dispatch a random number generator based on the provided array's backend.

    Parameters
    ----------
    array
        The array whose backend determines the RNG.

    Returns
    -------
        The appropriate random number generator for the array's backend.

    Raises
    ------
    NotImplementedError
        If the array backend is not supported.
    """
    xp = get_namespace(array)

    if xp.__name__ == "jax.numpy":
        import glass.jax  # noqa: PLC0415

        return glass.jax.Generator(seed=42)

    if xp.__name__ == "numpy":
        return xp.random.default_rng()  # type: ignore[no-any-return]

    if xp.__name__ == "array_api_strict":
        return Generator(seed=42)

    msg = "the array backend in not supported"
    raise NotImplementedError(msg)


class Generator:
    """
    NumPy random number generator returning array_api_strict Array.

    This class wraps NumPy's random number generator and returns arrays compatible
    with array_api_strict.
    """

    __slots__ = ("axp", "nxp", "rng")

    def __init__(
        self,
        seed: int | bool | AArray | None = None,  # noqa: FBT001
    ) -> None:
        """
        Initialize the Generator.

        Parameters
        ----------
        seed
            Seed for the random number generator.
        """
        import numpy as np  # noqa: PLC0415

        import array_api_strict  # noqa: PLC0415

        self.axp = array_api_strict
        self.nxp = np
        self.rng = np.random.default_rng(seed=seed)

    def random(
        self,
        size: Size = None,
        dtype: DTypeLike | None = None,
        out: AArray | None = None,
    ) -> AArray:
        """
        Return random floats in the half-open interval [0.0, 1.0).

        Parameters
        ----------
        size
            Output shape.
        dtype
            Desired data type.
        out
            Optional output array.

        Returns
        -------
            Array of random floats.
        """
        dtype = dtype if dtype is not None else self.nxp.float64
        return self.axp.asarray(self.rng.random(size, dtype, out))  # type: ignore[arg-type]

    def normal(
        self,
        loc: float | AArray = 0.0,
        scale: float | AArray = 1.0,
        size: Size = None,
    ) -> AArray:
        """
        Draw samples from a Normal distribution (mean=loc, stdev=scale).

        Parameters
        ----------
        loc
            Mean of the distribution.
        scale
            Standard deviation of the distribution.
        size
            Output shape.

        Returns
        -------
            Array of samples from the normal distribution.
        """
        return self.axp.asarray(self.rng.normal(loc, scale, size))

    def poisson(self, lam: float | AArray, size: Size = None) -> AArray:
        """
        Draw samples from a Poisson distribution.

        Parameters
        ----------
        lam
            Expected number of events.
        size
            Output shape.

        Returns
        -------
            Array of samples from the Poisson distribution.
        """
        return self.axp.asarray(self.rng.poisson(lam, size))

    def standard_normal(
        self,
        size: Size = None,
        dtype: DTypeLike | None = None,
        out: AArray | None = None,
    ) -> AArray:
        """
        Draw samples from a standard Normal distribution (mean=0, stdev=1).

        Parameters
        ----------
        size
            Output shape.
        dtype
            Desired data type.
        out
            Optional output array.

        Returns
        -------
            Array of samples from the standard normal distribution.
        """
        dtype = dtype if dtype is not None else self.nxp.float64
        return self.axp.asarray(self.rng.standard_normal(size, dtype, out))  # type: ignore[arg-type]

    def uniform(
        self,
        low: float | AArray = 0.0,
        high: float | AArray = 1.0,
        size: Size = None,
    ) -> AArray:
        """
        Draw samples from a Uniform distribution.

        Parameters
        ----------
        low
            Lower bound of the distribution.
        high
            Upper bound of the distribution.
        size : Size, optional
            Output shape.

        Returns
        -------
            Array of samples from the uniform distribution.
        """
        return self.axp.asarray(self.rng.uniform(low, high, size))


class XPAdditions:
    """
    Additional functions missing from both array-api-strict and array-api-extra.

    This class provides wrappers for common array operations such as integration,
    interpolation, and linear algebra, ensuring compatibility across NumPy, JAX,
    and array-api-strict backends.

    This is intended as a temporary solution. See https://github.com/glass-dev/glass/issues/645
    for details.
    """

    xp: ModuleType
    backend: str

    def __init__(self, xp: ModuleType) -> None:
        """
        Initialize XPAdditions with the given array namespace.

        Parameters
        ----------
        xp
            The array namespace module.
        """
        self.xp = xp

    def trapezoid(
        self, y: AnyArray, x: AnyArray = None, dx: float = 1.0, axis: int = -1
    ) -> AnyArray:
        """
        Integrate along the given axis using the composite trapezoidal rule.

        Parameters
        ----------
        y
            Input array to integrate.
        x
            Sample points corresponding to y.
        dx
            Spacing between sample points.
        axis
            Axis along which to integrate.

        Returns
        -------
            Integrated result.

        Raises
        ------
        NotImplementedError
            If the array backend is not supported.

        Notes
        -----
        See https://github.com/glass-dev/glass/issues/646
        """
        if self.xp.__name__ == "jax.numpy":
            import glass.jax  # noqa: PLC0415

            return glass.jax.trapezoid(y, x=x, dx=dx, axis=axis)

        if self.xp.__name__ == "numpy":
            return self.xp.trapezoid(y, x=x, dx=dx, axis=axis)

        if self.xp.__name__ == "array_api_strict":
            np = import_numpy(self.xp.__name__)

            # Using design principle of scipy (i.e. copy, use np, copy back)
            y_np = np.asarray(y, copy=True)
            x_np = np.asarray(x, copy=True)
            result_np = np.trapezoid(y_np, x_np, dx=dx, axis=axis)
            return self.xp.asarray(result_np, copy=True)

        msg = "the array backend in not supported"
        raise NotImplementedError(msg)

    def union1d(self, ar1: AnyArray, ar2: AnyArray) -> AnyArray:
        """
        Compute the set union of two 1D arrays.

        Parameters
        ----------
        ar1
            First input array.
        ar2
            Second input array.

        Returns
        -------
            The union of the two arrays.

        Raises
        ------
        NotImplementedError
            If the array backend is not supported.

        Notes
        -----
        See https://github.com/glass-dev/glass/issues/647
        """
        if self.xp.__name__ in {"numpy", "jax.numpy"}:
            return self.xp.union1d(ar1, ar2)

        if self.xp.__name__ == "array_api_strict":
            np = import_numpy(self.xp.__name__)

            # Using design principle of scipy (i.e. copy, use np, copy back)
            ar1_np = np.asarray(ar1, copy=True)
            ar2_np = np.asarray(ar2, copy=True)
            result_np = np.union1d(ar1_np, ar2_np)
            return self.xp.asarray(result_np, copy=True)

        msg = "the array backend in not supported"
        raise NotImplementedError(msg)

    def interp(  # noqa: PLR0913
        self,
        x: AnyArray,
        x_points: AnyArray,
        y_points: AnyArray,
        left: float | None = None,
        right: float | None = None,
        period: float | None = None,
    ) -> AnyArray:
        """
        One-dimensional linear interpolation for monotonically increasing sample points.

        Parameters
        ----------
        x
            The x-coordinates at which to evaluate the interpolated values.
        x_points
            The x-coordinates of the data points.
        y_points
            The y-coordinates of the data points.
        left
            Value to return for x < x_points[0].
        right
            Value to return for x > x_points[-1].
        period
            Period for periodic interpolation.

        Returns
        -------
            Interpolated values.

        Raises
        ------
        NotImplementedError
            If the array backend is not supported.

        Notes
        -----
        See https://github.com/glass-dev/glass/issues/650
        """
        if self.xp.__name__ in {"numpy", "jax.numpy"}:
            return self.xp.interp(
                x, x_points, y_points, left=left, right=right, period=period
            )

        if self.xp.__name__ == "array_api_strict":
            np = import_numpy(self.xp.__name__)

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

    def gradient(self, f: AnyArray) -> AnyArray:
        """
        Return the gradient of an N-dimensional array.

        Parameters
        ----------
        f
            Input array.

        Returns
        -------
            Gradient of the input array.

        Raises
        ------
        NotImplementedError
            If the array backend is not supported.

        Notes
        -----
        See https://github.com/glass-dev/glass/issues/648
        """
        if self.xp.__name__ in {"numpy", "jax.numpy"}:
            return self.xp.gradient(f)

        if self.xp.__name__ == "array_api_strict":
            np = import_numpy(self.xp.__name__)

            # Using design principle of scipy (i.e. copy, use np, copy back)
            f_np = np.asarray(f, copy=True)
            result_np = np.gradient(f_np)
            return self.xp.asarray(result_np, copy=True)

        msg = "the array backend in not supported"
        raise NotImplementedError(msg)

    def linalg_lstsq(
        self, a: AnyArray, b: AnyArray, rcond: float | None = None
    ) -> tuple[AnyArray, AnyArray, int, AnyArray]:
        """
        Solve a linear least squares problem.

        Parameters
        ----------
        a
            Coefficient matrix.
        b
            Ordinate or "dependent variable" values.
        rcond
            Cut-off ratio for small singular values.

        Returns
        -------
        x
            Least-squares solution. If b is two-dimensional, the solutions are in the K
            columns of x.

        residuals
            Sums of squared residuals: Squared Euclidean 2-norm for each column in b - a
            @ x. If the rank of a is < N or M <= N, this is an empty array. If b is
            1-dimensional, this is a (1,) shape array. Otherwise the shape is (K,).

        rank
            Rank of matrix a.

        s
            Singular values of a.

        Raises
        ------
        NotImplementedError
            If the array backend is not supported.

        Notes
        -----
        See https://github.com/glass-dev/glass/issues/649
        """
        if self.xp.__name__ in {"numpy", "jax.numpy"}:
            return self.xp.linalg.lstsq(a, b, rcond=rcond)  # type: ignore[no-any-return]

        if self.xp.__name__ == "array_api_strict":
            np = import_numpy(self.xp.__name__)

            # Using design principle of scipy (i.e. copy, use np, copy back)
            a_np = np.asarray(a, copy=True)
            b_np = np.asarray(b, copy=True)
            result_np = np.linalg.lstsq(a_np, b_np, rcond=rcond)
            return tuple(self.xp.asarray(res, copy=True) for res in result_np)

        msg = "the array backend in not supported"
        raise NotImplementedError(msg)

    def einsum(self, subscripts: str, *operands: AnyArray) -> AnyArray:
        """
        Evaluate the Einstein summation convention on the operands.

        Parameters
        ----------
        subscripts
            Specifies the subscripts for summation.
        *operands
            Arrays to be summed.

        Returns
        -------
            Result of the Einstein summation.

        Raises
        ------
        NotImplementedError
            If the array backend is not supported.

        Notes
        -----
        See https://github.com/glass-dev/glass/issues/657
        """
        if self.xp.__name__ in {"numpy", "jax.numpy"}:
            return self.xp.einsum(subscripts, *operands)

        if self.xp.__name__ == "array_api_strict":
            np = import_numpy(self.xp.__name__)

            # Using design principle of scipy (i.e. copy, use np, copy back)
            operands_np = (np.asarray(op, copy=True) for op in operands)
            result_np = np.einsum(subscripts, *operands_np)
            return self.xp.asarray(result_np, copy=True)

        msg = "the array backend in not supported"
        raise NotImplementedError(msg)

    def apply_along_axis(
        self,
        func1d: Callable[..., Any],
        axis: int,
        arr: AnyArray,
        *args: object,
        **kwargs: object,
    ) -> AnyArray:
        """
        Apply a function to 1-D slices along the given axis.

        Parameters
        ----------
        func1d
            Function to apply to 1-D slices.
        axis
            Axis along which to apply the function.
        arr
            Input array.
        *args
            Additional positional arguments to pass to func1d.
        **kwargs
            Additional keyword arguments to pass to func1d.

        Returns
        -------
            Result of applying the function along the axis.

        Raises
        ------
        NotImplementedError
            If the array backend is not supported.

        Notes
        -----
        See https://github.com/glass-dev/glass/issues/651

        """
        if self.xp.__name__ in {"numpy", "jax.numpy"}:
            return self.xp.apply_along_axis(func1d, axis, arr, *args, **kwargs)

        if self.xp.__name__ == "array_api_strict":
            # Import here to prevent users relying on numpy unless in this instance
            np = import_numpy(self.xp.__name__)

            return self.xp.asarray(
                np.apply_along_axis(func1d, axis, arr, *args, **kwargs), copy=True
            )

        msg = "the array backend in not supported"
        raise NotImplementedError(msg)
