"""
Array API Utilities for glass.
==============================

This module provides utility functions and classes for working with multiple array
backends in the glass project, including NumPy, JAX, and array-api-strict. It includes
functions for importing backends, determining array namespaces, dispatching random
number generators, and providing missing functionality for array-api-strict through the
xp_additions class.

Classes and functions in this module help ensure consistent behavior and compatibility
across different array libraries, and provide wrappers for common operations such as
integration, interpolation, and linear algebra.

"""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING, Any

import array_api_compat

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from types import ModuleType

    import numpy as np

    from glass._types import AnyArray, DTypeLike


class CompatibleBackendNotFoundError(Exception):
    """
    Exception raised when an array library backend that
    implements a requested function, is not found.

    """

    def __init__(self, missing_backend: str, users_backend: str | None) -> None:
        self.message = (
            f"{missing_backend} is required here as "
            "no alternative has been provided by the user."
            if users_backend is None
            else f"GLASS depends on functions not supported by {users_backend}"
        )
        super().__init__(self.message)


def import_numpy(backend: str | None = None) -> ModuleType:
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


def default_xp() -> ModuleType:
    """Returns the library backend we default to if none is specified by the user."""
    return import_numpy()


class xp_additions:  # noqa: N801
    """
    Additional functions missing from both array-api-strict and array-api-extra.

    This class provides wrappers for common array operations such as integration,
    interpolation, and linear algebra, ensuring compatibility across NumPy, JAX,
    and array-api-strict backends.

    This is intended as a temporary solution. See https://github.com/glass-dev/glass/issues/645
    for details.

    """

    @staticmethod
    def trapezoid(
        y: AnyArray,
        x: AnyArray | None = None,
        dx: float = 1.0,
        axis: int = -1,
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
        xp = array_api_compat.array_namespace(y, x, use_compat=False)

        if xp.__name__ == "jax.numpy":
            import glass.jax  # noqa: PLC0415

            return glass.jax.trapezoid(y, x=x, dx=dx, axis=axis)

        if xp.__name__ == "numpy":
            return xp.trapezoid(y, x=x, dx=dx, axis=axis)

        if xp.__name__ == "array_api_strict":
            np = import_numpy(xp.__name__)

            # Using design principle of scipy (i.e. copy, use np, copy back)
            y_np = np.asarray(y, copy=True)
            x_np = np.asarray(x, copy=True)
            result_np = np.trapezoid(y_np, x_np, dx=dx, axis=axis)
            return xp.asarray(result_np, copy=True)

        msg = "the array backend in not supported"
        raise NotImplementedError(msg)

    @staticmethod
    def interp(  # noqa: PLR0913
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
        xp = array_api_compat.array_namespace(x, x_points, y_points, use_compat=False)

        if xp.__name__ in {"numpy", "jax.numpy"}:
            return xp.interp(
                x,
                x_points,
                y_points,
                left=left,
                right=right,
                period=period,
            )

        if xp.__name__ == "array_api_strict":
            np = import_numpy(xp.__name__)

            # Using design principle of scipy (i.e. copy, use np, copy back)
            x_np = np.asarray(x, copy=True)
            x_points_np = np.asarray(x_points, copy=True)
            y_points_np = np.asarray(y_points, copy=True)
            result_np = np.interp(
                x_np,
                x_points_np,
                y_points_np,
                left=left,
                right=right,
                period=period,
            )
            return xp.asarray(result_np, copy=True)

        msg = "the array backend in not supported"
        raise NotImplementedError(msg)

    @staticmethod
    def gradient(f: AnyArray) -> AnyArray:
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
        xp = f.__array_namespace__()

        if xp.__name__ in {"numpy", "jax.numpy"}:
            return xp.gradient(f)

        if xp.__name__ == "array_api_strict":
            np = import_numpy(xp.__name__)
            # Using design principle of scipy (i.e. copy, use np, copy back)
            f_np = np.asarray(f, copy=True)
            result_np = np.gradient(f_np)
            return xp.asarray(result_np, copy=True)

        msg = "the array backend in not supported"
        raise NotImplementedError(msg)

    @staticmethod
    def linalg_lstsq(
        a: AnyArray,
        b: AnyArray,
        rcond: float | None = None,
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
        xp = array_api_compat.array_namespace(a, b, use_compat=False)

        if xp.__name__ in {"numpy", "jax.numpy"}:
            return xp.linalg.lstsq(a, b, rcond=rcond)  # type: ignore[no-any-return]

        if xp.__name__ == "array_api_strict":
            np = import_numpy(xp.__name__)

            # Using design principle of scipy (i.e. copy, use np, copy back)
            a_np = np.asarray(a, copy=True)
            b_np = np.asarray(b, copy=True)
            result_np = np.linalg.lstsq(a_np, b_np, rcond=rcond)
            return tuple(xp.asarray(res, copy=True) for res in result_np)

        msg = "the array backend in not supported"
        raise NotImplementedError(msg)

    @staticmethod
    def einsum(subscripts: str, *operands: AnyArray) -> AnyArray:
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
        xp = array_api_compat.array_namespace(*operands, use_compat=False)

        if xp.__name__ in {"numpy", "jax.numpy"}:
            return xp.einsum(subscripts, *operands)

        if xp.__name__ == "array_api_strict":
            np = import_numpy(xp.__name__)

            # Using design principle of scipy (i.e. copy, use np, copy back)
            operands_np = (np.asarray(op, copy=True) for op in operands)
            result_np = np.einsum(subscripts, *operands_np)
            return xp.asarray(result_np, copy=True)

        msg = "the array backend in not supported"
        raise NotImplementedError(msg)

    @staticmethod
    def apply_along_axis(
        func: Callable[..., Any],
        func_inputs: tuple[Any, ...],
        axis: int,
        arr: AnyArray,
        *args: object,
        **kwargs: object,
    ) -> AnyArray:
        """
        Apply a function to 1-D slices along the given axis.

        Rather than accepting a partial function as usual, the function and
        its inputs are passed in separately for better compatibility.

        Parameters
        ----------
        func
            Function to apply to 1-D slices.
        func_inputs
            All inputs to the func besides arr.
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
        xp = array_api_compat.array_namespace(arr, *func_inputs, use_compat=False)

        if xp.__name__ in {"numpy", "jax.numpy"}:
            func1d = functools.partial(func, *func_inputs)
            return xp.apply_along_axis(func1d, axis, arr, *args, **kwargs)

        if xp.__name__ == "array_api_strict":
            # Import here to prevent users relying on numpy unless in this instance
            np = import_numpy(xp.__name__)

            # Everything must be NumPy to avoid mismatches between array types
            inputs_np = (np.asarray(inp) for inp in func_inputs)
            func1d = functools.partial(func, *inputs_np)

            return xp.asarray(
                np.apply_along_axis(func1d, axis, arr, *args, **kwargs),
                copy=True,
            )

        msg = "the array backend in not supported"
        raise NotImplementedError(msg)

    @staticmethod
    def vectorize(
        pyfunc: Callable[..., Any],
        otypes: str | Sequence[DTypeLike],
        *,
        xp: ModuleType,
    ) -> Callable[..., Any]:
        """
        Returns an object that acts like pyfunc, but takes arrays as input.

        Parameters
        ----------
        pyfunc
            Python function to vectorize.
        otypes
            Output types.
        xp
            The array library backend to use for array operations.

        Returns
        -------
            Vectorized function.

        Raises
        ------
        NotImplementedError
            If the array backend is not supported.

        Notes
        -----
        See https://github.com/glass-dev/glass/issues/671

        """
        if xp.__name__ == "numpy":
            return xp.vectorize(pyfunc, otypes=otypes)  # type: ignore[no-any-return]

        if xp.__name__ in {"array_api_strict", "jax.numpy"}:
            # Import here to prevent users relying on numpy unless in this instance
            np = import_numpy(xp.__name__)

            return np.vectorize(pyfunc, otypes=otypes)  # type: ignore[no-any-return]

        msg = "the array backend in not supported"
        raise NotImplementedError(msg)

    @staticmethod
    def radians(deg_arr: AnyArray) -> AnyArray:
        """
        Convert angles from degrees to radians.

        Parameters
        ----------
        deg_arr
            Array of angles in degrees.

        Returns
        -------
            Array of angles in radians.

        Raises
        ------
        NotImplementedError
            If the array backend is not supported.

        """
        xp = deg_arr.__array_namespace__()

        if xp.__name__ in {"numpy", "jax.numpy"}:
            return xp.radians(deg_arr)

        if xp.__name__ == "array_api_strict":
            np = import_numpy(xp.__name__)
            return xp.asarray(np.radians(deg_arr))

        msg = "the array backend in not supported"
        raise NotImplementedError(msg)

    @staticmethod
    def degrees(rad_arr: AnyArray) -> AnyArray:
        """
        Convert angles from radians to degrees.

        Parameters
        ----------
        rad_arr
            Array of angles in radians.

        Returns
        -------
            Array of angles in degrees.

        Raises
        ------
        NotImplementedError
            If the array backend is not supported.

        """
        xp = rad_arr.__array_namespace__()

        if xp.__name__ in {"numpy", "jax.numpy"}:
            return xp.degrees(rad_arr)

        if xp.__name__ == "array_api_strict":
            np = import_numpy(xp.__name__)
            return xp.asarray(np.degrees(rad_arr))

        msg = "the array backend in not supported"
        raise NotImplementedError(msg)

    @staticmethod
    def ndindex(shape: tuple[int, ...], *, xp: ModuleType) -> np.ndindex:
        """
        Wrapper for numpy.ndindex.

        See relevant docs for details:
        - NumPy, https://numpy.org/doc/2.2/reference/generated/numpy.ndindex.html

        Parameters
        ----------
        shape
            Shape of the array to index.
        xp
            The array library backend to use for array operations.

        Raises
        ------
        NotImplementedError
            If the array backend is not supported.

        """
        if xp.__name__ == "numpy":
            return xp.ndindex(shape)  # type: ignore[no-any-return]

        if xp.__name__ in {"array_api_strict", "jax.numpy"}:
            np = import_numpy(xp.__name__)
            return np.ndindex(shape)  # type: ignore[no-any-return]

        msg = "the array backend in not supported"
        raise NotImplementedError(msg)
