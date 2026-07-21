import typing
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import TypeAlias

    import jaxtyping
    import numpy as np

    from array_api_strict._array_object import Array
    from array_api_strict._dtypes import DType

    P = typing.ParamSpec("P")
    R = typing.TypeVar("R")
    T = typing.TypeVar("T")

    AnyArray: TypeAlias = np.typing.NDArray[Any] | jaxtyping.Array | Array
    ComplexArray: TypeAlias = np.typing.NDArray[np.complex128] | jaxtyping.Array | Array
    DTypeLike: TypeAlias = np.typing.DTypeLike | jaxtyping.DTypeLike | DType
    FloatArray: TypeAlias = np.typing.NDArray[np.float64] | jaxtyping.Array | Array
    IntArray: TypeAlias = np.typing.NDArray[np.int64] | jaxtyping.Array | Array

    AngularPowerSpectra: TypeAlias = Sequence[AnyArray]
else:
    # Runtime fallbacks (for Sphinx / autodoc)
    # https://github.com/sphinx-doc/sphinx/issues/11991
    AnyArray = Any
    ComplexArray = Any
    DTypeLike = Any
    FloatArray = Any
    IntArray = Any

    AngularPowerSpectra = Any


class UnifiedGenerator(Protocol):
    """Defines the methods required for an RNG to be used within glass."""

    def random(
        self,
        size: int | tuple[int, ...] | None = None,
        dtype: DTypeLike | None = None,
        out: FloatArray | None = None,
    ) -> FloatArray:
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

    def normal(
        self,
        loc: float | FloatArray = 0.0,
        scale: float | FloatArray = 1.0,
        size: int | tuple[int, ...] | None = None,
    ) -> FloatArray:
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

    def poisson(
        self,
        lam: float | FloatArray,
        size: int | tuple[int, ...] | None = None,
    ) -> IntArray:
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

    def standard_normal(
        self,
        size: int | tuple[int, ...] | None = None,
        dtype: DTypeLike | None = None,
        out: FloatArray | None = None,
    ) -> FloatArray:
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

    def uniform(
        self,
        low: float | FloatArray = 0.0,
        high: float | FloatArray = 1.0,
        size: int | tuple[int, ...] | None = None,
    ) -> FloatArray:
        """
        Draw samples from a Uniform distribution.

        Parameters
        ----------
        low
            Lower bound of the distribution.
        high
            Upper bound of the distribution.
        size
            Output shape.

        Returns
        -------
            Array of samples from the uniform distribution.

        """

    def multinomial(
        self,
        n: int | IntArray,
        pvals: FloatArray,
        size: int | tuple[int, ...] | None = None,
    ) -> IntArray:
        """
        Draw samples from a multinomial distribution.

        Parameters
        ----------
        n
            Number of experiments.
        pvals
            Probabilities of each of the p different outcomes.
        size
            Output shape.

        Returns
        -------
            The drawn sample.

        """
