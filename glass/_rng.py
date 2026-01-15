"""
Random Number Generation Utilities for glass.
=============================================

This module includes functions for dispatching random number generators using consistent
seeds. The chhoice of rng generator is determined based on the array library chosen by
the user.

"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from types import ModuleType

    from numpy.typing import DTypeLike

    from array_api_strict._array_object import Array as AArray

    from glass._types import FloatArray, UnifiedGenerator


def rng_dispatcher(*, xp: ModuleType) -> UnifiedGenerator:
    """
    Dispatch a random number generator based on the provided array's backend.

    Parameters
    ----------
    xp
        The array library backend to use for array operations.

    Returns
    -------
        The appropriate random number generator for the array's backend.

    Raises
    ------
    NotImplementedError
        If the array backend is not supported.
    """
    seed = 42

    if xp.__name__ == "jax.numpy":
        import glass.jax  # noqa: PLC0415

        return glass.jax.Generator(seed=seed)

    if xp.__name__ == "numpy":
        return xp.random.default_rng(seed=seed)  # type: ignore[no-any-return]

    if xp.__name__ == "array_api_strict":
        return Generator(seed=seed)

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
        import numpy  # noqa: ICN001, PLC0415

        import array_api_strict  # noqa: PLC0415

        self.axp = array_api_strict
        self.nxp = numpy
        self.rng = self.nxp.random.default_rng(seed=seed)

    def random(
        self,
        size: int | tuple[int, ...] | None = None,
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
        loc: float | FloatArray = 0.0,
        scale: float | FloatArray = 1.0,
        size: int | tuple[int, ...] | None = None,
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

    def poisson(
        self,
        lam: float | AArray,
        size: int | tuple[int, ...] | None = None,
    ) -> AArray:
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
        size: int | tuple[int, ...] | None = None,
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
        size: int | tuple[int, ...] | None = None,
    ) -> AArray:
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
        return self.axp.asarray(self.rng.uniform(low, high, size))
