"""Module for random number generation utilities."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from types import ModuleType

    from glass._types import DTypeLike, FloatArray, IntArray, UnifiedGenerator


SEED = 42


def default_rng(
    *,
    seed: int | IntArray | None = None,
    xp: ModuleType,
) -> UnifiedGenerator:
    """
    Dispatch a random number generator for the array backend for a given seed.

    Parameters
    ----------
    xp
        The array library backend to use for array operations.
    seed
        Seed for the random number generator.

    Returns
    -------
        The appropriate random number generator for the array's backend.

    """
    if xp.__name__ == "jax.numpy":
        import glass.jax  # noqa: PLC0415

        return glass.jax.Generator(seed=seed)

    if xp.__name__ == "numpy":
        return xp.random.default_rng(seed=seed)

    return Generator(xp=xp, seed=seed)


def rng_dispatcher(*, xp: ModuleType) -> UnifiedGenerator:
    """
    Dispatch a random number generator for the array backend for the GLASS default seed.

    Parameters
    ----------
    xp
        The array library backend to use for array operations.

    Returns
    -------
        The appropriate random number generator for the array's backend.

    """
    return default_rng(seed=SEED, xp=xp)


class Generator:
    """
    NumPy random number generator returning Arrays of the given backend.

    This class wraps NumPy's random number generator and returns arrays compatible
    with the provided backend.

    """

    __slots__ = ("np", "rng", "xp")

    def __init__(
        self,
        xp: ModuleType,
        seed: int | IntArray = SEED,
    ) -> None:
        """
        Initialize the Generator.

        Parameters
        ----------
        xp
            The array library backend to return arrays from.
        seed
            Seed for the random number generator.

        """
        import numpy  # noqa: ICN001, PLC0415

        self.xp = xp
        self.np = numpy
        self.rng = default_rng(seed=seed, xp=xp)

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
        dtype = dtype if dtype is not None else self.np.float64
        return self.xp.asarray(self.rng.random(size, dtype, out))  # ty: ignore[no-matching-overload, too-many-positional-arguments]

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
        return self.xp.asarray(self.rng.normal(loc, scale, size))

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
        return self.xp.asarray(self.rng.poisson(lam, size))

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
        dtype = dtype if dtype is not None else self.np.float64
        return self.xp.asarray(self.rng.standard_normal(size, dtype, out))  # ty: ignore[no-matching-overload, too-many-positional-arguments]

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
        return self.xp.asarray(self.rng.uniform(low, high, size))

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
        return self.xp.asarray(self.rng.multinomial(n, pvals, size))
