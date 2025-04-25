"""JAX random number generation as a NumPy generator."""

import math
from threading import Lock
from typing import Literal, Self, TypeAlias

from jax import Array
from jax.dtypes import issubdtype, prng_key
from jax.numpy import array, broadcast_shapes, shape, uint8
from jax.random import (
    beta,
    binomial,
    bits,
    chisquare,
    choice,
    dirichlet,
    exponential,
    f,
    gamma,
    key,
    multivariate_normal,
    normal,
    permutation,
    poisson,
    randint,
    split,
    uniform,
)
from jax.typing import ArrayLike, DTypeLike

RealArray: TypeAlias = ArrayLike
Size: TypeAlias = int | tuple[int, ...] | None


def _s(size: Size, *bcast: ArrayLike) -> tuple[int, ...]:
    """
    Return a size, which can be a single int or None, as a shape, which
    is a tuple of int.
    """
    if size is None:
        if bcast:
            return broadcast_shapes(*map(shape, bcast))
        return ()
    if isinstance(size, int):
        return (size,)
    return size


class Generator:
    """Wrapper class for JAX random number generation."""

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

    def integers(
        self,
        low: int | ArrayLike,
        high: int | ArrayLike | None = None,
        size: Size = None,
        dtype: DTypeLike = int,
        endpoint: bool = False,
    ) -> Array:
        """
        Return random integers from the "discrete uniform" distribution
        of the specified dtype.  If *high* is None (the default), then
        results are from 0 to *low*.
        """
        if high is None:
            low, high = 0, low
        if endpoint:
            high = high + 1
        return randint(self.__key, _s(size), low, high, dtype)

    def random(self, size: Size = None, dtype: DTypeLike = float) -> Array:
        """Return random floats in the half-open interval [0.0, 1.0)."""
        return uniform(self.__key, _s(size), dtype)

    def choice(
        self,
        a: Array,
        size: Size = None,
        replace: bool = True,
        p: Array | None = None,
        axis: int = 0,
    ) -> Array:
        """Generate a random sample from a given array."""
        return choice(self.__key, a, _s(size), replace, p, axis)

    def bytes(self, length: int) -> bytes:
        """Return random bytes."""
        shape = (length // uint8.dtype.itemsize,)
        return bits(self.__key, shape, uint8).tobytes()

    def permutation(self, x: int | Array, axis: int = 0) -> Array:
        """Randomly permute a sequence, or return a permuted range."""
        return permutation(self.__key, x, axis, False)

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

    # geometric(p[, size])
    # Draw samples from the geometric distribution.

    # gumbel([loc, scale, size])
    # Draw samples from a Gumbel distribution.

    # hypergeometric(ngood, nbad, nsample[, size])
    # Draw samples from a Hypergeometric distribution.

    # laplace([loc, scale, size])
    # Draw samples from the Laplace or double exponential distribution with specified location (or mean) and scale (decay).

    # logistic([loc, scale, size])
    # Draw samples from a logistic distribution.

    # lognormal([mean, sigma, size])
    # Draw samples from a log-normal distribution.

    # logseries(p[, size])
    # Draw samples from a logarithmic series distribution.

    # multinomial(n, pvals[, size])
    # Draw samples from a multinomial distribution.

    # multivariate_hypergeometric(colors, nsample)
    # Generate variates from a multivariate hypergeometric distribution.

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

    # negative_binomial(n, p[, size])
    # Draw samples from a negative binomial distribution.

    # noncentral_chisquare(df, nonc[, size])
    # Draw samples from a noncentral chi-square distribution.

    # noncentral_f(dfnum, dfden, nonc[, size])
    # Draw samples from the noncentral F distribution.

    def normal(
        self, loc: float, scale: float, size: Size = None, dtype: DTypeLike = float
    ) -> Array:
        return loc + scale * normal(self.__key, _s(size), dtype)

    # pareto(a[, size])
    # Draw samples from a Pareto II (AKA Lomax) distribution with specified shape.

    def poisson(self, lam: float, size: Size = None, dtype: DTypeLike = float) -> Array:
        return poisson(self.__key, lam, size, dtype)

    # power(a[, size])
    # Draws samples in [0, 1] from a power distribution with positive exponent a - 1.

    # rayleigh([scale, size])
    # Draw samples from a Rayleigh distribution.

    # standard_cauchy([size])
    # Draw samples from a standard Cauchy distribution with mode = 0.

    # standard_exponential([size, dtype, method, out])
    # Draw samples from the standard exponential distribution.

    # standard_gamma(shape[, size, dtype, out])
    # Draw samples from a standard Gamma distribution.

    def standard_normal(self, size: Size = None, dtype: DTypeLike = float) -> Array:
        """Draw samples from a standard Normal distribution (mean=0, stdev=1)."""
        return normal(self.__key, _s(size), dtype)

    # standard_t(df[, size])
    # Draw samples from a standard Student's t distribution with df degrees of freedom.

    # triangular(left, mode, right[, size])
    # Draw samples from the triangular distribution over the interval [left, right].

    def uniform(
        self, low: int = 0, high: int = 1, size: Size = None, dtype: DTypeLike = float
    ) -> Array:
        """Draw samples from a Uniform distribution."""
        return uniform(self.__key, _s(size), dtype, low, high)

    # vonmises(mu, kappa[, size])
    # Draw samples from a von Mises distribution.

    # wald(mean, scale[, size])
    # Draw samples from a Wald, or inverse Gaussian, distribution.

    # weibull(a[, size])
    # Draw samples from a Weibull distribution.

    # zipf(a[, size])
    # Draw samples from a Zipf distribution.
