"""Wrapper for JAX RNG with a NumPy-like interface."""

from __future__ import annotations

import math
from threading import Lock
from typing import TYPE_CHECKING, TypeAlias

from typing_extensions import Self

if TYPE_CHECKING:
    from types import FunctionType

    from jaxtyping import Array, Integer, PRNGKeyArray, Shaped

    RealArray: TypeAlias = Array
    Size: TypeAlias = int | tuple[int, ...] | None


def _size(size: Size, *bcast: Array) -> tuple[int, ...]:
    """
    Return a size, which can be a single int or None, as a shape, which
    is a tuple of int.
    """
    import jax.numpy as jnp  # noqa: PLC0415
    if size is None:
        if bcast:
            return jnp.broadcast_shapes(*map(jnp.shape, bcast))  # type: ignore[no-any-return]
        return ()
    if isinstance(size, int):
        return (size,)
    return size


def trapezoid(y: Array, x: Array = None, dx: Array = 1.0, axis: int = -1) -> Array:
    """Wrapper for jax.scipy.integrate.trapezoid."""
    import jax.scipy  # noqa: PLC0415

    return jax.scipy.integrate.trapezoid(y, x=x, dx=dx, axis=axis)


def union1d(ar1: Array, ar2: Array) -> Array:
    """Wrapper for jax.numpy.trapezoid."""
    import jax.numpy as jnp  # noqa: PLC0415

    return jnp.union1d(ar1, ar2)


def interp(  # noqa: PLR0913
    x: Array,
    x_points: Array,
    y_points: Array,
    left: Array = None,
    right: Array = None,
    period: Array = None,
) -> Array:
    """Wrapper for jax.numpy.interp."""
    import jax.numpy as jnp  # noqa: PLC0415

    return jnp.interp(x, x_points, y_points, left=left, right=right, period=period)


def gradient(f: Array) -> Array:
    """Wrapper for jax.numpy.gradient."""
    import jax.numpy as jnp  # noqa: PLC0415

    return jnp.gradient(f)


def linalg_lstsq(
    a: Array, b: Array, rcond: float | None = None
) -> tuple[Array, Array, Array, Array]:
    """Wrapper for jax.numpy.linalg.lstsq."""
    import jax.numpy as jnp  # noqa: PLC0415

    return jnp.linalg.lstsq(a, b, rcond)  # type: ignore[no-any-return]


def einsum(subscripts: str, *operands: Array) -> Array:
    """Wrapper for jax.numpy.einsum."""
    import jax.numpy as jnp  # noqa: PLC0415

    return jnp.einsum(subscripts, *operands)


def apply_along_axis(
    func1d: FunctionType,
    axis: int,
    arr: Array,
    *args: object,
    **kwargs: object,
) -> Array:
    """Wrapper for jax.numpy.apply_along_axis."""
    import jax.numpy as jnp  # noqa: PLC0415

    return jnp.apply_along_axis(func1d, axis, arr, *args, **kwargs)


class Generator:
    """JAX random number generation as a NumPy generator."""

    __slots__ = ("key", "lock")
    key: PRNGKeyArray
    lock: Lock

    @classmethod
    def from_key(cls, key: PRNGKeyArray) -> Self:
        """Wrap a JAX random key."""
        import jax.dtypes  # noqa: PLC0415
        from jax.typing import ArrayLike  # noqa: PLC0415

        if not isinstance(key, ArrayLike) or not jax.dtypes.issubdtype(
            key.dtype, jax.dtypes.prng_key
        ):
            msg = "not a random key"
            raise ValueError(msg)
        rng = object.__new__(cls)
        rng.key = key
        rng.lock = Lock()
        return rng

    def __init__(self, seed: int | Array, *, impl: str | None = None) -> None:
        """Create a wrapper instance with a new key."""
        import jax.random  # noqa: PLC0415

        self.key = jax.random.key(seed, impl=impl)
        self.lock = Lock()

    @property
    def __key(self) -> Array:
        """Return next key for sampling while updating internal state."""
        import jax.random  # noqa: PLC0415

        with self.lock:
            self.key, key = jax.random.split(self.key)
        return key

    def split(self, size: Size = None) -> Array:
        """Split random key."""
        import jax.random  # noqa: PLC0415

        shape = _size(size)
        with self.lock:
            keys = jax.random.split(self.key, 1 + math.prod(shape))
            self.key = keys[0]
        return keys[1:].reshape(shape)

    def spawn(self, n_children: int) -> list[Self]:
        """Create new independent child generators."""
        import jax.random  # noqa: PLC0415

        with self.lock:
            self.key, *keys = jax.random.split(self.key, num=n_children + 1)
        return list(map(self.from_key, keys))

    def random(self, size: Size = None, dtype: Shaped[Array, ...] = float) -> Array:
        """Return random floats in the half-open interval [0.0, 1.0)."""
        import jax.random  # noqa: PLC0415

        return jax.random.uniform(self.__key, _size(size), dtype)

    def normal(
        self,
        loc: float = 0.0,
        scale: float = 1.0,
        size: Size = None,
        dtype: Shaped[Array, ...] = float,
    ) -> Array:
        """Draw samples from a Normal distribution (mean=loc, stdev=scale)."""
        import jax.random  # noqa: PLC0415

        return loc + scale * jax.random.normal(self.__key, _size(size), dtype)

    def poisson(
        self, lam: float, size: Size = None, dtype: Integer[Array, ...] = int
    ) -> Array:
        """Draw samples from a Poisson distribution."""
        import jax.random  # noqa: PLC0415

        return jax.random.poisson(self.__key, lam, size, dtype)

    def standard_normal(
        self, size: Size = None, dtype: Shaped[Array, ...] = float
    ) -> Array:
        """Draw samples from a standard Normal distribution (mean=0, stdev=1)."""
        import jax.random  # noqa: PLC0415

        return jax.random.normal(self.__key, _size(size), dtype)

    def uniform(
        self,
        low: int = 0,
        high: int = 1,
        size: Size = None,
        dtype: Shaped[Array, ...] = float,
    ) -> Array:
        """Draw samples from a Uniform distribution."""
        import jax.random  # noqa: PLC0415

        return jax.random.uniform(self.__key, _size(size), dtype, low, high)
