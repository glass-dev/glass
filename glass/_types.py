import typing
from typing import Any

if typing.TYPE_CHECKING:
    from collections.abc import Sequence

    import jaxtyping
    import numpy as np

    from array_api_strict._array_object import Array
    from array_api_strict._dtypes import DType

    import glass.jax
    from glass import _rng

    P = typing.ParamSpec("P")
    R = typing.TypeVar("R")
    T = typing.TypeVar("T")

    AnyArray: typing.TypeAlias = np.typing.NDArray[Any] | jaxtyping.Array | Array
    ComplexArray: typing.TypeAlias = (
        np.typing.NDArray[np.complex128] | jaxtyping.Array | Array
    )
    DTypeLike: typing.TypeAlias = np.typing.DTypeLike | jaxtyping.DTypeLike | DType
    FloatArray: typing.TypeAlias = (
        np.typing.NDArray[np.float64] | jaxtyping.Array | Array
    )
    IntArray: typing.TypeAlias = np.typing.NDArray[np.int64] | jaxtyping.Array | Array
    UnifiedGenerator: typing.TypeAlias = (
        np.random.Generator | glass.jax.Generator | _rng.Generator
    )

    AngularPowerSpectra: typing.TypeAlias = Sequence[AnyArray]
else:
    # Runtime fallbacks (for Sphinx / autodoc)
    # https://github.com/sphinx-doc/sphinx/issues/11991
    AnyArray = Any
    ComplexArray = Any
    DTypeLike = Any
    FloatArray = Any
    IntArray = Any
    UnifiedGenerator = Any

    AngularPowerSpectra = Any
