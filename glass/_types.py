from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import ParamSpec, TypeAlias, TypeVar

    import jaxtyping
    import numpy as np

    from array_api_strict._array_object import Array
    from array_api_strict._dtypes import DType

    import glass.jax
    from glass import _rng

    P = ParamSpec("P")
    R = TypeVar("R")
    T = TypeVar("T")

    AnyArray: TypeAlias = np.typing.NDArray[Any] | jaxtyping.Array | Array
    ComplexArray: TypeAlias = np.typing.NDArray[np.complex128] | jaxtyping.Array | Array
    DTypeLike: TypeAlias = np.typing.DTypeLike | jaxtyping.DTypeLike | DType
    FloatArray: TypeAlias = np.typing.NDArray[np.float64] | jaxtyping.Array | Array
    IntArray: TypeAlias = np.typing.NDArray[np.int64] | jaxtyping.Array | Array
    UnifiedGenerator: TypeAlias = (
        np.random.Generator | glass.jax.Generator | _rng.Generator
    )

    AngularPowerSpectra: TypeAlias = Sequence[AnyArray]
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
