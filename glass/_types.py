from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from typing import ParamSpec, TypeAlias, TypeVar

    import numpy as np
    from jaxtyping import Array as JAXArray
    from numpy.typing import NDArray

    from array_api_strict._array_object import Array as AArray

    import glass.jax
    from glass import _rng

    P = ParamSpec("P")
    R = TypeVar("R")
    T = TypeVar("T")

    AnyArray: TypeAlias = NDArray[Any] | JAXArray | AArray
    ComplexArray: TypeAlias = NDArray[np.complex128] | JAXArray | AArray
    FloatArray: TypeAlias = NDArray[np.float64] | JAXArray | AArray
    IntArray: TypeAlias = NDArray[np.int64] | JAXArray | AArray
    UnifiedGenerator: TypeAlias = (
        np.random.Generator | glass.jax.Generator | _rng.Generator
    )
else:
    # Runtime fallbacks (for Sphinx / autodoc)
    # https://github.com/sphinx-doc/sphinx/issues/11991
    AnyArray = Any
    ComplexArray = Any
    FloatArray = Any
    IntArray = Any
    UnifiedGenerator = Any
