from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from typing import ParamSpec, TypeAlias, TypeVar

    import numpy as np
    from jaxtyping import Array as JAXArray
    from numpy.typing import NDArray

    from array_api_strict._array_object import Array as AArray

    import glass._array_api_utils as _utils
    import glass.jax

    P = ParamSpec("P")
    R = TypeVar("R")
    T = TypeVar("T")

    AnyArray: TypeAlias = NDArray[Any] | JAXArray | AArray
    ComplexArray: TypeAlias = NDArray[np.complex128] | JAXArray | AArray
    DoubleArray: TypeAlias = NDArray[np.double] | JAXArray | AArray
    FloatArray: TypeAlias = NDArray[np.float64] | JAXArray | AArray
    IntArray: TypeAlias = NDArray[np.int_] | JAXArray | AArray
    UnifiedGenerator: TypeAlias = (
        np.random.Generator | glass.jax.Generator | _utils.Generator
    )
else:
    # Runtime fallbacks (for Sphinx / autodoc)
    # https://github.com/sphinx-doc/sphinx/issues/11991
    AnyArray = Any
    ComplexArray = Any
    DoubleArray = Any
    FloatArray = Any
    IntArray = Any
    UnifiedGenerator = Any
