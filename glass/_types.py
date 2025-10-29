from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any, TypeAlias

    import numpy as np
    from jaxtyping import Array as JAXArray
    from numpy.typing import NDArray

    from array_api_strict._array_object import Array as AArray

    import glass._array_api_utils as _utils
    import glass.jax

    AnyArray: TypeAlias = NDArray[Any] | JAXArray | AArray
    ComplexArray: TypeAlias = NDArray[np.complex128] | JAXArray | AArray
    DoubleArray: TypeAlias = NDArray[np.double] | JAXArray | AArray
    FloatArray: TypeAlias = NDArray[np.float64] | JAXArray | AArray
    IntArray: TypeAlias = NDArray[np.int_] | JAXArray | AArray

    UnifiedGenerator: TypeAlias = (
        np.random.Generator | glass.jax.Generator | _utils.Generator
    )
