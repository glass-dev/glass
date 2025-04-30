from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypeAlias

import numpy as np

from glass.jax import JAXGenerator

if TYPE_CHECKING:
    from types import ModuleType

    from jaxtyping import Array
    from numpy.typing import NDArray


def get_namespace(*arrays: NDArray[Any] | Array) -> ModuleType:
    """
    Return the array library (array namespace) of input arrays
    if they belong to the same library or raise a :class:`ValueError`
    if they do not.
    """
    namespace = arrays[0].__array_namespace__()
    if not all(array.__array_namespace__() == namespace for array in arrays):
        msg = "input arrays should belong to the same array library"
        raise ValueError(msg)

    return namespace


UnifiedGenerator: TypeAlias = np.random.Generator | JAXGenerator


def rng_dispatcher(array: NDArray[Any] | Array) -> UnifiedGenerator:
    """Dispatch RNG on the basis of the provided array."""
    if array.__array_namespace__().__name__ == "jax.numpy":
        return JAXGenerator(seed=42)
    return np.random.default_rng()
