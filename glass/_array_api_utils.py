from __future__ import annotations

from typing import TYPE_CHECKING, Any

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
    if any(
        array.__array_namespace__() != namespace
        for array in arrays
        if array is not None
    ):
        msg = "input arrays should belong to the same array library"
        raise ValueError(msg)

    return namespace
