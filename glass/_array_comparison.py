from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from glass._types import AnyArray


def assert_allclose(
    actual: AnyArray,
    desired: AnyArray,
    *,
    rtol: float = 1e-7,
    atol: float = 0,
) -> None:
    """Raise an AssertionError if two objects are not equal up to desired tolerance."""
    np.testing.assert_allclose(actual, desired, rtol=rtol, atol=atol)
