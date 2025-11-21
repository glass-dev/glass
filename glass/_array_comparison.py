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


def assert_array_almost_equal_nulp(
    actual: AnyArray,
    desired: AnyArray,
    *,
    nulp: int = 1,
) -> None:
    """Compare two arrays relatively to their spacing."""
    np.testing.assert_array_almost_equal_nulp(actual, desired, nulp=nulp)


def assert_array_equal(
    actual: AnyArray,
    desired: AnyArray,
) -> None:
    """Raise an AssertionError if two array objects are not equal."""
    np.testing.assert_array_equal(actual, desired)


def assert_array_less(
    actual: AnyArray,
    desired: AnyArray,
) -> None:
    """Raise an AssertionError if two array objects are not ordered by less than."""
    np.testing.assert_array_less(actual, desired)


def assert_equal(
    actual: AnyArray,
    desired: AnyArray,
) -> None:
    """Raises an AssertionError if two objects are not equal."""
    np.testing.assert_equal(actual, desired)
