from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

import glass.harmonics

if TYPE_CHECKING:
    from types import ModuleType

    from tests.fixtures.helper_classes import Compare


def test_multalm(compare: type[Compare], xp: ModuleType) -> None:
    # Call jax version of iternorm once jax version is written
    if xp.__name__ == "jax.numpy":
        pytest.skip("Arrays in multalm are not immutable, so do not support jax")

    # check output values and shapes

    alm = xp.asarray([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    bl = xp.asarray([2.0, 0.5, 1.0])
    alm_copy = xp.asarray(alm, copy=True)

    result = glass.harmonics.multalm(alm, bl)

    expected_result = xp.asarray([2.0, 1.0, 1.5, 4.0, 5.0, 6.0])
    compare.assert_allclose(result, expected_result)
    with pytest.raises(AssertionError, match="Not equal to tolerance"):
        compare.assert_allclose(alm_copy, result)

    # multiple with 1s

    alm = xp.asarray([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    bl = xp.ones(3)

    result = glass.harmonics.multalm(alm, bl)
    compare.assert_allclose(result, alm)

    # multiple with 0s

    bl = xp.asarray([0.0, 1.0, 0.0])

    result = glass.harmonics.multalm(alm, bl)

    expected_result = xp.asarray([0.0, 2.0, 3.0, 0.0, 0.0, 0.0])
    compare.assert_allclose(result, expected_result)

    # empty arrays

    alm = xp.asarray([])
    bl = xp.asarray([])

    result = glass.harmonics.multalm(alm, bl)
    compare.assert_allclose(result, alm)
