from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

import glass.harmonics

if TYPE_CHECKING:
    from types import ModuleType

    from pytest_benchmark.fixture import BenchmarkFixture


@pytest.mark.parametrize(
    ("alm_in", "bl_in", "expected_result"),
    [
        (
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            [2.0, 0.5, 1.0],
            [2.0, 1.0, 1.5, 4.0, 5.0, 6.0],
        ),
        (
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            [1.0, 1.0, 1.0],
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        ),
        (
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            [0.0, 1.0, 0.0],
            [0.0, 2.0, 3.0, 0.0, 0.0, 0.0],
        ),
        (
            [],
            [],
            [],
        ),
    ],
)
def test_multalm(
    xp: ModuleType,
    alm_in: list[float],
    bl_in: list[float],
    expected_result: list[float],
    benchmark: BenchmarkFixture,
) -> None:
    """Benchmarks for glass.harmonics.multalm."""
    # Call jax version of iternorm once jax version is written
    if xp.__name__ == "jax.numpy":
        pytest.skip("Arrays in multalm are not immutable, so do not support jax")

    # check output values and shapes

    alm = xp.asarray(alm_in)
    bl = xp.asarray(bl_in)

    result = benchmark(glass.harmonics.multalm, alm, bl)

    np.testing.assert_allclose(result, xp.asarray(expected_result))
