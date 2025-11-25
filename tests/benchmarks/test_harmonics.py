from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

import glass.harmonics

if TYPE_CHECKING:
    from types import ModuleType

    from pytest_benchmark.fixture import BenchmarkFixture


def test_multalm(
    xp: ModuleType,
    benchmark: BenchmarkFixture,
) -> None:
    """Benchmarks for glass.harmonics.multalm."""
    scale_factor = 100_000
    # Call jax version of iternorm once jax version is written
    if xp.__name__ == "jax.numpy":
        pytest.skip("Arrays in multalm are not immutable, so do not support jax")

    # check output values and shapes

    alm = xp.arange(scale_factor * 5, dtype=xp.float64)
    bl = xp.asarray(scale_factor * 3, dtype=xp.float64)

    result = benchmark(glass.harmonics.multalm, alm, bl)

    np.testing.assert_allclose(
        result[:5],
        xp.asarray([scale_factor * x for x in [0.0, 3.0, 6.0, 9.0, 12.0]]),
    )
