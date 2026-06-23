from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

import array_api_extra as xpx

glass_harmonics = pytest.importorskip(
    "glass.harmonics",
    reason="tests require glass.harmonics",
)


if TYPE_CHECKING:
    from types import ModuleType

    from pytest_benchmark.fixture import BenchmarkFixture


@pytest.mark.unstable
def test_multalm(
    benchmark: BenchmarkFixture,
    xp: ModuleType,
) -> None:
    """Regression tests for glass.harmonics.multalm."""
    scale_factor = 100_000

    alm = xp.arange(scale_factor * 5, dtype=xp.float64)
    bl = xp.asarray(scale_factor * 3, dtype=xp.float64)

    result = benchmark(glass_harmonics.multalm, alm, bl)

    xpx.testing.assert_equal(
        result[:5],
        xp.asarray([scale_factor * x for x in [0.0, 3.0, 6.0, 9.0, 12.0]]),
    )
