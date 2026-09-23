from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

import array_api_extra as xpx

if TYPE_CHECKING:
    from types import ModuleType

    from pytest_benchmark.fixture import BenchmarkFixture

glass_harmonics = pytest.importorskip(
    "glass.harmonics",
    reason="tests require glass.harmonics",
)


@pytest.mark.skipif(
    not hasattr(glass_harmonics, "multalm"),
    reason="glass.harmonics.multalm not implemented",
)
@pytest.mark.unstable
def test_multalm(
    benchmark: BenchmarkFixture,
    xp: ModuleType,
) -> None:
    """Regression tests for glass.harmonics.multalm."""
    alm = xp.arange(180_300, dtype=xp.float64)
    bl = xp.full(600, fill_value=2.0, dtype=xp.float64)

    result = benchmark(glass_harmonics.multalm, alm, bl)

    xpx.testing.assert_equal(
        result[:5],
        xp.asarray([0.0, 2.0, 4.0, 6.0, 8.0]),
    )
