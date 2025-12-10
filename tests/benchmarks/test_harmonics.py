from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

import glass.harmonics

if TYPE_CHECKING:
    from types import ModuleType

    from conftest import Compare
    from pytest_benchmark.fixture import BenchmarkFixture


@pytest.mark.unstable
def test_multalm(
    benchmark: BenchmarkFixture,
    compare: type[Compare],
    xp_benchmarks: ModuleType,
) -> None:
    """Benchmarks for glass.harmonics.multalm."""
    scale_factor = 100_000

    alm = xp_benchmarks.arange(scale_factor * 5, dtype=xp_benchmarks.float64)
    bl = xp_benchmarks.asarray(scale_factor * 3, dtype=xp_benchmarks.float64)

    result = benchmark(glass.harmonics.multalm, alm, bl)

    compare.assert_allclose(
        result[:5],
        xp_benchmarks.asarray([scale_factor * x for x in [0.0, 3.0, 6.0, 9.0, 12.0]]),
    )
