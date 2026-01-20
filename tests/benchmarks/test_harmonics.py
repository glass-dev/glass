from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

glass_harmonics = pytest.importorskip(
    "glass.harmonics",
    reason="tests require glass.harmonics",
)


if TYPE_CHECKING:
    from types import ModuleType

    from conftest import Compare
    from pytest_benchmark.fixture import BenchmarkFixture


@pytest.mark.unstable
def test_multalm(
    benchmark: BenchmarkFixture,
    compare: type[Compare],
    xpb: ModuleType,
) -> None:
    """Benchmarks for glass.harmonics.multalm."""
    scale_factor = 100_000

    alm = xpb.arange(scale_factor * 5, dtype=xpb.float64)
    bl = xpb.asarray(scale_factor * 3, dtype=xpb.float64)

    result = benchmark(glass_harmonics.multalm, alm, bl)

    compare.assert_allclose(
        result[:5],
        xpb.asarray([scale_factor * x for x in [0.0, 3.0, 6.0, 9.0, 12.0]]),
    )
