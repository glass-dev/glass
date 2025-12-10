from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

import glass

if TYPE_CHECKING:
    from types import ModuleType

    from conftest import Compare
    from pytest_benchmark.fixture import BenchmarkFixture


@pytest.mark.unstable
def test_radialwindow(
    benchmark: BenchmarkFixture,
    xpb: ModuleType,
    compare: Compare,
) -> None:
    """Benchmark for shells.RadialWindow."""
    # check zeff is computed when not provided
    arr_length = 100_000
    expected_zeff = 66_666.0

    wa = xpb.arange(arr_length)
    za = xpb.arange(arr_length)

    w = benchmark(glass.RadialWindow, za, wa)

    compare.assert_allclose(w.zeff, expected_zeff)
