from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

import glass.shells

if TYPE_CHECKING:
    from types import ModuleType

    from conftest import Compare
    from pytest_benchmark.fixture import BenchmarkFixture


@pytest.mark.stable
def test_radialwindow(
    benchmark: BenchmarkFixture,
    xp: ModuleType,
    compare: Compare,
) -> None:
    """Benchmark for shells.RadialWindow."""
    # check zeff is computed when not provided
    arr_length = 1_000_000
    expected_zeff = 666666.0

    wa = xp.arange(arr_length)
    za = xp.arange(arr_length)

    w = benchmark(glass.shells.RadialWindow, za, wa)

    compare.assert_allclose(w.zeff, expected_zeff)
