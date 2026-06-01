from __future__ import annotations

import typing

import pytest

import glass

if typing.TYPE_CHECKING:
    from types import ModuleType

    from pytest_benchmark.fixture import BenchmarkFixture

    from tests.fixtures.helper_classes import Compare


@pytest.mark.unstable
def test_radialwindow(
    benchmark: BenchmarkFixture,
    compare: type[Compare],
    xpb: ModuleType,
) -> None:
    """Benchmark for shells.RadialWindow."""
    # check zeff is computed when not provided
    arr_length = 100_000
    expected_zeff = 66_666.0

    wa = xpb.arange(arr_length)
    za = xpb.arange(arr_length)

    w = benchmark(glass.RadialWindow, za, wa)

    compare.assert_allclose(w.zeff, expected_zeff)
