from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

import array_api_extra as xpx

import glass

if TYPE_CHECKING:
    from types import ModuleType

    from pytest_benchmark.fixture import BenchmarkFixture


@pytest.mark.unstable
def test_radialwindow(
    benchmark: BenchmarkFixture,
    xpb: ModuleType,
) -> None:
    """Regression test for shells.RadialWindow."""
    # check zeff is computed when not provided
    arr_length = 100_000
    expected_zeff = xpb.asarray(66_666.0)

    wa = xpb.arange(arr_length)
    za = xpb.arange(arr_length)

    w = benchmark(glass.RadialWindow, za, wa)

    xpx.testing.assert_close(w.zeff, expected_zeff)
