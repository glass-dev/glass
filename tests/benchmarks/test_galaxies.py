from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

import glass

if TYPE_CHECKING:
    from types import ModuleType

    from pytest_benchmark.fixture import BenchmarkFixture


def test_redshifts(
    benchmark: BenchmarkFixture,
    xp: ModuleType,
) -> None:
    """Benchmark for galaxies.redhsifts."""
    if xp.__name__ == "jax.numpy":
        pytest.skip("Arrays in redshifts are not immutable, so do not support jax")
    scale_factor = 1_000
    # create a mock radial window function
    za = xp.linspace(0.0, 1.0, 20 * scale_factor)
    wa = xp.exp(-0.5 * (za - 0.5) ** 2 / 0.1**2)
    w = glass.shells.RadialWindow(za, wa)

    # sample redshifts (scalar)
    z = benchmark(glass.redshifts, 13 * scale_factor, w)
    assert z.shape == (13 * scale_factor,)
    assert xp.min(z) >= 0.0
    assert xp.max(z) <= 1.0
