from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

import glass

if TYPE_CHECKING:
    from types import ModuleType

    from pytest_benchmark.fixture import BenchmarkFixture


@pytest.mark.stable
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


@pytest.mark.stable
def test_redshifts_from_nz(
    benchmark: BenchmarkFixture,
    xp: ModuleType,
) -> None:
    """Benchmark for galaxies.redshifts_from_nz."""
    if xp.__name__ == "jax.numpy":
        pytest.skip(
            "Arrays in redshifts_from_nz are not immutable, so do not support jax",
        )
    scale_factor = 1_000
    # create a mock radial window function
    za = xp.linspace(0.0, 1.0, 20 * scale_factor)
    wa = xp.exp(-0.5 * (za - 0.5) ** 2 / 0.1**2)

    # sample redshifts (scalar)
    redshifts = benchmark(
        glass.redshifts_from_nz,
        13 * scale_factor,
        za,
        wa,
    )
    assert redshifts.shape == (13 * scale_factor,)
    assert xp.min(redshifts) >= 0.0
    assert xp.max(redshifts) <= 1.0
    assert xp.all((0 <= redshifts) & (redshifts <= 1))  # noqa: SIM300
