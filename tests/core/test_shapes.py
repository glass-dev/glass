from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

import glass

if TYPE_CHECKING:
    from types import ModuleType

    from glass._types import UnifiedGenerator
    from tests.fixtures.helper_classes import Compare


def test_triaxial_axis_ratio(urng: UnifiedGenerator, xp: ModuleType) -> None:
    # Pass floats without xp

    with pytest.raises(
        TypeError,
        match="array_namespace requires at least one non-scalar array input",
    ):
        glass.triaxial_axis_ratio(0.8, 0.4)

    # single axis ratio

    q = glass.triaxial_axis_ratio(0.8, 0.4, xp=xp)
    assert q.ndim == 0

    # many axis ratios

    q = glass.triaxial_axis_ratio(0.8, 0.4, size=1_000, xp=xp)
    assert q.shape == (1_000,)

    # explicit shape

    q = glass.triaxial_axis_ratio(0.8, 0.4, size=(10, 10), xp=xp)
    assert q.shape == (10, 10)

    # implicit size

    q1 = glass.triaxial_axis_ratio(xp.asarray([0.8, 0.9]), 0.4)
    q2 = glass.triaxial_axis_ratio(0.8, xp.asarray([0.4, 0.5]))
    assert q1.shape == q2.shape == (2,)

    # broadcasting rule

    q = glass.triaxial_axis_ratio(
        xp.asarray([[0.6, 0.7], [0.8, 0.9]]),
        xp.asarray([0.4, 0.5]),
    )
    assert q.shape == (2, 2)

    # random parameters and check that projection is
    # between largest and smallest possible value

    sorted_uniform_rnd = xp.sort(urng.uniform(0, 1, size=(2, 1_000)), axis=0)
    zeta = sorted_uniform_rnd[0, :]
    xi = sorted_uniform_rnd[1, :]
    qmin = xp.min(xp.stack([zeta, xi, xi / zeta]), axis=0)
    qmax = xp.max(xp.stack([zeta, xi, xi / zeta]), axis=0)
    q = glass.triaxial_axis_ratio(zeta, xi)
    assert xp.all((qmax >= q) & (q >= qmin))


def test_ellipticity_ryden04(urng: UnifiedGenerator, xp: ModuleType) -> None:
    # Pass floats without xp

    with pytest.raises(
        TypeError,
        match="array_namespace requires at least one non-scalar array input",
    ):
        glass.ellipticity_ryden04(-1.85, 0.89, 0.222, 0.056)

    # single ellipticity

    e = glass.ellipticity_ryden04(-1.85, 0.89, 0.222, 0.056, xp=xp)
    assert e.ndim == 0

    # test with rng

    e = glass.ellipticity_ryden04(-1.85, 0.89, 0.222, 0.056, rng=urng, xp=xp)
    assert e.ndim == 0

    # many ellipticities

    e = glass.ellipticity_ryden04(-1.85, 0.89, 0.222, 0.056, size=1_000, xp=xp)
    assert e.shape == (1_000,)

    # explicit shape

    e = glass.ellipticity_ryden04(-1.85, 0.89, 0.222, 0.056, size=(10, 10), xp=xp)
    assert e.shape == (10, 10)

    # implicit size

    e1 = glass.ellipticity_ryden04(-1.85, 0.89, xp.asarray([0.222, 0.333]), 0.056)
    e2 = glass.ellipticity_ryden04(-1.85, 0.89, 0.222, xp.asarray([0.056, 0.067]))
    e3 = glass.ellipticity_ryden04(xp.asarray([-1.85, -2.85]), 0.89, 0.222, 0.056)
    e4 = glass.ellipticity_ryden04(-1.85, xp.asarray([0.89, 1.001]), 0.222, 0.056)
    assert e1.shape == e2.shape == e3.shape == e4.shape == (2,)

    # broadcasting rule

    e = glass.ellipticity_ryden04(
        xp.asarray([-1.9, -2.9]),
        0.9,
        xp.asarray([[0.2, 0.3], [0.4, 0.5]]),
        0.1,
    )
    assert e.shape == (2, 2)

    # check that result is in the specified range

    e = glass.ellipticity_ryden04(0.0, 1.0, 0.222, 0.056, size=10, xp=xp)
    assert xp.all((xp.real(e) >= -1.0) & (xp.real(e) <= 1.0))

    e = glass.ellipticity_ryden04(0.0, 1.0, 0.0, 1.0, size=10, xp=xp)
    assert xp.all((xp.real(e) >= -1.0) & (xp.real(e) <= 1.0))


@pytest.mark.flaky(rerun=5, only_rerun=["AssertionError"])
def test_ellipticity_gaussian(
    compare: type[Compare],
    urng: UnifiedGenerator,
    xp: ModuleType,
) -> None:
    n = 1_000_000

    eps = glass.ellipticity_gaussian(n, 0.256, xp=xp)

    assert eps.shape == (n,)

    # Pass floats without xp

    with pytest.raises(
        TypeError,
        match="array_namespace requires at least one non-scalar array input",
    ):
        glass.ellipticity_gaussian(n, 0.256)

    # test with rng

    eps = glass.ellipticity_gaussian(n, 0.256, rng=urng, xp=xp)

    assert eps.shape == (n,)

    compare.assert_array_less(xp.abs(eps), 1)

    compare.assert_allclose(xp.std(xp.real(eps)), 0.256, atol=1e-3, rtol=0)
    compare.assert_allclose(xp.std(xp.imag(eps)), 0.256, atol=1e-3, rtol=0)

    eps = glass.ellipticity_gaussian(xp.asarray([n, n]), xp.asarray([0.128, 0.256]))

    assert eps.shape == (2 * n,)

    compare.assert_array_less(xp.abs(eps), 1)

    compare.assert_allclose(xp.std(xp.real(eps)[:n]), 0.128, atol=1e-3, rtol=0)
    compare.assert_allclose(xp.std(xp.imag(eps)[:n]), 0.128, atol=1e-3, rtol=0)
    compare.assert_allclose(xp.std(xp.real(eps)[n:]), 0.256, atol=1e-3, rtol=0)
    compare.assert_allclose(xp.std(xp.imag(eps)[n:]), 0.256, atol=1e-3, rtol=0)


def test_ellipticity_intnorm(
    compare: type[Compare],
    urng: UnifiedGenerator,
    xp: ModuleType,
) -> None:
    if xp.__name__ == "jax.numpy":
        pytest.skip(
            "Arrays in ellipticity_intnorm are not immutable, so do not support jax",
        )

    n = 1_000_000

    eps = glass.ellipticity_intnorm(n, 0.256, xp=xp)

    assert eps.shape == (n,)

    # Pass non-arrays without xp

    with pytest.raises(
        TypeError,
        match="array_namespace requires at least one non-scalar array input",
    ):
        glass.ellipticity_intnorm(n, 0.256)

    # test with rng

    eps = glass.ellipticity_intnorm(n, 0.256, rng=urng, xp=xp)

    assert eps.shape == (n,)

    compare.assert_array_less(xp.abs(eps), 1)

    compare.assert_allclose(xp.std(xp.real(eps)), 0.256, atol=1e-3, rtol=0)
    compare.assert_allclose(xp.std(xp.imag(eps)), 0.256, atol=1e-3, rtol=0)

    eps = glass.ellipticity_intnorm(xp.asarray([n, n]), xp.asarray([0.128, 0.256]))

    assert eps.shape == (2 * n,)

    compare.assert_array_less(xp.abs(eps), 1)

    compare.assert_allclose(xp.std(xp.real(eps)[:n]), 0.128, atol=1e-3, rtol=0)
    compare.assert_allclose(xp.std(xp.imag(eps)[:n]), 0.128, atol=1e-3, rtol=0)
    compare.assert_allclose(xp.std(xp.real(eps)[n:]), 0.256, atol=1e-3, rtol=0)
    compare.assert_allclose(xp.std(xp.imag(eps)[n:]), 0.256, atol=1e-3, rtol=0)

    with pytest.raises(ValueError, match="sigma must be between"):
        glass.ellipticity_intnorm(1, 0.71, xp=xp)
