from __future__ import annotations

import dataclasses
import math
from typing import TYPE_CHECKING

import numpy as np
import pytest

import glass

if TYPE_CHECKING:
    from types import ModuleType

    import glass._array_api_utils as _utils
    from glass.cosmology import Cosmology
    from tests.conftest import Compare


def test_distance_weight(
    compare: type[Compare],
    cosmo: Cosmology,
    xp: ModuleType,
) -> None:
    """Add unit tests for :class:`glass.DistanceWeight`."""
    z = xp.linspace(0, 1, 6)

    # check shape

    w = glass.DistanceWeight(cosmo)(z)
    compare.assert_array_equal(w.shape, z.shape)

    # check first value is 1

    assert w[0] == 1

    # check values are decreasing

    compare.assert_array_less(w[1:], w[:-1])


def test_volume_weight(
    compare: type[Compare],
    cosmo: Cosmology,
    xp: ModuleType,
) -> None:
    """Add unit tests for :class:`glass.VolumeWeight`."""
    z = xp.linspace(0, 1, 6)

    # check shape

    w = glass.VolumeWeight(cosmo)(z)
    compare.assert_array_equal(w.shape, z.shape)

    # check first value is 0

    assert w[0] == 0

    # check values are increasing

    compare.assert_array_less(w[:-1], w[1:])


def test_density_weight(compare: type[Compare], cosmo: Cosmology) -> None:
    """Add unit tests for :class:`glass.DensityWeight`."""
    z = np.linspace(0, 1, 6)

    # check shape

    w = glass.DensityWeight(cosmo)(z)
    compare.assert_array_equal(w.shape, z.shape)

    # check first value is 0

    assert w[0] == 0

    # check values are increasing

    compare.assert_array_less(w[:-1], w[1:])


def test_tophat_windows(xp: ModuleType) -> None:
    """Add unit tests for :func:`glass.tophat_windows`."""
    zb = xp.asarray([0.0, 0.1, 0.2, 0.5, 1.0, 2.0])
    dz = 0.005

    ws = glass.tophat_windows(zb, dz)

    assert len(ws) == zb.size - 1

    assert all(
        z0 == w.za[0] and zn == w.za[-1]
        for w, z0, zn in zip(ws, zb, zb[1:], strict=False)
    )

    assert all(
        zn <= z0 + w.za.size * dz <= zn + dz
        for w, z0, zn in zip(ws, zb, zb[1:], strict=False)
    )

    assert all(xp.all(w.wa == 1) for w in ws)


def test_linear_windows(compare: type[Compare], xp: ModuleType) -> None:
    """Add unit tests for :func:`glass.linear_windows`."""
    dz = 1e-2
    zgrid = xp.asarray(
        [
            0.0,
            0.20224358,
            0.42896272,
            0.69026819,
            1.0,
        ],
    )

    # check spacing of redshift grid

    ws = glass.linear_windows(zgrid)
    compare.assert_allclose(dz, xp.mean(xp.diff(ws[0].za)), atol=1e-2)

    # check number of windows

    assert len(ws) == zgrid.size - 2

    # check values of zeff

    compare.assert_allclose([w.zeff for w in ws], zgrid[1:-1])

    # check weight function input

    ws = glass.linear_windows(
        zgrid,
        weight=lambda _: 0,
    )
    for w in ws:
        compare.assert_allclose(w.wa, xp.zeros_like(w.wa))

    # check error raised

    with pytest.raises(ValueError, match="nodes must have at least 3 entries"):
        glass.linear_windows(xp.asarray([]))

    # check warning issued

    with pytest.warns(
        UserWarning,
        match="first triangular window does not start at z=0",
    ):
        glass.linear_windows(xp.asarray([0.1, 0.2, 0.3]))


def test_cubic_windows(compare: type[Compare], xp: ModuleType) -> None:
    """Add unit tests for :func:`glass.cubic_windows`."""
    dz = 1e-2
    zgrid = xp.asarray(
        [
            0.0,
            0.20224358,
            0.42896272,
            0.69026819,
            1.0,
        ],
    )

    # check spacing of redshift grid

    ws = glass.cubic_windows(zgrid)
    compare.assert_allclose(dz, xp.mean(xp.diff(ws[0].za)), atol=1e-2)

    # check number of windows

    assert len(ws) == zgrid.size - 2

    # check values of zeff

    compare.assert_allclose([w.zeff for w in ws], zgrid[1:-1])

    # check weight function input

    ws = glass.cubic_windows(
        zgrid,
        weight=lambda _: 0,
    )
    for w in ws:
        compare.assert_allclose(w.wa, xp.zeros_like(w.wa))

    # check error raised

    with pytest.raises(ValueError, match="nodes must have at least 3 entries"):
        glass.cubic_windows(xp.asarray([]))

    # check warning issued

    with pytest.warns(
        UserWarning,
        match="first cubic spline window does not start at z=0",
    ):
        glass.cubic_windows(xp.asarray([0.1, 0.2, 0.3]))


def test_restrict(uxpx: _utils.XPAdditions, xp: ModuleType) -> None:
    """Add unit tests for :func:`glass.restrict`."""
    # Gaussian test function
    z = xp.linspace(0.0, 5.0, 1000)
    f = xp.exp(-(((z - 2.0) / 0.5) ** 2) / 2)

    # window for restriction
    w = glass.RadialWindow(
        za=xp.asarray([1.0, 2.0, 3.0, 4.0]),
        wa=xp.asarray([0.0, 0.5, 0.5, 0.0]),
    )

    zr, fr = glass.restrict(z, f, w)

    assert zr[0] == w.za[0]
    assert zr[-1] == w.za[-1]

    assert fr[0] == fr[-1] == 0.0

    for zi, wi in zip(w.za, w.wa, strict=False):
        i = xp.searchsorted(zr, zi)
        assert zr[i] == zi

        # Using design principle of scipy (i.e. copy, use np, copy back)
        assert fr[i] == wi * uxpx.interp(zi, z, f)

    for zi, fi in zip(z, f, strict=False):
        if w.za[0] <= zi <= w.za[-1]:
            i = xp.searchsorted(zr, zi)
            assert zr[i] == zi
            assert fr[i] == fi * uxpx.interp(zi, w.za, w.wa)


@pytest.mark.parametrize("method", ["lstsq", "nnls", "restrict"])
def test_partition(
    compare: type[Compare],
    method: str,
    uxpx: _utils.XPAdditions,
    xp: ModuleType,
) -> None:
    """Add unit tests for :func:`glass.partition`."""
    if (xp.__name__ == "jax.numpy") and (method in {"nnls"}):
        pytest.skip(f"Arrays in {method} are not immutable, so do not support jax")

    shells = [
        glass.RadialWindow(xp.asarray([0.0, 1.0]), xp.asarray([1.0, 0.0]), 0.0),
        glass.RadialWindow(
            xp.asarray([0.0, 1.0, 2.0]),
            xp.asarray([0.0, 1.0, 0.0]),
            0.5,
        ),
        glass.RadialWindow(
            xp.asarray([1.0, 2.0, 3.0]),
            xp.asarray([0.0, 1.0, 0.0]),
            1.5,
        ),
        glass.RadialWindow(
            xp.asarray([2.0, 3.0, 4.0]),
            xp.asarray([0.0, 1.0, 0.0]),
            2.5,
        ),
        glass.RadialWindow(
            xp.asarray([3.0, 4.0, 5.0]),
            xp.asarray([0.0, 1.0, 0.0]),
            3.5,
        ),
        glass.RadialWindow(xp.asarray([4.0, 5.0]), xp.asarray([0.0, 1.0]), 5.0),
    ]

    z = xp.linspace(0.0, 5.0, 1000)
    k = 1.0 + xp.reshape(xp.arange(6.0), (3, 2, 1))
    fz = xp.exp(-z / k)

    assert fz.shape == (3, 2, 1000)

    part = glass.partition(z, fz, shells, method=method)

    assert part.shape == (len(shells), 3, 2)

    compare.assert_allclose(xp.sum(part, axis=0), uxpx.trapezoid(fz, z))


def test_redshift_grid_default_xp() -> None:
    """Add unit tests for :func:`glass.redshift_grid` with default xp."""
    zmin = 0
    zmax = 1

    # check num input

    num = 5
    z = glass.redshift_grid(zmin, zmax, num=5)
    assert z.size == num + 1

    # check dz input

    dz = 0.2
    z = glass.redshift_grid(zmin, zmax, dz=dz)
    assert z.size == math.ceil((zmax - zmin) / dz) + 1

    # check dz for spacing which results in a max value above zmax

    z = glass.redshift_grid(zmin, zmax, dz=0.3)
    assert zmax < z[-1]


def test_redshift_grid(xp: ModuleType) -> None:
    """Add unit tests for :func:`glass.redshift_grid`."""
    zmin = 0
    zmax = 1

    # check num input

    num = 5
    z = glass.redshift_grid(zmin, zmax, num=5, xp=xp)
    assert z.size == num + 1

    # check dz input

    dz = 0.2
    z = glass.redshift_grid(zmin, zmax, dz=dz, xp=xp)
    assert z.size == math.ceil((zmax - zmin) / dz) + 1

    # check dz for spacing which results in a max value above zmax

    z = glass.redshift_grid(zmin, zmax, dz=0.3, xp=xp)
    assert zmax < z[-1]

    # check error raised

    with pytest.raises(
        ValueError,
        match="exactly one of grid step size or number of steps must be given",
    ):
        glass.redshift_grid(zmin, zmax, xp=xp)

    with pytest.raises(
        ValueError,
        match="exactly one of grid step size or number of steps must be given",
    ):
        glass.redshift_grid(zmin, zmax, dz=dz, num=num, xp=xp)


def test_distance_grid(compare: type[Compare], cosmo: Cosmology) -> None:
    """Add unit tests for :func:`glass.distance_grid`."""
    zmin = 0
    zmax = 1

    # check num input

    num = 5
    x = glass.distance_grid(cosmo, zmin, zmax, num=5)
    assert len(x) == num + 1

    # check dz input

    dx = 0.2
    x = glass.distance_grid(cosmo, zmin, zmax, dx=dx)
    assert len(x) == math.ceil((zmax - zmin) / dx) + 1

    # check decrease in distance

    x = glass.distance_grid(cosmo, zmin, zmax, dx=0.3)
    compare.assert_array_less(x[1:], x[:-1])

    # check error raised

    with pytest.raises(
        ValueError,
        match="exactly one of grid step size or number of steps must be given",
    ):
        glass.distance_grid(cosmo, zmin, zmax)

    with pytest.raises(
        ValueError,
        match="exactly one of grid step size or number of steps must be given",
    ):
        glass.distance_grid(cosmo, zmin, zmax, dx=dx, num=num)


def test_combine(
    compare: type[Compare],
    uxpx: _utils.XPAdditions,
    xp: ModuleType,
) -> None:
    """Add unit tests for :func:`glass.combine`."""
    z = xp.linspace(0.0, 5.0, 1000)
    weights = xp.asarray(
        [1.0, 0.90595172, 0.81025465, 0.72003963, 0.63892872, 0.56796183],
    )
    shells = [
        glass.RadialWindow(xp.asarray([0.0, 1.0]), xp.asarray([1.0, 0.0]), 0.0),
        glass.RadialWindow(
            xp.asarray([0.0, 1.0, 2.0]),
            xp.asarray([0.0, 1.0, 0.0]),
            0.5,
        ),
        glass.RadialWindow(
            xp.asarray([1.0, 2.0, 3.0]),
            xp.asarray([0.0, 1.0, 0.0]),
            1.5,
        ),
        glass.RadialWindow(
            xp.asarray([2.0, 3.0, 4.0]),
            xp.asarray([0.0, 1.0, 0.0]),
            2.5,
        ),
        glass.RadialWindow(
            xp.asarray([3.0, 4.0, 5.0]),
            xp.asarray([0.0, 1.0, 0.0]),
            3.5,
        ),
        glass.RadialWindow(xp.asarray([4.0, 5.0]), xp.asarray([0.0, 1.0]), 5.0),
    ]

    result = glass.combine(z, weights, shells)

    assert result.shape == z.shape

    # Check sum of result
    compare.assert_allclose(sum(result), 929.267284)

    # Check integral w.r.t z has not changed
    compare.assert_allclose(uxpx.trapezoid(result, z), 4.643139, rtol=1e-6)


def test_radial_window_immutable(xp: ModuleType) -> None:
    """Checks the :class:`RadialWindow` class is immutable."""
    wa = xp.asarray([0.0, 1.0, 0.0])
    za = xp.asarray([0.0, 1.0, 2.0])
    zeff = 1.0

    w = glass.RadialWindow(za, wa, zeff)

    with pytest.raises(
        dataclasses.FrozenInstanceError,
        match="cannot assign to field 'za'",
    ):
        w.za = za  # type: ignore[misc]

    with pytest.raises(
        dataclasses.FrozenInstanceError,
        match="cannot assign to field 'wa'",
    ):
        w.wa = wa  # type: ignore[misc]

    with pytest.raises(
        dataclasses.FrozenInstanceError,
        match="cannot assign to field 'zeff'",
    ):
        w.zeff = zeff  # type: ignore[misc]


def test_radial_window_zeff_none(compare: type[Compare], xp: ModuleType) -> None:
    """Checks ``zeff`` is computed when not provided to :class:`RadialWindow`."""
    # check zeff is computed when not provided

    wa = xp.asarray([0.0, 1.0, 0.0])
    za = xp.asarray([0.0, 1.0, 2.0])

    w = glass.RadialWindow(za, wa)

    compare.assert_allclose(w.zeff, 1.0)

    # check zeff is NaN when redshift array is empty

    za = xp.asarray([])

    w = glass.RadialWindow(za, wa)

    assert math.isnan(w.zeff)
