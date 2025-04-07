import dataclasses
import math

import numpy as np
import pytest

from cosmology import Cosmology

import glass


def test_distance_weight(cosmo: Cosmology) -> None:
    """Add unit tests for :class:`glass.DistanceWeight`."""
    z = np.linspace(0, 1, 6)

    # check shape

    w = glass.DistanceWeight(cosmo)(z)
    np.testing.assert_array_equal(w.shape, z.shape)

    # check first value is 1

    assert w[0] == 1

    # check values are decreasing

    np.testing.assert_array_less(w[1:], w[:-1])


def test_volume_weight(cosmo: Cosmology) -> None:
    """Add unit tests for :class:`glass.VolumeWeight`."""
    z = np.linspace(0, 1, 6)

    # check shape

    w = glass.VolumeWeight(cosmo)(z)
    np.testing.assert_array_equal(w.shape, z.shape)

    # check first value is 0

    assert w[0] == 0

    # check values are increasing

    np.testing.assert_array_less(w[:-1], w[1:])


def test_density_weight(cosmo: Cosmology) -> None:
    """Add unit tests for :class:`glass.DensityWeight`."""
    z = np.linspace(0, 1, 6)

    # check shape

    w = glass.DensityWeight(cosmo)(z)
    np.testing.assert_array_equal(w.shape, z.shape)

    # check first value is 0

    assert w[0] == 0

    # check values are increasing

    np.testing.assert_array_less(w[:-1], w[1:])


def test_tophat_windows() -> None:
    """Add unit tests for :func:`glass.tophat_windows`."""
    zb = np.array([0.0, 0.1, 0.2, 0.5, 1.0, 2.0])
    dz = 0.005

    ws = glass.tophat_windows(zb, dz)

    assert len(ws) == len(zb) - 1

    assert all(
        z0 == w.za[0] and zn == w.za[-1]
        for w, z0, zn in zip(ws, zb, zb[1:], strict=False)
    )

    assert all(
        zn <= z0 + len(w.za) * dz <= zn + dz
        for w, z0, zn in zip(ws, zb, zb[1:], strict=False)
    )

    assert all(np.all(w.wa == 1) for w in ws)


def test_linear_windows() -> None:
    """Add unit tests for :func:`glass.linear_windows`."""
    dz = 1e-2
    zgrid = [
        0.0,
        0.20224358,
        0.42896272,
        0.69026819,
        1.0,
    ]

    # check spacing of redshift grid

    ws = glass.linear_windows(zgrid)
    np.testing.assert_allclose(dz, np.diff(ws[0].za).mean(), atol=1e-2)

    # check number of windows

    assert len(ws) == len(zgrid) - 2

    # check values of zeff

    np.testing.assert_array_equal([w.zeff for w in ws], zgrid[1:-1])

    # check weight function input

    ws = glass.linear_windows(
        zgrid,
        weight=lambda _: 0,  # type: ignore[arg-type, return-value]
    )
    for w in ws:
        np.testing.assert_array_equal(w.wa, np.zeros_like(w.wa))

    # check error raised

    with pytest.raises(ValueError, match="nodes must have at least 3 entries"):
        glass.linear_windows([])

    # check warning issued

    with pytest.warns(
        UserWarning, match="first triangular window does not start at z=0"
    ):
        glass.linear_windows([0.1, 0.2, 0.3])


def test_cubic_windows() -> None:
    """Add unit tests for :func:`glass.cubic_windows`."""
    dz = 1e-2
    zgrid = [
        0.0,
        0.20224358,
        0.42896272,
        0.69026819,
        1.0,
    ]

    # check spacing of redshift grid

    ws = glass.cubic_windows(zgrid)
    np.testing.assert_allclose(dz, np.diff(ws[0].za).mean(), atol=1e-2)

    # check number of windows

    assert len(ws) == len(zgrid) - 2

    # check values of zeff

    np.testing.assert_array_equal([w.zeff for w in ws], zgrid[1:-1])

    # check weight function input

    ws = glass.cubic_windows(
        zgrid,
        weight=lambda _: 0,  # type: ignore[arg-type, return-value]
    )
    for w in ws:
        np.testing.assert_array_equal(w.wa, np.zeros_like(w.wa))

    # check error raised

    with pytest.raises(ValueError, match="nodes must have at least 3 entries"):
        glass.cubic_windows([])

    # check warning issued

    with pytest.warns(
        UserWarning, match="first cubic spline window does not start at z=0"
    ):
        glass.cubic_windows([0.1, 0.2, 0.3])


def test_restrict() -> None:
    """Add unit tests for :func:`glass.restrict`."""
    # Gaussian test function
    z = np.linspace(0.0, 5.0, 1000)
    f = np.exp(-(((z - 2.0) / 0.5) ** 2) / 2)

    # window for restriction
    w = glass.RadialWindow(
        za=np.array([1.0, 2.0, 3.0, 4.0]),
        wa=np.array([0.0, 0.5, 0.5, 0.0]),
    )

    zr, fr = glass.restrict(z, f, w)

    assert zr[0] == w.za[0]
    assert zr[-1] == w.za[-1]

    assert fr[0] == fr[-1] == 0.0

    for zi, wi in zip(w.za, w.wa, strict=False):
        i = np.searchsorted(zr, zi)
        assert zr[i] == zi
        assert fr[i] == wi * np.interp(zi, z, f)

    for zi, fi in zip(z, f, strict=False):
        if w.za[0] <= zi <= w.za[-1]:
            i = np.searchsorted(zr, zi)
            assert zr[i] == zi
            assert fr[i] == fi * np.interp(zi, w.za, w.wa)


@pytest.mark.parametrize("method", ["lstsq", "nnls", "restrict"])
def test_partition(method: str) -> None:
    """Add unit tests for :func:`glass.partition`."""
    shells = [
        glass.RadialWindow(np.array([0.0, 1.0]), np.array([1.0, 0.0]), 0.0),
        glass.RadialWindow(np.array([0.0, 1.0, 2.0]), np.array([0.0, 1.0, 0.0]), 0.5),
        glass.RadialWindow(np.array([1.0, 2.0, 3.0]), np.array([0.0, 1.0, 0.0]), 1.5),
        glass.RadialWindow(np.array([2.0, 3.0, 4.0]), np.array([0.0, 1.0, 0.0]), 2.5),
        glass.RadialWindow(np.array([3.0, 4.0, 5.0]), np.array([0.0, 1.0, 0.0]), 3.5),
        glass.RadialWindow(np.array([4.0, 5.0]), np.array([0.0, 1.0]), 5.0),
    ]

    z = np.linspace(0.0, 5.0, 1000)
    k = 1 + np.arange(6).reshape(3, 2, 1)
    fz = np.exp(-z / k)

    assert fz.shape == (3, 2, 1000)

    part = glass.partition(z, fz, shells, method=method)

    assert part.shape == (len(shells), 3, 2)

    np.testing.assert_allclose(part.sum(axis=0), np.trapezoid(fz, z))


def test_redshift_grid() -> None:
    """Add unit tests for :func:`glass.redshift_grid`."""
    zmin = 0
    zmax = 1

    # check num input

    num = 5
    z = glass.redshift_grid(zmin, zmax, num=5)
    assert len(z) == num + 1

    # check dz input

    dz = 0.2
    z = glass.redshift_grid(zmin, zmax, dz=dz)
    assert len(z) == np.ceil((zmax - zmin) / dz) + 1

    # check dz for spacing which results in a max value above zmax

    z = glass.redshift_grid(zmin, zmax, dz=0.3)
    assert zmax < z[-1]

    # check error raised

    with pytest.raises(
        ValueError,
        match="exactly one of grid step size or number of steps must be given",
    ):
        glass.redshift_grid(zmin, zmax)

    with pytest.raises(
        ValueError,
        match="exactly one of grid step size or number of steps must be given",
    ):
        glass.redshift_grid(zmin, zmax, dz=dz, num=num)


def test_distance_grid(cosmo: Cosmology) -> None:
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
    assert len(x) == np.ceil((zmax - zmin) / dx) + 1

    # check decrease in distance

    x = glass.distance_grid(cosmo, zmin, zmax, dx=0.3)
    np.testing.assert_array_less(x[1:], x[:-1])

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


def test_combine() -> None:
    """Add unit tests for :func:`glass.combine`."""


def test_radial_window_immutable() -> None:
    """Checks the :class:`RadialWindow` class is immutable."""
    wa = np.array([0.0, 1.0, 0.0])
    za = np.array([0.0, 1.0, 2.0])
    zeff = 1.0

    w = glass.RadialWindow(za, wa, zeff)

    with pytest.raises(
        dataclasses.FrozenInstanceError, match="cannot assign to field 'za'"
    ):
        w.za = za

    with pytest.raises(
        dataclasses.FrozenInstanceError, match="cannot assign to field 'wa'"
    ):
        w.wa = wa

    with pytest.raises(
        dataclasses.FrozenInstanceError, match="cannot assign to field 'zeff'"
    ):
        w.zeff = zeff


def test_radial_window_zeff_none() -> None:
    """Checks ``zeff`` is computed when not provided to :class:`RadialWindow`."""
    # check zeff is computed when not provided

    wa = np.array([0.0, 1.0, 0.0])
    za = np.array([0.0, 1.0, 2.0])

    w = glass.RadialWindow(za, wa)

    np.testing.assert_equal(w.zeff, 1.0)

    # check zeff is 0.0 when redshift array is empty

    za = np.array([])

    w = glass.RadialWindow(za, wa)

    np.testing.assert_equal(w.zeff, math.nan)
