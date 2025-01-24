import numpy as np
import pytest

from cosmology import Cosmology

from glass import (
    RadialWindow,
    combine,  # noqa: F401
    cubic_windows,
    density_weight,
    distance_grid,
    distance_weight,
    linear_windows,
    partition,
    redshift_grid,
    restrict,
    tophat_windows,
    volume_weight,
)


def test_distance_weight(cosmo: Cosmology) -> None:
    """Add unit tests for :func:`distance_weight`."""
    z = np.linspace(0, 1, 6)

    # check shape

    w = distance_weight(z, cosmo)
    np.testing.assert_array_equal(w.shape, z.shape)

    # check first value is 1

    assert w[0] == 1

    # check values are decreasing

    np.testing.assert_array_less(w[1:], w[:-1])


def test_volume_weight(cosmo: Cosmology) -> None:
    """Add unit tests for :func:`volume_weight`."""
    z = np.linspace(0, 1, 6)

    # check shape

    w = volume_weight(z, cosmo)
    np.testing.assert_array_equal(w.shape, z.shape)

    # check first value is 0

    np.testing.assert_array_equal(w[0], 0)

    # check values are increasing

    np.testing.assert_array_less(w[:-1], w[1:])


def test_density_weight(cosmo: Cosmology) -> None:
    """Add unit tests for :func:`density_weight`."""
    z = np.linspace(0, 1, 6)

    # check shape

    w = density_weight(z, cosmo)
    np.testing.assert_array_equal(w.shape, z.shape)

    # check first value is 0

    assert w[0] == 0

    # check values are increasing

    np.testing.assert_array_less(w[:-1], w[1:])


def test_tophat_windows() -> None:
    """Add unit tests for :func:`tophat_windows`."""
    zb = np.array([0.0, 0.1, 0.2, 0.5, 1.0, 2.0])
    dz = 0.005

    ws = tophat_windows(zb, dz)

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
    """Add unit tests for :func:`linear_windows`."""
    dz = 1e-2
    zgrid = [
        0.0,
        0.20224358,
        0.42896272,
        0.69026819,
        1.0,
    ]

    # check spacing of redshift grid

    ws = linear_windows(zgrid)
    np.testing.assert_allclose(dz, np.diff(ws[0].za).mean(), atol=1e-2)

    # check number of windows

    np.testing.assert_array_equal(len(ws), len(zgrid) - 2)

    # check values of zeff

    np.testing.assert_array_equal([w.zeff for w in ws], zgrid[1:-1])

    # check weight function input

    ws = linear_windows(
        zgrid,
        weight=lambda _: 0,  # type: ignore[arg-type, return-value]
    )
    for w in ws:
        np.testing.assert_array_equal(w.wa, np.zeros_like(w.wa))

    # check error raised

    with pytest.raises(ValueError, match="nodes must have at least 3 entries"):
        linear_windows([])

    # check warning issued

    with pytest.warns(
        UserWarning, match="first triangular window does not start at z=0"
    ):
        linear_windows([0.1, 0.2, 0.3])


def test_cubic_windows() -> None:
    """Add unit tests for :func:`cubic_windows`."""
    dz = 1e-2
    zgrid = [
        0.0,
        0.20224358,
        0.42896272,
        0.69026819,
        1.0,
    ]

    # check spacing of redshift grid

    ws = cubic_windows(zgrid)
    np.testing.assert_allclose(dz, np.diff(ws[0].za).mean(), atol=1e-2)

    # check number of windows

    np.testing.assert_array_equal(len(ws), len(zgrid) - 2)

    # check values of zeff

    np.testing.assert_array_equal([w.zeff for w in ws], zgrid[1:-1])

    # check weight function input

    ws = cubic_windows(
        zgrid,
        weight=lambda _: 0,  # type: ignore[arg-type, return-value]
    )
    for w in ws:
        np.testing.assert_array_equal(w.wa, np.zeros_like(w.wa))

    # check error raised

    with pytest.raises(ValueError, match="nodes must have at least 3 entries"):
        cubic_windows([])

    # check warning issued

    with pytest.warns(
        UserWarning, match="first cubic spline window does not start at z=0"
    ):
        cubic_windows([0.1, 0.2, 0.3])


def test_restrict() -> None:
    """Add unit tests for :func:`restrict`."""
    # Gaussian test function
    z = np.linspace(0.0, 5.0, 1000)
    f = np.exp(-(((z - 2.0) / 0.5) ** 2) / 2)

    # window for restriction
    w = RadialWindow(
        za=np.array([1.0, 2.0, 3.0, 4.0]),
        wa=np.array([0.0, 0.5, 0.5, 0.0]),
    )

    zr, fr = restrict(z, f, w)

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
    """Add unit tests for :func:`partition`."""
    shells = [
        RadialWindow(np.array([0.0, 1.0]), np.array([1.0, 0.0]), 0.0),
        RadialWindow(np.array([0.0, 1.0, 2.0]), np.array([0.0, 1.0, 0.0]), 0.5),
        RadialWindow(np.array([1.0, 2.0, 3.0]), np.array([0.0, 1.0, 0.0]), 1.5),
        RadialWindow(np.array([2.0, 3.0, 4.0]), np.array([0.0, 1.0, 0.0]), 2.5),
        RadialWindow(np.array([3.0, 4.0, 5.0]), np.array([0.0, 1.0, 0.0]), 3.5),
        RadialWindow(np.array([4.0, 5.0]), np.array([0.0, 1.0]), 5.0),
    ]

    z = np.linspace(0.0, 5.0, 1000)
    k = 1 + np.arange(6).reshape(3, 2, 1)
    fz = np.exp(-z / k)

    assert fz.shape == (3, 2, 1000)

    part = partition(z, fz, shells, method=method)

    assert part.shape == (len(shells), 3, 2)

    np.testing.assert_allclose(part.sum(axis=0), np.trapezoid(fz, z))


def test_redshift_grid() -> None:
    """Add unit tests for :func:`redshift_grid`."""
    zmin = 0
    zmax = 1

    # check num input

    num = 5
    z = redshift_grid(zmin, zmax, num=5)
    assert len(z) == num + 1

    # check dz input

    dz = 0.2
    z = redshift_grid(zmin, zmax, dz=dz)
    assert len(z) == np.ceil((zmax - zmin) / dz) + 1

    # check dz for spacing which results in a max value above zmax

    z = redshift_grid(zmin, zmax, dz=0.3)
    assert z[-1] == zmax

    # check error raised

    with pytest.raises(
        ValueError,
        match="exactly one of grid step size or number of steps must be given",
    ):
        redshift_grid(zmin, zmax)

    with pytest.raises(
        ValueError,
        match="exactly one of grid step size or number of steps must be given",
    ):
        redshift_grid(zmin, zmax, dz=dz, num=num)


def test_distance_grid(cosmo: Cosmology) -> None:
    """Add unit tests for :func:`distance_grid`."""
    zmin = 0
    zmax = 1

    # check num input

    num = 5
    x = distance_grid(cosmo, zmin, zmax, num=5)
   assert len(x) == num + 1

    # check dz input

    dx = 0.2
    x = distance_grid(cosmo, zmin, zmax, dx=dx)
    assert len(x) == np.ceil((zmax - zmin) / dx) + 1

    # check decrease in distance

    x = distance_grid(cosmo, zmin, zmax, dx=0.3)
    np.testing.assert_array_less(x[1:], x[:-1])

    # check error raised

    with pytest.raises(
        ValueError,
        match="exactly one of grid step size or number of steps must be given",
    ):
        distance_grid(cosmo, zmin, zmax)

    with pytest.raises(
        ValueError,
        match="exactly one of grid step size or number of steps must be given",
    ):
        distance_grid(cosmo, zmin, zmax, dx=dx, num=num)


def test_combine() -> None:
    """Add unit tests for :func:`combine`."""
