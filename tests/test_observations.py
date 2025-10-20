from types import ModuleType

import healpix
import numpy as np
import pytest

import glass
from glass._array_api_utils import UnifiedGenerator


def test_vmap_galactic_ecliptic() -> None:
    """Add unit tests for :func:`glass.vmap_galactic_ecliptic`."""
    n_side = 4

    # check shape

    vmap = glass.vmap_galactic_ecliptic(n_side)
    np.testing.assert_array_equal(len(vmap), healpix.nside2npix(n_side))

    # no rotation

    vmap = glass.vmap_galactic_ecliptic(n_side, galactic=(0, 0), ecliptic=(0, 0))
    np.testing.assert_array_equal(vmap, np.zeros_like(vmap))

    # check errors raised

    with pytest.raises(TypeError, match="galactic stripe must be a pair of numbers"):
        glass.vmap_galactic_ecliptic(n_side, galactic=(1,))  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="ecliptic stripe must be a pair of numbers"):
        glass.vmap_galactic_ecliptic(n_side, ecliptic=(1,))  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="galactic stripe must be a pair of numbers"):
        glass.vmap_galactic_ecliptic(n_side, galactic=(1, 2, 3))  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="ecliptic stripe must be a pair of numbers"):
        glass.vmap_galactic_ecliptic(n_side, ecliptic=(1, 2, 3))  # type: ignore[arg-type]


def test_gaussian_nz(xp: ModuleType, urng: UnifiedGenerator) -> None:
    """Add unit tests for :func:`glass.gaussian_nz`."""
    mean = 0
    sigma = 1
    z = xp.linspace(0, 1, 11)

    # check passing in the norm

    nz = glass.gaussian_nz(z, mean, sigma, norm=0)
    assert nz == pytest.approx(xp.zeros_like(nz))

    # check the value of each entry is close to the norm

    norm = 1
    nz = glass.gaussian_nz(z, mean, sigma, norm=norm)
    assert xp.sum(nz) / nz.shape[0] == pytest.approx(norm, rel=1e-2)

    # check multidimensionality size

    nz = glass.gaussian_nz(
        z,
        xp.tile(xp.asarray(mean), z.shape),
        xp.tile(xp.asarray(sigma), z.shape),
        norm=urng.normal(size=z.shape),
    )
    assert nz.shape == (z.size, z.size)


def test_smail_nz(xp: ModuleType) -> None:
    """Add unit tests for :func:`glass.smail_nz`."""
    alpha = 1
    beta = 1
    mode = 1
    z = xp.linspace(0, 1, 11)

    # check passing in the norm

    pz = glass.smail_nz(z, mode, alpha, beta, norm=0)
    assert xp.all(pz == xp.zeros_like(pz))


def test_fixed_zbins() -> None:
    """Add unit tests for :func:`glass.fixed_zbins`."""
    zmin = 0
    zmax = 1

    # check nbins input

    nbins = 5
    expected_zbins = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]
    zbins = glass.fixed_zbins(zmin, zmax, nbins=nbins)
    np.testing.assert_array_equal(len(zbins), nbins)
    np.testing.assert_allclose(zbins, expected_zbins, rtol=1e-15)

    # check dz input

    dz = 0.2
    zbins = glass.fixed_zbins(zmin, zmax, dz=dz)
    np.testing.assert_array_equal(len(zbins), np.ceil((zmax - zmin) / dz))
    np.testing.assert_allclose(zbins, expected_zbins, rtol=1e-15)

    # check dz for spacing which results in a max value above zmax

    zbins = glass.fixed_zbins(zmin, zmax, dz=0.3)
    np.testing.assert_array_less(zmax, zbins[-1][1])

    # check error raised

    with pytest.raises(ValueError, match="exactly one of nbins and dz must be given"):
        glass.fixed_zbins(zmin, zmax, nbins=nbins, dz=dz)


def test_equal_dens_zbins() -> None:
    """Add unit tests for :func:`glass.equal_dens_zbins`."""
    z = np.linspace(0, 1, 11)
    nbins = 5

    # check expected zbins returned

    expected_zbins = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]
    zbins = glass.equal_dens_zbins(z, np.ones_like(z), nbins)
    np.testing.assert_allclose(zbins, expected_zbins, rtol=1e-15)

    # check output shape

    np.testing.assert_array_equal(len(zbins), nbins)


def test_tomo_nz_gausserr() -> None:
    """Add unit tests for :func:`glass.tomo_nz_gausserr`."""
    sigma_0 = 0.1
    z = np.linspace(0, 1, 11)
    zbins = [(0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]

    # check zeros returned

    binned_nz = glass.tomo_nz_gausserr(z, np.zeros_like(z), sigma_0, zbins)
    np.testing.assert_array_equal(binned_nz, np.zeros_like(binned_nz))

    # check the shape of the output

    np.testing.assert_array_equal(binned_nz.shape, (len(zbins), len(z)))
