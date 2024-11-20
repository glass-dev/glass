import numpy as np
import pytest

from glass import (  # type: ignore[attr-defined]
    equal_dens_zbins,
    fixed_zbins,
    gaussian_nz,
    smail_nz,
    tomo_nz_gausserr,
    vmap_galactic_ecliptic,
)


def test_vmap_galactic_ecliptic() -> None:
    """Add unit tests for vmap_galactic_ecliptic."""
    # check errors raised

    with pytest.raises(TypeError, match="galactic stripe must be a pair of numbers"):
        vmap_galactic_ecliptic(1, galactic=(1, 2, 3))  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="ecliptic stripe must be a pair of numbers"):
        vmap_galactic_ecliptic(1, ecliptic=(1, 2, 3))  # type: ignore[arg-type]


def test_gaussian_nz(rng: np.random.Generator) -> None:
    """Add unit tests for gaussian_nz."""
    mean = 0
    sigma = 1
    z = np.linspace(0, 1, 11)

    # check passing in the norm

    nz = gaussian_nz(z, mean, sigma, norm=0)
    np.testing.assert_array_equal(nz, np.zeros_like(nz))

    # check the value of each entry is close to the norm

    norm = 1
    nz = gaussian_nz(z, mean, sigma, norm=norm)
    np.testing.assert_allclose(nz.sum() / nz.shape, norm, rtol=1e-2)

    # check multidimensionality size

    nz = gaussian_nz(
        z,
        np.tile(mean, z.shape),
        np.tile(sigma, z.shape),
        norm=rng.normal(size=z.shape),
    )
    np.testing.assert_array_equal(nz.shape, (len(z), len(z)))


def test_smail_nz() -> None:
    """Add unit tests for smail_nz."""
    alpha = 1
    beta = 1
    mode = 1
    z = np.linspace(0, 1, 11)

    # check passing in the norm

    pz = smail_nz(z, mode, alpha, beta, norm=0)
    np.testing.assert_array_equal(pz, np.zeros_like(pz))


def test_fixed_zbins() -> None:
    """Add unit tests for fixed_zbins."""
    # check error raised

    with pytest.raises(ValueError, match="exactly one of nbins and dz must be given"):
        fixed_zbins(0, 1, nbins=10, dz=0.1)


def test_equal_dens_zbins() -> None:
    """Add unit tests for equal_dens_zbins."""
    z = np.linspace(0, 1, 11)
    nbins = 5

    # check expected zbins returned

    zbins = equal_dens_zbins(z, np.ones_like(z), nbins)
    expected_zbins = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]
    np.testing.assert_allclose(zbins, expected_zbins, rtol=1e-15)

    # check output shape

    np.testing.assert_array_equal(len(zbins), nbins)


def test_tomo_nz_gausserr() -> None:
    """Add unit tests for tomo_nz_gausserr."""
    sigma_0 = 0.1
    z = np.linspace(0, 1, 11)
    zbins = [(0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]

    # check zeros returned

    binned_nz = tomo_nz_gausserr(z, np.zeros_like(z), sigma_0, zbins)
    np.testing.assert_array_equal(binned_nz, np.zeros_like(binned_nz))

    # check the shape of the output

    np.testing.assert_array_equal(binned_nz.shape, (len(zbins), len(z)))
