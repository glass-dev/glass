import numpy as np
import pytest

from glass import fixed_zbins, gaussian_nz, smail_nz, vmap_galactic_ecliptic


def test_vmap_galactic_ecliptic() -> None:
    """Add unit tests for vmap_galactic_ecliptic."""
    # test errors raised

    with pytest.raises(TypeError, match="galactic stripe must be a pair of numbers"):
        vmap_galactic_ecliptic(1, galactic=(1, 2, 3))  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="ecliptic stripe must be a pair of numbers"):
        vmap_galactic_ecliptic(1, ecliptic=(1, 2, 3))  # type: ignore[arg-type]


def test_gaussian_nz(rng: np.random.Generator) -> None:
    """Add unit tests for gaussian_nz."""
    z = np.linspace(0, 1, 11)
    mean = 0
    sigma = 1

    # check passing in the norm

    nz = gaussian_nz(z, mean, sigma, norm=0)
    np.testing.assert_array_equal(nz, np.zeros(nz.shape))

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
    z = np.linspace(0, 1, 11)
    mode = 1
    alpha = 1
    beta = 1

    # check passing in the norm

    pz = smail_nz(z, mode, alpha, beta, norm=0)
    np.testing.assert_array_equal(pz, np.zeros(pz.shape))


def test_fixed_zbins() -> None:
    """Add unit tests for fixed_zbins."""
    # test error raised

    with pytest.raises(ValueError, match="exactly one of nbins and dz must be given"):
        fixed_zbins(0, 1, nbins=10, dz=0.1)


def test_equal_dens_zbins() -> None:
    """Add unit tests for equal_dens_zbins."""


def test_tomo_nz_gausserr() -> None:
    """Add unit tests for tomo_nz_gausserr."""
