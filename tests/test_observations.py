import healpix
import numpy as np
import pytest
from numpy.typing import NDArray

import glass
import glass.observations


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


def test_gaussian_nz(rng: np.random.Generator) -> None:
    """Add unit tests for :func:`glass.gaussian_nz`."""
    mean = 0
    sigma = 1
    z = np.linspace(0, 1, 11)

    # check passing in the norm

    nz = glass.gaussian_nz(z, mean, sigma, norm=0)
    np.testing.assert_array_equal(nz, np.zeros_like(nz))

    # check the value of each entry is close to the norm

    norm = 1
    nz = glass.gaussian_nz(z, mean, sigma, norm=norm)
    np.testing.assert_allclose(nz.sum() / nz.shape, norm, rtol=1e-2)

    # check multidimensionality size

    nz = glass.gaussian_nz(
        z,
        np.tile(mean, z.shape),
        np.tile(sigma, z.shape),
        norm=rng.normal(size=z.shape),
    )
    np.testing.assert_array_equal(nz.shape, (len(z), len(z)))


def test_smail_nz() -> None:
    """Add unit tests for :func:`glass.smail_nz`."""
    alpha = 1
    beta = 1
    mode = 1
    z = np.linspace(0, 1, 11)

    # check passing in the norm

    pz = glass.smail_nz(z, mode, alpha, beta, norm=0)
    np.testing.assert_array_equal(pz, np.zeros_like(pz))


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


@pytest.mark.parametrize(
    ("vardepth_map", "n_bins", "zbins", "index", "expected_mask"),
    [
        (
            np.array([[1.0, 2.0], [3.0, 4.0]]),
            2,
            [(0.0, 0.5), (0.5, 1.0)],
            (0, 0),
            np.array([1.0, 2.0]),
        ),
        (
            np.array([[1.0, 2.0], [3.0, 4.0]]),
            2,
            [(0.0, 0.5), (0.5, 1.0)],
            (1, 1),
            np.array([3.0, 4.0]),
        ),
    ],
    ids=["test_valid_index_1", "test_valid_index_2"],
)
def test_getitem_happy_path(
    vardepth_map: NDArray[np.float64],
    n_bins: int,
    zbins: list[tuple[float, float]],
    index: tuple[int, int],
    expected_mask: NDArray[np.float64],
) -> None:
    # Arrange
    mask = glass.observations.AngularVariableDepthMask(vardepth_map, n_bins, zbins)

    # Act
    result = mask[index]

    # Assert
    np.testing.assert_array_equal(result, expected_mask)


@pytest.mark.parametrize(
    ("vardepth_map", "n_bins", "zbins", "index", "expected_error"),
    [
        (
            np.array([[1.0, 2.0], [3.0, 4.0]]),
            2,
            [(0.0, 0.5), (0.5, 1.0)],
            (2, 0),
            ValueError,
        ),
        (
            np.array([[1.0, 2.0], [3.0, 4.0]]),
            2,
            [(0.0, 0.5), (0.5, 1.0)],
            (0, 2),
            ValueError,
        ),
        (
            np.array([[1.0, 2.0], [3.0, 4.0]]),
            2,
            [(0.0, 0.5), (0.5, 1.0)],
            0,
            TypeError,
        ),
    ],
    ids=[
        "test_invalid_index_1",
        "test_invalid_index_2",
        "test_invalid_index_type",
    ],
)
def test_getitem_error_cases(
    vardepth_map: NDArray[np.float64],
    n_bins: int,
    zbins: list[tuple[float, float]],
    index: tuple[int, int],
    expected_error: type[BaseException],
) -> None:
    # Arrange
    mask = glass.observations.AngularVariableDepthMask(vardepth_map, n_bins, zbins)

    # Act & Assert
    with pytest.raises(expected_error):
        mask[index]
