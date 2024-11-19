import pytest

from glass import fixed_zbins, vmap_galactic_ecliptic


def test_vmap_galactic_ecliptic() -> None:
    """Add unit tests for vmap_galactic_ecliptic."""
    # test errors raised
    with pytest.raises(TypeError, match="galactic stripe must be a pair of numbers"):
        vmap_galactic_ecliptic(1, galactic=(1, 2, 3))  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="ecliptic stripe must be a pair of numbers"):
        vmap_galactic_ecliptic(1, ecliptic=(1, 2, 3))  # type: ignore[arg-type]


def test_gaussian_nz() -> None:
    """Add unit tests for gaussian_nz."""


def test_smail_nz() -> None:
    """Add unit tests for smail_nz."""


def test_fixed_zbins() -> None:
    """Add unit tests for fixed_zbins."""
    # test error raised
    with pytest.raises(ValueError, match="exactly one of nbins and dz must be given"):
        fixed_zbins(0, 1, nbins=10, dz=0.1)


def test_equal_dens_zbins() -> None:
    """Add unit tests for equal_dens_zbins."""


def test_tomo_nz_gausserr() -> None:
    """Add unit tests for tomo_nz_gausserr."""
