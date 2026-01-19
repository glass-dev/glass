from __future__ import annotations

from typing import TYPE_CHECKING

import healpix
import healpy
import numpy as np
import pytest

import glass._array_api_utils as _utils
import glass.healpix as hp

if TYPE_CHECKING:
    from types import ModuleType

    from glass._types import UnifiedGenerator
    from tests.fixtures.helper_classes import Compare, HealpixInputs


def test_alm2map() -> None:
    """Compare ``glass.healpix.alm2map`` against ``healpy.alm2map``."""
    pass  # noqa: PIE790


def test_alm2map_spin() -> None:
    """Compare ``glass.healpix.alm2map_spin`` against ``healpy.alm2map_spin``."""
    pass  # noqa: PIE790


def test_almxfl() -> None:
    """Compare ``glass.healpix.almxfl`` against ``healpy.almxfl``."""
    pass  # noqa: PIE790


def test_ang2pix() -> None:
    """Compare ``glass.healpix.ang2pix`` against ``healpix.ang2pix``."""
    pass  # noqa: PIE790


def test_ang2vec() -> None:
    """Compare ``glass.healpix.ang2vec`` against ``healpix.ang2vec``."""
    pass  # noqa: PIE790


def test_get_nside(
    healpix_inputs: type[HealpixInputs],
    urng: UnifiedGenerator,
) -> None:
    """Compare ``glass.healpix.get_nside`` against ``healpy.get_nside``."""
    kappa = healpix_inputs.kappa(urng)
    assert healpy.get_nside(np.asarray(kappa)) == hp.get_nside(kappa)


def test_map2alm(
    compare: type[Compare],
    healpix_inputs: type[HealpixInputs],
    urng: UnifiedGenerator,
) -> None:
    """Compare ``glass.healpix.map2alm`` against ``healpy.map2alm``."""
    kappa = healpix_inputs.kappa(urng)
    compare.assert_array_equal(
        healpy.map2alm(
            np.asarray(kappa),
            lmax=healpix_inputs.lmax,
            pol=False,
            use_pixel_weights=True,
        ),
        hp.map2alm(
            kappa,
            lmax=healpix_inputs.lmax,
            pol=False,
            use_pixel_weights=True,
        ),
    )


def test_npix2nside(
    healpix_inputs: type[HealpixInputs],
) -> None:
    """Compare ``glass.healpix.npix2nside`` against ``healpix.npix2nside``."""
    assert healpix.npix2nside(healpix_inputs.npix) == hp.npix2nside(healpix_inputs.npix)


def test_nside2npix(
    healpix_inputs: type[HealpixInputs],
) -> None:
    """Compare ``glass.healpix.nside2npix`` against ``healpix.nside2npix``."""
    assert healpix.nside2npix(healpix_inputs.nside) == hp.nside2npix(
        healpix_inputs.nside
    )


@pytest.mark.parametrize("pol", [False, True])
def test_pixwin(
    compare: type[Compare],
    healpix_inputs: type[HealpixInputs],
    pol: bool,  # noqa: FBT001
    xp: ModuleType,
) -> None:
    """Compare ``glass.healpix.pixwin`` against ``healpy.pixwin``."""
    old = healpy.pixwin(healpix_inputs.nside, lmax=healpix_inputs.lmax, pol=pol)
    new = hp.pixwin(healpix_inputs.nside, lmax=healpix_inputs.lmax, pol=pol, xp=xp)

    # Normalize to tuple
    old = old if isinstance(old, tuple) else (old,)
    new = new if isinstance(new, tuple) else (new,)

    for i in range(len(old)):
        compare.assert_array_equal(old[i], new[i])


def test_query_strip(
    compare: type[Compare],
    healpix_inputs: type[HealpixInputs],
    xp: ModuleType,
) -> None:
    """
    Compare ``glass.healpix.query_strip`` against ``healpy.query_strip``.

    The behaviour of ``query_strip`` has been changed subtly. Previously it
    returned the indices of the pixels within the strip. Now it returns a mask
    array indicating which pixels are within the strip.
    """
    old = np.zeros(healpix_inputs.npix)
    old[healpy.query_strip(healpix_inputs.nside, *healpix_inputs.thetas)] = 0
    new = np.zeros(healpix_inputs.npix, dtype=np.int64)
    new *= 1 - hp.query_strip(healpix_inputs.nside, healpix_inputs.thetas, xp=xp)
    compare.assert_array_equal(old, new)


def test_randang(
    compare: type[Compare],
    healpix_inputs: type[HealpixInputs],
    xp: ModuleType,
    urng: UnifiedGenerator,
) -> None:
    """
    Compare ``glass.healpix.randang`` against ``healpix.randang``.

    ``healpix.randang`` consumes the random numbers from RNG, changing its
    internal state. So the ``rng`` must be re-initialized before each call.
    """
    ipix = healpix_inputs.ipix(urng, xp)
    old = healpix.randang(
        healpix_inputs.nside, ipix, lonlat=True, rng=_utils.rng_dispatcher(xp=np)
    )
    new = hp.randang(
        healpix_inputs.nside, ipix, lonlat=True, rng=_utils.rng_dispatcher(xp=np)
    )
    assert len(old) == len(new)
    compare.assert_array_equal(old[0], new[0])
    compare.assert_array_equal(old[1], new[1])


@pytest.mark.parametrize("coord", ["CE", "GC"])
def test_rotate_map_pixel(
    compare: type[Compare],
    coord: str,
    healpix_inputs: type[HealpixInputs],
    urng: UnifiedGenerator,
    xp: ModuleType,
) -> None:
    """
    Compare ``glass.healpix.Rotator.rotate_map_pixel`` against
    ``healpy.Rotator.rotate_map_pixel``.
    """  # noqa: D205
    kappa = healpix_inputs.kappa(urng)
    compare.assert_array_equal(
        healpy.Rotator(coord=coord).rotate_map_pixel(np.asarray(kappa)),
        hp.Rotator(coord=coord, xp=xp).rotate_map_pixel(kappa),
    )
