from __future__ import annotations

import math
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


@pytest.mark.parametrize(
    ("pixwin", "pol"),
    [
        (False, False),
        (False, True),
        (True, False),
        (True, True),
    ],
)
def test_alm2map_individual(
    compare: type[Compare],
    healpix_inputs: type[HealpixInputs],
    pixwin: bool,  # noqa: FBT001
    pol: bool,  # noqa: FBT001
    urng: UnifiedGenerator,
) -> None:
    """Compare ``glass.healpix.alm2map`` against ``healpy.alm2map``."""
    alm = healpix_inputs.alm(rng=urng)
    compare.assert_array_equal(
        healpy.alm2map(
            np.asarray(alm),
            healpix_inputs.nside,
            lmax=healpix_inputs.lmax,
            pixwin=pixwin,
            pol=pol,
        ),
        hp.alm2map(
            alm,
            healpix_inputs.nside,
            lmax=healpix_inputs.lmax,
            pixwin=pixwin,
            pol=pol,
        ),
    )


@pytest.mark.parametrize(
    ("pixwin", "pol"),
    [
        (False, False),
        (False, True),
        (True, False),
        (True, True),
    ],
)
def test_alm2map_sequence(
    compare: type[Compare],
    healpix_inputs: type[HealpixInputs],
    pixwin: bool,  # noqa: FBT001
    pol: bool,  # noqa: FBT001
    urng: UnifiedGenerator,
) -> None:
    """Compare ``glass.healpix.alm2map`` against ``healpy.alm2map``."""
    alm = healpix_inputs.alm(rng=urng)
    blm = healpix_inputs.alm(rng=urng)
    clm = healpix_inputs.alm(rng=urng)
    old = healpy.alm2map(
        [np.asarray(alm), np.asarray(blm), np.asarray(clm)],
        healpix_inputs.nside,
        lmax=healpix_inputs.lmax,
        pixwin=pixwin,
        pol=pol,
    )
    new = hp.alm2map(
        [alm, blm, clm],
        healpix_inputs.nside,
        lmax=healpix_inputs.lmax,
        pixwin=pixwin,
        pol=pol,
    )
    compare.assert_array_equal(old, new)


@pytest.mark.parametrize("spin", [1, 2])
def test_alm2map_spin(
    compare: type[Compare],
    healpix_inputs: type[HealpixInputs],
    spin: int,
    urng: UnifiedGenerator,
) -> None:
    """Compare ``glass.healpix.alm2map_spin`` against ``healpy.alm2map_spin``."""
    alm = healpix_inputs.alm(rng=urng)
    blm = healpix_inputs.alm(rng=urng)
    old = healpy.alm2map_spin(
        [alm, blm],
        healpix_inputs.nside,
        spin,
        healpix_inputs.lmax,
    )
    new = hp.alm2map_spin([alm, blm], healpix_inputs.nside, spin, healpix_inputs.lmax)
    assert type(old) is type(new)
    assert len(old) == len(new)
    for i in range(len(old)):
        compare.assert_array_equal(old[i], new[i])


def test_almxfl(
    compare: type[Compare],
    healpix_inputs: type[HealpixInputs],
    urng: UnifiedGenerator,
) -> None:
    """Compare ``glass.healpix.almxfl`` against ``healpy.almxfl``."""
    alm = healpix_inputs.alm(rng=urng)
    fl = healpix_inputs.fl(rng=urng)
    compare.assert_array_equal(
        healpy.almxfl(alm, fl),
        hp.almxfl(alm, fl),
    )


@pytest.mark.parametrize(
    ("lonlat", "max_phi", "max_theta"),
    [
        (False, math.pi, math.pi / 2),
        (True, 180, 90),
    ],
)
def test_ang2pix(  # noqa: PLR0913
    compare: type[Compare],
    healpix_inputs: type[HealpixInputs],
    lonlat: bool,  # noqa: FBT001
    max_phi: float,
    max_theta: float,
    urng: UnifiedGenerator,
    xp: ModuleType,
) -> None:
    """Compare ``glass.healpix.ang2pix`` against ``healpix.ang2pix``."""
    thetas = healpix_inputs.longitudes(max_theta, rng=urng)
    phis = healpix_inputs.latitudes(max_phi, rng=urng)
    old = healpix.ang2pix(healpix_inputs.nside, thetas, phis, lonlat=lonlat)
    new = hp.ang2pix(healpix_inputs.nside, thetas, phis, lonlat=lonlat, xp=xp)
    compare.assert_array_equal(old, new)


@pytest.mark.parametrize(
    ("lonlat", "max_phi", "max_theta"),
    [
        (False, math.pi, math.pi / 2),
        (True, 180, 90),
    ],
)
def test_ang2vec(  # noqa: PLR0913
    compare: type[Compare],
    healpix_inputs: type[HealpixInputs],
    lonlat: bool,  # noqa: FBT001
    max_phi: float,
    max_theta: float,
    urng: UnifiedGenerator,
    xp: ModuleType,
) -> None:
    """Compare ``glass.healpix.ang2vec`` against ``healpix.ang2vec``."""
    thetas = healpix_inputs.longitudes(max_theta, rng=urng)
    phis = healpix_inputs.latitudes(max_phi, rng=urng)
    old = healpix.ang2vec(thetas, phis, lonlat=lonlat)
    new = hp.ang2vec(thetas, phis, lonlat=lonlat, xp=xp)
    assert type(old) is type(new)
    assert len(old) == len(new)
    for i in range(len(old)):
        compare.assert_array_equal(old[i], new[i])


def test_get_nside(
    healpix_inputs: type[HealpixInputs],
    urng: UnifiedGenerator,
) -> None:
    """Compare ``glass.healpix.get_nside`` against ``healpy.get_nside``."""
    kappa = healpix_inputs.kappa(rng=urng)
    assert healpy.get_nside(np.asarray(kappa)) == hp.get_nside(kappa)


@pytest.mark.parametrize(
    ("pol", "use_pixel_weights"),
    [
        (False, False),
        (False, True),
        (True, False),
        (True, True),
    ],
)
def test_map2alm_individual(
    compare: type[Compare],
    healpix_inputs: type[HealpixInputs],
    pol: bool,  # noqa: FBT001
    urng: UnifiedGenerator,
    use_pixel_weights: bool,  # noqa: FBT001
) -> None:
    """Compare ``glass.healpix.map2alm`` against ``healpy.map2alm``."""
    kappa = healpix_inputs.kappa(rng=urng)
    compare.assert_array_equal(
        healpy.map2alm(
            np.asarray(kappa),
            lmax=healpix_inputs.lmax,
            pol=pol,
            use_pixel_weights=use_pixel_weights,
        ),
        hp.map2alm(
            kappa,
            lmax=healpix_inputs.lmax,
            pol=pol,
            use_pixel_weights=use_pixel_weights,
        ),
    )


@pytest.mark.parametrize(
    ("pol", "use_pixel_weights"),
    [
        (False, False),
        (False, True),
        (True, False),
        (True, True),
    ],
)
def test_map2alm_sequence(
    compare: type[Compare],
    healpix_inputs: type[HealpixInputs],
    pol: bool,  # noqa: FBT001
    urng: UnifiedGenerator,
    use_pixel_weights: bool,  # noqa: FBT001
) -> None:
    """Compare ``glass.healpix.map2alm`` against ``healpy.map2alm``."""
    kappa1 = healpix_inputs.kappa(rng=urng)
    kappa2 = healpix_inputs.kappa(rng=urng)
    kappa3 = healpix_inputs.kappa(rng=urng)
    compare.assert_array_equal(
        healpy.map2alm(
            [np.asarray(kappa1), np.asarray(kappa2), np.asarray(kappa3)],
            lmax=healpix_inputs.lmax,
            pol=pol,
            use_pixel_weights=use_pixel_weights,
        ),
        hp.map2alm(
            [kappa1, kappa2, kappa3],
            lmax=healpix_inputs.lmax,
            pol=pol,
            use_pixel_weights=use_pixel_weights,
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
        healpix_inputs.nside,
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

    assert len(old) == len(new)
    for i in range(len(old)):
        compare.assert_array_equal(old[i], new[i])


@pytest.mark.parametrize("thetas", [(20, 80), (30, 90)])
def test_query_strip(
    compare: type[Compare],
    healpix_inputs: type[HealpixInputs],
    thetas: tuple[int, int],
    xp: ModuleType,
) -> None:
    """
    Compare ``glass.healpix.query_strip`` against ``healpy.query_strip``.

    The behaviour of ``query_strip`` has been changed subtly. Previously it
    returned the indices of the pixels within the strip. Now it returns a mask
    array indicating which pixels are within the strip.
    """
    old = np.ones(healpix_inputs.npix)
    old[healpy.query_strip(healpix_inputs.nside, *thetas)] = 0
    new = np.ones(healpix_inputs.npix, dtype=np.int64)
    new *= 1 - hp.query_strip(healpix_inputs.nside, thetas, xp=xp)
    compare.assert_array_equal(old, new)


@pytest.mark.parametrize("lonlat", [False, True])
def test_randang(
    compare: type[Compare],
    healpix_inputs: type[HealpixInputs],
    lonlat: bool,  # noqa: FBT001
    xp: ModuleType,
    urng: UnifiedGenerator,
) -> None:
    """
    Compare ``glass.healpix.randang`` against ``healpix.randang``.

    ``healpix.randang`` consumes the random numbers from RNG, changing its
    internal state. So the ``rng`` must be re-initialized before each call.
    """
    ipix = healpix_inputs.ipix(rng=urng, xp=xp)
    old = healpix.randang(
        healpix_inputs.nside,
        ipix,
        lonlat=lonlat,
        rng=_utils.rng_dispatcher(xp=np),
    )
    new = hp.randang(
        healpix_inputs.nside,
        ipix,
        lonlat=lonlat,
        rng=_utils.rng_dispatcher(xp=np),
    )
    assert type(old) is type(new)
    assert len(old) == len(new)
    for i in range(len(old)):
        compare.assert_array_equal(old[i], new[i])


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
    kappa = healpix_inputs.kappa(rng=urng)
    compare.assert_array_equal(
        healpy.Rotator(coord=coord).rotate_map_pixel(np.asarray(kappa)),
        hp.Rotator(coord=coord, xp=xp).rotate_map_pixel(kappa),
    )
