from __future__ import annotations

from typing import TYPE_CHECKING

import healpix
import healpy
import numpy as np

import glass.healpix as hp

if TYPE_CHECKING:
    from types import ModuleType

    from glass._types import UnifiedGenerator
    from tests.fixtures.helper_classes import Compare

NPIX = 192
NSIDE = 4
THETAS = (30, 90)

def test_alm2map() -> None:
    """Compare ``glass.healpix.alm2map`` against ``healpy.alm2map``."""
    pass

def test_alm2map_spin() -> None:
    """Compare ``glass.healpix.alm2map_spin`` against ``healpy.alm2map_spin``."""
    pass

def test_almxfl() -> None:
    """Compare ``glass.healpix.almxfl`` against ``healpy.almxfl``."""
    pass

def test_ang2pix() -> None:
    """Compare ``glass.healpix.ang2pix`` against ``healpix.ang2pix``."""
    pass

def test_ang2vec() -> None:
    """Compare ``glass.healpix.ang2vec`` against ``healpix.ang2vec``."""
    pass

def test_get_nside(urng: UnifiedGenerator) -> None:
    """Compare ``glass.healpix.get_nside`` against ``healpy.get_nside``."""
    kappa = urng.normal(10, size=hp.nside2npix(NSIDE))
    assert hp.get_nside(kappa) == healpy.get_nside(np.asarray(kappa))

def test_map2alm() -> None:
    """Compare ``glass.healpix.map2alm`` against ``healpy.map2alm``."""
    pass

def test_npix2nside() -> None:
    """Compare ``glass.healpix.npix2nside`` against ``healpix.npix2nside``."""
    assert hp.npix2nside(NPIX) == healpix.npix2nside(NPIX)

def test_nside2npix() -> None:
    """Compare ``glass.healpix.nside2npix`` against ``healpix.nside2npix``."""
    assert hp.nside2npix(NSIDE) == healpix.nside2npix(NSIDE)

def test_pixwin() -> None:
    """Compare ``glass.healpix.pixwin`` against ``healpy.pixwin``."""
    pass

def test_query_strip(compare: type[Compare], xp: ModuleType) -> None:
    """
    Compare ``glass.healpix.query_strip`` against ``healpy.query_strip``.

    The beahviour of ``query_strip`` has been changed subtly. Previously it
    returned the indices of the pixels within the strip. Now it returns a mask
    array indicating which pixels are within the strip.
    """
    output = np.zeros(NPIX, dtype=np.int64)
    output[healpy.query_strip(NSIDE, *THETAS)] = 1
    compare.assert_array_equal(
        hp.query_strip(NSIDE, THETAS, xp=xp),
        xp.asarray(output),
    )

def test_randang() -> None:
    """Compare ``glass.healpix.randang`` against ``healpix.randang``."""
    pass

def test_rotator() -> None:
    """Compare ``glass.healpix.Rotator`` against ``healpy.Rotator``."""
    pass
