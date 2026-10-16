from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

import array_api_extra as xpx

import glass.harmonics
import glass.healpix as hp

if TYPE_CHECKING:
    from types import ModuleType


def test_multalm(xp: ModuleType) -> None:
    # check output values and shapes

    alm = xp.asarray([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    bl = xp.asarray([2.0, 0.5, 1.0])
    alm_copy = xp.asarray(alm, copy=True)

    result = glass.harmonics.multalm(alm, bl)

    expected_result = xp.asarray([2.0, 1.0, 3.0, 2.0, 5.0, 6.0])
    xpx.testing.assert_equal(result, expected_result)
    with pytest.raises(AssertionError, match="Not equal to tolerance"):
        xpx.testing.assert_close(alm_copy, result)

    # multiple with 1s

    alm = xp.asarray([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    bl = xp.ones(3)

    result = glass.harmonics.multalm(alm, bl)
    xpx.testing.assert_equal(result, alm)

    # multiple with 0s

    bl = xp.asarray([0.0, 1.0, 0.0])

    result = glass.harmonics.multalm(alm, bl)

    expected_result = xp.asarray([0.0, 2.0, 0.0, 4.0, 0.0, 0.0])
    xpx.testing.assert_equal(result, expected_result)

    # empty arrays

    alm = xp.asarray([])
    bl = xp.asarray([])

    result = glass.harmonics.multalm(alm, bl)
    xpx.testing.assert_equal(result, alm)


def test_inverse_transform_healpy(xp: ModuleType) -> None:
    alm = xp.asarray([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    nside = 1
    lmax = 2

    expected = hp._alm2map(alm, nside=nside, lmax=lmax)
    actual = glass.harmonics.inverse_transform(alm, lmax=lmax, nside=nside)
    xpx.testing.assert_equal(actual, expected)


@pytest.mark.parametrize("spin", [1, 2])
def test_inverse_transform_healpy_spin(
    xp: ModuleType,
    spin: int,
) -> None:
    alm = xp.asarray([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    lmax = 2
    nside = 1

    result = hp._alm2map_spin([alm, xp.zeros_like(alm)], nside, spin, lmax)
    expected = result[0] + 1j * result[1]
    actual = glass.harmonics.inverse_transform(alm, lmax=lmax, nside=nside, spin=spin)
    xpx.testing.assert_equal(actual, expected)


def test_transform_healpy(xp: ModuleType) -> None:
    maps = xp.asarray([1.0] * 12)
    lmax = 2

    expected = hp._map2alm(
        maps,
        lmax=lmax,
        # using pixel weights is the default in the transform
        use_pixel_weights=True,
    )
    actual = glass.harmonics.transform(maps, lmax=lmax)
    xpx.testing.assert_equal(actual, expected)
