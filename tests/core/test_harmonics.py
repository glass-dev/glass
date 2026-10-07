from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

import array_api_extra as xpx

import glass.harmonics
import glass.healpix as hp
from tests._optional_dependencies import HAVE_S2FFT

if TYPE_CHECKING:
    from types import ModuleType

    from glass._types import UnifiedGenerator
    from tests.fixtures.helper_classes import HealpixInputs


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

    expected = hp.alm2map(alm, nside=nside, lmax=lmax)
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

    result = hp.alm2map_spin([alm, xp.zeros_like(alm)], nside, spin, lmax)
    expected = result[0] + 1j * result[1]
    actual = glass.harmonics.inverse_transform(alm, lmax=lmax, nside=nside, spin=spin)
    xpx.testing.assert_equal(actual, expected)


@pytest.mark.skipif(not HAVE_S2FFT, reason="test requires s2fft")
def test_inverse_transform_s2fft(
    healpix_inputs: type[HealpixInputs],
    jnp: ModuleType,
    rng: UnifiedGenerator,
) -> None:
    alm = healpix_inputs.alm(rng=rng)

    expected = glass.harmonics.inverse_transform(
        alm,
        lmax=healpix_inputs.lmax,
        nside=healpix_inputs.nside,
    )
    actual = glass.harmonics.inverse_transform(
        jnp.asarray(alm),
        lmax=healpix_inputs.lmax,
        nside=healpix_inputs.nside,
    )
    xpx.testing.assert_close(actual, jnp.asarray(expected), atol=1e-13, rtol=0)


@pytest.mark.skipif(not HAVE_S2FFT, reason="test requires s2fft")
@pytest.mark.parametrize("spin", [1, 2])
def test_inverse_transform_s2fft_spin(
    healpix_inputs: type[HealpixInputs],
    jnp: ModuleType,
    rng: UnifiedGenerator,
    spin: int,
) -> None:
    # keep well below the map resolution to isolate spin conventions from
    # S2FFT's larger HEALPix errors near the fixture's full bandlimit
    lmax = 2
    alm_size = (lmax + 1) * (lmax + 2) // 2
    alm = healpix_inputs.alm(rng=rng)[:alm_size]

    expected = glass.harmonics.inverse_transform(
        alm,
        lmax=lmax,
        nside=healpix_inputs.nside,
        spin=spin,
    )
    actual = glass.harmonics.inverse_transform(
        jnp.asarray(alm),
        lmax=lmax,
        nside=healpix_inputs.nside,
        spin=spin,
    )
    xpx.testing.assert_close(actual, jnp.asarray(expected), atol=1e-14, rtol=0)


def test_transform_healpy() -> None:
    maps = np.asarray([1.0] * 12)
    nside = 1
    lmax = 2

    expected = hp.map2alm(maps, lmax=lmax)
    actual = glass.harmonics.transform(maps, lmax=lmax, nside=nside)
    xpx.testing.assert_equal(actual, expected)
