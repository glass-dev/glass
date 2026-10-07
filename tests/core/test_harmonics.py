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


def test_inverse_transform_healpy() -> None:
    alm = np.asarray([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    lmax = 2
    nside = 1

    expected = hp.alm2map(alm, nside=nside, lmax=lmax)
    actual = glass.harmonics.inverse_transform(alm, lmax=lmax, nside=nside)

    xpx.testing.assert_equal(actual, expected)


@pytest.mark.parametrize("spin", [1, 2])
def test_inverse_transform_healpy_spin(spin: int) -> None:
    alm = np.asarray([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    lmax = 2
    nside = 1

    result = hp.alm2map_spin([alm, np.zeros_like(alm)], nside, spin, lmax)
    expected = result[0] + 1j * result[1]
    actual = glass.harmonics.inverse_transform(alm, lmax=lmax, nside=nside, spin=spin)

    xpx.testing.assert_equal(actual, expected)


@pytest.mark.skipif(not HAVE_S2FFT, reason="test requires s2fft")
def test_inverse_transform_s2fft(jnp: ModuleType) -> None:
    # use real m=0 coefficients and complex coefficients for m > 0
    alm = np.arange(78, dtype=np.complex128) / 78
    alm[12:] += 1j * np.arange(66) / 78
    lmax = 11
    nside = 4

    expected = glass.harmonics.inverse_transform(
        alm,
        lmax=lmax,
        nside=nside,
    )
    actual = glass.harmonics.inverse_transform(
        jnp.asarray(alm),
        lmax=lmax,
        nside=nside,
    )

    assert expected.__array_namespace__() == np
    assert actual.__array_namespace__() == jnp
    xpx.testing.assert_close(actual, jnp.asarray(expected), atol=1e-13, rtol=0)


@pytest.mark.skipif(not HAVE_S2FFT, reason="test requires s2fft")
@pytest.mark.parametrize("spin", [1, 2])
def test_inverse_transform_s2fft_spin(
    jnp: ModuleType,
    spin: int,
) -> None:
    alm = np.asarray([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    lmax = 2
    nside = 1

    expected = glass.harmonics.inverse_transform(
        alm,
        lmax=lmax,
        nside=nside,
        spin=spin,
    )
    actual = glass.harmonics.inverse_transform(
        jnp.asarray(alm),
        lmax=lmax,
        nside=nside,
        spin=spin,
    )

    assert expected.__array_namespace__() == np
    assert actual.__array_namespace__() == jnp
    xpx.testing.assert_close(actual, jnp.asarray(expected), atol=1e-14, rtol=0)


def test_transform_healpy() -> None:
    maps = np.asarray([1.0] * 12)
    lmax = 2
    nside = 1

    expected = hp.map2alm(
        maps,
        lmax=lmax,
        # using pixel weights is the default in the transform
        use_pixel_weights=True,
    )
    actual = glass.harmonics.transform(maps, lmax=lmax, nside=nside)

    xpx.testing.assert_equal(actual, expected)


@pytest.mark.skipif(not HAVE_S2FFT, reason="test requires s2fft")
def test_transform_s2fft(jnp: ModuleType) -> None:
    kappa = np.linspace(-1.0, 1.0, 192)
    lmax = 11
    nside = 4

    expected = glass.harmonics.transform(
        kappa,
        lmax=lmax,
        nside=nside,
    )
    actual = glass.harmonics.transform(
        jnp.asarray(kappa),
        lmax=lmax,
        nside=nside,
    )

    assert expected.__array_namespace__() == np
    assert actual.__array_namespace__() == jnp
    xpx.testing.assert_close(actual, jnp.asarray(expected), atol=1e-14, rtol=0)


@pytest.mark.skipif(not HAVE_S2FFT, reason="test requires s2fft")
def test_transform_s2fft_low_bandlimit(jnp: ModuleType) -> None:
    low_lmax = 2
    nside = 4
    full_lmax = 2 * nside - 1
    kappa = jnp.linspace(-1.0, 1.0, 12 * nside**2)

    low_alms = glass.harmonics.transform(
        kappa,
        lmax=low_lmax,
        nside=nside,
    )
    full_alms = glass.harmonics.transform(
        kappa,
        lmax=full_lmax,
        nside=nside,
    )
    # HEALPix packs alms in blocks of increasing m, then increasing ell
    full_indices = [
        m * (2 * full_lmax + 1 - m) // 2 + ell
        for m in range(low_lmax + 1)
        for ell in range(m, low_lmax + 1)
    ]

    n_low_alms = sum(ell + 1 for ell in range(low_lmax + 1))
    assert low_alms.shape == (n_low_alms,)
    xpx.testing.assert_equal(low_alms, full_alms[jnp.asarray(full_indices)])
