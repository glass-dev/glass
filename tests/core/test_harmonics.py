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


def test_inverse_transform_matches_healpy(xp: ModuleType) -> None:
    if xp.__name__ == "jax.numpy" and not HAVE_S2FFT:
        pytest.skip("test require s2fft")

    nside = 2
    lmax = 2
    alm = xp.asarray([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])

    expected = hp._alm2map(alm, nside=nside, lmax=lmax)
    actual = glass.harmonics.inverse_transform(alm, lmax=lmax, nside=nside)

    xpx.testing.assert_close(actual, expected, atol=1e-14, rtol=0)


@pytest.mark.parametrize("spin", [1, 2])
def test_spin_inverse_transform_matches_healpy(
    spin: int,
    xp: ModuleType,
) -> None:
    if xp.__name__ == "jax.numpy" and not HAVE_S2FFT:
        pytest.skip("test require s2fft")

    nside = 2
    lmax = 2
    alm = xp.asarray([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])

    result = hp._alm2map_spin([alm, xp.zeros_like(alm)], nside, spin, lmax)
    expected = result[0] + 1j * result[1]
    actual = glass.harmonics.inverse_transform(alm, lmax=lmax, nside=nside, spin=spin)

    xpx.testing.assert_close(actual, expected, atol=1e-13, rtol=0)


@pytest.mark.skipif(not HAVE_S2FFT, reason="test requires s2fft")
def test_inverse_transform_matches_numpy_with_s2fft(jnp: ModuleType) -> None:
    nside = 4
    lmax = 11
    # use real m=0 coefficients and complex coefficients for m > 0
    alm = np.arange(78, dtype=np.complex128) / 78
    alm[12:] += 1j * np.arange(66) / 78

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
def test_spin_inverse_transform_matches_numpy_with_s2fft(
    jnp: ModuleType,
    spin: int,
) -> None:
    nside = 2
    lmax = 2
    alm = np.asarray([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])

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
    xpx.testing.assert_close(actual, jnp.asarray(expected), atol=1e-13, rtol=0)


def test_transform_matches_healpy(xp: ModuleType) -> None:
    if xp.__name__ == "jax.numpy" and not HAVE_S2FFT:
        pytest.skip("test require s2fft")

    nside = 2
    lmax = 2
    maps = xp.asarray([1.0] * (12 * nside**2))

    expected = hp._map2alm(maps, lmax=lmax)
    actual = glass.harmonics.transform(maps, lmax=lmax, nside=nside)

    xpx.testing.assert_close(actual, expected, atol=1e-15, rtol=0)


@pytest.mark.skipif(not HAVE_S2FFT, reason="test requires s2fft")
def test_jax_transform_requires_nside_with_s2fft(jnp: ModuleType) -> None:
    lmax = 11
    kappa = np.linspace(-1.0, 1.0, 192)

    with pytest.raises(ValueError, match=r"nside must be specified when using JAX."):
        glass.harmonics.transform(jnp.asarray(kappa), lmax=lmax)


@pytest.mark.skipif(not HAVE_S2FFT, reason="test requires s2fft")
def test_jax_transform_matches_numpy_with_s2fft(jnp: ModuleType) -> None:
    nside = 4
    lmax = 11
    kappa = np.linspace(-1.0, 1.0, 192)

    expected = glass.harmonics.transform(kappa, lmax=lmax)
    actual = glass.harmonics.transform(jnp.asarray(kappa), lmax=lmax, nside=nside)

    assert expected.__array_namespace__() == np
    assert actual.__array_namespace__() == jnp
    xpx.testing.assert_close(actual, jnp.asarray(expected), atol=1e-14, rtol=0)


@pytest.mark.skipif(not HAVE_S2FFT, reason="test requires s2fft")
def test_low_bandlimit_transform_matches_full_transform(jnp: ModuleType) -> None:
    nside = 4
    low_lmax = 2
    high_lmax = 2 * nside - 1
    kappa = jnp.linspace(-1.0, 1.0, 12 * nside**2)

    low_alms = glass.harmonics.transform(
        kappa,
        lmax=low_lmax,
        nside=nside,
    )
    full_alms = glass.harmonics.transform(
        kappa,
        lmax=high_lmax,
        nside=nside,
    )
    # HEALPix packs alms in blocks of increasing m, then increasing ell
    full_indices = [
        m * (2 * high_lmax + 1 - m) // 2 + ell
        for m in range(low_lmax + 1)
        for ell in range(m, low_lmax + 1)
    ]

    n_low_alms = sum(ell + 1 for ell in range(low_lmax + 1))
    assert low_alms.shape == (n_low_alms,)
    xpx.testing.assert_equal(low_alms, full_alms[jnp.asarray(full_indices)])
