from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

import array_api_extra as xpx

import glass.harmonics

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


def test_inverse_transform_default_spin(xp: ModuleType) -> None:
    alm = xp.asarray([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    nside = 1
    lmax = 2

    result = glass.harmonics.inverse_transform(alm, lmax=lmax, nside=nside)
    assert result.shape[0] == 12
    assert xp.isdtype(result.dtype, "real floating")


def test_inverse_transform_spin_1(xp: ModuleType) -> None:
    alm = xp.asarray([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    nside = 1
    lmax = 2

    # test with spin 1
    result = glass.harmonics.inverse_transform(alm, lmax=lmax, nside=nside, spin=1)
    assert result.shape[0] == 12
    assert xp.isdtype(result.dtype, "complex floating")


def test_inverse_transform_spin_2(xp: ModuleType) -> None:
    alm = xp.asarray([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    nside = 1
    lmax = 2

    result = glass.harmonics.inverse_transform(alm, lmax=lmax, nside=nside, spin=2)
    assert result.shape[0] == 12
    assert xp.isdtype(result.dtype, "complex floating")


def test_inverse_transform_spin_0(xp: ModuleType) -> None:
    alm = xp.asarray([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    nside = 1
    lmax = 2

    result = glass.harmonics.inverse_transform(alm, lmax=lmax, nside=nside, spin=0)
    assert result.shape[0] == 12
    assert xp.isdtype(result.dtype, "real floating")


def test_transform(xp: ModuleType) -> None:
    maps = xp.asarray([1.0] * 12)
    lmax = 2

    result = glass.harmonics.transform(maps, lmax=lmax)
    assert result.shape[0] == 6
    assert xp.isdtype(result.dtype, "complex floating")
